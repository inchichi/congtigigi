import { openProjectDirectory } from './openProjectDirectory'
import {
  buildTileClusterEntities,
  findAllStyleTargetCells,
  findFileByRelativeSource,
  findObjectKindCells,
  findTileClusterDetail,
  isTileClusterEntity,
  loadGame,
  resolveRelativePath,
  type GameFile,
  type LoadedGame,
  type LoadedGameMap
} from './loadGame'
import { extractTmxTilesetImageInfo } from './tmxTileEntities'
import { analyzeGame, type GameAnalysis } from './analyzeGame'
import { extractTmxLayerNames, extractTmxObjects, type TmxObject } from './tmxObjects'
import { readLocalStorage, writeLocalStorage } from './safeStorage'
import { ANTHROPIC_MODEL } from './anthropicGenerate'
import {
  PROVIDER_LABEL,
  PROVIDER_MODELS,
  detectProvider,
  getProviderModel,
  setProviderModel,
  validateApiKey,
  type LlmProvider
} from './llmProvider'
import {
  appendEventEvaluation,
  clearEventEvaluations,
  loadEventEvaluations,
  type EventEvaluation,
  type EventEvaluationVerdict
} from './eventEvaluator'
import { buildSessionMetrics, type SessionGenerationTally } from './sessionMetrics'
import {
  createStyleTransferModal,
  type StyleTransferMapObject
} from './createStyleTransferModal'
import type { GameEntity, GenerationResult } from './gameAdapter'

// 하드코딩 어댑터가 엔티티를 못 찾은 미지의 게임을, LLM 분석이 찾은 editable 그룹으로 채운다.
const buildEntitiesFromAnalysis = (
  files: GameFile[],
  analysis: GameAnalysis
): LoadedGameMap[] => {
  const editableKindByGroup = new Map<string, string>()
  for (const entityGroup of analysis.entity_groups) {
    if (entityGroup.editable) {
      editableKindByGroup.set(entityGroup.group, entityGroup.kind)
    }
  }

  return files
    .filter((file) => file.name.endsWith('.tmx'))
    .map((file) => {
      const id = file.name.replace(/\.tmx$/u, '')
      // 한 맵이 깨졌다고 분석 기반 트리 재구성을 통째로 죽이지 않는다(loadGame과 동일한 격리).
      let objects: TmxObject[] = []
      try {
        objects = extractTmxObjects(file.text)
      } catch {
        // 파싱 실패 맵 → 엔티티 0개
      }
      const entities = objects
        .filter(
          (object) =>
            editableKindByGroup.has(object.group) && object.name.length > 0
        )
        .map((object) => ({
          id: `${object.group}-${object.id}`,
          name: object.name,
          kind: editableKindByGroup.get(object.group) ?? 'entity',
          mapId: id
        }))
        // 분석으로 트리를 갈아끼워도 타일 군집(보기 전용 구조물)은 유지한다 — loadGame과 동일.
        .concat(buildTileClusterEntities(id, file, files, objects))
      return { id, name: id, file: file.path, entities, layers: extractTmxLayerNames(file.text) }
    })
}

type CreateEditorAppInput = {
  mountElement: HTMLElement
  initialFiles: GameFile[]
  gamePreviewUrl: string
}

const API_KEY_STORAGE_KEY = 'my-sample-rpg:anthropic-api-key'
const MODEL_STORAGE_PREFIX = 'my-sample-rpg:model:'

const KIND_ICON: Record<string, string> = {
  npc: '👤',
  monster: '👹',
  sign: '🪧',
  portal: '🚪',
  chest: '📦',
  loot: '💰',
  building: '🏠',
  character: '🧍',
  // 타일 군집(tmxTileEntities)으로 인식되는 종류들.
  tent: '⛺',
  clocktower: '🕰',
  fountain: '⛲',
  lamp: '🏮',
  banner: '🚩',
  tree: '🌳',
  hedge: '🌿',
  flower: '🌸',
  prop: '🧺',
  rock: '🪨',
  stairs: '🪜',
  wall: '🧱',
  window: '🪟'
}

// 보기 전용 요소(몬스터·표지판·포털 등)에 붙는 짧은 한국어 종류 라벨.
const KIND_LABEL: Record<string, string> = {
  npc: 'NPC',
  monster: '몬스터',
  sign: '표지판',
  portal: '포털',
  chest: '상자',
  loot: '전리품',
  building: '건물',
  object: '객체',
  character: '캐릭터',
  // 타일 군집(tmxTileEntities)으로 인식되는 종류들.
  tent: '천막',
  clocktower: '시계탑',
  fountain: '분수',
  lamp: '가로등',
  banner: '깃발',
  tree: '나무',
  hedge: '생울타리',
  flower: '화단',
  prop: '소품',
  rock: '바위',
  stairs: '계단',
  wall: '벽',
  window: '창문'
}

// 트리 그룹핑용 종류 정규화. LLM 분석이 'NPC'처럼 대소문자를 섞어 줄 수 있어 소문자로 맞추고,
// enemy는 라벨·아이콘이 '몬스터'로 같아 monster 그룹에 합친다(그래서 위 맵에는 enemy 키가 없다).
const groupKindOf = (kind: string): string => {
  const normalized = kind.trim().toLowerCase()
  return normalized === 'enemy' ? 'monster' : normalized
}

// 부분 스타일 변환 대상에서 제외할 종류: 캐릭터·표지판·포털은 점 객체(스프라이트)라
// 타일셋 패치 방식의 대상이 아니다. NPC는 LLM 생성 대상으로 이미 클릭이 점유돼 있다.
const STYLE_TARGET_EXCLUDED_KINDS = new Set(['npc', 'monster', 'sign', 'portal'])

const el = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string
): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag)
  node.className = className
  if (text !== undefined) {
    node.textContent = text
  }
  return node
}

// ---- 디자인 토큰 ----
// 타입 스케일은 3단으로 고정한다: LABEL(11px 섹션 eyebrow) · text-xs(메타) · text-sm(본문/컨트롤).
// 예전엔 0.65/0.7/0.72/0.8rem 등이 뒤섞여 글자 크기가 들쭉날쭉했다.
const LABEL =
  'text-[11px] font-semibold uppercase tracking-wider text-zinc-400'
const CARD =
  'rounded-xl border border-white/10 bg-white/[0.03] p-4 flex flex-col gap-2'
const FIELD_INPUT =
  'w-full rounded-lg border border-white/15 bg-black/40 px-3 py-2.5 text-sm text-zinc-100 outline-none transition placeholder:text-zinc-500 focus:border-indigo-500/60 focus:ring-2 focus:ring-indigo-500/25'
const PRIMARY_BUTTON =
  'rounded-lg px-4 py-2 bg-indigo-500 text-white text-sm font-medium shadow-sm shadow-indigo-500/30 transition hover:bg-indigo-400 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none'
const GHOST_BUTTON =
  'rounded-lg px-3.5 py-2 bg-white/[0.04] text-zinc-300 text-sm border border-white/10 transition hover:bg-white/[0.08] hover:text-zinc-100 hover:border-white/20 disabled:opacity-40 disabled:cursor-not-allowed'
const ENTITY_BASE =
  'truncate text-left rounded-lg px-2.5 py-2 text-sm text-zinc-300 transition hover:bg-white/[0.06] hover:text-zinc-100'
const ENTITY_ACTIVE =
  'truncate text-left rounded-lg px-2.5 py-2 text-sm bg-indigo-500/15 text-indigo-100 ring-1 ring-inset ring-indigo-500/30 transition'
const ENTITY_GROUP_HEADER =
  'w-full flex items-center gap-1.5 text-left rounded-lg px-2.5 py-2 text-sm text-zinc-200 font-medium transition hover:bg-white/[0.06] hover:text-zinc-100'

export const createEditorApp = ({
  mountElement,
  initialFiles,
  gamePreviewUrl
}: CreateEditorAppInput): void => {
  let game: LoadedGame = loadGame(initialFiles)
  let currentFiles: GameFile[] = initialFiles
  let apiKey = readLocalStorage(API_KEY_STORAGE_KEY) ?? ''
  let selectedEntity: GameEntity | undefined
  let currentResult: GenerationResult | undefined
  let currentAnalysis: GameAnalysis | undefined
  let isGenerating = false
  let isAnalyzing = false
  let entityButtons: Array<{ entity: GameEntity; node: HTMLButtonElement }> = []
  // 라이브 게임(iframe)이 보고한 현재 맵 id. 트리를 이 맵의 요소만으로 좁힌다(showAllMaps면 전부).
  // 게임의 sceneId 와 에디터 map.id(=tmx 파일명)가 같아 직접 비교한다. 게임이 맵을 바꿀 때마다
  // 'game:scene-changed' 메시지로 갱신된다.
  let currentMapId: string | undefined
  let showAllMaps = false
  // 트리의 종류별 그룹(NPC/몬스터 등) 펼침 상태. 키는 `${mapId}:${kind}` — 트리를 다시 그려도 유지된다.
  // 기본은 접힘: 요소를 쭉 나열하면 목록이 길어 보기 불편하다는 피드백에 따른 동작.
  const expandedGroups = new Set<string>()
  // 세션 내 생성 결과 누적(최신 우선, 최대 10개). 데모에서 여러 생성을 비교·재선택하려는 용도.
  const HISTORY_LIMIT = 10
  let history: Array<{ n: number; result: GenerationResult }> = []
  let historyCounter = 0
  // Evaluator(사람 이진 평가, 회의 #3/#7): 생성/검증과 분리된 품질 판정. 단일 지표 acceptance_rate.
  // 평가 판정은 결과 객체 동일성으로 기억한다(단일 슬롯이면 히스토리에서 옛 결과를 다시 골라 재평가 →
  // 중복 집계되어 acceptance_rate가 오염됨). WeakMap이라 참조가 사라진 결과는 알아서 GC된다.
  let evaluations: EventEvaluation[] = loadEventEvaluations()
  let verdictByResult = new WeakMap<GenerationResult, EventEvaluationVerdict>()
  // 이번 세션 집계: 생성 수 + Validator 통과 수. 프로젝트를 바꾸면 초기화한다.
  let sessionTally: SessionGenerationTally = { generations: 0, validatorPasses: 0 }

  // ---------- shell ----------
  // w-screen이 아니라 w-full — 100vw는 세로 스크롤바 폭을 포함해 가로 스크롤을 만든다.
  const root = el('div', 'h-screen w-full flex flex-col bg-zinc-950 text-zinc-100 overflow-hidden')

  const header = el('header', 'h-12 shrink-0 flex items-center justify-between px-4 border-b border-white/10 bg-zinc-900/60')
  const brand = el('div', 'flex items-center gap-2')
  brand.append(
    el('span', 'w-2.5 h-2.5 rounded-full bg-indigo-500'),
    el('span', 'text-sm font-semibold tracking-tight whitespace-nowrap', 'Scenario Editor'),
    // 게임 이름·모델 배지는 좁은 화면에선 숨긴다 — 헤더가 넘치면 설정 버튼이 밀려난다.
    el('span', 'hidden sm:inline text-xs text-zinc-600', '·')
  )
  const gameLabel = el('span', 'hidden sm:inline text-xs text-zinc-400 truncate', game.adapter.name)
  brand.append(gameLabel)
  // 모델 배지 — 입력한 키의 provider(Claude/GPT)에 따라 동적으로 갱신된다.
  const modelBadge = el(
    'span',
    'hidden md:inline-block text-[11px] rounded-full px-2 py-0.5 bg-indigo-500/10 text-indigo-300 border border-indigo-500/20',
    `Claude · ${ANTHROPIC_MODEL}`
  )
  brand.append(modelBadge)
  const connection = el('div', 'flex items-center gap-2 text-xs text-zinc-400')
  const connectionDot = el('span', 'w-2 h-2 rounded-full bg-zinc-600')
  const connectionLabel = el('span', '', '게임 로딩...')
  connection.append(connectionDot, connectionLabel)
  const settingsButton = el('button', 'rounded-lg px-2.5 py-1 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100', '⚙ 설정') as HTMLButtonElement
  settingsButton.type = 'button'
  // 음소거 토글 — 소리는 게임(iframe)이 내므로 postMessage로 즉시 끄고, 게임 리로드/에디터
  // 재시작에도 유지되도록 게임의 오디오 설정(localStorage, 같은 origin 공유)에 함께 기록한다.
  const AUDIO_SETTINGS_KEY = 'my-sample-rpg:audio-settings'
  const readStoredMuted = (): boolean => {
    try {
      const settings = JSON.parse(readLocalStorage(AUDIO_SETTINGS_KEY) ?? '{}') as { isMuted?: unknown }
      return settings.isMuted === true
    } catch {
      return false
    }
  }
  let isGameMuted = readStoredMuted()
  const muteButton = el('button', 'rounded-lg px-2.5 py-1 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100') as HTMLButtonElement
  muteButton.type = 'button'
  const renderMuteButton = (): void => {
    muteButton.textContent = isGameMuted ? '🔇' : '🔊'
    muteButton.title = isGameMuted ? '음소거 해제' : '게임 소리 끄기'
    muteButton.setAttribute('aria-pressed', String(isGameMuted))
  }
  renderMuteButton()
  muteButton.addEventListener('click', () => {
    isGameMuted = !isGameMuted
    let settings: Record<string, unknown> = {}
    try {
      settings = JSON.parse(readLocalStorage(AUDIO_SETTINGS_KEY) ?? '{}') as Record<string, unknown>
    } catch {
      settings = {}
    }
    writeLocalStorage(AUDIO_SETTINGS_KEY, JSON.stringify({ ...settings, isMuted: isGameMuted }))
    iframe.contentWindow?.postMessage({ type: 'editor:set-mute', isMuted: isGameMuted }, '*')
    renderMuteButton()
  })

  // AdaIN 스타일 트랜스퍼 — 로컬 Python 서비스(/api/style 프록시)로 이미지를 변환하는 독립 모달.
  const styleTransfer = createStyleTransferModal()
  const headerRight = el('div', 'flex items-center gap-3')
  headerRight.append(muteButton, styleTransfer.openButton, settingsButton, connection)
  header.append(brand, headerRight)

  // LLM 챗 스타일 배치: 가운데가 라이브 게임(위 가득) + 프롬프트(아래), 오른쪽이 생성 결과.
  // 3열은 md(≥768px)부터 바로 적용한다 — 이전엔 lg부터여서, 브라우저 줌을 쓰는 일반 노트북
  // 창이 "결과가 하단 전폭" 배치로 떨어지며 게임 세로 공간을 잃었다(게임이 납작한 띠가 됨).
  //  - md(≥768px): [엔티티 트리 | 게임+프롬프트 | 생성 결과] 3열 (lg부터는 사이드가 약간 넓어짐)
  //  - 그 미만: 트리 → 게임+프롬프트 → 생성 결과 세로 스택
  const body = el(
    'div',
    'flex-1 min-h-0 grid ' +
      'grid-cols-1 grid-rows-[auto_minmax(0,1.4fr)_minmax(0,1fr)] [grid-template-areas:"tree""main""side"] ' +
      // 결과 사이드는 코드 확인용이라 좁게 잡는다(26%/30%는 게임 화면을 잡아먹는다는 피드백).
      'md:grid-cols-[minmax(170px,210px)_minmax(0,1fr)_minmax(220px,20%)] md:grid-rows-[minmax(0,1fr)] md:[grid-template-areas:"tree_main_side"] ' +
      'lg:grid-cols-[minmax(200px,250px)_minmax(0,1fr)_minmax(250px,22%)]'
  )

  // ---------- left: project tree ----------
  // 스택 배치(<md)에선 아래 경계선 + 높이 제한(목록이 길면 자체 스크롤), 옆 배치에선 오른쪽 경계선.
  const tree = el(
    'aside',
    '[grid-area:tree] min-w-0 max-h-[35vh] border-b md:max-h-none md:border-b-0 md:border-r border-white/10 flex flex-col min-h-0'
  )
  const treeHeader = el('div', 'p-3 border-b border-white/10 flex flex-col gap-2')
  const openButton = el('button', 'rounded-lg px-3 py-2 bg-white/5 text-sm text-zinc-200 text-left transition hover:bg-white/10', '📂 게임 폴더 열기') as HTMLButtonElement
  openButton.type = 'button'
  const analyzeButton = el('button', 'rounded-lg px-3 py-2 bg-indigo-500/10 text-sm text-indigo-200 text-left transition hover:bg-indigo-500/20 disabled:opacity-50', '🔍 LLM 게임 분석') as HTMLButtonElement
  analyzeButton.type = 'button'
  const resetButton = el('button', 'rounded-lg px-3 py-1.5 bg-white/5 text-xs text-zinc-400 text-left transition hover:bg-white/10 hover:text-zinc-200', '🏠 내 게임으로 복귀') as HTMLButtonElement
  resetButton.type = 'button'
  // 프로젝트 버튼(폴더 열기/분석/복귀)은 설정 모달로 이동했다. 사이드바는 엔티티 목록만.
  const treeHeaderTop = el('div', 'flex items-center justify-between gap-2')
  treeHeaderTop.append(el('div', LABEL, '엔티티'))
  // 라이브 게임이 맵을 보고한 뒤에만 의미가 있는 토글(현재 맵만 ↔ 전체 맵). 그 전엔 숨긴다.
  const mapFilterToggle = el('button', 'text-[11px] text-zinc-500 transition hover:text-zinc-300', '전체 보기') as HTMLButtonElement
  mapFilterToggle.type = 'button'
  mapFilterToggle.hidden = true
  treeHeaderTop.append(mapFilterToggle)
  // 게임과의 동기화 상태(현재 맵 이름)를 보여주는 줄. 연결 전엔 대기 메시지.
  const treeSyncLine = el('div', 'text-[11px] text-zinc-500', '게임과 연결 대기 중…')
  treeHeader.append(treeHeaderTop, treeSyncLine)
  const treeList = el('div', 'flex-1 overflow-auto p-3 flex flex-col gap-3')
  tree.append(treeHeader, treeList)

  // ---------- center: 라이브 게임(위) + 프롬프트 컴포저(아래) ----------
  // min-w-0: grid 자식의 기본 min-width:auto 때문에 내용이 열을 밀어내는 것 방지.
  const center = el('main', '[grid-area:main] min-w-0 min-h-0 flex flex-col')
  const targetLine = el('div', 'text-sm text-zinc-400')
  const analysisPanel = el('div', 'rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-4 flex flex-col gap-1.5')
  analysisPanel.hidden = true
  const supportNote = el('div', 'rounded-xl border border-amber-500/30 bg-amber-500/10 px-3.5 py-2.5 text-xs text-amber-200')

  const apiKeyField = el('label', 'flex flex-col gap-1.5')
  apiKeyField.append(el('span', LABEL, 'API 키 — Claude 또는 GPT (자동 감지)'))
  const apiKeyInput = el('input', FIELD_INPUT) as HTMLInputElement
  apiKeyInput.type = 'password'
  apiKeyInput.placeholder = 'sk-ant-… (Claude)  또는  sk-… (GPT)'
  apiKeyInput.autocomplete = 'off'
  apiKeyInput.value = apiKey
  apiKeyField.append(apiKeyInput)
  // 키 유효성 피드백(입력 시 디바운스로 갱신). 빈 문자열이면 자리만 차지하지 않게 둔다.
  const apiKeyStatus = el('span', 'text-xs text-zinc-500', '')
  apiKeyField.append(apiKeyStatus)

  // 모델 선택 — 키는 모델을 정하지 않으므로, 감지된 provider의 모델 중에서 고른다(저장됨).
  const modelField = el('label', 'flex flex-col gap-1.5')
  modelField.append(el('span', LABEL, '모델'))
  const modelSelect = el('select', `${FIELD_INPUT} cursor-pointer`) as HTMLSelectElement
  modelField.append(modelSelect)

  const promptField = el('label', 'flex flex-col gap-1.5')
  promptField.append(el('span', LABEL, '자연어 프롬프트  ·  ⌘/Ctrl+Enter로 생성'))
  // 챗 컴포저처럼 기본 2줄 높이 — 필요하면 손잡이로 늘릴 수 있다(resize-y).
  const promptInput = el('textarea', `${FIELD_INPUT} min-h-[60px] resize-y`) as HTMLTextAreaElement
  promptInput.placeholder = '예: 대장장이가 새로 만든 검을 자랑하는 대화'
  promptField.append(promptInput)

  const actions = el('div', 'flex flex-wrap items-center gap-2')
  const generateButton = el('button', PRIMARY_BUTTON, '생성') as HTMLButtonElement
  generateButton.type = 'button'
  const applyButton = el('button', GHOST_BUTTON, '게임에 적용') as HTMLButtonElement
  applyButton.type = 'button'
  const copyButton = el('button', GHOST_BUTTON, '⧉ 복사') as HTMLButtonElement
  copyButton.type = 'button'
  const exportButton = el('button', GHOST_BUTTON, '↓ 내보내기') as HTMLButtonElement
  exportButton.type = 'button'
  actions.append(generateButton, applyButton, copyButton, exportButton)

  const status = el('div', 'text-sm text-zinc-400 min-h-[1.25rem]')
  const validationLine = el('div', 'text-xs')
  validationLine.hidden = true

  const resultWrap = el('div', 'flex flex-col gap-1.5 flex-1 min-h-0')
  resultWrap.append(el('span', LABEL, '생성 결과'))
  const result = el('pre', 'm-0 flex-1 min-h-0 overflow-auto rounded-lg border border-white/10 bg-black/40 p-3.5 text-xs leading-relaxed text-zinc-300 whitespace-pre-wrap break-words')
  resultWrap.append(result)

  // ---------- Evaluator (사람 이진 평가) ----------
  const evaluationWrap = el('div', CARD)
  evaluationWrap.hidden = true
  // flex-wrap: 좁은 사이드바에선 지표 줄이 라벨 옆에 끼어 두 줄 컬럼으로 뭉개지는 대신 제 줄로 내려간다.
  const evaluationTop = el('div', 'flex flex-wrap items-center justify-between gap-x-2 gap-y-1')
  evaluationTop.append(
    el('span', LABEL, 'Evaluator · 사람 이진 평가')
  )
  const acceptanceStat = el('span', 'text-xs text-zinc-400')
  evaluationTop.append(acceptanceStat)
  const evaluationButtons = el('div', 'flex flex-wrap items-center gap-2')
  const acceptButton = el('button', GHOST_BUTTON, '👍 수용') as HTMLButtonElement
  acceptButton.type = 'button'
  const rejectButton = el('button', GHOST_BUTTON, '👎 거부') as HTMLButtonElement
  rejectButton.type = 'button'
  const evaluationVerdict = el('span', 'text-xs flex-1')
  const resetEvaluationsButton = el('button', 'text-[11px] text-zinc-500 transition hover:text-zinc-300', '누적 기록 초기화') as HTMLButtonElement
  resetEvaluationsButton.type = 'button'
  evaluationButtons.append(acceptButton, rejectButton, evaluationVerdict, resetEvaluationsButton)
  evaluationWrap.append(evaluationTop, evaluationButtons)

  const historyWrap = el('div', 'flex flex-col gap-1.5')
  historyWrap.hidden = true
  const historyHeader = el('div', 'flex items-center justify-between')
  historyHeader.append(
    el('span', LABEL, '생성 히스토리')
  )
  const clearHistoryButton = el('button', 'text-[11px] text-zinc-500 transition hover:text-zinc-300', '비우기') as HTMLButtonElement
  clearHistoryButton.type = 'button'
  historyHeader.append(clearHistoryButton)
  const historyList = el('div', 'flex flex-col gap-1')
  historyWrap.append(historyHeader, historyList)

  // ---------- center 상단: live game preview ----------
  // min-h 바닥: 어떤 창 크기에서도 게임이 HUD만 보이는 납작한 띠로 짓눌리지 않게 한다.
  const preview = el('section', 'flex-1 min-h-[200px] md:min-h-[300px] min-w-0 flex flex-col')
  const previewBar = el('div', 'h-9 shrink-0 flex items-center justify-between gap-2 px-3 border-b border-white/10 bg-zinc-900/40 min-w-0')
  previewBar.append(el('span', 'text-xs text-zinc-400 truncate', '🎮 라이브 게임 (실제 게임 실행 중)'))
  const previewActions = el('div', 'flex items-center gap-2 shrink-0')
  // 맵 전환 — 프리뷰는 항상 my-sample-rpg를 실행하므로 그 게임의 씬(마을/사냥터/동굴)을 바꾼다.
  const previewScenes = [
    { id: 'town', label: '마을' },
    { id: 'hunting-ground', label: '사냥터' },
    { id: 'cave', label: '동굴' }
  ]
  const mapSwitcher = el('div', 'flex items-center gap-1')
  const popoutButton = el('button', 'text-xs text-zinc-400 transition hover:text-zinc-100', '↗ 새 창') as HTMLButtonElement
  popoutButton.type = 'button'
  const reloadButton = el('button', 'text-xs text-zinc-400 transition hover:text-zinc-100', '↻ 새로고침') as HTMLButtonElement
  reloadButton.type = 'button'
  previewActions.append(mapSwitcher, popoutButton, reloadButton)
  previewBar.append(previewActions)
  const iframe = el('iframe', 'flex-1 w-full border-0 bg-black') as HTMLIFrameElement
  iframe.src = gamePreviewUrl
  iframe.title = '게임 프리뷰'
  iframe.addEventListener('load', () => {
    connectionDot.className = 'w-2 h-2 rounded-full bg-emerald-400'
    connectionLabel.textContent = '게임 연결됨'
  })
  // iframe 정의 후 맵 버튼을 채운다 — 클릭하면 게임에 씬 전환 메시지를 보낸다.
  mapSwitcher.append(
    ...previewScenes.map((scene) => {
      const button = el(
        'button',
        'text-[11px] rounded px-2 py-0.5 bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100',
        scene.label
      ) as HTMLButtonElement
      button.type = 'button'
      button.addEventListener('click', () => {
        iframe.contentWindow?.postMessage(
          { type: 'editor:switch-scene', sceneId: scene.id },
          '*'
        )
      })
      return button
    })
  )
  preview.append(previewBar, iframe)

  // ---------- center 하단: 프롬프트 컴포저 (LLM 챗의 입력창처럼 게임 바로 아래) ----------
  // 챗 입력창처럼 낮게 유지한다 — 컴포저가 높을수록 게임이 그만큼 낮아진다.
  // max-h+스크롤: 창이 낮을 때 컴포저가 게임 영역을 통째로 밀어내지 않게 한다.
  const composer = el('div', 'shrink-0 max-h-[45%] overflow-y-auto border-t border-white/10 p-3 flex flex-col gap-2')
  composer.append(targetLine, supportNote, promptField, actions, status)
  center.append(preview, composer)

  // ---------- right: 생성 결과 사이드바 ----------
  // 옆 배치(md부터, body 그리드와 동일 기준)에선 왼쪽 경계선, 아래 배치(<md)에선 위 경계선.
  const side = el(
    'aside',
    '[grid-area:side] min-w-0 min-h-0 overflow-y-auto p-4 flex flex-col gap-4 border-t md:border-t-0 md:border-l border-white/10'
  )
  side.append(analysisPanel, validationLine, resultWrap, evaluationWrap, historyWrap)

  body.append(tree, center, side)
  // ---------- settings modal (헤더 ⚙) ----------
  // API 키·폴더 열기·분석·복귀는 상시 노출 대신 여기로 모은다. 메인은 편집에 집중.
  const settingsBackdrop = el('div', 'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4')
  // 숨김은 hidden 속성 대신 인라인 display로 제어한다 — `flex` 클래스의 display:flex가 [hidden]을
  // 덮어써 안 닫히는 사고를 막는다(인라인 스타일이 항상 이긴다).
  settingsBackdrop.style.display = 'none'
  const settingsPanel = el('div', 'w-full max-w-md rounded-2xl border border-white/10 bg-zinc-900 p-5 flex flex-col gap-4 shadow-2xl')
  const settingsTop = el('div', 'flex items-center justify-between')
  settingsTop.append(el('span', 'text-sm font-semibold tracking-tight', '⚙ 설정'))
  const settingsClose = el('button', 'text-zinc-500 text-sm transition hover:text-zinc-200', '✕') as HTMLButtonElement
  settingsClose.type = 'button'
  settingsTop.append(settingsClose)
  const projectControls = el('div', 'flex flex-col gap-2')
  projectControls.append(el('div', LABEL, '프로젝트'), openButton, analyzeButton, resetButton)
  settingsPanel.append(settingsTop, apiKeyField, modelField, el('div', 'h-px bg-white/10'), projectControls)
  settingsBackdrop.append(settingsPanel)

  const closeSettings = (): void => {
    settingsBackdrop.style.display = 'none'
  }
  settingsButton.addEventListener('click', () => {
    settingsBackdrop.style.display = 'flex'
  })
  settingsClose.addEventListener('click', closeSettings)
  settingsBackdrop.addEventListener('click', (event) => {
    // 패널 바깥(백드롭)을 클릭했을 때만 닫는다.
    if (event.target === settingsBackdrop) {
      closeSettings()
    }
  })
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && settingsBackdrop.style.display !== 'none') {
      closeSettings()
    }
  })

  root.append(header, body, settingsBackdrop, styleTransfer.backdrop)
  mountElement.append(root)

  // ---------- behavior ----------
  const setStatus = (message: string): void => {
    status.textContent = message
  }

  // 파싱 실패한 맵이 있으면 상태 메시지 끝에 붙일 경고(없으면 빈 문자열). loadGame이 throw 대신
  // game.parseErrors로 모아주므로, 에디터가 통째로 안 뜨는 일 없이 실패를 사용자에게 알린다.
  const parseErrorNote = (): string =>
    game.parseErrors.length > 0
      ? ` · ⚠️ 파싱 실패 맵 ${game.parseErrors.length}개: ${game.parseErrors.join(', ')}`
      : ''

  const renderAnalysis = (): void => {
    if (!currentAnalysis) {
      analysisPanel.hidden = true
      return
    }

    analysisPanel.hidden = false
    const analysis = currentAnalysis
    analysisPanel.replaceChildren(
      el('div', 'text-[11px] font-semibold uppercase tracking-wider text-indigo-300', '🔍 LLM 게임 분석'),
      el('div', 'text-sm text-zinc-100 font-medium', `${analysis.game_name} · ${analysis.engine}`),
      el('div', 'text-xs text-zinc-400', `콘텐츠 모델: ${analysis.content_model}`),
      el('div', 'text-xs text-zinc-400', `적용 전략: ${analysis.apply_strategy}`),
      ...analysis.entity_groups.map((entityGroup) =>
        el(
          'div',
          'text-xs text-zinc-500',
          `• ${entityGroup.group} → ${entityGroup.kind}${entityGroup.editable ? ' (편집 가능)' : ''}`
        )
      )
    )
  }

  const runAnalyze = async (): Promise<void> => {
    if (isAnalyzing) {
      return
    }

    if (apiKey.trim().length === 0) {
      setStatus('분석하려면 먼저 Claude(Anthropic) API 키를 입력하세요.')
      return
    }

    isAnalyzing = true
    analyzeButton.disabled = true
    analyzeButton.textContent = '분석 중...'
    setStatus('LLM이 게임을 분석 중...')

    const filesAtStart = currentFiles
    try {
      const analysis = await analyzeGame({ apiKey: apiKey.trim(), files: filesAtStart })
      // 분석 중 다른 프로젝트를 열었으면 이 결과는 버린다(레이스 방지).
      if (currentFiles !== filesAtStart) {
        return
      }
      currentAnalysis = analysis
      // 하드코딩 어댑터가 엔티티를 못 찾았으면(미지의 게임), 분석 결과로 트리를 채운다.
      // 타일 군집(보기 전용 구조물)은 세지 않는다 — 장식만 있는 미지의 게임에서 분석 결과가
      // 트리에 반영되지 못하게 막아버린다.
      const totalEntities = game.maps.reduce(
        (sum, map) =>
          sum + map.entities.filter((entity) => !isTileClusterEntity(entity)).length,
        0
      )
      if (totalEntities === 0) {
        game = { ...game, maps: buildEntitiesFromAnalysis(filesAtStart, analysis) }
        // 트리를 새 엔티티로 갈아끼우므로, 이전 대상의 생성 결과·히스토리·세션 집계는 모두 무효 처리한다
        // (open/reset과 동일한 초기화 묶음 — 지표가 폐기된 생성을 계속 세지 않도록).
        selectedEntity = undefined
        currentResult = undefined
        history = []
        historyCounter = 0
        sessionTally = { generations: 0, validatorPasses: 0 }
        expandedGroups.clear()
        renderTree()
      }
      renderAnalysis()
      render()
      setStatus(`분석 완료: ${analysis.game_name} (${analysis.engine})`)
    } catch (error) {
      setStatus(`분석 실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isAnalyzing = false
      analyzeButton.disabled = false
      analyzeButton.textContent = '🔍 LLM 게임 분석'
    }
  }

  // 트리에서 클릭한 타일 군집(나무·분수·가로등 등)을 부분 스타일 변환 대상으로 변환한다.
  // 셀·타일 id를 되찾고, 타일셋 .tsx에서 이미지 경로·격자 정보를 읽는다. 실패하면 undefined —
  // 호출부가 상태줄로 알린다. 서비스는 src/assets 안만 다루므로 다른 폴더로 연 게임은 대상 외.
  const buildStyleObjectTarget = (
    map: LoadedGameMap,
    entity: GameEntity
  ): StyleTransferMapObject | undefined => {
    const mapFile = currentFiles.find((file) => file.path === map.file)
    if (!mapFile) {
      return undefined
    }
    let objects: TmxObject[] = []
    try {
      objects = extractTmxObjects(mapFile.text)
    } catch {
      return undefined
    }
    // 타일 군집(좌표 id)은 군집 재추출로, 영역 오브젝트(건물·분수·나무 장식 등)는
    // 사각형 안의 같은 종류 타일 수집으로 셀 목록을 얻는다.
    const detail = isTileClusterEntity(entity)
      ? findTileClusterDetail(mapFile, currentFiles, objects, entity.id)
      : findObjectKindCells(mapFile, currentFiles, objects, entity)
    if (!detail || detail.cells.length === 0 || detail.tilesetSource === undefined) {
      return undefined
    }
    const tsxFile = findFileByRelativeSource(currentFiles, mapFile.path, detail.tilesetSource)
    const info = tsxFile ? extractTmxTilesetImageInfo(tsxFile.text) : undefined
    if (!tsxFile || !info) {
      return undefined
    }
    const imagePath = resolveRelativePath(tsxFile.path, info.imageSource)
    if (!imagePath.startsWith('src/assets/')) {
      return undefined
    }
    const kind = groupKindOf(entity.kind)
    return {
      label: `${KIND_ICON[kind] ?? '•'} ${entity.name}`,
      tilesetImagePath: imagePath,
      tileWidth: info.tileWidth,
      tileHeight: info.tileHeight,
      columns: info.columns,
      cells: detail.cells,
      sharedOutsideCells: detail.sharedOutsideCells
    }
  }

  // 맵 인식 시점의 자동 누끼 추출: 현재 맵의 변환 가능 오브젝트들의 셀 정보를 모아 서비스에
  // 배치로 보낸다. 서비스가 타일을 조립해 투명 PNG로 저장하고(이미 추출된 키는 스킵),
  // 모달의 '추출 오브젝트' 탭이 그 목록을 쓴다. 백그라운드 fetch라 에디터 UI는 멈추지 않고,
  // 서비스가 꺼져 있으면 조용히 무시한다. 성공한 맵은 세션 내 재전송하지 않는다.
  const extractedMapIds = new Set<string>()
  const extractMapObjectsInBackground = (mapId: string): void => {
    if (game.adapter.id !== 'my-sample-rpg' || extractedMapIds.has(mapId)) {
      return
    }
    const map = game.maps.find((candidate) => candidate.id === mapId)
    if (!map) {
      return
    }
    // 준비(파싱)는 scene-changed 핸들러의 페인트를 막지 않게 타이머로 미루고,
    // 엔티티별 재파싱 대신 일괄 수집(맵당 파싱 2회)으로 메인 스레드 점유를 줄인다.
    window.setTimeout(() => {
      const mapFile = currentFiles.find((file) => file.path === map.file)
      if (!mapFile) {
        return
      }
      let objects: TmxObject[] = []
      try {
        objects = extractTmxObjects(mapFile.text)
      } catch {
        return
      }
      const styleable = map.entities.filter(
        (entity) => !STYLE_TARGET_EXCLUDED_KINDS.has(groupKindOf(entity.kind))
      )
      const cellsByEntityId = findAllStyleTargetCells(mapFile, currentFiles, objects, styleable)

      // 타일셋 .tsx 해석은 source별로 1회만.
      type ResolvedTileset = { imagePath: string; tileWidth: number; tileHeight: number; columns: number }
      const tilesetBySource = new Map<string, ResolvedTileset | undefined>()
      const resolveTileset = (source: string): ResolvedTileset | undefined => {
        if (!tilesetBySource.has(source)) {
          const tsxFile = findFileByRelativeSource(currentFiles, mapFile.path, source)
          const info = tsxFile ? extractTmxTilesetImageInfo(tsxFile.text) : undefined
          const imagePath = tsxFile && info ? resolveRelativePath(tsxFile.path, info.imageSource) : undefined
          tilesetBySource.set(
            source,
            info && imagePath && imagePath.startsWith('src/assets/')
              ? { imagePath, tileWidth: info.tileWidth, tileHeight: info.tileHeight, columns: info.columns }
              : undefined
          )
        }
        return tilesetBySource.get(source)
      }

      const targets: Array<StyleTransferMapObject & { id: string }> = []
      for (const entity of styleable) {
        const detail = cellsByEntityId.get(entity.id)
        if (!detail || detail.cells.length === 0 || detail.tilesetSource === undefined) {
          continue
        }
        const tileset = resolveTileset(detail.tilesetSource)
        if (!tileset) {
          continue
        }
        const kind = groupKindOf(entity.kind)
        targets.push({
          id: entity.id,
          label: `${KIND_ICON[kind] ?? '•'} ${entity.name}`,
          tilesetImagePath: tileset.imagePath,
          tileWidth: tileset.tileWidth,
          tileHeight: tileset.tileHeight,
          columns: tileset.columns,
          cells: detail.cells,
          sharedOutsideCells: detail.sharedOutsideCells
        })
      }
      if (targets.length === 0) {
        return
      }
      // 현재 데이터는 맵당 타일셋이 하나라 첫 대상 기준으로 묶는다(다른 타일셋 대상은 제외).
      const first = targets[0]
      const sameTileset = targets.filter(
        (candidate) => candidate.tilesetImagePath === first.tilesetImagePath
      )
      void fetch('/api/style/extract-objects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tileset_path: first.tilesetImagePath,
          tile_width: first.tileWidth,
          tile_height: first.tileHeight,
          columns: first.columns,
          objects: sameTileset.map((candidate) => ({
            id: candidate.id,
            label: candidate.label,
            cells: candidate.cells,
            sharedOutsideCells: candidate.sharedOutsideCells
          }))
        })
      })
        .then((response) => {
          if (response.ok) {
            extractedMapIds.add(mapId)
          }
        })
        .catch(() => undefined)
    }, 0)
  }

  const renderTree = (): void => {
    entityButtons = []
    const groups: HTMLElement[] = []

    // 게임이 보고한 현재 맵을 트리의 기준으로 삼는다. 그 맵을 game.maps에서 찾으면(=같은 프로젝트)
    // 기본적으로 그 맵의 요소만 보여준다. 다른 게임 폴더를 열어 매칭이 안 되면 전체를 보여준다.
    const focusMap =
      currentMapId !== undefined
        ? game.maps.find((map) => map.id === currentMapId)
        : undefined
    const mapsToShow = focusMap && !showAllMaps ? [focusMap] : game.maps

    // 동기화 상태줄·토글 갱신. focusMap이 있을 때만 토글이 의미가 있다.
    if (focusMap) {
      mapFilterToggle.hidden = false
      mapFilterToggle.textContent = showAllMaps ? '현재 맵만' : '전체 보기'
      treeSyncLine.replaceChildren(
        el('span', 'inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5 align-middle'),
        showAllMaps
          ? document.createTextNode('전체 맵 표시 중')
          : el('span', 'text-zinc-300', `현재 맵: ${focusMap.name}`)
      )
    } else {
      mapFilterToggle.hidden = true
      // 연결 전(undefined)엔 대기 메시지, 매칭 안 되는 프로젝트면 줄을 비운다.
      treeSyncLine.textContent =
        currentMapId === undefined ? '게임과 연결 대기 중…' : ''
    }

    for (const map of mapsToShow) {
      // 객체도 레이어도 없는 맵만 건너뛴다(예: 파싱 실패). 몬스터만 있는 맵·지형만 있는 맵도 보여준다.
      if (map.entities.length === 0 && map.layers.length === 0) {
        continue
      }

      const group = el('div', 'flex flex-col gap-1')
      const isCurrent = map.id === currentMapId
      group.append(
        el(
          'div',
          'text-xs text-zinc-300 font-medium px-1',
          `🗺 ${map.name}${isCurrent ? ' · 현재 맵' : ''}`
        )
      )

      // 같은 종류끼리 접이식 그룹으로 묶는다(쭉 나열하면 길어서 보기 불편하다는 피드백).
      // NPC 그룹을 맨 위로, 나머지는 맵에 등장한 순서대로.
      const byKind = new Map<string, GameEntity[]>()
      for (const entity of map.entities) {
        const kind = groupKindOf(entity.kind)
        const list = byKind.get(kind)
        if (list) {
          list.push(entity)
        } else {
          byKind.set(kind, [entity])
        }
      }
      const kindEntries = [...byKind.entries()].sort(
        (a, b) => (a[0] === 'npc' ? 0 : 1) - (b[0] === 'npc' ? 0 : 1)
      )

      // 어떤 엔티티를 생성 대상으로 고를 수 있나: 타일 군집(보기 전용 구조물)은 제외하고,
      // rpg는 대화 NPC만(profile.npcs 큐레이션과 일치), 그 외 게임(legend-of-lua·분석된 게임)은
      // 어댑터/분석이 찾은 모든 엔티티가 공용 대사 생성의 대상이다.
      const isSelectableEntity = (entity: GameEntity): boolean =>
        !isTileClusterEntity(entity) &&
        (game.adapter.id === 'my-sample-rpg'
          ? groupKindOf(entity.kind) === 'npc'
          : true)

      const selectableCount = map.entities.filter(isSelectableEntity).length
      for (const [kind, entities] of kindEntries) {
        const groupKey = `${map.id}:${kind}`
        // 선택된 NPC가 속한 그룹은 이번 렌더에서만 펼쳐 보인다(접혀 있으면 선택 표시가 가려진다).
        // Set에는 쓰지 않는다 — 영구 펼침으로 만들면 사용자가 접어도 다음 렌더마다 되돌아간다.
        const containsSelected =
          selectedEntity !== undefined && entities.some((entity) => entity === selectedEntity)
        const expanded = expandedGroups.has(groupKey) || containsSelected

        const headerButton = el('button', ENTITY_GROUP_HEADER) as HTMLButtonElement
        headerButton.type = 'button'
        headerButton.setAttribute('aria-expanded', String(expanded))
        const arrow = el('span', 'w-3 shrink-0 text-[10px] text-zinc-500', expanded ? '▾' : '▸')
        arrow.setAttribute('aria-hidden', 'true')
        headerButton.append(
          arrow,
          el('span', 'truncate', `${KIND_ICON[kind] ?? '•'} ${KIND_LABEL[kind] ?? kind}`),
          el('span', 'ml-auto shrink-0 text-[10px] tabular-nums text-zinc-500', String(entities.length))
        )

        const body = el('div', 'flex flex-col gap-0.5 pl-3')
        body.id = `entity-group-${groupKey}`.replace(/[^A-Za-z0-9_-]/gu, '-')
        headerButton.setAttribute('aria-controls', body.id)
        body.hidden = !expanded
        // 토글은 이 그룹의 DOM만 만지고 트리를 다시 그리지 않는다 — 선택 상태·버튼 참조가 그대로 유지된다.
        headerButton.addEventListener('click', () => {
          const nextExpanded = body.hidden
          if (nextExpanded) {
            expandedGroups.add(groupKey)
          } else {
            expandedGroups.delete(groupKey)
          }
          body.hidden = !nextExpanded
          arrow.textContent = nextExpanded ? '▾' : '▸'
          headerButton.setAttribute('aria-expanded', String(nextExpanded))
        })

        for (const entity of entities) {
          if (isSelectableEntity(entity)) {
            // 생성 대상은 클릭 가능한 버튼으로 — 선택하면 그 엔티티로 생성한다.
            const node = el('button', ENTITY_BASE, entity.name) as HTMLButtonElement
            node.type = 'button'
            node.addEventListener('click', () => {
              selectedEntity = entity
              // 대상을 바꾸면 이전 생성 결과는 무효 — 새로 생성하게 한다.
              currentResult = undefined
              render()
            })
            entityButtons.push({ entity, node })
            body.append(node)
          } else if (
            game.adapter.id === 'my-sample-rpg' &&
            !STYLE_TARGET_EXCLUDED_KINDS.has(groupKindOf(entity.kind))
          ) {
            // 타일 구조물·장식 오브젝트: LLM 생성 대상은 아니지만, 클릭하면 그 오브젝트만 스타일 변환한다.
            const node = el(
              'button',
              'flex items-center gap-1 text-left rounded-lg px-2.5 py-2 text-sm text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-200'
            ) as HTMLButtonElement
            node.type = 'button'
            node.title = '클릭하면 이 오브젝트를 스타일 변환합니다 (같은 타일을 쓰는 다른 곳도 함께 바뀔 수 있습니다)'
            node.append(
              el('span', 'truncate', entity.name),
              el('span', 'ml-auto shrink-0 text-[10px]', '🎨')
            )
            node.addEventListener('click', () => {
              const target = buildStyleObjectTarget(map, entity)
              if (target) {
                styleTransfer.openForMapObject(target)
              } else {
                setStatus('이 오브젝트의 타일 정보를 읽지 못해 스타일 변환을 열 수 없습니다.')
              }
            })
            body.append(node)
          } else {
            // 몬스터·표지판·포털(및 다른 게임의 구조물)은 맵에 있음을 보여주되(보기 전용), 생성 대상은 아니다.
            const row = el('div', 'truncate rounded-lg px-2.5 py-2 text-sm text-zinc-400', entity.name)
            body.append(row)
          }
        }

        group.append(headerButton, body)
      }

      // 요소는 있는데 생성 대상이 하나도 없는 맵(사냥터·동굴 등)에선, 왜 클릭할 게 없는지 알려준다.
      if (selectableCount === 0 && map.entities.length > 0) {
        group.append(
          el('div', 'px-1 text-[11px] text-zinc-500 italic', '생성 대상이 없는 맵 — 위 요소는 보기 전용입니다.')
        )
      }

      // "ground" 같은 타일/지형 레이어 — 객체가 아니라 맵 자체의 구성. 보기 전용 정보로 한 줄에 보여준다.
      if (map.layers.length > 0) {
        group.append(
          el('div', 'px-1 pt-0.5 text-[11px] text-zinc-500', `🗂 타일 레이어: ${map.layers.join(' · ')}`)
        )
      }

      groups.push(group)
    }

    if (groups.length === 0) {
      const message =
        focusMap && !showAllMaps
          ? `현재 맵(${focusMap.name})에서 읽을 요소가 없습니다. ‘전체 보기’로 다른 맵을 볼 수 있어요.`
          : '로드된 맵이 없습니다. "게임 폴더 열기"로 프로젝트를 여세요.'
      groups.push(el('div', 'text-xs text-zinc-500 leading-relaxed', message))
    }

    treeList.replaceChildren(...groups)
  }

  const renderHistory = (): void => {
    if (history.length === 0) {
      historyWrap.hidden = true
      return
    }

    historyWrap.hidden = false
    historyList.replaceChildren(
      ...history.map((entry) => {
        const active = entry.result === currentResult
        // truncate: 라벨은 snake_case 강제라 줄바꿈 지점이 없어, 좁은 사이드바에 가로 스크롤을 만든다.
        const node = el(
          'button',
          active
            ? 'truncate text-left rounded-md px-2.5 py-1.5 text-xs bg-indigo-500/15 text-indigo-200 transition'
            : 'truncate text-left rounded-md px-2.5 py-1.5 text-xs text-zinc-400 transition hover:bg-white/5 hover:text-zinc-100'
        ) as HTMLButtonElement
        node.type = 'button'
        const mark = entry.result.issues.length === 0 ? '✅' : '⚠️'
        // 라벨은 LLM/열린 파일에서 온 임의 값이라 textContent로만 넣는다(주입/깨짐 방지).
        node.textContent = `#${entry.n} ${mark} ${entry.result.label}`
        node.addEventListener('click', () => {
          currentResult = entry.result
          render()
        })
        return node
      })
    )
  }

  const renderEvaluation = (): void => {
    if (!currentResult) {
      evaluationWrap.hidden = true
      return
    }

    evaluationWrap.hidden = false
    // 자동 Validator 통과율(이번 세션) + 사람 수용률(누적)을 한 줄로(회의의 두 평가 개념).
    // 둘은 시간 범위가 다르다: validatorPass는 sessionTally(프로젝트 전환 시 초기화), 수용률은
    // localStorage에 영속되는 평가 전체(누적 acceptance_rate 목표용)라 라벨로 범위를 구분한다.
    const metrics = buildSessionMetrics(sessionTally, evaluations)
    const validatorPercent = Math.round(metrics.validatorPassRate * 100)
    const acceptancePercent = Math.round(metrics.acceptanceRate * 100)
    acceptanceStat.textContent =
      `세션 생성 ${metrics.generations} · Validator 통과 ${validatorPercent}%` +
      (metrics.acceptanceTotal === 0
        ? ' · 누적 수용 평가 없음'
        : ` · 누적 수용률 ${acceptancePercent}%${metrics.meetsAcceptanceGoal ? ' ✅' : ''}`)

    // 현재 결과가 이미 평가됐으면(객체 단위로 기억) 그 판정을 보여주고 버튼을 잠근다(중복 집계 방지).
    const verdict = verdictByResult.get(currentResult)
    const evaluated = verdict !== undefined
    acceptButton.disabled = evaluated
    rejectButton.disabled = evaluated
    acceptButton.className =
      verdict === 'acceptable'
        ? 'rounded-lg px-3.5 py-2 bg-emerald-500/15 text-emerald-200 text-sm border border-emerald-500/30'
        : GHOST_BUTTON
    rejectButton.className =
      verdict === 'not_acceptable'
        ? 'rounded-lg px-3.5 py-2 bg-rose-500/15 text-rose-200 text-sm border border-rose-500/30'
        : GHOST_BUTTON
    if (evaluated) {
      evaluationVerdict.className =
        verdict === 'acceptable' ? 'text-xs text-emerald-300' : 'text-xs text-rose-300'
      evaluationVerdict.textContent =
        verdict === 'acceptable' ? '· 이 결과를 수용함' : '· 이 결과를 거부함'
    } else {
      evaluationVerdict.textContent = ''
    }
  }

  const runResetEvaluations = (): void => {
    clearEventEvaluations()
    evaluations = []
    // 영속 기록을 비웠으니 현재 결과의 잠금(verdict)도 함께 풀어 정합성을 맞춘다.
    verdictByResult = new WeakMap<GenerationResult, EventEvaluationVerdict>()
    renderEvaluation()
    setStatus('누적 평가 기록을 초기화했습니다.')
  }

  const runEvaluate = (verdict: EventEvaluationVerdict): void => {
    if (!currentResult || verdictByResult.has(currentResult)) {
      return
    }

    evaluations = appendEventEvaluation({
      event_id: `${currentResult.label || 'generation'}-${evaluations.length + 1}`,
      event_name: currentResult.label,
      verdict,
      reason: '',
      evaluated_at: Date.now()
    })
    verdictByResult.set(currentResult, verdict)
    renderEvaluation()
    const metrics = buildSessionMetrics(sessionTally, evaluations)
    setStatus(
      `평가 기록됨(${verdict === 'acceptable' ? '수용' : '거부'}) · 누적 수용률 ${Math.round(
        metrics.acceptanceRate * 100
      )}%`
    )
  }

  function render(): void {
    gameLabel.textContent = game.adapter.name

    if (game.adapter.supportsApply) {
      supportNote.hidden = true
    } else {
      supportNote.hidden = false
      supportNote.textContent = `${game.adapter.name}: 생성은 되지만 라이브 적용은 아직 지원되지 않습니다 (Stage 3). 결과는 미리보기로 확인하세요.`
    }

    // 엔티티 이름/맵은 열린 TMX에서 온 임의 값이므로 textContent로만 넣는다(주입/깨짐 방지).
    if (selectedEntity) {
      targetLine.replaceChildren(
        document.createTextNode('대상: '),
        el('span', 'text-indigo-300 font-medium', selectedEntity.name),
        document.createTextNode(' '),
        el('span', 'text-zinc-500', `(${selectedEntity.kind} · ${selectedEntity.mapId})`)
      )
    } else {
      targetLine.replaceChildren(
        el('span', 'text-zinc-400', '왼쪽에서 엔티티를 선택하면 그 대상으로 생성합니다.')
      )
    }

    for (const { entity, node } of entityButtons) {
      // id가 아니라 참조로 비교한다 — id는 맵이 달라도 겹칠 수 있어(같은 TMX 이름·그룹-번호 조합)
      // '전체 보기'에서 다른 맵의 동명 NPC까지 선택된 것처럼 칠해진다.
      node.className = entity === selectedEntity ? ENTITY_ACTIVE : ENTITY_BASE
    }

    if (!currentResult) {
      validationLine.hidden = true
    } else if (currentResult.issues.length === 0) {
      validationLine.hidden = false
      validationLine.className = 'text-xs text-emerald-300'
      validationLine.replaceChildren(
        document.createTextNode('✅ Validator (생성과 분리된 자동 검증): 통과')
      )
    } else {
      validationLine.hidden = false
      validationLine.className = 'text-xs text-amber-300 flex flex-col gap-0.5'
      // 이슈 문자열은 Validator가 만든 값이지만 안전하게 textContent(el)로만 넣는다.
      validationLine.replaceChildren(
        el('div', 'font-medium', `⚠️ Validator (생성과 분리된 자동 검증): ${currentResult.issues.length}건`),
        ...currentResult.issues.map((issue) => el('div', 'pl-3 text-amber-300/80', `• ${issue}`))
      )
    }

    generateButton.textContent = isGenerating ? '생성 중...' : '생성'
    generateButton.disabled =
      isGenerating || apiKey.trim().length === 0 || promptInput.value.trim().length === 0
    // 검증(issues)이 적용을 막지 않는다 — 사용자 요청대로 검증과 무관하게 바로 적용 가능.
    applyButton.disabled = isGenerating || !currentResult?.apply
    copyButton.disabled = !currentResult || isGenerating
    exportButton.disabled = !currentResult || isGenerating
    result.textContent = currentResult ? currentResult.preview : '생성 결과가 여기에 표시됩니다.'
    renderEvaluation()
    renderHistory()
  }

  const runGenerate = async (): Promise<void> => {
    if (isGenerating) {
      return
    }

    if (apiKey.trim().length === 0) {
      setStatus('먼저 Claude(Anthropic) API 키를 입력하세요.')
      return
    }

    if (promptInput.value.trim().length === 0) {
      setStatus('생성할 내용을 자연어 프롬프트에 입력하세요.')
      return
    }

    isGenerating = true
    // 생성은 비동기다. 도중에 다른 프로젝트를 열거나(runOpenProject) 복귀(runReset)하면, 늦게 도착한
    // 이 결과를 새 게임에 섞으면 안 된다(히스토리/집계 오염 + 옛 게임에 묶인 apply() 클로저). 시작 시점의
    // 프로젝트 정체성을 캡처해 커밋 전에 검사한다(runAnalyze의 filesAtStart 가드와 동일).
    const filesAtStart = currentFiles
    setStatus(`${game.adapter.name}로 생성 중...`)
    render()

    try {
      const result = await game.adapter.generate({
        apiKey: apiKey.trim(),
        userPrompt: promptInput.value,
        entity: selectedEntity,
        profile: game.profile,
        gameContext: currentAnalysis
          ? `${currentAnalysis.game_name} (${currentAnalysis.engine}). 콘텐츠 모델: ${currentAnalysis.content_model}`
          : undefined
      })
      // 생성 중 프로젝트가 바뀌었으면 이 결과는 버린다.
      if (currentFiles !== filesAtStart) {
        return
      }
      currentResult = result
      historyCounter += 1
      history = [{ n: historyCounter, result }, ...history].slice(0, HISTORY_LIMIT)
      // 세션 지표 집계: 생성 1건 + (Validator 통과면) 통과 1건.
      sessionTally = {
        generations: sessionTally.generations + 1,
        validatorPasses: sessionTally.validatorPasses + (result.issues.length === 0 ? 1 : 0)
      }
      setStatus(`생성 완료: ${result.label}`)
    } catch (error) {
      // 프로젝트가 바뀐 뒤 도착한 실패는 새 게임의 상태를 건드리지 않는다.
      if (currentFiles !== filesAtStart) {
        return
      }
      const message = error instanceof Error ? error.message : String(error)
      setStatus(`생성 실패: ${message}`)
    } finally {
      // isGenerating은 이 호출이 소유하므로 항상 해제하고 다시 그린다. 프로젝트가 바뀌었어도 render()는
      // 현재(=새) 게임 상태를 그대로 반영하므로 안전하다(생성 버튼 disabled 갱신 등).
      isGenerating = false
      render()
    }
  }

  const runApply = (): void => {
    if (!currentResult?.apply) {
      return
    }

    // apply()는 localStorage 저장을 동반해 실패할 수 있다. 조용히 죽지 않고 상태로 알린다.
    try {
      currentResult.apply()
      setStatus('게임에 적용됨 — 오른쪽 라이브 프리뷰에 즉시 반영됩니다.')
    } catch (error) {
      setStatus(`적용 실패: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const runCopy = async (): Promise<void> => {
    if (!currentResult) {
      return
    }

    // clipboard API는 비보안 컨텍스트·권한 거부에서 없거나 reject될 수 있어 방어한다.
    try {
      await navigator.clipboard.writeText(currentResult.preview)
      setStatus('생성 결과를 클립보드에 복사했습니다.')
    } catch {
      setStatus('클립보드 복사에 실패했습니다(브라우저 권한/보안 컨텍스트 확인).')
    }
  }

  const runClearHistory = (): void => {
    history = []
    historyCounter = 0
    renderHistory()
    setStatus('생성 히스토리를 비웠습니다.')
  }

  const runExport = (): void => {
    if (!currentResult) {
      return
    }

    const fileName = `${currentResult.label || 'generated'}.json`
    const blob = new Blob([currentResult.preview], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = fileName
    link.click()
    URL.revokeObjectURL(url)
    setStatus(`내보냄: ${fileName}`)
  }

  const runOpenProject = async (): Promise<void> => {
    try {
      const files = await openProjectDirectory()
      const loaded = loadGame(files)

      if (loaded.maps.length === 0) {
        setStatus('선택한 폴더에서 .tmx 맵을 찾지 못했습니다.')
        return
      }

      game = loaded
      currentFiles = files
      selectedEntity = undefined
      currentResult = undefined
      currentAnalysis = undefined
      history = []
      historyCounter = 0
      sessionTally = { generations: 0, validatorPasses: 0 }
      // 그룹 펼침 상태도 프로젝트 단위 — 맵 id(tmx 파일명)가 프로젝트끼리 겹쳐서, 안 비우면
      // 이전 게임에서 펼친 상태가 새 게임 트리에 그대로 묻어 나온다.
      expandedGroups.clear()
      renderTree()
      renderAnalysis()
      render()
      // 엔티티(어댑터가 찾은 개체)와 타일 구조물(보기 전용)을 나눠 세서, 수치가 부풀어 보이지 않게 한다.
      const allEntities = game.maps.flatMap((map) => map.entities)
      const tileCount = allEntities.filter(isTileClusterEntity).length
      const entityCount = allEntities.length - tileCount
      setStatus(
        `프로젝트 로드: ${game.adapter.name} · 맵 ${game.maps.length}개 · 엔티티 ${entityCount}개` +
          `${tileCount > 0 ? ` · 구조물 ${tileCount}개` : ''}${parseErrorNote()}`
      )
      // 토큰이 있으면 LLM이 이 게임을 자동 분석한다(네 아이디어: 열면 LLM이 이해).
      if (apiKey.trim().length > 0) {
        void runAnalyze()
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      setStatus(error instanceof Error ? error.message : String(error))
    }
  }

  const runReset = (): void => {
    game = loadGame(initialFiles)
    currentFiles = initialFiles
    selectedEntity = undefined
    currentResult = undefined
    currentAnalysis = undefined
    history = []
    historyCounter = 0
    sessionTally = { generations: 0, validatorPasses: 0 }
    expandedGroups.clear()
    renderTree()
    renderAnalysis()
    render()
    setStatus(`내 게임으로 복귀했습니다.${parseErrorNote()}`)
  }

  // 감지된 provider의 모델 목록으로 드롭다운을 채우고, 현재 선택값을 맞춘다.
  const populateModelSelect = (provider: LlmProvider): void => {
    const current = getProviderModel(provider)
    const models = PROVIDER_MODELS[provider]
    // 저장된 값이 목록에 없으면(예: 옛 커스텀) 맨 앞에 추가해 선택을 보존한다.
    const options = models.includes(current) ? models : [current, ...models]
    modelSelect.replaceChildren(
      ...options.map((modelId) => {
        const option = el('option', '', modelId) as HTMLOptionElement
        option.value = modelId
        return option
      })
    )
    modelSelect.value = current
  }

  // 입력한 키의 provider를 감지해 모델 배지·드롭다운을 갱신하고, /v1/models로 유효성을 확인해 피드백한다.
  let apiKeyCheckSeq = 0
  const refreshApiKeyStatus = async (): Promise<void> => {
    const key = apiKey.trim()
    const provider = detectProvider(key)
    if (provider) {
      populateModelSelect(provider)
      modelBadge.textContent = `${PROVIDER_LABEL[provider]} · ${getProviderModel(provider)}`
    }
    if (key.length === 0) {
      apiKeyStatus.className = 'text-xs text-zinc-500'
      apiKeyStatus.textContent = '키를 입력하세요.'
      return
    }
    const seq = ++apiKeyCheckSeq
    apiKeyStatus.className = 'text-xs text-zinc-400'
    apiKeyStatus.textContent = '🔍 확인 중...'
    const check = await validateApiKey(key)
    // 확인 중 더 최신 입력이 있었으면 이 결과는 버린다(레이스 방지).
    if (seq !== apiKeyCheckSeq) {
      return
    }
    apiKeyStatus.className =
      check.status === 'valid'
        ? 'text-xs text-emerald-300'
        : check.status === 'invalid'
          ? 'text-xs text-rose-300'
          : 'text-xs text-amber-300'
    const icon =
      check.status === 'valid' ? '✓' : check.status === 'invalid' ? '✗' : 'ℹ'
    apiKeyStatus.textContent = `${icon} ${check.message}`
  }

  let apiKeyDebounce: ReturnType<typeof setTimeout> | undefined
  apiKeyInput.addEventListener('input', () => {
    apiKey = apiKeyInput.value
    // 저장이 막혀도(프라이빗 모드 등) 입력·생성 흐름은 끊기지 않게 한다. 키는 메모리에 유지된다.
    const persisted = writeLocalStorage(API_KEY_STORAGE_KEY, apiKey)
    render()
    if (!persisted && apiKey.length > 0) {
      setStatus('API 키를 저장하지 못했습니다(브라우저 저장소 차단). 이번 세션에만 사용됩니다.')
    }
    if (apiKeyDebounce !== undefined) {
      clearTimeout(apiKeyDebounce)
    }
    apiKeyDebounce = setTimeout(() => {
      void refreshApiKeyStatus()
    }, 500)
  })
  // 모델 선택 → 현재 provider에 적용하고 저장. (키가 정하는 게 아니라 사용자가 고른다)
  modelSelect.addEventListener('change', () => {
    const provider = detectProvider(apiKey) ?? 'anthropic'
    setProviderModel(provider, modelSelect.value)
    writeLocalStorage(`${MODEL_STORAGE_PREFIX}${provider}`, modelSelect.value)
    const detected = detectProvider(apiKey)
    if (detected) {
      modelBadge.textContent = `${PROVIDER_LABEL[detected]} · ${getProviderModel(detected)}`
    }
  })

  // 저장된 모델(provider별)을 복원한 뒤, 저장돼 있던 키가 있으면 검증해 배지·모델·상태를 채운다.
  for (const provider of ['anthropic', 'openai'] as const) {
    const storedModel = readLocalStorage(`${MODEL_STORAGE_PREFIX}${provider}`)
    if (storedModel) {
      setProviderModel(provider, storedModel)
    }
  }
  // 키가 없어도 기본 provider 모델 목록은 채워 둔다(빈 드롭다운 방지).
  populateModelSelect(detectProvider(apiKey) ?? 'anthropic')
  void refreshApiKeyStatus()
  resetButton.addEventListener('click', runReset)
  generateButton.addEventListener('click', () => {
    void runGenerate()
  })
  // 프롬프트가 비면 생성 버튼도 비활성(눌러보고 실패하는 대신). 전체 re-render 없이 버튼만 갱신.
  promptInput.addEventListener('input', () => {
    generateButton.disabled =
      isGenerating || apiKey.trim().length === 0 || promptInput.value.trim().length === 0
  })
  // ⌘/Ctrl+Enter로 빠르게 생성(데모 흐름용). runGenerate가 자체 가드(키·프롬프트·생성중)를 가진다.
  promptInput.addEventListener('keydown', (event) => {
    // 한글 IME 조합 중의 Enter(후보 확정)는 가로채지 않는다 — 에디터 전체가 한국어 입력이다.
    if (event.isComposing) {
      return
    }
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      void runGenerate()
    }
  })
  applyButton.addEventListener('click', runApply)
  copyButton.addEventListener('click', () => {
    void runCopy()
  })
  clearHistoryButton.addEventListener('click', runClearHistory)
  acceptButton.addEventListener('click', () => {
    runEvaluate('acceptable')
  })
  rejectButton.addEventListener('click', () => {
    runEvaluate('not_acceptable')
  })
  resetEvaluationsButton.addEventListener('click', runResetEvaluations)
  exportButton.addEventListener('click', runExport)
  // 폴더 열기/분석은 결과가 중앙(분석 패널·상태줄)에 나오므로, 설정 모달을 닫아 그걸 가리지 않게 한다.
  openButton.addEventListener('click', () => {
    closeSettings()
    void runOpenProject()
  })
  analyzeButton.addEventListener('click', () => {
    closeSettings()
    void runAnalyze()
  })
  popoutButton.addEventListener('click', () => {
    window.open(gamePreviewUrl, 'game-window', 'width=1280,height=720')
  })
  reloadButton.addEventListener('click', () => {
    iframe.src = gamePreviewUrl
  })
  // 현재 맵만 ↔ 전체 맵 토글. 트리만 다시 그리되, 보이는 버튼의 선택 강조는 render()가 다시 입힌다.
  mapFilterToggle.addEventListener('click', () => {
    showAllMaps = !showAllMaps
    renderTree()
    render()
  })
  // 라이브 게임(iframe)이 맵을 바꾸면 그 맵의 요소만 트리에 보여준다. 게임은 bootstrapScene에서
  // 부모(에디터)로 'game:scene-changed'를 쏜다(초기 로드·포털 이동·맵 버튼 모두 포함).
  window.addEventListener('message', (event) => {
    // 게임 iframe에서 온 메시지만 신뢰한다(브라우저 확장 등 다른 출처 무시).
    if (event.source !== iframe.contentWindow) {
      return
    }
    const data = event.data as { type?: unknown; sceneId?: unknown } | null
    if (
      !data ||
      data.type !== 'game:scene-changed' ||
      typeof data.sceneId !== 'string' ||
      data.sceneId === currentMapId
    ) {
      return
    }
    currentMapId = data.sceneId
    // 게임이 새 맵으로 가면 자동으로 그 맵에 다시 집중한다(전체 보기 해제).
    showAllMaps = false
    renderTree()
    render()
    // 맵 인식 시점의 자동 누끼 추출 — 백그라운드라 UI를 막지 않는다.
    extractMapObjectsInBackground(currentMapId)
  })

  renderTree()
  renderAnalysis()
  render()
  if (game.parseErrors.length > 0) {
    setStatus(`기본 맵 일부를 읽지 못했습니다${parseErrorNote()}`)
  }
  // 데모 흐름: 키가 없으면 키 입력에, 있으면 바로 프롬프트에 포커스.
  ;(apiKey.trim().length > 0 ? promptInput : apiKeyInput).focus()
}
