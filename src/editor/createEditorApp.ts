import { openProjectDirectory } from './openProjectDirectory'
import { loadGame, type GameFile, type LoadedGame, type LoadedGameMap } from './loadGame'
import { buildMapPreviewInputs } from './buildMapPreviewInputs'
import {
  createTiledMapPreview,
  type TiledMapPreview
} from './createTiledMapPreview'
import { createGameBridge, type BridgeStatus } from './gameBridge'
import { analyzeGame, type GameAnalysis } from './analyzeGame'
import { extractTmxObjects, type TmxObject } from './tmxObjects'
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
      return { id, name: id, file: file.path, entities }
    })
}

type CreateEditorAppInput = {
  mountElement: HTMLElement
  initialFiles: GameFile[]
  gamePreviewUrl: string
}

const API_KEY_STORAGE_KEY = 'my-sample-rpg:anthropic-api-key'
const MODEL_STORAGE_PREFIX = 'my-sample-rpg:model:'
const BRIDGE_URL_STORAGE_KEY = 'my-sample-rpg:game-bridge-url'
// 실행 중인 외부 게임(Love2D 등)의 로컬 HTTP 브리지 기본 주소.
const DEFAULT_BRIDGE_URL = 'http://localhost:17320'
// love.js로 빌드한 게임의 웹 URL(예: /legend-of-lua/). 설정하면 그 게임을 패널에서 직접 플레이한다.
const WEB_BUILD_URL_STORAGE_KEY = 'my-sample-rpg:web-build-url'

const KIND_ICON: Record<string, string> = {
  npc: '👤',
  enemy: '👹',
  chest: '📦',
  loot: '💰'
}

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
  'text-[11px] font-semibold uppercase tracking-wider text-zinc-500'
const CARD =
  'rounded-xl border border-white/10 bg-white/[0.03] p-4 flex flex-col gap-2'
const FIELD_INPUT =
  'w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2.5 text-sm text-zinc-100 outline-none transition placeholder:text-zinc-600 focus:border-indigo-500/60 focus:ring-2 focus:ring-indigo-500/25'
const PRIMARY_BUTTON =
  'rounded-lg px-4 py-2 bg-indigo-500 text-white text-sm font-medium shadow-sm shadow-indigo-500/30 transition hover:bg-indigo-400 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none'
const GHOST_BUTTON =
  'rounded-lg px-3.5 py-2 bg-white/[0.04] text-zinc-300 text-sm border border-white/10 transition hover:bg-white/[0.08] hover:text-zinc-100 hover:border-white/20 disabled:opacity-40 disabled:cursor-not-allowed'
const ENTITY_BASE =
  'text-left rounded-lg px-2.5 py-2 text-sm text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-100'
const ENTITY_ACTIVE =
  'text-left rounded-lg px-2.5 py-2 text-sm bg-indigo-500/15 text-indigo-100 ring-1 ring-inset ring-indigo-500/30 transition'

// 폴더에서 만든 이미지 object URL을 정리한다(새 프로젝트를 열 때 옛 URL 누수 방지).
const revokePreviewObjectUrls = (files: GameFile[]): void => {
  for (const file of files) {
    if (file.url?.startsWith('blob:')) {
      URL.revokeObjectURL(file.url)
    }
  }
}

export const createEditorApp = ({
  mountElement,
  initialFiles,
  gamePreviewUrl
}: CreateEditorAppInput): void => {
  let game: LoadedGame = loadGame(initialFiles)
  let currentFiles: GameFile[] = initialFiles
  let apiKey = readLocalStorage(API_KEY_STORAGE_KEY) ?? ''
  let selectedEntity: GameEntity | undefined
  // 왼쪽 엔티티 트리를 한 맵으로 좁히는 필터(맵 버튼으로 고른다). undefined면 전체 맵을 보여준다.
  let treeMapFilter: string | undefined
  let currentResult: GenerationResult | undefined
  let currentAnalysis: GameAnalysis | undefined
  let isGenerating = false
  let isAnalyzing = false
  let entityButtons: Array<{ entity: GameEntity; node: HTMLButtonElement }> = []
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
  const root = el('div', 'h-screen w-screen flex flex-col bg-zinc-950 text-zinc-100 overflow-hidden')

  const header = el('header', 'h-12 shrink-0 flex items-center justify-between px-4 border-b border-white/10 bg-zinc-900/60')
  const brand = el('div', 'flex items-center gap-2')
  brand.append(
    el('span', 'w-2.5 h-2.5 rounded-full bg-indigo-500'),
    el('span', 'text-sm font-semibold tracking-tight', 'Scenario Editor'),
    el('span', 'text-xs text-zinc-600', '·')
  )
  const gameLabel = el('span', 'text-xs text-zinc-400', game.adapter.name)
  brand.append(gameLabel)
  // 모델 배지 — 입력한 키의 provider(Claude/GPT)에 따라 동적으로 갱신된다.
  const modelBadge = el(
    'span',
    'text-[11px] rounded-full px-2 py-0.5 bg-indigo-500/10 text-indigo-300 border border-indigo-500/20',
    `Claude · ${ANTHROPIC_MODEL}`
  )
  brand.append(modelBadge)
  const connection = el('div', 'flex items-center gap-2 text-xs text-zinc-400')
  const connectionDot = el('span', 'w-2 h-2 rounded-full bg-zinc-600')
  const connectionLabel = el('span', '', '게임 로딩...')
  connection.append(connectionDot, connectionLabel)
  const settingsButton = el('button', 'rounded-lg px-2.5 py-1 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100', '⚙ 설정') as HTMLButtonElement
  settingsButton.type = 'button'
  const headerRight = el('div', 'flex items-center gap-3')
  headerRight.append(settingsButton, connection)
  header.append(brand, headerRight)

  const body = el('div', 'flex-1 grid grid-cols-[240px_1fr_minmax(360px,38%)] min-h-0')

  // ---------- left: project tree ----------
  const tree = el('aside', 'border-r border-white/10 flex flex-col min-h-0')
  const treeHeader = el('div', 'p-3 border-b border-white/10 flex flex-col gap-2')
  const openButton = el('button', 'rounded-lg px-3 py-2 bg-white/5 text-sm text-zinc-200 text-left transition hover:bg-white/10', '📂 게임 폴더 열기') as HTMLButtonElement
  openButton.type = 'button'
  const analyzeButton = el('button', 'rounded-lg px-3 py-2 bg-indigo-500/10 text-sm text-indigo-200 text-left transition hover:bg-indigo-500/20 disabled:opacity-50', '🔍 LLM 게임 분석') as HTMLButtonElement
  analyzeButton.type = 'button'
  const resetButton = el('button', 'rounded-lg px-3 py-1.5 bg-white/5 text-xs text-zinc-400 text-left transition hover:bg-white/10 hover:text-zinc-200', '🏠 내 게임으로 복귀') as HTMLButtonElement
  resetButton.type = 'button'
  // 프로젝트 버튼(폴더 열기/분석/복귀)은 설정 모달로 이동했다. 사이드바는 엔티티 목록만.
  treeHeader.append(el('div', LABEL, '엔티티'))
  const treeList = el('div', 'flex-1 overflow-auto p-3 flex flex-col gap-3')
  tree.append(treeHeader, treeList)

  // ---------- center: generation ----------
  const center = el('main', 'p-5 flex flex-col gap-4 min-h-0 overflow-y-auto')
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
  const promptInput = el('textarea', `${FIELD_INPUT} min-h-[110px] resize-y`) as HTMLTextAreaElement
  promptInput.placeholder = '예: 대장장이가 새로 만든 검을 자랑하는 대화'
  promptField.append(promptInput)

  const actions = el('div', 'flex items-center gap-2')
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
  const evaluationTop = el('div', 'flex items-center justify-between gap-2')
  evaluationTop.append(
    el('span', LABEL, 'Evaluator · 사람 이진 평가')
  )
  const acceptanceStat = el('span', 'text-xs text-zinc-400')
  evaluationTop.append(acceptanceStat)
  const evaluationButtons = el('div', 'flex items-center gap-2')
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

  center.append(targetLine, analysisPanel, supportNote, promptField, actions, status, validationLine, resultWrap, evaluationWrap, historyWrap)

  // ---------- right: live game preview ----------
  // 내 게임(my-sample-rpg)은 웹 런타임이 있어 iframe으로 "실제 실행"을 보여주지만, 다른 게임
  // (예: Love2D의 legend-of-lua)은 브라우저에서 런타임을 못 돌린다. 대신 본게임과 똑같은 Pixi
  // Tiled 렌더 경로로 그 게임의 맵을 라이브로 그려 보여준다. iframe(내 게임)과 Pixi 캔버스(다른
  // 게임 맵)를 한 자리에 겹쳐 두고 모드에 따라 바꿔 켠다.
  const preview = el('section', 'border-l border-white/10 flex flex-col min-w-0')
  // 맵 버튼이 많아도 바가 안 잘리도록: [제목(잘림)] [맵 버튼(가로 스크롤)] [새창·새로고침(고정)].
  const previewBar = el('div', 'h-9 shrink-0 flex items-center gap-2 px-3 border-b border-white/10 bg-zinc-900/40')
  const previewTitle = el('span', 'text-xs text-zinc-400 shrink-0 whitespace-nowrap truncate max-w-[30%]', '🎮 라이브 게임 (실제 게임 실행 중)')
  // 내 게임 iframe 모드에서 전환할 씬(마을/사냥터/동굴).
  const rpgPreviewScenes = [
    { id: 'town', label: '마을' },
    { id: 'hunting-ground', label: '사냥터' },
    { id: 'cave', label: '동굴' }
  ]
  // 맵 버튼 줄: 남는 공간을 차지하고 넘치면 가로 스크롤(macOS 오버레이 스크롤바라 높이 영향 없음).
  const mapSwitcher = el('div', 'flex items-center gap-1 flex-1 min-w-0 overflow-x-auto')
  const previewActions = el('div', 'flex items-center gap-2 shrink-0')
  const popoutButton = el('button', 'text-xs text-zinc-400 transition hover:text-zinc-100', '↗ 새 창') as HTMLButtonElement
  popoutButton.type = 'button'
  const reloadButton = el('button', 'text-xs text-zinc-400 transition hover:text-zinc-100', '↻ 새로고침') as HTMLButtonElement
  reloadButton.type = 'button'
  previewActions.append(popoutButton, reloadButton)
  previewBar.append(previewTitle, mapSwitcher, previewActions)
  // 세로 휠로도 맵 버튼 줄을 좌우로 스크롤할 수 있게 한다(트랙패드 가로 스크롤은 기본 지원).
  mapSwitcher.addEventListener('wheel', (event) => {
    if (event.deltaY !== 0 && mapSwitcher.scrollWidth > mapSwitcher.clientWidth) {
      mapSwitcher.scrollLeft += event.deltaY
      event.preventDefault()
    }
  })

  const previewBody = el('div', 'flex-1 relative min-h-0')
  const iframe = el('iframe', 'absolute inset-0 w-full h-full border-0 bg-black') as HTMLIFrameElement
  iframe.src = gamePreviewUrl
  iframe.title = '게임 프리뷰'
  iframe.addEventListener('load', () => {
    // iframe이 게임을 띄우는 모드(rpg 웹게임 / love.js 웹빌드)면 연결됨으로 표시한다.
    if (isRpgPreviewMode() || isWebBuildMode()) {
      connectionDot.className = 'w-2 h-2 rounded-full bg-emerald-400'
      connectionLabel.textContent = '게임 연결됨'
    }
  })
  // 다른 게임의 맵을 Pixi로 그릴 호스트. 드래그 팬을 위해 기본 커서를 grab으로.
  const mapPreviewHost = el('div', 'absolute inset-0 bg-black overflow-hidden cursor-grab')
  mapPreviewHost.style.display = 'none'
  // 렌더 진행/실패 안내(이미지 누락, 미지원 인코딩 등).
  const previewMessage = el('div', 'absolute inset-0 flex items-center justify-center p-6 text-center text-xs text-zinc-400 pointer-events-none')
  previewMessage.style.display = 'none'
  previewBody.append(iframe, mapPreviewHost, previewMessage)
  preview.append(previewBar, previewBody)

  body.append(tree, center, preview)

  // ---------- live preview behavior ----------
  // 내 게임(rpg)은 iframe 실행. 다른 게임은: love.js 웹 빌드 URL이 있으면 그걸 패널에서 플레이,
  // 없으면 맵을 Pixi로 렌더(정적 미리보기).
  let webBuildUrl = (readLocalStorage(WEB_BUILD_URL_STORAGE_KEY) ?? '').trim()
  const isRpgPreviewMode = (): boolean => game.adapter.id === 'my-sample-rpg'
  // 다른 게임 + love.js 빌드 URL이 있으면 패널에서 실제 게임을 iframe으로 플레이한다.
  const isWebBuildMode = (): boolean =>
    !isRpgPreviewMode() && webBuildUrl.length > 0

  let activeMapPreview: TiledMapPreview | undefined
  // 맵 전환이 빠르게 겹쳐도 늦게 끝난 렌더가 패널을 덮지 않게 토큰으로 최신 요청만 커밋한다.
  let mapPreviewToken = 0
  let selectedPreviewMapId: string | undefined

  const setPreviewMessage = (message: string | undefined): void => {
    if (message === undefined) {
      previewMessage.style.display = 'none'
      previewMessage.textContent = ''
      return
    }

    previewMessage.style.display = 'flex'
    previewMessage.textContent = message
  }

  const destroyMapPreview = (): void => {
    activeMapPreview?.destroy()
    activeMapPreview = undefined
  }

  const MAP_BUTTON_BASE =
    'shrink-0 whitespace-nowrap text-[11px] rounded px-2 py-0.5 bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100'
  const MAP_BUTTON_ACTIVE =
    'shrink-0 whitespace-nowrap text-[11px] rounded px-2 py-0.5 bg-indigo-500/20 border border-indigo-500/40 text-indigo-100 transition'

  // 스위처 버튼들을 추적해, 클릭 시 전체를 다시 그리지 않고 활성 표시만 바꾼다.
  // (replaceChildren로 매번 다시 그리면 가로 스크롤이 0으로 리셋되고 버튼이 커서 밑에서 움직여
  //  다음 클릭이 엉뚱하게 떨어진다.) id가 undefined인 버튼은 "전체".
  let mapSwitcherButtons: Array<{ id: string | undefined; node: HTMLButtonElement }> = []

  const updateMapSwitcherActive = (): void => {
    for (const { id, node } of mapSwitcherButtons) {
      node.className = id === treeMapFilter ? MAP_BUTTON_ACTIVE : MAP_BUTTON_BASE
    }
  }

  const makeMapButton = (
    id: string | undefined,
    label: string,
    onClick: () => void
  ): HTMLButtonElement => {
    const button = el('button', MAP_BUTTON_BASE, label) as HTMLButtonElement
    button.type = 'button'
    // 부분만 보이는 버튼을 클릭할 때 브라우저가 포커스로 자동 스크롤(=튕김)하면서 버튼이 커서 밑에서
    // 움직여 click이 안 먹는다. 포커스를 막으면 자동 스크롤이 사라지고, click은 그대로 발생한다.
    button.addEventListener('mousedown', (event) => event.preventDefault())
    button.addEventListener('click', onClick)
    mapSwitcherButtons.push({ id, node: button })
    return button
  }

  // 스위처를 처음부터 다시 그린다(모드/게임 변경 시에만 호출 — 클릭 시엔 updateMapSwitcherActive 사용).
  const renderMapSwitcher = (): void => {
    mapSwitcherButtons = []
    const children: HTMLButtonElement[] = []

    // "전체": 트리 맵 필터 해제(전체 맵 표시).
    children.push(
      makeMapButton(undefined, '전체', () => {
        treeMapFilter = undefined
        renderTree()
        updateMapSwitcherActive()
      })
    )

    if (isRpgPreviewMode()) {
      // 씬 = 맵. 누르면 iframe 씬 전환 + 그 맵으로 트리를 좁힌다.
      for (const scene of rpgPreviewScenes) {
        children.push(
          makeMapButton(scene.id, scene.label, () => {
            treeMapFilter = scene.id
            renderTree()
            updateMapSwitcherActive()
            iframe.contentWindow?.postMessage(
              { type: 'editor:switch-scene', sceneId: scene.id },
              '*'
            )
          })
        )
      }
    } else {
      // 다른 게임: 맵 버튼은 트리를 그 맵으로 좁힌다. 맵 미리보기 모드면 Pixi 렌더, love.js 플레이
      // 모드면 게임에 맵 전환 요청(미리보기로 갈아끼우지 않음).
      for (const map of game.maps) {
        children.push(
          makeMapButton(map.id, map.name, () => {
            treeMapFilter = map.id
            renderTree()
            updateMapSwitcherActive()
            if (isWebBuildMode()) {
              iframe.contentWindow?.postMessage(
                { type: 'editor:goto-map', mapId: map.id, mapName: map.name },
                '*'
              )
            } else {
              void renderMapPreview(map.id)
            }
          })
        )
      }
    }

    mapSwitcher.replaceChildren(...children)
    updateMapSwitcherActive()
  }

  const showRpgPreview = (): void => {
    destroyMapPreview()
    setPreviewMessage(undefined)
    mapPreviewHost.style.display = 'none'
    iframe.style.display = 'block'
    popoutButton.style.display = 'inline'
    if (iframe.src !== new URL(gamePreviewUrl, location.href).href) {
      iframe.src = gamePreviewUrl
    }
    previewTitle.textContent = '🎮 라이브 게임 (실제 게임 실행 중)'
    connectionDot.className = 'w-2 h-2 rounded-full bg-emerald-400'
    connectionLabel.textContent = '게임 연결됨'
    renderMapSwitcher()
  }

  // love.js 웹 빌드를 패널에서 직접 플레이한다(별도 창 없이, my-sample-rpg처럼 iframe 안에서).
  const showWebGamePreview = (): void => {
    destroyMapPreview()
    setPreviewMessage(undefined)
    mapPreviewHost.style.display = 'none'
    iframe.style.display = 'block'
    popoutButton.style.display = 'inline'
    if (iframe.src !== new URL(webBuildUrl, location.href).href) {
      iframe.src = webBuildUrl
    }
    previewTitle.textContent = `🎮 라이브 게임 (love.js) — ${game.adapter.name}`
    // 연결 표시는 iframe load에서 갱신. 로딩 동안엔 연결 중으로 둔다.
    connectionDot.className = 'w-2 h-2 rounded-full bg-amber-400'
    connectionLabel.textContent = '게임 로딩…'
    renderMapSwitcher()
  }

  // 브리지 게임은 헤더 연결 표시를 브리지 상태가 소유한다 — 맵 프리뷰가 그걸 덮지 않게 한다.
  const markMapPreviewConnection = (ok: boolean): void => {
    if (game.adapter.applyMode === 'bridge') {
      return
    }

    connectionDot.className = ok
      ? 'w-2 h-2 rounded-full bg-emerald-400'
      : 'w-2 h-2 rounded-full bg-amber-400'
    connectionLabel.textContent = ok ? '맵 미리보기' : '맵 미리보기 불가'
  }

  const renderMapPreview = async (mapId: string): Promise<void> => {
    const targetMap = game.maps.find((map) => map.id === mapId)

    if (!targetMap) {
      return
    }

    selectedPreviewMapId = mapId
    iframe.style.display = 'none'
    popoutButton.style.display = 'none'
    mapPreviewHost.style.display = 'block'
    previewTitle.textContent = `🗺 맵 미리보기 — ${targetMap.name}`
    // 스위처는 syncPreviewToGame에서 한 번 그려둔다. 여기선 활성 표시만 갱신(스크롤 보존).
    updateMapSwitcherActive()

    const inputs = buildMapPreviewInputs(currentFiles, targetMap.file)

    if (!inputs.ok) {
      destroyMapPreview()
      setPreviewMessage(inputs.error)
      markMapPreviewConnection(false)
      return
    }

    const token = (mapPreviewToken += 1)
    setPreviewMessage('맵 렌더링 중...')
    destroyMapPreview()

    try {
      const instance = await createTiledMapPreview({
        mountElement: mapPreviewHost,
        map: inputs.inputs.map,
        imageUrls: inputs.inputs.imageUrls
      })

      // 렌더 중에 더 최근 요청(맵 전환/리셋)이 들어왔으면 이 결과는 버린다.
      if (token !== mapPreviewToken) {
        instance.destroy()
        return
      }

      activeMapPreview = instance
      setPreviewMessage(undefined)
      markMapPreviewConnection(true)
    } catch (error) {
      if (token !== mapPreviewToken) {
        return
      }

      setPreviewMessage(
        `맵을 렌더링하지 못했습니다: ${error instanceof Error ? error.message : String(error)}`
      )
      markMapPreviewConnection(false)
    }
  }

  // 게임이 바뀔 때(열기/복귀) 프리뷰를 그 게임에 맞게 동기화한다.
  const syncPreviewToGame = (): void => {
    if (isRpgPreviewMode()) {
      showRpgPreview()
      return
    }

    // love.js 웹 빌드가 설정돼 있으면 그 게임을 패널에서 직접 플레이한다.
    if (isWebBuildMode()) {
      showWebGamePreview()
      return
    }

    // 웹 빌드가 없으면 맵을 Pixi로 렌더(정적 미리보기). 엔티티가 있는 맵을 우선(없으면 첫 맵).
    const firstMap =
      game.maps.find((map) => map.entities.length > 0) ?? game.maps[0]

    if (!firstMap) {
      destroyMapPreview()
      setPreviewMessage('이 게임에서 렌더할 맵을 찾지 못했습니다.')
      return
    }

    renderMapSwitcher() // 스위처를 이 게임의 맵으로 한 번 그린다(이후 클릭은 활성 표시만 갱신).
    void renderMapPreview(firstMap.id)
  }

  // ---------- live game bridge (별도 프로세스 게임용) ----------
  // my-sample-rpg는 같은 origin localStorage로 적용하지만, Love2D 같은 외부 프로세스 게임은
  // 실행 중인 게임의 로컬 HTTP 브리지로 생성물을 보낸다. 연결 상태는 헤더 표시등이 보여준다.
  let bridgeStatus: BridgeStatus = 'disconnected'

  const applyBridgeStatusToIndicator = (): void => {
    if (bridgeStatus === 'connected') {
      connectionDot.className = 'w-2 h-2 rounded-full bg-emerald-400'
      connectionLabel.textContent = '게임 연결됨'
    } else if (bridgeStatus === 'connecting') {
      connectionDot.className = 'w-2 h-2 rounded-full bg-amber-400'
      connectionLabel.textContent = '게임 연결 중…'
    } else {
      connectionDot.className = 'w-2 h-2 rounded-full bg-zinc-600'
      connectionLabel.textContent = '게임 미연결'
    }
  }

  const bridge = createGameBridge({
    baseUrl: readLocalStorage(BRIDGE_URL_STORAGE_KEY) ?? DEFAULT_BRIDGE_URL,
    onStatusChange: (next) => {
      bridgeStatus = next
      // 브리지 게임이고 웹빌드 모드가 아닐 때만 헤더 표시등을 브리지 상태로 갱신한다.
      if (game.adapter.applyMode === 'bridge' && !isWebBuildMode()) {
        applyBridgeStatusToIndicator()
      }
      // 적용 버튼 활성/지원 안내가 연결 상태에 의존하므로 다시 그린다.
      render()
    }
  })

  // 브리지 적용 게임이고 웹빌드(love.js)로 패널에서 직접 플레이하는 게 아니면 폴링을 켠다.
  // love.js 모드에선 게임이 iframe 안에 있으므로 HTTP 브리지(별도 프로세스용)는 끈다.
  const syncBridgeForGame = (): void => {
    if (game.adapter.applyMode === 'bridge' && !isWebBuildMode()) {
      bridge.start()
      applyBridgeStatusToIndicator()
    } else {
      bridge.stop()
    }
  }

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

  // 외부 게임(Love2D 등)의 라이브 브리지 주소. 게임이 띄운 로컬 HTTP 서버를 가리킨다.
  const bridgeField = el('label', 'flex flex-col gap-1.5')
  bridgeField.append(el('span', LABEL, '게임 브리지 URL — 외부 게임(Love2D 등) 라이브 적용'))
  const bridgeInput = el('input', FIELD_INPUT) as HTMLInputElement
  bridgeInput.type = 'text'
  bridgeInput.placeholder = DEFAULT_BRIDGE_URL
  bridgeInput.value = bridge.getBaseUrl()
  bridgeInput.spellcheck = false
  bridgeField.append(bridgeInput)
  bridgeInput.addEventListener('change', () => {
    const url = bridgeInput.value.trim() || DEFAULT_BRIDGE_URL
    bridge.setBaseUrl(url)
    writeLocalStorage(BRIDGE_URL_STORAGE_KEY, url)
    bridgeInput.value = bridge.getBaseUrl()
  })

  // love.js로 빌드한 게임의 웹 URL. 넣으면 그 게임을 패널에서 직접 플레이한다(비우면 맵 미리보기).
  const webBuildField = el('label', 'flex flex-col gap-1.5')
  webBuildField.append(el('span', LABEL, 'love.js 웹 빌드 URL — 패널에서 게임 직접 플레이(예: /legend-of-lua/)'))
  const webBuildInput = el('input', FIELD_INPUT) as HTMLInputElement
  webBuildInput.type = 'text'
  webBuildInput.placeholder = '/legend-of-lua/'
  webBuildInput.value = webBuildUrl
  webBuildInput.spellcheck = false
  webBuildField.append(webBuildInput)
  webBuildInput.addEventListener('change', () => {
    webBuildUrl = webBuildInput.value.trim()
    writeLocalStorage(WEB_BUILD_URL_STORAGE_KEY, webBuildUrl)
    // 현재 보고 있는 게임이 외부 게임이면 즉시 모드를 다시 맞춘다(미리보기 ↔ 플레이).
    syncPreviewToGame()
    syncBridgeForGame()
    render()
  })

  settingsPanel.append(settingsTop, apiKeyField, modelField, bridgeField, webBuildField, el('div', 'h-px bg-white/10'), projectControls)
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

  root.append(header, body, settingsBackdrop)
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
      const totalEntities = game.maps.reduce((sum, map) => sum + map.entities.length, 0)
      if (totalEntities === 0) {
        game = { ...game, maps: buildEntitiesFromAnalysis(filesAtStart, analysis) }
        // 트리를 새 엔티티로 갈아끼우므로, 이전 대상의 생성 결과·히스토리·세션 집계는 모두 무효 처리한다
        // (open/reset과 동일한 초기화 묶음 — 지표가 폐기된 생성을 계속 세지 않도록).
        selectedEntity = undefined
        currentResult = undefined
        history = []
        historyCounter = 0
        sessionTally = { generations: 0, validatorPasses: 0 }
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

  const renderTree = (): void => {
    entityButtons = []
    const groups: HTMLElement[] = []

    // 맵 필터가 켜져 있으면 그 맵만, 아니면 전체 맵을 보여준다.
    const mapsToShow = treeMapFilter
      ? game.maps.filter((map) => map.id === treeMapFilter)
      : game.maps

    for (const map of mapsToShow) {
      if (map.entities.length === 0) {
        continue
      }

      const group = el('div', 'flex flex-col gap-1')
      group.append(el('div', 'text-xs text-zinc-300 font-medium px-1', `🗺 ${map.name}`))

      for (const entity of map.entities) {
        const node = el('button', ENTITY_BASE, `${KIND_ICON[entity.kind] ?? '•'} ${entity.name}`) as HTMLButtonElement
        node.type = 'button'
        node.addEventListener('click', () => {
          selectedEntity = entity
          // 대상을 바꾸면 이전 생성 결과는 무효 — 새로 생성하게 한다.
          currentResult = undefined
          render()
        })
        entityButtons.push({ entity, node })
        group.append(node)
      }

      groups.push(group)
    }

    if (groups.length === 0) {
      groups.push(
        el(
          'div',
          'text-xs text-zinc-500 leading-relaxed',
          treeMapFilter
            ? '이 맵에는 표시할 엔티티가 없습니다.'
            : '로드된 엔티티가 없습니다. "게임 폴더 열기"로 프로젝트를 여세요.'
        )
      )
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
        const node = el(
          'button',
          active
            ? 'text-left rounded-md px-2.5 py-1.5 text-xs bg-indigo-500/15 text-indigo-200 transition'
            : 'text-left rounded-md px-2.5 py-1.5 text-xs text-zinc-400 transition hover:bg-white/5 hover:text-zinc-100'
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

    // 적용 안내: 같은 origin 웹게임/love.js 패널게임은 안내 불필요, 브리지 게임은 연결 상태를
    // 알려주고, 그 외는 미지원.
    if (game.adapter.applyMode === 'local-storage' || isWebBuildMode()) {
      supportNote.hidden = true
    } else if (game.adapter.applyMode === 'bridge') {
      supportNote.hidden = false
      supportNote.textContent =
        bridgeStatus === 'connected'
          ? `${game.adapter.name}: 게임 브리지 연결됨 — '게임에 적용'하면 실행 중인 게임에 라이브 반영됩니다.`
          : `${game.adapter.name}: 게임을 실행하고 브리지를 켜세요(기본 ${bridge.getBaseUrl()}). 연결되면 '게임에 적용'이 활성화됩니다. (또는 설정에서 love.js 웹 빌드 URL을 넣으면 패널에서 바로 플레이됩니다.)`
    } else {
      supportNote.hidden = false
      supportNote.textContent = `${game.adapter.name}: 생성은 되지만 라이브 적용은 아직 지원되지 않습니다. 결과는 미리보기로 확인하세요.`
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
        el('span', 'text-zinc-500', '왼쪽에서 엔티티를 선택하면 그 대상으로 생성합니다.')
      )
    }

    for (const { entity, node } of entityButtons) {
      node.className = entity.id === selectedEntity?.id ? ENTITY_ACTIVE : ENTITY_BASE
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
    // 같은 origin 웹게임은 apply()로, love.js 패널게임은 iframe postMessage로, 브리지 게임은
    // 연결돼 있을 때 bridgePayload로 적용한다.
    const canApplyLocal = currentResult?.apply != null
    const canApplyWeb = isWebBuildMode() && currentResult?.bridgePayload != null
    const canApplyBridge =
      game.adapter.applyMode === 'bridge' &&
      !isWebBuildMode() &&
      bridgeStatus === 'connected' &&
      currentResult?.bridgePayload != null
    applyButton.disabled =
      isGenerating || (!canApplyLocal && !canApplyWeb && !canApplyBridge)
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

  const runApply = async (): Promise<void> => {
    if (!currentResult) {
      return
    }

    // 같은 origin 웹게임(my-sample-rpg): localStorage로 적용. 저장 실패할 수 있어 상태로 알린다.
    if (currentResult.apply) {
      try {
        currentResult.apply()
        setStatus('게임에 적용됨 — 오른쪽 라이브 프리뷰에 즉시 반영됩니다.')
      } catch (error) {
        setStatus(`적용 실패: ${error instanceof Error ? error.message : String(error)}`)
      }
      return
    }

    // love.js로 패널에서 플레이 중인 게임: 같은 페이지 iframe이므로 postMessage로 보낸다.
    // 게임-쪽 love.js 빌드가 'editor:apply' 메시지를 받아 적용한다(docs/legend-of-lua-love-js.md).
    if (isWebBuildMode() && currentResult.bridgePayload) {
      iframe.contentWindow?.postMessage(
        { type: 'editor:apply', payload: currentResult.bridgePayload },
        '*'
      )
      setStatus('패널의 게임에 적용을 보냈습니다 (love.js).')
      return
    }

    // 브리지 게임(Love2D 등): 실행 중인 게임의 HTTP 브리지로 전송한다.
    if (game.adapter.applyMode === 'bridge' && currentResult.bridgePayload) {
      if (bridgeStatus !== 'connected') {
        setStatus(
          `게임 브리지가 연결되지 않았습니다. 게임을 실행하고 브리지(${bridge.getBaseUrl()})를 켜세요.`
        )
        return
      }

      setStatus('실행 중인 게임에 적용 중…')
      const applyResult = await bridge.apply(currentResult.bridgePayload)
      setStatus(
        applyResult.ok
          ? '게임에 적용됨 — 실행 중인 게임에 라이브 반영되었습니다.'
          : `적용 실패: ${applyResult.error ?? '게임이 적용을 거부했습니다.'}`
      )
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

      const previousFiles = currentFiles
      game = loaded
      currentFiles = files
      selectedEntity = undefined
      currentResult = undefined
      currentAnalysis = undefined
      selectedPreviewMapId = undefined
      treeMapFilter = undefined
      history = []
      historyCounter = 0
      sessionTally = { generations: 0, validatorPasses: 0 }
      renderTree()
      renderAnalysis()
      render()
      syncPreviewToGame()
      syncBridgeForGame()
      revokePreviewObjectUrls(previousFiles)
      const entityCount = game.maps.reduce((sum, map) => sum + map.entities.length, 0)
      setStatus(
        `프로젝트 로드: ${game.adapter.name} · 맵 ${game.maps.length}개 · 엔티티 ${entityCount}개${parseErrorNote()}`
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
    const previousFiles = currentFiles
    game = loadGame(initialFiles)
    currentFiles = initialFiles
    selectedEntity = undefined
    currentResult = undefined
    currentAnalysis = undefined
    selectedPreviewMapId = undefined
    treeMapFilter = undefined
    history = []
    historyCounter = 0
    sessionTally = { generations: 0, validatorPasses: 0 }
    renderTree()
    renderAnalysis()
    render()
    syncPreviewToGame()
    syncBridgeForGame()
    revokePreviewObjectUrls(previousFiles)
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
  applyButton.addEventListener('click', () => void runApply())
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
    // 별도 창으로 띄울 때도 현재 패널이 보여주는 게임(rpg면 내 게임, 웹빌드면 love.js 게임)을 연다.
    window.open(
      isWebBuildMode() ? webBuildUrl : gamePreviewUrl,
      'game-window',
      'width=1280,height=720'
    )
  })
  reloadButton.addEventListener('click', () => {
    if (isRpgPreviewMode()) {
      iframe.src = gamePreviewUrl
      return
    }

    // love.js 플레이 모드: iframe 게임을 다시 로드한다.
    if (isWebBuildMode()) {
      iframe.src = webBuildUrl
      return
    }

    // 맵 프리뷰 모드: 현재 맵을 다시 렌더한다.
    if (selectedPreviewMapId) {
      void renderMapPreview(selectedPreviewMapId)
    }
  })

  renderTree()
  renderAnalysis()
  render()
  syncPreviewToGame()
  syncBridgeForGame()
  if (game.parseErrors.length > 0) {
    setStatus(`기본 맵 일부를 읽지 못했습니다${parseErrorNote()}`)
  }
  // 데모 흐름: 키가 없으면 키 입력에, 있으면 바로 프롬프트에 포커스.
  ;(apiKey.trim().length > 0 ? promptInput : apiKeyInput).focus()
}
