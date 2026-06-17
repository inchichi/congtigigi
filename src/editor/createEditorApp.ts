import { openProjectDirectory } from './openProjectDirectory'
import {
  buildTileClusterEntities,
  isTileClusterEntity,
  loadGame,
  type GameFile,
  type LoadedGame,
  type LoadedGameMap
} from './loadGame'
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
import { editorIcon, type EditorIconName } from './editorIcons'
import type { GameEntity, GenerationFeedback, GenerationResult } from './gameAdapter'
import { generateQuestCandidates, type QuestCandidate } from './questCandidates'
import { type DryRunReport } from './dryRunEventApply'
import { generateQuestJson } from './questJsonGenerator'
import { dryRunQuestApply } from './dryRunQuestApply'
import { convertGeneratedQuestToDefinition } from './questCodeGenerator'
import { createGeneratedQuestValidationIssues } from './questJsonSchema'
import { replacePendingQuests } from './pendingQuests'
import { buildLuaQuestCatalog } from './luaQuestCatalog'
import { generateLuaQuestCandidates } from './luaQuestCandidates'
import { generateLuaQuestJson } from './luaQuestJsonGenerator'
import { dryRunLuaQuestApply } from './dryRunLuaQuestApply'
import {
  renderGeneratedLuaQuestModule,
  convertLuaQuestToBridgePayload
} from './luaQuestCodeGenerator'
import { createGeneratedLuaQuestValidationIssues } from './luaQuestSchema'
import { generateLuaNpcJson } from './luaNpcJsonGenerator'
import { dryRunLuaNpcApply } from './dryRunLuaNpcApply'
import {
  renderEditorNpcSpawnEntry,
  convertLuaNpcToGameEntity,
  convertLuaNpcToSpawnBridgePayload
} from './luaNpcCodeGenerator'
import { createGeneratedLuaNpcValidationIssues } from './luaNpcSchema'
import { createGameBridge } from './gameBridge'

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

const findEntityById = (game: LoadedGame, entityId: string): GameEntity | undefined =>
  game.maps
    .flatMap((map) => map.entities)
    .find((entity) => entity.id === entityId || entity.name === entityId)

const isLegendOfLuaGame = (game: LoadedGame): boolean => game.adapter.id === 'legend-of-lua'

const API_KEY_STORAGE_KEY = 'my-sample-rpg:anthropic-api-key'
const MODEL_STORAGE_PREFIX = 'my-sample-rpg:model:'

// 종류별 게임풍 SVG 아이콘 매핑(editorIcons.ts에서 손으로 그린 것들). emoji는 쓰지 않는다.
const KIND_ICON: Record<string, EditorIconName> = {
  npc: 'npc',
  monster: 'monster',
  enemy: 'monster',
  sign: 'sign',
  portal: 'portal',
  chest: 'chest',
  loot: 'loot',
  building: 'building',
  character: 'character',
  // 타일 군집(tmxTileEntities)으로 인식되는 종류들.
  tent: 'tent',
  clocktower: 'clocktower',
  fountain: 'fountain',
  lamp: 'lamp',
  banner: 'banner',
  tree: 'tree',
  hedge: 'hedge',
  flower: 'flower',
  prop: 'prop',
  rock: 'rock',
  stairs: 'stairs',
  wall: 'wall',
  window: 'window'
}

// 보기 전용 요소(몬스터·표지판·포털 등)에 붙는 짧은 한국어 종류 라벨.
const KIND_LABEL: Record<string, string> = {
  npc: 'NPC',
  monster: '몬스터',
  enemy: '몬스터',
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

// 에셋 카테고리(표시 전용) — 인물/건축물/장식물/환경 4층으로 묶어 정보 구조를 만든다.
const CATEGORY_ORDER = ['인물', '건축물', '장식물', '환경'] as const
const CATEGORY_OF: Record<string, string> = {
  npc: '인물',
  character: '인물',
  monster: '인물',
  enemy: '인물',
  building: '건축물',
  sign: '건축물',
  portal: '건축물',
  clocktower: '건축물',
  tent: '건축물',
  window: '건축물',
  stairs: '건축물',
  tree: '환경',
  hedge: '환경',
  wall: '환경',
  rock: '환경'
}
const categoryOf = (kind: string): string => CATEGORY_OF[kind] ?? '장식물'

// 표시용 이름 정리(표시 전용) — 내부 id 느낌의 이름('villager_a' 등)을 발표용 라벨로 바꾼다.
// 우선순위: 짧은 원본 이름 그대로 → 흔한 영문 키워드 한글화 → 구분자/확장자 정리.
const displayNameOf = (rawName: string): string => {
  const cleaned = rawName
    .replace(/\.(png|jpe?g|json|tmx|lua)$/iu, '')
    .replace(/[_-]+/gu, ' ')
    .trim()
  const lower = cleaned.toLowerCase()
  const villager = lower.match(/^villager\s*([a-z0-9]*)$/u)
  if (villager) {
    const suffix = (villager[1] ?? '').toUpperCase()
    return suffix ? `주민 ${suffix}` : '주민'
  }
  if (lower.startsWith('blacksmith')) {
    return '대장장이'
  }
  if (lower.startsWith('merchant') || lower.startsWith('vendor')) {
    return '상인'
  }
  if (lower.startsWith('mage') || lower.startsWith('wizard')) {
    return '마법사'
  }
  if (lower.startsWith('guard')) {
    return '경비병'
  }
  if (lower.startsWith('santa')) {
    return '산타'
  }
  return cleaned
}

// NPC 역할별 아이콘(표시 전용) — 이름 키워드로 추정한다. 사용자는 이름보다 아이콘으로 먼저 구분한다.
const npcIconFor = (name: string): EditorIconName => {
  const lower = name.toLowerCase()
  if (name.includes('마법') || lower.includes('mage') || lower.includes('wizard')) {
    return 'orb'
  }
  if (name.includes('대장') || lower.includes('smith')) {
    return 'sword'
  }
  if (name.includes('경비') || name.includes('기사') || lower.includes('guard') || lower.includes('knight')) {
    return 'shield'
  }
  if (name.includes('상인') || name.includes('상점') || lower.includes('merchant') || lower.includes('shop') || lower.includes('vendor')) {
    return 'loot'
  }
  return 'npc'
}

// 트리 그룹핑용 종류 정규화. LLM 분석이 'NPC'처럼 대소문자를 섞어 줄 수 있어 소문자로 맞추고,
// enemy는 라벨·아이콘이 '몬스터'로 같아 monster 그룹에 합친다(그래서 위 맵에는 enemy 키가 없다).
const groupKindOf = (kind: string): string => {
  const normalized = kind.trim().toLowerCase()
  return normalized === 'enemy' ? 'monster' : normalized
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
// 상용 MMORPG 월드 에디터 톤: 게임 화면이 주인공, UI는 짙은 갈색/회색 + 은은한 금색(#c48a4a)으로
// 보조한다. 패널은 반투명(rgba(37,33,31,0.9))이라 게임과 경쟁하지 않는다.
// 팔레트 — 배경 #1e1e1e · 패널 #252526 · hover #302a26 · 테두리 #333336 · 강조 #c48a4a
//          텍스트 #d4d4d4 · 보조 텍스트 #9d9d9d.
const LABEL = 'text-[11px] font-semibold tracking-wide text-[#9d9d9d]'
// 큰 영역(왼쪽 에셋/게임 카드/입력 카드/결과 카드)이 공유하는 패널 골격 — 은은한 그라데이션 테두리.
const PANEL = 'rounded-xl box-grad-border [--bgb:rgba(37,37,38,0.97)] text-[#d4d4d4]'
const CARD =
  'rounded-lg border border-[#d9a85c]/25 bg-[#1e1e1e]/60 p-3 flex flex-col gap-2'
// 자연어 입력창 — 어두운 속지에 갈색 테두리, focus 시 금색.
const FIELD_INPUT =
  'w-full rounded-[12px] border-2 border-[#dca14b]/45 bg-[#1a1a1a] px-4 py-3 text-[12px] text-[#d4d4d4] outline-none transition placeholder:text-[#919191] focus:border-[#d09a4c] focus:shadow-[0_0_18px_rgba(208,154,76,0.18)]'
// 액션 버튼 — 생성이 화면의 메인 CTA(가장 크고 눈에 띄는 금색), 적용(초록)은 그보다 작게.
// CTA 계층: 생성(Primary, 금색 그라데이션+glow) > 적용(Secondary, 금테+어두운 브라운) > 복사/내보내기(보조).
const PRIMARY_BUTTON =
  'rounded-xl h-[52px] min-w-[240px] px-6 flex items-center justify-center bg-gradient-to-b from-[#e8b96a] to-[#c9883a] text-[#241608] border-2 border-[#8c5b26] shadow-[inset_0_1px_0_rgba(255,255,255,0.35),0_0_16px_rgba(225,178,100,0.3)] transition duration-150 hover:brightness-[1.1] hover:-translate-y-px hover:shadow-[inset_0_1px_0_rgba(255,255,255,0.45),0_8px_22px_rgba(225,178,100,0.42)] active:translate-y-0 disabled:opacity-45 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:brightness-100'
const APPLY_BUTTON =
  'rounded-xl h-[46px] px-5 flex items-center justify-center bg-[#2d2d30] text-[#d9a85c] text-[15px] font-semibold border-2 border-[#c9923f] shadow-[inset_0_1px_0_rgba(255,255,255,0.08),0_0_8px_rgba(217,168,92,0.15)] transition duration-150 hover:bg-[#333333] hover:-translate-y-px hover:shadow-[0_0_10px_rgba(217,168,92,0.25)] active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:shadow-none'
const GHOST_BUTTON =
  'rounded-lg h-[40px] px-3.5 bg-[#2d2d30] text-[#9d9d9d] text-[13px] border border-[#d9a85c]/25 opacity-80 transition duration-150 hover:opacity-100 hover:bg-[#333333] hover:text-[#d4d4d4] hover:border-[#d5a14f]/50 active:border-[#d9a85c] disabled:opacity-50 disabled:cursor-not-allowed'
// 종류 카드 — 게임 에디터의 선택 카드: 아이콘(46px) + 이름 + '8개' 카운트. 한 화면에 10개 이상 보이게 낮춘다.
// 에셋 종류 카드 — 기본/hover는 테두리 없이 면(배경)으로만 구분, 선택된 카드만 금색 테두리.
const KIND_CARD =
  'relative h-[78px] flex flex-col items-center justify-center gap-1 rounded-xl px-2 text-center bg-[#2d2d30] border-2 border-transparent transition duration-150'
const KIND_CARD_CLICKABLE =
  'cursor-pointer hover:bg-[#333333] hover:shadow-[0_4px_12px_rgba(0,0,0,0.25)] hover:-translate-y-px'
// 선택/펼침된 카드: 밝은 금색 테두리 + 살짝 밝은 그라데이션 + 은은한 glow.
const KIND_CARD_ACTIVE =
  'border-[#d9a85c] bg-gradient-to-b from-[#3a281b] to-[#2d2d30] shadow-[0_0_12px_rgba(217,168,92,0.35)]'
// 구성원 pill(34px, 한 줄에 2개) — 짧은 한글 표시 이름은 잘리지 않고, 긴 이름은 툴팁으로 보완.
const ENTITY_BASE =
  'h-[34px] min-w-0 flex items-center gap-1.5 rounded-lg px-2 text-left bg-[#2d2d30] border border-[#d9a85c]/18 text-[12px] text-[#e8d5a5] transition hover:bg-[#333333] hover:border-[#d5a14f]'
const ENTITY_ACTIVE =
  'h-[34px] min-w-0 flex items-center gap-1.5 rounded-lg px-2 text-left bg-gradient-to-b from-[#3a281b] to-[#2d2d30] border-2 border-[#d7a14a] shadow-[0_0_10px_rgba(215,161,74,0.4)] text-[12px] text-[#e0e0e0] transition'
// 게임 미리보기 위 맵 탭(마을/사냥터/동굴) — 둥근 나무 탭, 선택된 탭만 금색 그라데이션.
const SCENE_TAB =
  'h-[26px] flex items-center gap-1.5 text-[14px] leading-none rounded-lg px-3 py-1 bg-[#2d2d30] border border-[#d9a85c]/22 text-[#9d9d9d] transition hover:bg-[#333333] hover:text-[#ead8b6]'
const SCENE_TAB_ACTIVE =
  'h-[26px] flex items-center gap-1.5 text-[14px] leading-none rounded-lg px-3 py-1 bg-gradient-to-b from-[#d9a85c] to-[#9a6a2f] border border-[#f3d88b]/70 text-[#1e1e1e] shadow-[inset_0_1px_0_rgba(255,255,255,0.25)] transition'
// 진행 단계 캡슐(선택→작성→생성→확인→적용) — 숫자 원형 배지 + 라벨 구조.
// 현재 단계: 금색 테두리+은은한 glow 펄스, 숫자는 금색 채움 / 완료: ✓ / 미완료: 보조 정보 수준.
const STEP_PILL =
  'group h-[30px] inline-flex items-center gap-1.5 text-[13px] rounded-full pl-1 pr-3 bg-[#2d2d30] border border-[#d9a85c]/22 transition hover:border-[#c98a3a]/70'
const STEP_PILL_ACTIVE =
  'step-pulse h-[30px] inline-flex items-center gap-1.5 text-[13px] rounded-full pl-1 pr-3 bg-gradient-to-b from-[#d9a85c] to-[#9a6a2f] border border-[#f3d88b]/75 transition'
const STEP_PILL_DONE =
  'h-[30px] inline-flex items-center gap-1.5 text-[13px] rounded-full pl-1 pr-3 bg-[#2c2115] border border-[#d5a55a]/40 transition'
// 숫자 원형 배지 — 활성은 살짝 크고 금색 채움 + 밝은 숫자.
const STEP_NUM =
  'w-[18px] h-[18px] shrink-0 flex items-center justify-center rounded-full bg-[#333333] text-[10px] leading-none text-[#9d9d9d] transition'
const STEP_NUM_ACTIVE =
  'w-[20px] h-[20px] shrink-0 flex items-center justify-center rounded-full bg-[#1e1e1e]/25 border border-[#9a6a2f]/70 text-[11px] leading-none text-[#1e1e1e] font-semibold transition'
const STEP_NUM_DONE =
  'w-[18px] h-[18px] shrink-0 flex items-center justify-center rounded-full bg-[#3a2c1a] text-[10px] leading-none text-[#e6cf9a] transition'
const STEP_TEXT = 'text-[#8a8a8a] transition group-hover:text-[#b8b8b8]'
const STEP_TEXT_ACTIVE = 'text-[#1e1e1e] font-semibold'
const STEP_TEXT_DONE = 'text-[#e6cf9a]'

// ---- 설정 모달(VSCode Dark 설정창) 토큰 ----
// 메인 에디터(차콜)보다 두세 단계 밝은 연회색 계층 — 모달이 떠 있을 때 명확히 분리돼 보인다.
const SETTINGS_SECTION = 'rounded-2xl border-2 border-[#5a5a61] bg-[#45454b] p-4 flex flex-col gap-3'
const SETTINGS_LABEL = 'text-base font-semibold text-[#e6e6e6]'
const SETTINGS_INPUT =
  'w-full rounded-lg border-2 border-[#5a5a61] bg-[#2e2e33] px-3 py-2.5 text-base text-[#e6e6e6] outline-none transition placeholder:text-[#9a9a9a] focus:border-[#569cd6] focus:ring-2 focus:ring-[#569cd6]/30'
// 프로젝트 기본 버튼 — 연회색 보조 버튼 톤.
const SETTINGS_BUTTON =
  'flex items-center justify-center gap-2.5 rounded-xl min-h-[48px] px-4 bg-[#4a4a50] text-[#e6e6e6] text-lg border border-[#5e5e66] transition hover:bg-[#56565c] hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:hover:translate-y-0'
// 메인 액션(AI 게임 분석)만 VSCode 블루 강조.
const SETTINGS_BUTTON_SPECIAL =
  'flex items-center justify-center gap-2.5 rounded-xl min-h-[48px] px-4 bg-[#0e639c] text-white text-lg border border-[#1177bb] transition hover:bg-[#1177bb] hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 disabled:hover:translate-y-0'
// 제공사 표시 칩(Claude/GPT) — 키로 자동 감지되므로 클릭은 안 되고, 감지된 쪽만 블루 테두리.
const PROVIDER_CHIP =
  'rounded-xl px-5 py-2 text-[18px] leading-none bg-[#45454b] text-[#d9d9d9] border-2 border-[#5e5e66]'
const PROVIDER_CHIP_ACTIVE =
  'rounded-xl px-5 py-2 text-[18px] leading-none bg-[#54545b] text-white border-2 border-[#569cd6]'
// 모델 목록은 작은 태그로 — 선택된 것만 블루 테두리.
const MODEL_CHIP =
  'rounded-md px-2 py-1 text-[12px] leading-none bg-[#3f3f45] text-[#d9d9d9] border border-[#5a5a61] transition hover:bg-[#4a4a50]'
const MODEL_CHIP_ACTIVE =
  'rounded-md px-2 py-1 text-[12px] leading-none bg-[#4a4a50] text-white border border-[#569cd6] ring-1 ring-[#569cd6]/40'

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
  // 에셋 검색어(표시 전용) — 왼쪽 트리를 이름으로 실시간 필터링한다.
  let assetQuery = ''
  // 트리의 종류별 그룹(NPC/몬스터 등) 펼침 상태. 키는 `${mapId}:${kind}` — 트리를 다시 그려도 유지된다.
  // 기본은 접힘: 요소를 쭉 나열하면 목록이 길어 보기 불편하다는 피드백에 따른 동작.
  const expandedGroups = new Set<string>()
  // 카테고리(인물/건축물/장식물/환경) 접힘 상태 — 표시 전용.
  const collapsedCategories = new Set<string>()
  // 세션 내 생성 결과 누적(최신 우선, 최대 10개). 데모에서 여러 생성을 비교·재선택하려는 용도.
  const HISTORY_LIMIT = 10
  let history: Array<{ n: number; result: GenerationResult }> = []
  let historyCounter = 0
  // Evaluator(사람 이진 평가, 회의 #3/#7): 생성/검증과 분리된 품질 판정. 단일 지표 acceptance_rate.
  // 평가 판정은 결과 객체 동일성으로 기억한다(단일 슬롯이면 히스토리에서 옛 결과를 다시 골라 재평가 →
  // 중복 집계되어 acceptance_rate가 오염됨). WeakMap이라 참조가 사라진 결과는 알아서 GC된다.
  let evaluations: EventEvaluation[] = loadEventEvaluations()
  let verdictByResult = new WeakMap<GenerationResult, EventEvaluationVerdict>()
  // 결과 보드의 "적용 상태" 표시 전용 — 게임에 적용된 결과를 기억한다(로직에는 영향 없음).
  const appliedResults = new WeakSet<GenerationResult>()
  // 빈 보드의 '오늘 작업' 통계 표시 전용 — 이번 세션의 적용 횟수.
  let appliedCount = 0
  // '최근 작업' 시간 표시 전용 — 결과가 처음 화면에 잡힌 시각(생성 직후 render에서 기록).
  const resultTimes = new WeakMap<GenerationResult, string>()
  // 이번 세션 집계: 생성 수 + Validator 통과 수. 프로젝트를 바꾸면 초기화한다.
  let sessionTally: SessionGenerationTally = { generations: 0, validatorPasses: 0 }
  // 퀘스트 모드(2단계 생성): '퀘스트' 빠른시작을 고르면 켜진다. 1단계는 자연어 후보 N개를 만들고,
  // 유저가 하나를 골라 2단계에서 그 후보만 이벤트 JSON으로 만든 뒤 드라이런 검증→적용한다.
  let candidateMode = false
  // NPC 생성 모드(legend-of-lua 전용 단일 생성): 'NPC 추가' 빠른시작을 legend 게임에서 고르면 켜진다.
  // '이야기 생성'이 새 NPC를 한 번에 생성→검증하고, 통과하면 game.maps에 주입해 카탈로그/트리가 인식한다.
  let luaNpcMode = false
  // NPC 생성은 "새로 만드는" 작업이라 기존 대상을 고를 필요가 없다 — 이 모드에선 대상 선택 단계를
  // 건너뛰고(흐름·안내가 "대상 선택 필요"에 멈추지 않게) 바로 작성/생성으로 진행한다.
  const isTargetOptional = (): boolean => luaNpcMode
  let candidates: QuestCandidate[] = []
  let selectedCandidateIndex: number | undefined
  // 인라인 요약 편집 중인 후보 인덱스(표시 전용).
  let editingCandidateIndex: number | undefined
  // 후보 재생성(피드백 루프) 반복 횟수.
  let candidateIteration = 0
  // 마지막 2단계 결과의 무결성 검증(드라이런) 리포트. 퀘스트 모드에서 적용 게이트로 쓴다.
  let currentDryRun: DryRunReport | undefined
  // 후보 카드 다시 그리기 훅 — 실제 구현은 핸들러 정의부에서 할당한다(render에서 안전히 호출하려 let).
  let renderCandidates: () => void = () => {}

  // ---------- shell ----------
  // w-screen이 아니라 w-full — 100vw는 세로 스크롤바 폭을 포함해 가로 스크롤을 만든다.
  const root = el('div', 'h-screen w-full flex flex-col bg-[#1e1e1e] text-[#d4d4d4] overflow-hidden')

  // 헤더는 얇고 어두운 도구 바 — 시선은 아래 게임 화면으로 가게 한다.
  // 게임 화면이 주인공이도록 헤더는 낮게 압축한다.
  const header = el('header', 'settings-game-font select-none shrink-0 flex items-center justify-between gap-3 px-4 py-1.5 border-b border-[#d9a85c]/28 bg-[#252526] text-[#d4d4d4]')
  const brand = el('div', 'flex items-center gap-2.5 min-w-0')
  const brandText = el('div', 'flex flex-col gap-0.5 min-w-0')
  const brandTitleRow = el('div', 'flex items-center gap-2 min-w-0')
  brandTitleRow.append(
    el('span', 'text-[16px] font-semibold leading-none tracking-tight whitespace-nowrap text-[#e6e6e6]', '마을 이야기 공방')
  )
  // 프로젝트명은 작은 나무 팻말 배지로(좁은 화면에선 숨김 — 헤더가 넘치면 설정 버튼이 밀려난다).
  const gameLabel = el('span', 'hidden sm:inline-block text-[11px] rounded-md px-2 py-0.5 bg-[#d9a85c]/[0.18] border border-[#d9a85c]/35 text-[#f3d88b] truncate', game.adapter.name)
  brandTitleRow.append(gameLabel)
  brandText.append(
    brandTitleRow,
    // 3순위 서브카피 — 제목·프로젝트명보다 한 단계 아래로(여백 +3px, 75% 투명도).
    el('span', 'hidden md:block mt-[3px] text-[10px] leading-none text-[#9d9d9d] opacity-75', '마을을 바꾸고 이야기를 만들어보세요')
  )
  brand.append(editorIcon('building', 20), brandText)
  // 모델 배지 — 입력한 키의 provider(Claude/GPT)에 따라 동적으로 갱신된다.
  // 모델명은 게임 분위기를 깨서 헤더 대신 설정 모달의 고급 설정 안에 산다.
  const modelBadge = el(
    'span',
    'self-start text-[12px] rounded-md px-2 py-0.5 bg-[#45454b] border border-[#5a5a61] text-[#b8b8b8]',
    `Claude · ${ANTHROPIC_MODEL}`
  )
  // 연결 상태 배지 — 28px 캡슐. 연결되면 초록(#72d36b)으로 바뀐다(iframe load 리스너에서 갱신).
  const connection = el('div', 'h-7 flex items-center gap-1.5 text-[11px] rounded-full px-2.5 bg-[#2d2d30] border border-[#d9a85c]/28 text-[#9d9d9d]')
  const connectionDot = el('span', 'w-2 h-2 rounded-full bg-[#6e6e6e]')
  const connectionLabel = el('span', '', '접속 중...')
  connection.append(connectionDot, connectionLabel)
  // 설정 — 톱니 아이콘만 있는 32px 원형 버튼.
  const settingsButton = el('button', 'w-8 h-8 shrink-0 flex items-center justify-center rounded-full bg-[#2d2d30] border border-[#d9a85c]/28 transition hover:bg-[#302a26] hover:border-[#c48a4a]/40') as HTMLButtonElement
  settingsButton.setAttribute('aria-label', '설정')
  settingsButton.append(editorIcon('gear', 16))
  settingsButton.type = 'button'
  const headerRight = el('div', 'flex items-center gap-2 shrink-0')
  headerRight.append(connection, settingsButton)
  header.append(brand, headerRight)

  // LLM 챗 스타일 배치: 가운데가 라이브 게임(위 가득) + 프롬프트(아래), 오른쪽이 생성 결과.
  // 3열은 md(≥768px)부터 바로 적용한다 — 이전엔 lg부터여서, 브라우저 줌을 쓰는 일반 노트북
  // 창이 "결과가 하단 전폭" 배치로 떨어지며 게임 세로 공간을 잃었다(게임이 납작한 띠가 됨).
  //  - md(≥768px): [엔티티 트리 | 게임+프롬프트 | 생성 결과] 3열 (lg부터는 사이드가 약간 넓어짐)
  //  - 그 미만: 트리 → 게임+프롬프트 → 생성 결과 세로 스택
  const body = el(
    'div',
    // 패널 사이 여백은 최소로 — 게임 화면에 최대한 면적을 준다(헤더 쪽 위 여백은 절반).
    'flex-1 min-h-0 grid gap-2 px-2 pb-2 pt-1 ' +
      'grid-cols-1 grid-rows-[auto_minmax(0,1.4fr)_minmax(0,1fr)] [grid-template-areas:"tree""main""side"] ' +
      // 게임 화면이 화면 대부분을 차지하도록 좌우 사이드를 좁게 못 박는다.
      // 왼쪽은 아이콘 카드 2열(카드 약 90px+)이 들어가야 해서 최소 폭을 조금 더 준다.
      'md:grid-cols-[minmax(210px,16%)_minmax(0,1fr)_minmax(195px,16%)] md:grid-rows-[minmax(0,1fr)] md:[grid-template-areas:"tree_main_side"]'
  )

  // ---------- left: project tree ----------
  // 스택 배치(<md)에선 높이 제한(목록이 길면 자체 스크롤), md부터는 왼쪽 열 카드.
  const tree = el(
    'aside',
    // settings-game-font: 왼쪽 패널도 ESC/설정 모달과 같은 둥근 픽셀 폰트를 쓴다.
    // select-none: 라벨이 드래그로 파랗게 선택되면 UI 오류처럼 보인다 — 패널 전체 선택 금지.
    `settings-game-font select-none [grid-area:tree] ${PANEL} min-w-0 max-h-[35vh] md:max-h-none flex flex-col min-h-0 overflow-hidden`
  )
  // 상단 정보는 압축 — 에셋 목록이 더 위에서부터 보이게.
  const treeHeader = el('div', 'px-2.5 pt-2 pb-1.5 border-b border-[#d9a85c]/25 flex flex-col gap-1')
  // 설정 모달의 프로젝트 버튼 — 모바일 게임 버튼(그라데이션 + 아이콘 + hover 떠오름).
  const openButton = el('button', SETTINGS_BUTTON) as HTMLButtonElement
  openButton.append(editorIcon('folder', 22), el('span', '', '게임 폴더 열기'))
  openButton.type = 'button'
  const analyzeButton = el('button', SETTINGS_BUTTON_SPECIAL) as HTMLButtonElement
  // 분석 버튼은 진행 상태에 따라 라벨만 갈아끼운다(textContent를 통째로 바꾸면 아이콘이 날아간다).
  const analyzeLabel = el('span', '', 'AI 게임 분석')
  analyzeButton.append(editorIcon('orb', 22), analyzeLabel)
  analyzeButton.type = 'button'
  const resetButton = el('button', SETTINGS_BUTTON) as HTMLButtonElement
  resetButton.append(editorIcon('building', 20), el('span', '', '게임으로 복귀'))
  resetButton.type = 'button'
  // 프로젝트 버튼(폴더 열기/분석/복귀)은 설정 모달로 이동했다. 사이드바는 엔티티 목록만.
  const treeHeaderTop = el('div', 'flex items-center justify-between gap-2')
  const treeTitle = el('div', 'flex items-center gap-2 text-[15px] font-semibold tracking-wide text-[#e6e6e6]')
  treeTitle.append(editorIcon('map', 16), el('span', '', '현재 맵 에셋'))
  treeHeaderTop.append(treeTitle)
  // 라이브 게임이 맵을 보고한 뒤에만 의미가 있는 토글(현재 맵만 ↔ 전체 맵). 그 전엔 숨긴다.
  const mapFilterToggle = el('button', 'text-[11px] text-[#9d9d9d] transition hover:text-[#d4d4d4]', '전체 보기') as HTMLButtonElement
  mapFilterToggle.type = 'button'
  mapFilterToggle.hidden = true
  treeHeaderTop.append(mapFilterToggle)
  // 선택 대상 카드 — 종류별 개수는 아래 카드 그리드가 보여주므로 헤더엔 타겟만 남긴다.
  // 선택 대상은 한 줄 정보바(36px) — 🎯 선택 대상 : 이름.
  // 선택 대상은 보조 정보 — 카드가 아니라 '현재 맵' 줄과 같은 레벨의 한 줄 텍스트.
  const summaryCard = el('div', 'h-[28px] px-1 flex items-center gap-1.5 opacity-80')
  summaryCard.append(editorIcon('target', 13))
  const targetText = el('div', 'flex items-baseline gap-1.5 min-w-0')
  const targetValue = el('div', 'text-[11px] font-medium text-[#e2bd8c] truncate')
  targetText.append(el('div', 'text-[11px] font-medium tracking-wide text-[#b59458] leading-none whitespace-nowrap', '선택 대상 :'), targetValue)
  summaryCard.append(targetText)
  // 선택 대상 표시 갱신(UI 표시 전용).
  const updateSummary = (): void => {
    targetValue.textContent = selectedEntity
      ? selectedEntity.name
      : isTargetOptional()
        ? '새 NPC 생성 (대상 불필요)'
        : '없음'
    // 선택된 동안에만 금색 — 강조는 "지금 선택된 것"에만 쓴다는 규칙.
    // 선택 없음 → 비활성처럼 흐리게, 선택 있음 → 일반 밝기의 금색.
    targetValue.className = selectedEntity
      ? 'text-[11px] font-medium text-[#e2bd8c] truncate'
      : 'text-[11px] font-medium text-[#777777]/70 truncate'
  }
  // 게임과의 동기화 상태(현재 맵 이름)를 보여주는 줄. 연결 전엔 대기 메시지.
  const treeSyncLine = el('div', 'text-[12px] font-medium text-[#c9a96b]', '게임과 연결 대기 중…')
  // 짧은 안내문 — 눈에 띄지 않는 흐린 브라운, 카드 그리드를 방해하지 않는 한 줄.
  // 에셋 검색 — 이름으로 실시간 필터링. NPC가 수십 개로 늘어나도 탐색기처럼 쓸 수 있다.
  // 검색이 패널의 첫 행동으로 보이게 — 40px 높이 + 살짝 밝은 배경 + focus 금색.
  const assetSearch = el('input', 'h-[40px] w-full rounded-[10px] border border-[#d9a85c]/28 bg-[#1a1a1a] px-3 text-[12px] text-[#d4d4d4] outline-none transition placeholder:text-[#919191] focus:border-[#d9a85c] focus:shadow-[0_0_8px_rgba(217,168,92,0.2)]') as HTMLInputElement
  assetSearch.type = 'search'
  assetSearch.placeholder = '🔍 NPC · 건물 · 오브젝트 검색'
  assetSearch.addEventListener('input', () => {
    assetQuery = assetSearch.value
    renderTree()
    render()
  })
  // 순서: 제목 → 검색 → 현재 맵 → 선택 대상 — 정보는 두 줄만, 설명문은 없앤다.
  treeHeader.append(treeHeaderTop, assetSearch, treeSyncLine, summaryCard)
  const treeList = el('div', 'flex-1 overflow-auto p-3 flex flex-col gap-3')
  tree.append(treeHeader, treeList)

  // ---------- center: 라이브 게임(위) + 프롬프트 컴포저(아래) ----------
  // min-w-0: grid 자식의 기본 min-width:auto 때문에 내용이 열을 밀어내는 것 방지.
  const center = el('main', '[grid-area:main] min-w-0 min-h-0 flex flex-col gap-2')
  // 선택 대상 배지 — 입력창 위 오른쪽의 작은 상태 캡슐(render()가 내용을 채운다).
  const targetLine = el('div', 'flex items-center')
  const analysisPanel = el('div', 'rounded-lg border border-[#d9a85c]/28 bg-[#252526] p-3 flex flex-col gap-1.5 text-[#d4d4d4]')
  analysisPanel.hidden = true
  const supportNote = el('div', 'rounded-lg border border-[#d9a85c]/22 bg-[#3a3122] px-3 py-2 text-xs text-[#d8b270]')

  const apiKeyField = el('label', 'flex flex-col gap-2')
  apiKeyField.append(el('span', SETTINGS_LABEL, 'API 키 — Claude 또는 GPT (자동 감지)'))
  const apiKeyInput = el('input', SETTINGS_INPUT) as HTMLInputElement
  apiKeyInput.type = 'password'
  apiKeyInput.placeholder = 'sk-ant-… (Claude)  또는  sk-… (GPT)'
  apiKeyInput.autocomplete = 'off'
  apiKeyInput.value = apiKey
  apiKeyField.append(apiKeyInput)
  // 키 유효성 피드백(입력 시 디바운스로 갱신). 빈 문자열이면 자리만 차지하지 않게 둔다.
  const apiKeyStatus = el('span', 'text-sm font-medium text-[#9d9d9d]', '')
  apiKeyField.append(apiKeyStatus)

  // 모델 선택 — 키는 모델을 정하지 않으므로, 감지된 provider의 모델 중에서 고른다(저장됨).
  // 드롭다운 대신 게임식 선택 버튼. 실제 상태는 숨겨진 select가 그대로 들고 있어
  // 기존 change 리스너·저장 로직이 전혀 바뀌지 않는다(칩 클릭 → select 값 변경 + change 디스패치).
  const modelField = el('div', 'flex flex-col gap-2.5')
  modelField.append(el('span', SETTINGS_LABEL, '모델 선택'))
  const providerChips: Record<LlmProvider, HTMLElement> = {
    anthropic: el('span', PROVIDER_CHIP, 'Claude'),
    openai: el('span', PROVIDER_CHIP, 'GPT')
  }
  const providerRow = el('div', 'flex items-center gap-2')
  providerRow.append(providerChips.anthropic, providerChips.openai)
  const modelChips = el('div', 'flex flex-wrap gap-2')
  const modelSelect = el('select', 'hidden') as HTMLSelectElement
  modelField.append(providerRow, modelChips, modelSelect)

  // 필요하면 손잡이로 더 늘릴 수 있다(resize-y). placeholder는 예시 목록 형태.
  const promptField = el('label', 'flex flex-col gap-1.5')
  // 화면에서 가장 강한 입력 요소 — 낮게 유지(최대 96px), 내용이 길면 스크롤.
  const promptInput = el('textarea', `${FIELD_INPUT} min-h-[80px] max-h-[96px] overflow-y-auto resize-y leading-[1.6]`) as HTMLTextAreaElement
  promptInput.placeholder =
    '예) 마법사가 플레이어에게 위험을 경고하는 대사를 추가해줘\n예) 숨겨진 퀘스트를 만들어줘\n예) 나무를 가을 분위기로 바꿔줘'
  promptField.append(promptInput)

  // 입력창 자동 채움 — 빈 화면보다 예시를 고쳐 쓰는 게 훨씬 쉽다.
  // (입력 이벤트를 쏴서 생성 버튼 활성화·진행 표시도 함께 갱신.)
  const fillPrompt = (text: string): void => {
    promptInput.value = text
    promptInput.dispatchEvent(new Event('input'))
    promptInput.focus()
  }

  // 빠른 시작 — 아이콘 카드. 클릭하면 입력창이 채워지고 카드에 '✓ 선택됨' 상태가 남아
  // "지금 내가 뭘 만드는 중인지"가 보인다(표시 전용 상태).
  // primary: 대표 액션(핵심 기능) 표시 — 카드가 한 단계 강조된다.
  const SUGGESTIONS = [
    { label: 'NPC 대사', desc: '대화 생성', text: '마법사가 플레이어에게 경고하는 대사를 추가해줘', primary: true },
    { label: '퀘스트', desc: '의뢰 생성', text: '마을 주민이 부탁하는 숨겨진 퀘스트를 만들어줘', primary: true },
    { label: '스타일 변경', desc: '외형 수정', text: '이 나무를 가을 분위기의 나무로 바꿔줘', primary: false },
    { label: 'NPC 추가', desc: '주민 생성', text: '마을에 새로운 주민 NPC를 추가해줘', primary: false }
  ]
  let activeSuggestion: string | undefined
  // 빠른 템플릿 선택 — 기본은 완전 중립(회색 테두리, 금색 없음). 약한 hover, 금색은 active(클릭)만.
  const QUICK_CARD =
    'h-[52px] flex flex-col items-start justify-center gap-1 rounded-lg px-3 text-left bg-[#2d2d30] border border-[#3c3c3c] transition duration-[180ms] ease-out hover:bg-[#333333] hover:border-[#4a4a4a]'
  // 대표 액션(NPC 대사·퀘스트)도 기본은 다른 카드와 똑같은 중립으로 둔다(초기 진입 시
  // 선택된 것처럼 미리 강조되면 안 됨). 강조는 active(클릭)에만.
  const QUICK_CARD_PRIMARY = QUICK_CARD
  const QUICK_CARD_ACTIVE =
    'h-[52px] flex flex-col items-start justify-center gap-1 rounded-lg px-3 text-left bg-gradient-to-b from-[#e7b15a]/15 to-[#e7b15a]/5 border border-[#e7b15a] shadow-[0_0_12px_rgba(231,177,90,0.18)] transition duration-[180ms] ease-out hover:-translate-y-[2px]'
  const quickStart = el('div', 'flex flex-wrap items-center gap-1.5')
  quickStart.append(el('span', 'text-[10px] leading-none text-[#777777]', '빠른 시작'))
  const suggestionRow = el('div', 'flex flex-wrap gap-1.5')
  const quickCards = SUGGESTIONS.map((suggestion) => {
    const card = el('button', suggestion.primary ? QUICK_CARD_PRIMARY : QUICK_CARD) as HTMLButtonElement
    card.type = 'button'
    const badge = el('span', 'text-[10px] leading-none text-[#e7b15a]', '✓')
    badge.hidden = true
    const titleLine = el('span', 'flex items-center gap-1.5')
    // 라벨도 기본은 중립 회색 — 금색 글자색은 선택 강조에만 쓴다.
    titleLine.append(
      el('span', 'text-[14px] leading-none text-[#d4d4d4]', suggestion.label),
      badge
    )
    card.append(
      titleLine,
      el('span', 'text-[11px] leading-none text-[#777777] opacity-65', suggestion.desc)
    )
    suggestionRow.append(card)
    return { label: suggestion.label, text: suggestion.text, primary: suggestion.primary, card, badge }
  })
  const updateQuickCards = (): void => {
    for (const { label, primary, card, badge } of quickCards) {
      const active = label === activeSuggestion
      card.className = active ? QUICK_CARD_ACTIVE : primary ? QUICK_CARD_PRIMARY : QUICK_CARD
      badge.hidden = !active
    }
  }
  for (const quickCard of quickCards) {
    quickCard.card.addEventListener('click', () => {
      activeSuggestion = quickCard.label
      // '퀘스트'만 2단계(후보→선택→검증) 모드. 다른 빠른시작은 기존 단일 생성 흐름.
      candidateMode = quickCard.label === '퀘스트'
      // 'NPC 추가'는 legend-of-lua에서만 "진짜 NPC 생성"(검증+카탈로그 주입) 모드로 바뀐다.
      // 그 외 게임에선 기존 대사 생성(entity_lines) 흐름 그대로다.
      luaNpcMode = quickCard.label === 'NPC 추가' && isLegendOfLuaGame(game)
      candidates = []
      selectedCandidateIndex = undefined
      editingCandidateIndex = undefined
      currentDryRun = undefined
      if (!candidateMode && activeBoardTab === 'candidates') {
        activeBoardTab = undefined
      }
      fillPrompt(quickCard.text)
      updateQuickCards()
      render()
    })
  }
  quickStart.append(suggestionRow)

  // 추천 의뢰 — 한 줄 칩. 설명은 툴팁(title)으로, 클릭하면 그대로 입력창에 들어간다.
  // quest: true인 추천 의뢰는 빠른시작 '퀘스트' 카드처럼 2단계(후보→선택) 모드로 들어간다.
  const RECOMMENDED: {
    title: string
    desc: string
    text: string
    quest?: boolean
  }[] = [
    { title: '📜 마법사의 경고', desc: '플레이어에게 위험 경고', text: '마법사가 플레이어에게 위험을 경고하는 대사를 추가해줘' },
    { title: '📜 숨겨진 퀘스트', desc: '새로운 보상 의뢰', text: '마을 주민이 부탁하는 숨겨진 퀘스트를 만들어줘', quest: true },
    { title: '📜 계절 변화', desc: '가을 분위기로 변경', text: '이 나무를 가을 분위기의 나무로 바꿔줘' }
  ]
  // 도움말처럼 보이게: 더 어두운 배경 + 흐린 테두리 + 좌측 배지 — 입력창과 즉시 구분된다.
  const recommendBoard = el('div', 'flex flex-wrap items-center gap-1.5 rounded-lg bg-[#181818] border border-white/[0.06] px-2 py-1.5')
  recommendBoard.append(
    el('span', 'rounded px-2 py-0.5 text-[11px] leading-none text-[#d5b87a] bg-[#dca14b]/10 border border-[#dca14b]/25', '추천 의뢰')
  )
  for (const item of RECOMMENDED) {
    const chip = el('button', 'h-[26px] flex items-center rounded-full px-2.5 text-[11px] leading-none text-[#d5b87a]/75 bg-white/[0.02] border border-[#d9a85c]/18 transition hover:border-[#e7b15a] hover:bg-[#333333] hover:text-[#f2dfb3]', item.title) as HTMLButtonElement
    chip.type = 'button'
    chip.title = item.desc
    chip.addEventListener('click', () => {
      // 퀘스트형 추천은 빠른시작 '퀘스트' 카드와 동일하게 2단계(후보→선택) 모드로 들어간다.
      const isQuest = item.quest === true
      candidateMode = isQuest
      // 추천 의뢰엔 NPC 생성형이 없다 — NPC 모드는 항상 끈다.
      luaNpcMode = false
      candidates = []
      selectedCandidateIndex = undefined
      editingCandidateIndex = undefined
      currentDryRun = undefined
      activeSuggestion = isQuest ? '퀘스트' : undefined
      if (!candidateMode && activeBoardTab === 'candidates') {
        activeBoardTab = undefined
      }
      fillPrompt(item.text)
      updateQuickCards()
      render()
    })
    recommendBoard.append(chip)
  }

  const actions = el('div', 'flex flex-wrap items-end gap-2')
  // 메인 CTA — 적용/복사/내보내기보다 살짝만 크게(42px). 라벨 span만 갱신한다.
  const generateButton = el('button', PRIMARY_BUTTON) as HTMLButtonElement
  const generateLabel = el('span', 'text-[18px] font-bold leading-none', '✨ 이야기 생성')
  generateButton.append(generateLabel)
  generateButton.type = 'button'
  const applyButton = el('button', APPLY_BUTTON, '적용') as HTMLButtonElement
  applyButton.type = 'button'
  const copyButton = el('button', GHOST_BUTTON, '복사') as HTMLButtonElement
  copyButton.type = 'button'
  const exportButton = el('button', GHOST_BUTTON, '내보내기') as HTMLButtonElement
  exportButton.type = 'button'
  // 큰 액션 2개(왼쪽) + 보조 2개(오른쪽) + 단축키 힌트.
  // 좌측 = 주요 작업(생성+적용, 설명은 버튼 아래) / 우측 = 보조 작업(복사·내보내기·안내).
  const primaryRow = el('div', 'flex flex-wrap items-center gap-2')
  primaryRow.append(generateButton, applyButton)
  const primaryGroup = el('div', 'flex flex-col gap-1')
  primaryGroup.append(
    primaryRow,
    el('span', 'text-[11px] leading-[1.3] text-[#777777] opacity-70', '선택한 대상에 새로운 이야기를 생성합니다.')
  )
  const utilityGroup = el('div', 'flex flex-wrap items-center gap-1.5')
  utilityGroup.append(
    copyButton,
    exportButton,
    // 결과 보드로 이어지는 시선 안내 + 단축키를 한 줄에 병합.
    el('span', 'hidden sm:inline text-[10px] text-[#777777]', '생성 후 → 결과 보드에서 확인 · ⌘/Ctrl+Enter')
  )
  actions.append(primaryGroup, el('div', 'flex-1'), utilityGroup)

  const status = el('div', 'text-sm text-[#9d9d9d] min-h-[1.25rem]')
  const validationLine = el('div', 'text-xs')
  validationLine.hidden = true

  // ---------- 결과 보드: 위 목록(4줄) + 아래 단일 상세 창 (퀘스트 로그식 마스터-디테일) ----------
  type BoardTab = 'lua' | 'files' | 'verify' | 'apply' | 'candidates'
  // 표시 전용 상태 — 어떤 항목의 상세를 보여줄지. 처음엔 미선택("항목을 선택하세요").
  let activeBoardTab: BoardTab | undefined
  // 상태 카드형 목록(52px): 제목 + 짧은 상태 텍스트 + 우측 화살표. hover에서 화살표도 같이 강조.
  const BOARD_ROW =
    'group h-[52px] shrink-0 flex items-center gap-2 rounded-xl border border-[#d9a85c]/22 bg-[#2d2d30] px-3 text-left transition duration-150 hover:bg-[#333333] hover:border-[#d9a85c]/70'
  const BOARD_ROW_ACTIVE =
    'group h-[52px] shrink-0 flex items-center gap-2 rounded-xl border border-[#d9a85c] bg-[#3a2416] px-3 text-left transition duration-150 shadow-[0_0_10px_rgba(217,168,92,0.3)]'
  // 상세 창의 항목별 내용 — 항상 하나의 상세 창 안에서 토글된다(새 창 생성 금지).
  const makeDetailView = (title: string): { view: HTMLElement; body: HTMLElement } => {
    const view = el('div', 'flex flex-col gap-2')
    view.hidden = true
    view.append(el('div', 'text-[15px] text-[#e8d5a5] [text-shadow:0_1px_0_rgba(0,0,0,0.35)] pb-1.5 border-b border-[#d9a85c]/20', title))
    const body = el('div', 'flex flex-col gap-1')
    view.append(body)
    return { view, body }
  }
  const luaView = makeDetailView('생성된 Lua 코드')
  const luaStatus = el('div', 'text-[12px] text-[#9d9d9d]', '생성 후 표시됩니다')
  const result = el('pre', 'm-0 max-h-[36vh] overflow-auto text-[12px] leading-relaxed text-[#d4d4d4] whitespace-pre-wrap break-words')
  result.hidden = true
  luaView.body.append(luaStatus, result)
  const filesView = makeDetailView('변경 예정 파일')
  const filesStatus = el('div', 'text-[12px] text-[#9d9d9d]', '변경 파일 없음')
  filesView.body.append(filesStatus)
  const verifyView = makeDetailView('검증 결과')
  // 무결성 검증(드라이런) 단계별 결과 — verify 상세에 validationLine과 함께 표시(render가 채움).
  const dryRunBox = el('div', 'flex flex-col gap-1 pt-1')
  dryRunBox.hidden = true
  // 퀘스트 1단계 후보 카드 영역(render의 renderCandidates가 채움).
  const candidatesView = makeDetailView('퀘스트 후보 (하나를 선택하세요)')
  const applyView = makeDetailView('적용 상태')
  const applyStatus = el('div', 'text-[12px] text-[#9d9d9d]', '대기 중')
  applyView.body.append(applyStatus)

  // 위쪽 목록 — 클릭하면 아래 상세 창의 내용만 바뀐다.
  const boardList = el('div', 'flex flex-col gap-1.5')
  const BOARD_TABS: Array<{ id: BoardTab; label: string }> = [
    { id: 'lua', label: '생성된 Lua 코드' },
    { id: 'files', label: '변경 예정 파일' },
    { id: 'verify', label: '검증 결과' },
    { id: 'apply', label: '적용 상태' }
  ]
  // 아래 상세 창 — 깊은 차콜 + 중립 테두리의 둥근 카드 하나.
  const boardDetail = el('div', 'flex-1 min-h-[240px] rounded-2xl border border-[#d9a85c]/28 bg-[#1a1a1a] p-3.5 flex flex-col overflow-auto')
  // 빈 상태엔 검은 공간 대신 '오늘 작업' 요약을 보여준다(값은 render()가 갱신).
  // '오늘 작업' 통계 카드 — 결과 카드 4장 바로 아래에 붙는 독립 카드(라벨/숫자 분리, 숫자 강조).
  const todayCard = el('div', 'shrink-0 w-full rounded-lg border border-[#d9a85c]/22 bg-[#2d2d30] px-3 py-2.5 flex flex-col gap-1.5 text-left')
  const todayStats = el('div', 'flex flex-col gap-1.5')
  const statGen = el('span', 'text-[12px] font-bold leading-none text-[#e8d5a5]', '0')
  const statPass = el('span', 'text-[12px] font-bold leading-none text-[#e8d5a5]', '0')
  const statApply = el('span', 'text-[12px] font-bold leading-none text-[#e8d5a5]', '0')
  const statGenRow = el('div', 'flex items-center justify-between')
  statGenRow.append(el('span', 'text-[11px] font-medium leading-none text-[#9d9d9d]', '생성 요청'), statGen)
  const statPassRow = el('div', 'flex items-center justify-between')
  statPassRow.append(el('span', 'text-[11px] font-medium leading-none text-[#9d9d9d]', '검증 통과'), statPass)
  const statApplyRow = el('div', 'flex items-center justify-between')
  statApplyRow.append(el('span', 'text-[11px] font-medium leading-none text-[#9d9d9d]', '게임 적용'), statApply)
  todayStats.append(statGenRow, statPassRow, statApplyRow)
  todayCard.append(
    el('div', 'text-[12px] leading-none text-[#e8d5a5]', '오늘 작업'),
    todayStats
  )
  // '최근 작업' 카드 — 오늘 작업과 같은 톤의 작은 기록 카드(작업명 ── 시간 한 줄). 빈 상태에서도 낮게.
  const recentCard = el('div', 'shrink-0 w-full min-h-[84px] rounded-lg border border-[#d9a85c]/22 bg-[#2d2d30] px-3 py-2.5 flex flex-col gap-1.5 text-left')
  const recentList = el('div', 'flex flex-col gap-1.5')
  recentCard.append(
    el('div', 'text-[12px] leading-none text-[#e8d5a5]', '최근 작업'),
    recentList
  )
  // 빈 상태 안내 — 설명 패널처럼 보이게 텍스트 블록을 중앙보다 살짝 위에 둔다.
  const detailPlaceholder = el('div', 'flex-1 flex flex-col items-center justify-center gap-1.5 pt-2 pb-8 text-center leading-relaxed')
  detailPlaceholder.append(
    el('div', 'text-[13px] font-semibold text-[#d9a85c]', '결과 없음'),
    el('div', 'whitespace-pre-line text-[11px] leading-[1.5] text-[#777777] opacity-70', '왼쪽 패널에서 에셋을 선택하고\n이야기를 생성하면 여기에 표시됩니다.'),
    el('div', 'text-[11px] leading-[1.5] text-[#9d9d9d] opacity-80', '생성 → 검증 → 적용 결과를 이 영역에서 확인할 수 있습니다.')
  )
  boardDetail.append(detailPlaceholder, candidatesView.view, luaView.view, filesView.view, verifyView.view, applyView.view)
  const boardRows = BOARD_TABS.map((tab) => {
    const row = el('button', BOARD_ROW) as HTMLButtonElement
    // 제목(좌) + 상태 캡슐 배지(우, render()가 채움) + ▸ 화살표(클릭하면 아래 상세가 열린다는 신호).
    const rowStatus = el('span', 'h-[20px] flex items-center rounded-full px-2 text-[10px] font-semibold leading-none whitespace-nowrap bg-[#333333] text-[#9d9d9d]', '대기')
    row.append(
      el('span', 'truncate text-[13px] leading-none text-[#e8d5a5]', tab.label),
      el('span', 'ml-auto flex items-center gap-1.5'),
      rowStatus,
      el('span', 'text-[11px] text-[#777777] transition group-hover:text-[#e7b15a]', '▸')
    )
    row.type = 'button'
    boardList.append(row)
    return { id: tab.id, row, status: rowStatus }
  })
  const updateBoard = (): void => {
    for (const { id, row, status } of boardRows) {
      const active = id === activeBoardTab
      row.className = active ? BOARD_ROW_ACTIVE : BOARD_ROW
      // 선택된 행은 상태 배지도 금색 링으로 함께 강조(표시 전용).
      status.classList.toggle('ring-1', active)
      status.classList.toggle('ring-[#e7b15a]/60', active)
    }
    detailPlaceholder.hidden = activeBoardTab !== undefined
    candidatesView.view.hidden = activeBoardTab !== 'candidates'
    luaView.view.hidden = activeBoardTab !== 'lua'
    filesView.view.hidden = activeBoardTab !== 'files'
    verifyView.view.hidden = activeBoardTab !== 'verify'
    applyView.view.hidden = activeBoardTab !== 'apply'
  }
  for (const { id, row } of boardRows) {
    row.addEventListener('click', () => {
      activeBoardTab = id
      updateBoard()
    })
  }

  // ---------- Evaluator (사람 이진 평가) ----------
  const evaluationWrap = el('div', CARD)
  evaluationWrap.hidden = true
  // flex-wrap: 좁은 사이드바에선 지표 줄이 라벨 옆에 끼어 두 줄 컬럼으로 뭉개지는 대신 제 줄로 내려간다.
  const evaluationTop = el('div', 'flex flex-wrap items-center justify-between gap-x-2 gap-y-1')
  evaluationTop.append(
    el('span', LABEL, 'Evaluator · 사람 이진 평가')
  )
  const acceptanceStat = el('span', 'text-xs text-[#9d9d9d]')
  evaluationTop.append(acceptanceStat)
  const evaluationButtons = el('div', 'flex flex-wrap items-center gap-2')
  const acceptButton = el('button', GHOST_BUTTON, '수용') as HTMLButtonElement
  acceptButton.type = 'button'
  const rejectButton = el('button', GHOST_BUTTON, '거부') as HTMLButtonElement
  rejectButton.type = 'button'
  const evaluationVerdict = el('span', 'text-xs flex-1')
  const resetEvaluationsButton = el('button', 'text-[11px] text-[#9d9d9d] transition hover:text-[#d4d4d4]', '누적 기록 초기화') as HTMLButtonElement
  resetEvaluationsButton.type = 'button'
  evaluationButtons.append(acceptButton, rejectButton, evaluationVerdict, resetEvaluationsButton)
  evaluationWrap.append(evaluationTop, evaluationButtons)

  const historyWrap = el('div', 'flex flex-col gap-1.5')
  historyWrap.hidden = true
  const historyHeader = el('div', 'flex items-center justify-between')
  historyHeader.append(
    el('span', LABEL, '생성 히스토리')
  )
  const clearHistoryButton = el('button', 'text-[11px] text-[#9d9d9d] transition hover:text-[#d4d4d4]', '비우기') as HTMLButtonElement
  clearHistoryButton.type = 'button'
  historyHeader.append(clearHistoryButton)
  const historyList = el('div', 'flex flex-col gap-1')
  historyWrap.append(historyHeader, historyList)

  // ---------- center 상단: live game preview ----------
  // min-h 바닥: 어떤 창 크기에서도 게임이 HUD만 보이는 납작한 띠로 짓눌리지 않게 한다.
  // 게임 화면이 이 화면의 주인공 — 프레임(위 탭 바·아래 단계 바)은 얇고 어둡게 유지한다.
  const preview = el('section', `flex-1 min-h-[200px] md:min-h-[300px] min-w-0 flex flex-col ${PANEL} overflow-hidden`)
  // 게임 화면 위에 붙은 작은 RPG 조작 패널 — 짧은 제목 + 나무 탭 + 아이콘 버튼.
  const previewBar = el('div', 'settings-game-font select-none h-9 shrink-0 flex items-center justify-between gap-2 px-3 border-b border-[#d9a85c]/22 bg-[#252526] min-w-0')
  // 시선이 요청 패널로 먼저 가도록 게임 화면 헤더는 한 톤 차분하게.
  const previewTitle = el('span', 'flex items-center gap-2 truncate min-w-0')
  previewTitle.append(
    editorIcon('map', 14),
    el('span', 'truncate text-[13px] font-medium leading-none text-[#9d9d9d]', '게임 화면'),
    el('span', 'hidden lg:inline text-[10px] leading-none text-[#7a6a52]', '현재 실행 중')
  )
  // 맵 요약(현재 맵 + 개체 수) — 텍스트 나열 대신 정보 칩(pill)로 분리해 한눈에 읽히게.
  // 줄바꿈 금지: 공간이 모자라면 overflow-hidden으로 끝 칩부터 잘린다(헤더 높이 고정).
  const STAT_CHIP =
    'h-[22px] shrink-0 inline-flex items-center gap-1 px-2 rounded-full border border-[#d9a85c]/35 bg-[#2d2d30]/65 text-[11px] font-semibold leading-none text-[#e8d3a3] whitespace-nowrap'
  const STAT_CHIP_MAP =
    'h-[22px] shrink-0 inline-flex items-center px-2.5 rounded-full border border-[#d9a85c]/50 bg-[#d9a85c]/[0.18] text-[11px] font-semibold leading-none text-[#f3d88b] whitespace-nowrap'
  // translate-y-[2px]: 헤더 수직 중앙에 더 가깝게 정렬.
  const previewStats = el('span', 'hidden md:flex flex-1 items-center justify-center gap-1.5 leading-none min-w-0 overflow-hidden translate-y-[2px]')
  // 표시 전용: 현재 맵의 NPC/건물/포털/기타 개수를 한 줄로 요약한다(render()가 갱신).
  const updatePreviewStats = (): void => {
    const focusMap =
      currentMapId !== undefined
        ? game.maps.find((map) => map.id === currentMapId)
        : undefined
    const map = focusMap ?? game.maps[0]
    if (!map) {
      previewStats.replaceChildren()
      return
    }
    const counts = { npc: 0, building: 0, portal: 0, other: 0 }
    for (const entity of map.entities) {
      const kind = groupKindOf(entity.kind)
      if (kind === 'npc') {
        counts.npc += 1
      } else if (kind === 'building') {
        counts.building += 1
      } else if (kind === 'portal') {
        counts.portal += 1
      } else {
        counts.other += 1
      }
    }
    // 라벨은 살짝 흐리게, 숫자는 굵은 금색으로 — 수치가 먼저 읽히게(표시 전용).
    const statChip = (label: string, value: number): HTMLElement => {
      const chip = el('span', STAT_CHIP)
      chip.append(
        el('span', 'opacity-75', label),
        el('span', 'font-bold text-[#f3d88b]', String(value))
      )
      return chip
    }
    const mapChip = el('span', STAT_CHIP_MAP, map.name)
    mapChip.title = `현재 맵: ${map.name}`
    previewStats.replaceChildren(
      mapChip,
      statChip('NPC', counts.npc),
      statChip('건물', counts.building),
      statChip('포털', counts.portal),
      statChip('오브젝트', counts.other)
    )
  }
  previewBar.append(previewTitle, previewStats)
  const previewActions = el('div', 'flex items-center gap-1.5 shrink-0')
  // 맵 전환 — 프리뷰는 항상 my-sample-rpg를 실행하므로 그 게임의 씬(마을/사냥터/동굴)을 바꾼다.
  // 탭마다 게임풍 아이콘: 마을→집, 사냥터→검, 동굴→수정.
  const previewScenes: Array<{ id: string; label: string; icon: EditorIconName }> = [
    { id: 'town', label: '마을', icon: 'building' },
    { id: 'hunting-ground', label: '사냥터', icon: 'sword' },
    { id: 'cave', label: '동굴', icon: 'crystal' }
  ]
  const mapSwitcher = el('div', 'flex items-center gap-1')
  // 새 창/새로고침은 아이콘 버튼으로 — 의미는 title(툴팁)로 유지한다.
  const popoutButton = el('button', 'w-[26px] h-[26px] flex items-center justify-center rounded-lg bg-[#2d2d30] border border-[#d9a85c]/22 text-[13px] leading-none text-[#9d9d9d] transition hover:bg-[#333333] hover:text-[#ead8b6]', '↗') as HTMLButtonElement
  popoutButton.type = 'button'
  popoutButton.title = '새 창에서 열기'
  const reloadButton = el('button', 'w-[26px] h-[26px] flex items-center justify-center rounded-lg bg-[#2d2d30] border border-[#d9a85c]/22 text-[13px] leading-none text-[#9d9d9d] transition hover:bg-[#333333] hover:text-[#ead8b6]', '↻') as HTMLButtonElement
  reloadButton.type = 'button'
  reloadButton.title = '게임 새로고침'
  previewActions.append(mapSwitcher, popoutButton, reloadButton)
  previewBar.append(previewActions)
  // 게임 스테이지 — 16:9 게임 화면을 패널 안에 '맞춰 넣는'(contain) 방식. 게임의 대사 박스가
  // 화면 상단에 그려지는데, 예전 cover-크롭은 그 상단을 잘라 대사가 가려졌다. 그래서 화면 전체가
  // 항상 보이도록 contain으로 바꾼다 — 남는 가장자리는 검은 레터박스로 둔다. 중앙 정렬.
  // ---------- 게임에 맞춰 프리뷰 iframe 소스 결정 ----------
  // love.js 웹빌드가 있는 게임(legend-of-lua)은 그 빌드를, 그 외(my-sample-rpg)는 기본 게임 URL을
  // iframe에 띄운다. 이전에 설정에서 저장한 웹빌드 URL(localStorage)이 있으면 그것을 우선한다.
  const WEB_BUILD_URL_STORAGE_KEY = 'my-sample-rpg:web-build-url'
  const previewSrcForGame = (): string => {
    const stored = (readLocalStorage(WEB_BUILD_URL_STORAGE_KEY) ?? '').trim()
    const webBuild =
      stored.length > 0 ? stored : (game.adapter.defaultWebBuildUrl ?? '').trim()
    return webBuild.length > 0 ? webBuild : gamePreviewUrl
  }
  const previewStage = el('div', 'relative flex-1 min-h-0 flex items-center justify-center overflow-hidden bg-[#181818]')
  const iframe = el('iframe', 'shrink-0 border-0 bg-black') as HTMLIFrameElement
  iframe.src = previewSrcForGame()
  iframe.title = '게임 프리뷰'
  previewStage.append(iframe)
  // love.js 빌드는 wasm·게임 데이터를 받느라 첫 로드에 시간이 걸린다 — 빈 화면 대신 로딩 안내를
  // 띄우고, iframe load 이벤트에서 지운다.
  const previewLoading = el(
    'div',
    'absolute inset-0 z-10 flex items-center justify-center text-[13px] text-[#ead8b6] bg-[#181818]/85 pointer-events-none',
    '🎮 게임 로딩 중…'
  )
  previewLoading.style.display = 'none'
  previewStage.append(previewLoading)
  // 표시 전용: 스테이지 크기가 바뀔 때마다 16:9 게임 화면을 패널 안에 '맞춰 넣는'(contain) 크기로
  // 다시 맞춘다. 가로/세로 중 더 빡빡한 쪽에 맞춰 전체가 보이게 하고(잘림 없음), 남는 쪽은
  // 레터박스로 둔다. 상단 대사 박스가 잘리지 않는 것이 우선이라 cover-크롭/TOP_TRIM은 쓰지 않는다.
  const fitGameFrame = (): void => {
    const stageWidth = previewStage.clientWidth
    const stageHeight = previewStage.clientHeight
    if (stageWidth <= 0 || stageHeight <= 0) {
      return
    }
    const width = Math.min(stageWidth, (stageHeight * 16) / 9)
    iframe.style.width = `${Math.floor(width)}px`
    iframe.style.height = `${Math.floor((width * 9) / 16)}px`
  }
  new ResizeObserver(fitGameFrame).observe(previewStage)
  iframe.addEventListener('load', () => {
    connection.className = 'h-7 flex items-center gap-1.5 text-[11px] rounded-full px-2.5 bg-[#72d36b]/10 border border-[#72d36b]/50 text-[#9fe296]'
    connectionDot.className = 'w-2 h-2 rounded-full bg-[#72d36b] shadow-[0_0_6px_rgba(114,211,107,0.8)]'
    connectionLabel.textContent = 'AI 연결됨'
    previewLoading.style.display = 'none'
  })

  // 게임이 바뀌면(폴더 열기/복귀) 프리뷰 iframe을 그 게임 URL로 다시 가리킨다. URL이 같으면
  // 불필요한 재로드를 피한다(같은 게임 재선택 등).
  const syncPreviewToGame = (): void => {
    const src = previewSrcForGame()
    if (iframe.src === new URL(src, location.href).href) {
      return
    }
    previewLoading.style.display = 'flex'
    iframe.src = src
  }
  // 상단 맵/씬 버튼은 로드된 게임에 맞춰 구성한다:
  // - my-sample-rpg: 큐레이션된 씬(마을/사냥터/동굴) + 아이콘, 'editor:switch-scene' 전송
  //   (게임이 'game:scene-changed'로 되보고 → currentMapId 갱신).
  // - 그 외(legend-of-lua 등): 실제 맵 목록(game.maps), 'editor:goto-map' 전송. 되보고가 없으므로
  //   버튼이 currentMapId의 주인 — 클릭 시 직접 집중·강조한다.
  let mapSwitcherButtons: Array<{ id: string; button: HTMLButtonElement }> = []
  // 게임이 보고/선택한 현재 맵의 탭을 금색으로 강조한다(표시 전용). rpg는 연결 전 기본 맵(town).
  const updateSceneTabs = (): void => {
    const activeId =
      currentMapId ?? (game.adapter.id === 'my-sample-rpg' ? 'town' : undefined)
    for (const { id, button } of mapSwitcherButtons) {
      button.className = id === activeId ? SCENE_TAB_ACTIVE : SCENE_TAB
    }
  }
  const renderMapSwitcher = (): void => {
    const isRpg = game.adapter.id === 'my-sample-rpg'
    const entries: Array<{ id: string; label: string; icon: EditorIconName }> = isRpg
      ? previewScenes
      : game.maps.map((map) => ({ id: map.id, label: map.name, icon: 'map' as const }))
    mapSwitcherButtons = entries.map((entry) => {
      const button = el('button', SCENE_TAB) as HTMLButtonElement
      button.append(editorIcon(entry.icon, 12), el('span', '', entry.label))
      button.type = 'button'
      button.addEventListener('click', () => {
        if (isRpg) {
          iframe.contentWindow?.postMessage(
            { type: 'editor:switch-scene', sceneId: entry.id },
            '*'
          )
          return
        }
        currentMapId = entry.id
        showAllMaps = false
        iframe.contentWindow?.postMessage(
          { type: 'editor:goto-map', mapId: entry.id, mapName: entry.label },
          '*'
        )
        renderTree()
        render()
        updateSceneTabs()
      })
      return { id: entry.id, button }
    })
    mapSwitcher.replaceChildren(...mapSwitcherButtons.map((b) => b.button))
    updateSceneTabs()
  }
  renderMapSwitcher()
  // 진행 단계 바 — RPG 퀘스트 진행도처럼 ①~⑤ 번호 캡슐 + 화살표.
  const FLOW_STEPS = ['선택', '작성', '생성', '확인', '적용']
  const stepBar = el('div', 'flex flex-wrap items-center gap-1.5')
  const stepPills = FLOW_STEPS.map((label, index) => {
    const pill = el('span', STEP_PILL)
    const num = el('span', STEP_NUM, String(index + 1))
    const text = el('span', STEP_TEXT, label)
    pill.append(num, text)
    return { pill, num, text }
  })
  // 단계 사이 연결선(→) — 진행된 구간은 금색, 남은 구간은 어두운 브론즈로 칠한다.
  const stepArrows: HTMLElement[] = []
  stepPills.forEach(({ pill }, index) => {
    if (index > 0) {
      const arrow = el('span', 'text-[11px] text-[#d9a85c]/28 transition', '→')
      stepArrows.push(arrow)
      stepBar.append(arrow)
    }
    stepBar.append(pill)
  })
  // 진행 단계를 실제 상태와 연결(표시 전용): 완료엔 숫자 대신 ✓, 생성 중엔 로딩 문구.
  const updateStepBar = (): void => {
    const applied = currentResult !== undefined && appliedResults.has(currentResult)
    const hasPrompt = promptInput.value.trim().length > 0
    // 선택 전 0 → 작성 중 1 → 생성(작성 완료/생성 중) 2 → 확인 3 → 적용 후엔 5(전부 완료).
    // NPC 생성처럼 대상이 불필요한 모드는 0단계(대상 선택)를 건너뛴다.
    const activeStep = !selectedEntity && !isTargetOptional()
      ? 0
      : isGenerating
        ? 2
        : currentResult
          ? applied
            ? 5
            : 3
          : hasPrompt
            ? 2
            : 1
    stepPills.forEach(({ pill, num, text }, index) => {
      const isDone = index < activeStep
      const isActive = index === activeStep
      pill.className = isActive ? STEP_PILL_ACTIVE : isDone ? STEP_PILL_DONE : STEP_PILL
      num.className = isActive ? STEP_NUM_ACTIVE : isDone ? STEP_NUM_DONE : STEP_NUM
      num.textContent = isDone ? '✓' : String(index + 1)
      text.className = isActive ? STEP_TEXT_ACTIVE : isDone ? STEP_TEXT_DONE : STEP_TEXT
      text.textContent =
        isActive && index === 2 && isGenerating
          ? '생성 중...'
          : isDone && index === 4
            ? '적용 완료'
            : FLOW_STEPS[index] ?? ''
    })
    // 연결선 진행색: i번째 화살표는 i단계가 완료됐을 때 금색이 된다.
    stepArrows.forEach((arrow, index) => {
      arrow.className =
        index < activeStep
          ? 'text-[11px] text-[#d9a85c] transition'
          : 'text-[11px] text-[#d9a85c]/28 transition'
    })
  }
  preview.append(previewBar, previewStage)

  // ---------- center 하단: 프롬프트 컴포저 (마을 게시판/퀘스트 보드 카드) ----------
  // max-h+스크롤: 창이 낮을 때 컴포저가 게임 영역을 통째로 밀어내지 않게 한다.
  // settings-game-font: ESC/왼쪽 패널과 같은 둥근 픽셀 폰트로 통일.
  // 화면의 주인공 — 다른 패널보다 밝은 금색 그라데이션 테두리(과한 glow 없이 은은하게).
  // 세로를 아끼려 제목·설명을 한 줄에 같이 두고, 높이 상한도 낮춰 게임 화면에 공간을 양보한다.
  // 화면의 주인공 '마을 의뢰서' — 다른 패널보다 밝은 배경 + 2px 금색 테두리,
  // 입력에 포커스되면 은은한 발광(focus-within)으로 "여기에 쓰면 된다"가 바로 보이게.
  // 게임 화면이 주인공 — 의뢰서는 화면의 약 1/3 이하로 압축한다.
  const composer = el('div', 'settings-game-font shrink-0 max-h-[36%] overflow-y-auto rounded-xl box-grad-border box-grad-border--strong box-grad-border--thick [--bgb:#252526] text-[#d4d4d4] p-2.5 flex flex-col gap-1.5 transition focus-within:shadow-[0_0_20px_rgba(222,170,90,0.25)]')
  // 제목은 하나, 설명도 한 줄만 — 정보를 줄여 흐름(선택→작성→생성)이 먼저 읽히게.
  const composerTitle = el('div', 'flex flex-wrap items-baseline gap-x-2.5 gap-y-0.5 min-w-0')
  const composerTitleRow = el('div', 'flex items-center gap-2')
  composerTitleRow.append(
    editorIcon('scroll', 20),
    el('span', 'text-[18px] font-semibold leading-none text-[#f3d7a2]', '마을 의뢰서')
  )
  composerTitle.append(
    composerTitleRow,
    el('span', 'text-[13px] text-[#9d9d9d] opacity-75', '선택한 대상에게 원하는 이야기나 변화를 작성하세요.')
  )
  // 우측 상단: 진행 상태 — 완료는 초록 '…완료', 현재는 금색 '…중', 미완료는 흐리게.
  const COMPOSER_STEPS = [
    { todo: '① 대상 선택', doing: '① 대상 선택 중', done: '① 대상 선택 완료' },
    { todo: '② 요청 작성', doing: '② 요청 작성 중', done: '② 요청 작성 완료' },
    { todo: '③ 생성 대기', doing: '③ 생성 대기', done: '③ 생성 완료' }
  ]
  const composerStepEls = COMPOSER_STEPS.map((step) =>
    el('span', 'text-[11px] text-[#9d9d9d] opacity-45', step.todo)
  )
  const composerSteps = el('div', 'hidden md:flex items-center gap-2')
  composerSteps.append(...composerStepEls)
  // 진행 상태 갱신(표시 전용): 선택 전 → ①, 요청 비었으면 → ②, 채워지면 → ③.
  const updateComposerSteps = (): void => {
    const current =
      !selectedEntity && !isTargetOptional()
        ? 0
        : promptInput.value.trim().length === 0
          ? 1
          : 2
    composerStepEls.forEach((node, index) => {
      const step = COMPOSER_STEPS[index]
      if (!step) {
        return
      }
      if (index < current) {
        node.className = 'text-[11px] text-[#78c26d] opacity-60'
        node.textContent = step.done
      } else if (index === current) {
        // 현재 단계만 금색 알약으로 또렷하게.
        node.className = 'text-[11px] leading-none text-white bg-[#d09a4c] rounded-full px-2 py-1'
        node.textContent = step.doing
      } else {
        node.className = 'text-[11px] text-[#9d9d9d] opacity-40'
        node.textContent = step.todo
      }
    })
  }
  const composerRight = el('div', 'flex items-center gap-3 shrink-0')
  composerRight.append(composerSteps, targetLine)
  const composerTop = el('div', 'flex flex-wrap items-start justify-between gap-2')
  composerTop.append(composerTitle, composerRight)
  // 제목 → 빠른 시작 → 입력창 → 추천 의뢰 → 생성 버튼: 보조 행은 전부 다른 줄에 병합했다.
  // 최근 생성 결과 한 줄 — 처음엔 발표용 예시, 실제 결과가 생기면 그 라벨로 바뀐다(render()가 갱신).
  const recentResultLine = el('div', 'text-[10px] leading-[1.4] text-[#777777]')
  composer.append(composerTop, supportNote, quickStart, promptField, recommendBoard, actions, status, recentResultLine)
  center.append(preview, composer)

  // ---------- right: 생성 결과 사이드바 ----------
  // 결과 사이드는 보조 정보 — 패널 자체를 본문보다 살짝 더 어둡게 가라앉힌다.
  const side = el(
    'aside',
    // settings-game-font: 왼쪽 패널·컴포저와 같은 둥근 픽셀 폰트로 통일.
    'settings-game-font [grid-area:side] rounded-xl box-grad-border [--bgb:#252526] text-[#d4d4d4] min-w-0 min-h-0 overflow-y-auto p-3.5 flex flex-col gap-3'
  )
  const sideTitle = el('div', 'flex items-center gap-2 text-[15px] font-semibold text-[#e6e6e6] pb-2 border-b border-[#d5a14f]/30')
  sideTitle.append(el('span', '', '결과 보드'))
  // 결과가 없을 때만 보이는 작은 안내문(큰 빈 카드 대신).
  const boardHint = el('div', 'text-[11px] text-[#777777]', '생성 결과와 검증 상태를 확인하세요.')
  // 검증 표시(render()가 갱신)는 검증 섹션 본문 안에 산다. 검증 전엔 대기 한 줄.
  const validationEmpty = el('div', 'text-[12px] text-[#9d9d9d]', '대기 중')
  verifyView.body.append(validationEmpty, validationLine, dryRunBox)
  side.append(
    sideTitle,
    analysisPanel,
    boardHint,
    boardList,
    todayCard,
    recentCard,
    boardDetail,
    evaluationWrap,
    historyWrap
  )

  body.append(tree, center, side)
  // ---------- settings modal (헤더 ⚙) ----------
  // API 키·폴더 열기·분석·복귀는 상시 노출 대신 여기로 모은다. 메인은 편집에 집중.
  const settingsBackdrop = el('div', 'fixed inset-0 z-50 bg-black/60 backdrop-blur flex items-center justify-center p-4')
  // 숨김은 hidden 속성 대신 인라인 display로 제어한다 — `flex` 클래스의 display:flex가 [hidden]을
  // 덮어써 안 닫히는 사고를 막는다(인라인 스타일이 항상 이긴다).
  settingsBackdrop.style.display = 'none'
  // VSCode 설정창 톤: 차콜 패널 + 중립 테두리 + 깊은 그림자(크기·여백은 기존 그대로).
  const settingsPanel = el('div', 'settings-game-font relative w-full max-w-[600px] rounded-[24px] border border-[#57575e] bg-[#3a3a3f] text-[#e0e0e0] p-6 pt-5 flex flex-col gap-4 shadow-[0_10px_40px_rgba(0,0,0,0.55)] max-h-[90vh] overflow-y-auto')
  const settingsTitle = el('div', 'flex items-center justify-center gap-2.5 pb-3 border-b border-[#5a5a61]')
  settingsTitle.append(
    editorIcon('gear', 26),
    el('span', 'text-[27px] leading-none tracking-wide text-[#e6e6e6]', '공방 설정')
  )
  // 우측 상단 원형 닫기 버튼 — 기본은 연회색, hover 시 VSCode 닫기 레드.
  const settingsClose = el('button', 'absolute top-3.5 right-3.5 w-9 h-9 rounded-full bg-[#4a4a50] text-[#d9d9d9] text-base font-semibold leading-none border border-[#5e5e66] transition hover:bg-[#c42b1c] hover:text-white hover:-translate-y-0.5 active:translate-y-0', '✕') as HTMLButtonElement
  settingsClose.type = 'button'
  const modelSection = el('div', SETTINGS_SECTION)
  modelSection.append(modelField)
  const projectSection = el('div', SETTINGS_SECTION)
  projectSection.append(el('span', SETTINGS_LABEL, '프로젝트'), openButton, analyzeButton, resetButton)
  // 고급 설정(API 키) — 일반 사용자에겐 보이지 않게 기본 접힘.
  // 개발자용 영역이라 기본적으로 눈에 띄지 않게 — 작고 연한 토글.
  const advancedToggle = el('button', 'self-start text-[13px] text-[#9d9d9d] transition hover:text-[#cccccc]', '▸ 고급 설정 (API 키)') as HTMLButtonElement
  advancedToggle.type = 'button'
  const advancedBody = el('div', SETTINGS_SECTION)
  advancedBody.hidden = true
  // API 키 + 현재 모델 배지 — 모델명은 헤더 대신 여기서만 보인다.
  advancedBody.append(apiKeyField, modelBadge)
  advancedToggle.addEventListener('click', () => {
    advancedBody.hidden = !advancedBody.hidden
    advancedToggle.textContent = `${advancedBody.hidden ? '▸' : '▾'} 고급 설정 (API 키)`
  })
  settingsPanel.append(settingsTitle, settingsClose, modelSection, projectSection, advancedToggle, advancedBody)
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

  // 진행 단계 스트립 — 헤더 바로 아래 한 줄(퀘스트 진행 UI, 화살표 없음).
  const stepStrip = el('div', 'settings-game-font select-none shrink-0 flex px-4 py-1 border-b border-[#d9a85c]/22 bg-[#252526]')
  stepStrip.append(stepBar)
  root.append(header, stepStrip, body, settingsBackdrop)
  mountElement.append(root)

  // ---------- behavior ----------
  const setStatus = (message: string): void => {
    status.textContent = message
  }

  // 파싱 실패한 맵이 있으면 상태 메시지 끝에 붙일 경고(없으면 빈 문자열). loadGame이 throw 대신
  // game.parseErrors로 모아주므로, 에디터가 통째로 안 뜨는 일 없이 실패를 사용자에게 알린다.
  const parseErrorNote = (): string =>
    game.parseErrors.length > 0
      ? ` · 파싱 실패 맵 ${game.parseErrors.length}개: ${game.parseErrors.join(', ')}`
      : ''

  const renderAnalysis = (): void => {
    if (!currentAnalysis) {
      analysisPanel.hidden = true
      return
    }

    analysisPanel.hidden = false
    const analysis = currentAnalysis
    analysisPanel.replaceChildren(
      el('div', 'text-[11px] font-semibold tracking-wide text-[#c48a4a]', 'LLM 게임 분석'),
      el('div', 'text-sm text-[#d4d4d4] font-medium', `${analysis.game_name} · ${analysis.engine}`),
      el('div', 'text-xs text-[#9d9d9d]', `콘텐츠 모델: ${analysis.content_model}`),
      el('div', 'text-xs text-[#9d9d9d]', `적용 전략: ${analysis.apply_strategy}`),
      ...analysis.entity_groups.map((entityGroup) =>
        el(
          'div',
          'text-xs text-[#777777]',
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
    analyzeLabel.textContent = '분석 중...'
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
      analyzeLabel.textContent = 'AI 게임 분석'
    }
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
        el('span', 'inline-block w-1.5 h-1.5 rounded-full bg-[#7ba368] mr-1.5 align-middle'),
        showAllMaps
          ? document.createTextNode('전체 맵 표시 중')
          : el('span', 'text-[#d4d4d4]', `현재 맵: ${focusMap.name}`)
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
      const mapTitle = el('div', 'flex items-center gap-1.5 text-[12px] text-[#d4d4d4] font-semibold tracking-wide px-1')
      mapTitle.append(
        editorIcon('map', 13),
        el('span', 'truncate', `${map.name}${isCurrent ? ' · 현재 맵' : ''}`)
      )
      group.append(mapTitle)

      // 검색어가 있으면 이름으로 실시간 필터링(표시 전용).
      const query = assetQuery.trim().toLowerCase()
      const visibleEntities = query
        ? map.entities.filter((entity) => entity.name.toLowerCase().includes(query))
        : map.entities
      // 검색 중인데 이 맵에 일치하는 에셋이 없으면 맵 자체를 건너뛴다.
      if (query && visibleEntities.length === 0) {
        continue
      }

      // 같은 종류끼리 접이식 그룹으로 묶는다(쭉 나열하면 길어서 보기 불편하다는 피드백).
      // NPC 그룹을 맨 위로, 나머지는 맵에 등장한 순서대로.
      const byKind = new Map<string, GameEntity[]>()
      for (const entity of visibleEntities) {
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

      const selectableCount = visibleEntities.filter(isSelectableEntity).length
      // 종류별 "아이콘 카드" 2열 그리드 — 파일 탐색기식 세로 리스트 대신 게임 건설 메뉴처럼.
      // 카드는 큰 아이콘(52px)이 먼저 보이고, 이름·개수는 아래 작은 캡션으로만 붙는다.
      // 카테고리(인물/건축물/장식물)로 한 층 더 묶는다 — 정보 구조가 한눈에 읽히게.
      const byCategory = new Map<string, Array<[string, GameEntity[]]>>()
      for (const entry of kindEntries) {
        const category = categoryOf(entry[0])
        const list = byCategory.get(category)
        if (list) {
          list.push(entry)
        } else {
          byCategory.set(category, [entry])
        }
      }
      for (const category of CATEGORY_ORDER) {
        const entriesInCategory = byCategory.get(category)
        if (!entriesInCategory || entriesInCategory.length === 0) {
          continue
        }
        // 그룹 제목 — 작고 은은한 금색 라벨 + 얇은 구분선. 클릭하면 접기/펼치기(검색 중엔 항상 펼침).
        const categoryCollapsed = query.length === 0 && collapsedCategories.has(category)
        const categoryHeader = el('button', 'w-full flex items-center gap-1.5 mt-1 px-1 py-1 text-left text-[13px] leading-none tracking-[0.5px] text-[#d9a85c] border-y border-[#d9a85c]/20 transition hover:text-[#f3d7a2]') as HTMLButtonElement
        categoryHeader.type = 'button'
        categoryHeader.append(
          el('span', 'text-[10px] leading-none text-[#c9a96b]', categoryCollapsed ? '▸' : '▾'),
          el('span', '', category)
        )
        categoryHeader.addEventListener('click', () => {
          if (collapsedCategories.has(category)) {
            collapsedCategories.delete(category)
          } else {
            collapsedCategories.add(category)
          }
          renderTree()
          render()
        })
        group.append(categoryHeader)
        if (categoryCollapsed) {
          continue
        }
        const kindGrid = el('div', 'grid grid-cols-2 gap-2')
        group.append(kindGrid)
        for (const [kind, entities] of entriesInCategory) {
          const groupKey = `${map.id}:${kind}`
        const selectable = entities.filter(isSelectableEntity)
        // 선택된 NPC가 속한 종류는 이번 렌더에서만 펼쳐 보인다(접혀 있으면 선택 표시가 가려진다).
        // Set에는 쓰지 않는다 — 영구 펼침으로 만들면 사용자가 접어도 다음 렌더마다 되돌아간다.
        const containsSelected =
          selectedEntity !== undefined && entities.some((entity) => entity === selectedEntity)
        // 검색 중에는 결과를 바로 보여줘야 하므로 자동으로 펼친다.
        const expanded =
          selectable.length > 0 &&
          (query.length > 0 || expandedGroups.has(groupKey) || containsSelected)

        const card = el(
          selectable.length > 0 ? 'button' : 'div',
          `${KIND_CARD}${
            containsSelected || expanded
              ? ` ${KIND_CARD_ACTIVE}`
              : selectable.length > 0
                ? ` ${KIND_CARD_CLICKABLE}`
                : ''
          }`
        )
        // 카드 구성: 아이콘 → 이름(+펼침 화살표) → '8명/5개' 카운트. 전부 중앙 정렬.
        const cardLabel = el('div', 'flex items-center justify-center gap-1')
        cardLabel.append(
          el('span', 'whitespace-nowrap text-[12px] leading-none font-semibold tracking-wide text-[#d4d4d4]', KIND_LABEL[kind] ?? kind)
        )
        if (selectable.length > 0) {
          cardLabel.append(el('span', 'text-[9px] leading-none text-[#777777]', expanded ? '▾' : '▸'))
        }
        const countUnit = kind === 'npc' || kind === 'character' || kind === 'monster' ? '명' : '개'
        // NPC는 가장 중요한 에셋 — '8명 존재'처럼 조금 더 살아있는 표현.
        const countText = kind === 'npc' ? `${entities.length}명 존재` : `${entities.length}${countUnit}`
        card.append(
          editorIcon(KIND_ICON[kind] ?? 'prop', 44),
          cardLabel,
          el('div', 'whitespace-nowrap text-[10px] leading-none text-[#777777]', countText)
        )
        kindGrid.append(card)

        // 보기 전용 종류(나무·가로등 등)는 카드로 개수만 보여주고 끝 — 펼칠 목록이 없다.
        if (selectable.length === 0) {
          continue
        }

        const cardButton = card as HTMLButtonElement
        cardButton.type = 'button'
        cardButton.setAttribute('aria-expanded', String(expanded))
        cardButton.addEventListener('click', () => {
          if (expandedGroups.has(groupKey)) {
            expandedGroups.delete(groupKey)
          } else {
            expandedGroups.add(groupKey)
          }
          renderTree()
          render()
        })

        // 펼친 종류의 구성원 선택 그리드 — 카드 바로 아래 한 줄 전체를 쓴다(인벤토리 상세 칸 느낌).
        if (expanded) {
          const memberGrid = el('div', 'col-span-2 grid grid-cols-2 gap-1.5 rounded-lg border border-[#c98a3a]/25 bg-[#252526] p-1.5')
          for (const entity of selectable) {
            const node = el('button', entity === selectedEntity ? ENTITY_ACTIVE : ENTITY_BASE) as HTMLButtonElement
            node.type = 'button'
            // 이름이 길어도 hover로 전체 이름·타입을 볼 수 있다(CSS 툴팁).
            node.setAttribute('data-tip', `${entity.name} · ${KIND_LABEL[kind] ?? kind}`)
            // NPC는 역할(마법사/대장장이/상인/경비) 아이콘으로 먼저 구분되게 한다.
            const memberIcon = kind === 'npc' ? npcIconFor(entity.name) : (KIND_ICON[kind] ?? 'prop')
            node.append(
              editorIcon(memberIcon, 16),
              el('span', 'truncate', displayNameOf(entity.name))
            )
            node.addEventListener('click', () => {
              selectedEntity = entity
              // 대상을 바꾸면 이전 생성 결과는 무효 — 새로 생성하게 한다.
              currentResult = undefined
              render()
            })
            entityButtons.push({ entity, node })
            memberGrid.append(node)
          }
          kindGrid.append(memberGrid)
        }
        }
      }

      // 요소는 있는데 생성 대상이 하나도 없는 맵(사냥터·동굴 등)에선, 왜 클릭할 게 없는지 알려준다.
      if (selectableCount === 0 && map.entities.length > 0) {
        group.append(
          el('div', 'px-1 text-[11px] text-[#777777] italic', '생성 대상이 없는 맵 — 위 요소는 보기 전용입니다.')
        )
      }

      // "ground" 같은 타일/지형 레이어 — 객체가 아니라 맵 자체의 구성. 보기 전용 정보로 한 줄에 보여준다.
      if (map.layers.length > 0) {
        const layersLine = el('div', 'px-1 pt-0.5 flex items-center gap-1.5 text-[11px] text-[#777777]')
        layersLine.append(
          editorIcon('layers', 12),
          el('span', 'truncate', `타일 레이어: ${map.layers.join(' · ')}`)
        )
        group.append(layersLine)
      }

      groups.push(group)
    }

    if (groups.length === 0) {
      const message =
        assetQuery.trim().length > 0
          ? `'${assetQuery.trim()}' 검색 결과가 없습니다.`
          : focusMap && !showAllMaps
            ? `현재 맵(${focusMap.name})에서 읽을 요소가 없습니다. ‘전체 보기’로 다른 맵을 볼 수 있어요.`
            : '로드된 맵이 없습니다. "게임 폴더 열기"로 프로젝트를 여세요.'
      groups.push(el('div', 'text-xs text-[#9d9d9d] leading-relaxed', message))
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
            ? 'truncate text-left rounded-md px-2.5 py-1.5 text-xs bg-[#c48a4a]/15 text-[#e2bd8c] ring-1 ring-inset ring-[#c48a4a]/40 transition'
            : 'truncate text-left rounded-md px-2.5 py-1.5 text-xs text-[#9d9d9d] transition hover:bg-[#302a26] hover:text-[#d4d4d4]'
        ) as HTMLButtonElement
        node.type = 'button'
        const mark = entry.result.issues.length === 0 ? '✓' : '!'
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
        : ` · 누적 수용률 ${acceptancePercent}%${metrics.meetsAcceptanceGoal ? ' ✓' : ''}`)

    // 현재 결과가 이미 평가됐으면(객체 단위로 기억) 그 판정을 보여주고 버튼을 잠근다(중복 집계 방지).
    const verdict = verdictByResult.get(currentResult)
    const evaluated = verdict !== undefined
    acceptButton.disabled = evaluated
    rejectButton.disabled = evaluated
    acceptButton.className =
      verdict === 'acceptable'
        ? 'rounded-lg h-8 px-3 bg-[#4e6b42]/25 text-[#a3bd92] text-sm border border-[#4e6b42]/50'
        : GHOST_BUTTON
    rejectButton.className =
      verdict === 'not_acceptable'
        ? 'rounded-lg h-8 px-3 bg-[#8a4a3e]/25 text-[#d49a8c] text-sm border border-[#8a4a3e]/50'
        : GHOST_BUTTON
    if (evaluated) {
      evaluationVerdict.className =
        verdict === 'acceptable' ? 'text-xs text-[#a3bd92]' : 'text-xs text-[#d49a8c]'
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

    if (game.adapter.applyMode !== 'none') {
      supportNote.hidden = true
    } else {
      supportNote.hidden = false
      supportNote.textContent = `${game.adapter.name}: 생성은 되지만 라이브 적용은 아직 지원되지 않습니다 (Stage 3). 결과는 미리보기로 확인하세요.`
    }

    // 엔티티 이름/맵은 열린 TMX에서 온 임의 값이므로 textContent로만 넣는다(주입/깨짐 방지).
    if (selectedEntity) {
      // 선택됨: '선택 대상 / OO 선택됨' 카드 — 금색 테두리 + 은은한 발광 + 페이드 전환.
      const selectedCard = el('span', 'fade-in inline-flex items-center gap-2 rounded-lg px-3 py-1.5 bg-[#b78446]/15 border border-[#d5a14f] shadow-[0_0_8px_rgba(213,161,79,0.25)] max-w-full')
      const selectedText = el('span', 'flex flex-col gap-1 min-w-0')
      selectedText.append(
        el('span', 'text-[10px] leading-none text-[#9d9d9d]', '선택 대상'),
        el('span', 'text-[14px] leading-none text-[#f6e4b8] truncate', `${selectedEntity.name} 선택됨`)
      )
      selectedCard.append(
        editorIcon(KIND_ICON[groupKindOf(selectedEntity.kind)] ?? 'target', 20),
        selectedText
      )
      targetLine.replaceChildren(selectedCard)
    } else {
      // 선택 전: 오류처럼 보이지 않게 흐린 회색 안내 톤. NPC 생성처럼 대상이 불필요한 모드에선
      // "대상 선택 필요" 대신 "대상 불필요"로 안내해 흐름이 막힌 것처럼 보이지 않게 한다.
      const emptyCardBadge = el('span', 'fade-in inline-flex items-center gap-2 rounded-lg px-3 py-1.5 bg-[#2d2d30]/70 border border-[#d9a85c]/20 opacity-80')
      emptyCardBadge.append(
        editorIcon('target', 16),
        el(
          'span',
          'text-[12px] leading-none text-[#9d9d9d]',
          isTargetOptional() ? '새 NPC 생성 — 대상 선택 불필요' : '대상 선택 필요'
        )
      )
      targetLine.replaceChildren(emptyCardBadge)
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
      validationLine.className = 'text-[12px] leading-relaxed text-[#8fc96a]'
      validationLine.replaceChildren(
        document.createTextNode('✓ 자동 검증 통과')
      )
    } else {
      validationLine.hidden = false
      validationLine.className = 'text-[12px] leading-relaxed text-[#d9a64f] flex flex-col gap-0.5'
      // 이슈 문자열은 Validator가 만든 값이지만 안전하게 textContent(el)로만 넣는다.
      validationLine.replaceChildren(
        el('div', '', `! 자동 검증 ${currentResult.issues.length}건 확인 필요`),
        ...currentResult.issues.map((issue) => el('div', 'pl-3 text-[#d9a64f]/80', `• ${issue}`))
      )
    }
    // 검증 전 안내문은 검증 표시와 반대로 토글(둘 다 검증 섹션 본문 안).
    validationEmpty.hidden = !validationLine.hidden

    // 무결성 검증(드라이런) 단계별 결과 — 퀘스트 2단계 생성 후에만 채워진다.
    if (currentDryRun) {
      dryRunBox.hidden = false
      dryRunBox.replaceChildren(
        el(
          'div',
          'text-[12px] font-semibold pt-1 ' +
            (currentDryRun.ok ? 'text-[#8fc96a]' : 'text-[#e06c6c]'),
          `무결성 검증(드라이런) — ${currentDryRun.ok ? '통과' : '실패'}`
        ),
        ...currentDryRun.steps.map((step) => {
          const icon = step.status === 'ok' ? '✓' : step.status === 'warn' ? '⚠' : '✗'
          const color =
            step.status === 'ok'
              ? 'text-[#8fc96a]'
              : step.status === 'warn'
                ? 'text-[#d9a64f]'
                : 'text-[#e06c6c]'
          return el('div', `text-[11px] leading-relaxed ${color}`, `${icon} ${step.label} — ${step.detail}`)
        }),
        ...currentDryRun.jsonIssues.map((issue) =>
          el('div', 'text-[11px] leading-relaxed text-[#d9a64f]/80 pl-3', `• ${issue}`)
        ),
        el(
          'div',
          'text-[10px] leading-[1.4] text-[#777777] pt-1',
          '※ 에디터 프로필 기준 시뮬레이션입니다. 실제 게임 상태와 다를 수 있습니다.'
        )
      )
    } else {
      dryRunBox.hidden = true
    }

    generateLabel.textContent = isGenerating ? '✨ 생성 중...' : '✨ 이야기 생성'
    generateButton.disabled =
      isGenerating || apiKey.trim().length === 0 || promptInput.value.trim().length === 0
    // 단일 흐름: 검증(issues)이 적용을 막지 않는다(기존 동작 유지). 퀘스트 모드에서는 무결성
    // 검증(드라이런)이 통과해야만 적용을 허용한다 — 사용자가 정한 "검증 통과 시 라이브 적용".
    // 적용 가능 = localStorage apply()가 있거나(rpg) 브리지 페이로드가 있을 때(legend-of-lua).
    applyButton.disabled =
      isGenerating ||
      (!currentResult?.apply && !currentResult?.bridgePayload) ||
      (candidateMode && currentDryRun?.ok !== true)
    copyButton.disabled = !currentResult || isGenerating
    exportButton.disabled = !currentResult || isGenerating
    // 결과 보드 채우기(표시 전용): 목록 4줄은 항상 보이고, 상세 창 내용만 갱신된다.
    // 목록 카드 우측 상태 배지(캡슐) — 색으로 상태가 한눈에 들어온다.
    const BADGE = 'h-[20px] flex items-center rounded-full px-2 text-[10px] font-semibold leading-none whitespace-nowrap bg-[#333333]'
    const appliedNow = currentResult !== undefined && appliedResults.has(currentResult)
    for (const { id, status } of boardRows) {
      if (id === 'lua') {
        status.textContent = currentResult ? 'Completed' : 'Ready'
        status.className = `${BADGE} ${currentResult ? 'text-[#8fc96a]' : 'text-[#d9a85c]'}`
      } else if (id === 'files') {
        status.textContent = currentResult ? '1개' : '0개'
        status.className = `${BADGE} ${currentResult ? 'text-[#e8d5a5]' : 'text-[#9d9d9d]'}`
      } else if (id === 'verify') {
        // 퀘스트 모드면 드라이런 통과/실패를 우선 표시, 아니면 기존 자동 검증 배지.
        if (currentDryRun) {
          status.textContent = currentDryRun.ok ? '검증 통과' : '검증 실패'
          status.className = `${BADGE} ${currentDryRun.ok ? 'text-[#8fc96a]' : 'text-[#e06c6c]'}`
        } else {
          status.textContent = !currentResult
            ? '0 Errors'
            : currentResult.issues.length === 0
              ? 'Passed'
              : `${currentResult.issues.length} Issues`
          status.className = !currentResult
            ? `${BADGE} text-[#9d9d9d]`
            : currentResult.issues.length === 0
              ? `${BADGE} text-[#8fc96a]`
              : `${BADGE} text-[#d9a64f]`
        }
      } else {
        status.textContent = !currentResult ? 'Ready' : appliedNow ? '적용 완료' : '적용 전'
        status.className = appliedNow ? `${BADGE} text-[#8fc96a]` : `${BADGE} text-[#9d9d9d]`
      }
    }
    // 빈 상태의 '오늘 작업' 요약(이번 세션 집계) + 최근 작업 목록.
    statGen.textContent = String(sessionTally.generations)
    statPass.textContent = String(sessionTally.validatorPasses)
    statApply.textContent = String(appliedCount)
    // 최근 작업 목록(표시 전용) — '작업명 ── 시간' 한 줄. 시간은 처음 표시된 시각을 기억한다.
    if (history.length === 0) {
      // 발표용 세션 샘플 — 실제 기록이 생기면 아래 분기로 대체된다(저장 기능 아님, 표시 전용).
      const SAMPLE_RECENT = [
        { label: 'NPC 대사 생성', time: '11:24' },
        { label: '건물 스타일 변경', time: '11:19' },
        { label: '포털 생성', time: '11:12' }
      ]
      recentList.replaceChildren(
        ...SAMPLE_RECENT.map((sample) => {
          // 시간 → 작업명 순서: 실제 작업 로그처럼 읽힌다.
          const rowItem = el('div', 'flex items-center gap-2')
          rowItem.append(
            el('span', 'shrink-0 text-[10px] leading-none tabular-nums text-[#d9a85c]/80', sample.time),
            el('span', 'truncate text-[11px] leading-none text-[#9d9d9d]', sample.label)
          )
          return rowItem
        })
      )
    } else {
      recentList.replaceChildren(
        ...history.slice(0, 3).map((entry) => {
          let timeLabel = resultTimes.get(entry.result)
          if (timeLabel === undefined) {
            const now = new Date()
            timeLabel = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`
            resultTimes.set(entry.result, timeLabel)
          }
          const rowItem = el('div', 'flex items-center gap-2')
          rowItem.append(
            el('span', 'shrink-0 text-[10px] leading-none tabular-nums text-[#d9a85c]/80', timeLabel),
            el('span', 'truncate text-[11px] leading-none text-[#9d9d9d]', entry.result.label)
          )
          return rowItem
        })
      )
    }
    luaStatus.hidden = currentResult !== undefined
    result.hidden = currentResult === undefined
    result.textContent = currentResult ? currentResult.preview : ''
    filesStatus.textContent = currentResult
      ? `${currentResult.exportFileExtension === 'lua' ? 'Lua 코드' : '이벤트'}: ${currentResult.label}`
      : '변경 파일 없음'
    if (!currentResult) {
      applyStatus.className = 'text-[12px] text-[#9d9d9d]'
      applyStatus.textContent = '대기 중'
    } else if (
      !currentResult.apply &&
      !currentResult.bridgePayload &&
      currentResult.exportFileExtension === 'lua'
    ) {
      applyStatus.className = 'text-[12px] text-[#9d9d9d]'
      applyStatus.textContent = '코드만 생성됨'
    } else if (appliedResults.has(currentResult)) {
      applyStatus.className = 'text-[12px] text-[#8fc96a]'
      applyStatus.textContent = '적용 완료'
    } else {
      applyStatus.className = 'text-[12px] text-[#9d9d9d]'
      applyStatus.textContent = '적용 전'
    }
    // 최근 생성 결과 표시(표시 전용): 실제 결과 우선, 없으면 발표용 예시 한 줄.
    recentResultLine.textContent = currentResult
      ? currentResult.exportFileExtension === 'lua'
        ? `최근 생성 결과 · "${currentResult.label}" Lua 퀘스트 코드가 생성되었습니다.`
        : `최근 생성 결과 · "${currentResult.label}" 이벤트가 생성되었습니다.`
      : '최근 생성 결과 · "마법사가 플레이어에게 마을 북쪽 숲의 위험을 경고하는 대사가 생성되었습니다."'
    // 결과가 생겼는데 아직 아무 항목도 안 골랐으면 Lua 코드 상세를 자동으로 연다.
    if (currentResult && activeBoardTab === undefined) {
      activeBoardTab = 'lua'
    }
    updateBoard()
    // 퀘스트 후보 카드의 선택/버튼 상태를 현재 상태에 맞춰 다시 그린다(표시 전용).
    if (candidateMode) {
      renderCandidates()
    }
    // 표시 전용 UI 동기화: 요약 카드 · 맵 탭 강조 · 진행 단계 바 · 컴포저 진행 상태 · 맵 통계.
    updateComposerSteps()
    updatePreviewStats()
    updateSummary()
    updateSceneTabs()
    updateStepBar()
    renderEvaluation()
    renderHistory()
  }

  // ---------- 퀘스트 모드: 1단계 후보 ↔ 2단계 후보로 생성 + 드라이런 검증 ----------
  const CANDIDATE_CARD =
    'w-full flex flex-col gap-1 rounded-xl border border-[#d9a85c]/22 bg-[#2d2d30] px-3 py-2.5 text-left transition hover:border-[#d9a85c]/70 hover:bg-[#333333]'
  const CANDIDATE_CARD_ACTIVE =
    'w-full flex flex-col gap-1 rounded-xl border border-[#e7b15a] bg-[#3a2416] px-3 py-2.5 text-left shadow-[0_0_10px_rgba(217,168,92,0.3)]'

  // 1단계: 자연어 후보 N개 생성(피드백이 있으면 재생성). 결과 보드의 '퀘스트 후보' 상세에 카드로 뜬다.
  const runGenerateCandidates = async (
    feedback?: GenerationFeedback
  ): Promise<void> => {
    const isLegend = isLegendOfLuaGame(game)
    const profile = game.profile
    const catalog = isLegend ? buildLuaQuestCatalog(game) : undefined

    if (!profile && !catalog) {
      setStatus('이 게임의 구조 정보가 없습니다.')
      return
    }
    isGenerating = true
    const filesAtStart = currentFiles
    setStatus(isLegend ? 'Lua 퀘스트 후보 생성 중...' : '퀘스트 후보 생성 중...')
    activeBoardTab = 'candidates'
    render()

    try {
      const result = isLegend
        ? await generateLuaQuestCandidates({
            apiKey: apiKey.trim(),
            userPrompt: promptInput.value,
            catalog: catalog!,
            entity: selectedEntity,
            gameContext: currentAnalysis
              ? `${currentAnalysis.game_name} (${currentAnalysis.engine}). 콘텐츠 모델: ${currentAnalysis.content_model}`
              : undefined,
            feedback
          })
        : await generateQuestCandidates({
            apiKey: apiKey.trim(),
            userPrompt: promptInput.value,
            profile: profile!,
            entity: selectedEntity,
            gameContext: currentAnalysis
              ? `${currentAnalysis.game_name} (${currentAnalysis.engine}). 콘텐츠 모델: ${currentAnalysis.content_model}`
              : undefined,
            feedback
          })
      if (currentFiles !== filesAtStart) {
        return
      }
      candidates = result
      selectedCandidateIndex = undefined
      editingCandidateIndex = undefined
      currentResult = undefined
      currentDryRun = undefined
      setStatus(
        result.length > 0
          ? isLegend
            ? `Lua 후보 ${result.length}개 생성됨 — 하나를 골라 "이 후보로 생성"을 누르세요.`
            : `후보 ${result.length}개 생성됨 — 하나를 골라 "이 후보로 생성"을 누르세요.`
          : '후보를 생성하지 못했습니다. 다시 시도하세요.'
      )
    } catch (error) {
      if (currentFiles !== filesAtStart) {
        return
      }
      setStatus(`후보 생성 실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isGenerating = false
      render()
    }
  }

  const runRegenerateCandidates = async (): Promise<void> => {
    if (isGenerating) {
      return
    }
    candidateIteration += 1
    await runGenerateCandidates({
      previousOutput: JSON.stringify(candidates),
      validatorIssues: [],
      rejectionReason: '다른 방향의 후보들로 다시 제안해줘',
      iteration: candidateIteration
    })
  }

  // 2단계: 고른 후보로만 실제 이벤트 JSON 생성 → 드라이런 검증 → 통과 시 적용 가능.
  const runGenerateFromCandidate = async (): Promise<void> => {
    if (isGenerating || selectedCandidateIndex === undefined) {
      return
    }
    const candidate = candidates[selectedCandidateIndex]
    if (!candidate) {
      return
    }
    const isLegend = isLegendOfLuaGame(game)
    const profile = game.profile
    const catalog = isLegend ? buildLuaQuestCatalog(game) : undefined

    if (!profile && !catalog) {
      setStatus('이 게임의 구조 정보가 없습니다.')
      return
    }
    const effectiveEntity =
      candidate.target_hint
        ? findEntityById(game, candidate.target_hint) ??
          (selectedEntity?.kind === 'npc' ? selectedEntity : undefined)
        : selectedEntity?.kind === 'npc'
          ? selectedEntity
          : undefined
    const selectedNpcId =
      effectiveEntity?.kind === 'npc' ? effectiveEntity.id : undefined
    isGenerating = true
    const filesAtStart = currentFiles
    setStatus(
      isLegend
        ? `"${candidate.title}" 후보로 Lua 퀘스트 생성 중...`
        : `"${candidate.title}" 후보로 단계별 퀘스트 생성 중...`
    )
    render()

    try {
      if (isLegend) {
        const quest = await generateLuaQuestJson({
          apiKey: apiKey.trim(),
          userPrompt: promptInput.value,
          catalog: catalog!,
          candidate,
          entity: effectiveEntity
        })
        if (currentFiles !== filesAtStart) {
          return
        }
        currentDryRun = dryRunLuaQuestApply(quest, catalog!, {
          selectedEntityId: selectedNpcId
        })
        const issues = createGeneratedLuaQuestValidationIssues(quest, catalog!).map(
          (issue) => `${issue.path} - ${issue.message}`
        )
        const result: GenerationResult = {
          label: quest.title || quest.quest_id,
          preview: renderGeneratedLuaQuestModule(quest),
          issues,
          apply: null,
          exportFileExtension: 'lua',
          // 라이브 적용용: 실행 중인 legend-of-lua에 브리지로 보낼 퀘스트 페이로드.
          bridgePayload: convertLuaQuestToBridgePayload(quest)
        }
        currentResult = result
        historyCounter += 1
        history = [{ n: historyCounter, result }, ...history].slice(0, HISTORY_LIMIT)
        sessionTally = {
          generations: sessionTally.generations + 1,
          validatorPasses:
            sessionTally.validatorPasses + (result.issues.length === 0 ? 1 : 0)
        }
        activeBoardTab = 'lua'
        setStatus(
          currentDryRun && !currentDryRun.ok
            ? `생성됨 — Lua 무결성 검증 실패. '검증 결과'에서 위치를 확인하고 다시 시도하세요.`
            : `생성 완료: ${result.label} — Lua 무결성 검증 통과`
        )
      } else {
        // 퀘스트 모드: 고른 후보로 "목표 포함 진짜 퀘스트"를 생성한다(대사 이벤트가 아님).
        const quest = await generateQuestJson({
          apiKey: apiKey.trim(),
          userPrompt: promptInput.value,
          profile: profile!,
          candidate,
          entity: effectiveEntity
        })
        if (currentFiles !== filesAtStart) {
          return
        }
        // 무결성 검증(드라이런): 목표/기버/보상 타깃이 런타임에서 추적되는 값인지 단계별 점검.
        currentDryRun = dryRunQuestApply(quest, profile!, {
          selectedEntityId: selectedNpcId
        })
        const issues = createGeneratedQuestValidationIssues(quest, profile!, {
          selectedEntityId: selectedNpcId
        }).map((issue) => `${issue.path} - ${issue.message}`)
        // 결과 보드/적용 경로가 쓰는 GenerationResult 형태로 감싼다(apply는 런타임 퀘스트로 변환·저장).
        const result: GenerationResult = {
          label: quest.title || quest.quest_id,
          preview: JSON.stringify(quest, null, 2),
          issues,
          apply: () => {
            replacePendingQuests([
              convertGeneratedQuestToDefinition(quest, profile!)
            ])
          },
          bridgePayload: null
        }
        currentResult = result
        historyCounter += 1
        history = [{ n: historyCounter, result }, ...history].slice(0, HISTORY_LIMIT)
        sessionTally = {
          generations: sessionTally.generations + 1,
          validatorPasses:
            sessionTally.validatorPasses + (result.issues.length === 0 ? 1 : 0)
        }
        activeBoardTab = 'verify'
        setStatus(
          currentDryRun && !currentDryRun.ok
            ? `생성됨 — 무결성 검증 실패. '검증 결과'에서 위치를 확인하고 다시 시도하세요.`
            : `생성 완료: ${result.label} — 무결성 검증 통과`
        )
      }
    } catch (error) {
      if (currentFiles !== filesAtStart) {
        return
      }
      setStatus(`생성 실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isGenerating = false
      render()
    }
  }

  // 후보 카드 렌더(상태 훅에 할당). 카드 클릭=선택, 수정=요약 인라인 편집, 하단에 생성/재생성 버튼.
  renderCandidates = (): void => {
    candidatesView.body.replaceChildren()
    if (candidates.length === 0) {
      candidatesView.body.append(
        el(
          'div',
          'text-[12px] text-[#9d9d9d]',
          isGenerating ? '후보 생성 중...' : '후보가 없습니다. "이야기 생성"으로 후보를 만드세요.'
        )
      )
      return
    }
    candidates.forEach((candidate, index) => {
      const selected = index === selectedCandidateIndex
      const card = el('div', selected ? CANDIDATE_CARD_ACTIVE : CANDIDATE_CARD)
      const header = el('div', 'flex items-center gap-2')
      const titleEl = el('div', 'flex-1 text-[13px] font-semibold text-[#e8d5a5] truncate')
      titleEl.textContent = candidate.title || `후보 ${index + 1}`
      const editBtn = el(
        'button',
        'shrink-0 text-[11px] text-[#9d9d9d] transition hover:text-[#e8d5a5]',
        editingCandidateIndex === index ? '완료' : '수정'
      ) as HTMLButtonElement
      editBtn.type = 'button'
      editBtn.addEventListener('click', (event) => {
        event.stopPropagation()
        editingCandidateIndex = editingCandidateIndex === index ? undefined : index
        renderCandidates()
      })
      header.append(titleEl, editBtn)
      card.append(header)
      if (editingCandidateIndex === index) {
        const textarea = el(
          'textarea',
          `${FIELD_INPUT} min-h-[64px] text-[12px] leading-[1.5]`
        ) as HTMLTextAreaElement
        textarea.value = candidate.summary
        textarea.addEventListener('click', (event) => event.stopPropagation())
        textarea.addEventListener('input', () => {
          candidates[index] = { ...candidate, summary: textarea.value }
        })
        card.append(textarea)
      } else {
        const summaryEl = el(
          'div',
          'text-[12px] leading-[1.5] text-[#bfbfbf] whitespace-pre-wrap'
        )
        summaryEl.textContent = candidate.summary
        card.append(summaryEl)
      }
      if (candidate.target_hint) {
        const chip = el(
          'span',
          'self-start rounded-full px-2 py-0.5 text-[10px] leading-none text-[#d5b87a] bg-[#dca14b]/10 border border-[#dca14b]/25'
        )
        chip.textContent = `대상: ${candidate.target_hint}`
        card.append(chip)
      }
      card.addEventListener('click', () => {
        if (candidate.target_hint) {
          const targetEntity = findEntityById(game, candidate.target_hint)
          if (targetEntity?.kind === 'npc') {
            selectedEntity = targetEntity
          }
        }
        selectedCandidateIndex = index
        renderCandidates()
        render()
      })
      candidatesView.body.append(card)
    })
    const actionRow = el('div', 'flex flex-wrap items-center gap-2 pt-1')
    const genFromBtn = el('button', APPLY_BUTTON, '이 후보로 생성') as HTMLButtonElement
    genFromBtn.type = 'button'
    genFromBtn.disabled = selectedCandidateIndex === undefined || isGenerating
    genFromBtn.addEventListener('click', () => void runGenerateFromCandidate())
    const regenBtn = el('button', GHOST_BUTTON, '후보 다시 제안') as HTMLButtonElement
    regenBtn.type = 'button'
    regenBtn.disabled = isGenerating
    regenBtn.addEventListener('click', () => void runRegenerateCandidates())
    actionRow.append(genFromBtn, regenBtn)
    candidatesView.body.append(actionRow)
  }

  // NPC 생성(legend-of-lua): 새 NPC를 한 번에 생성→드라이런 검증→Lua 출력한다. 검증을 통과하면
  // 생성 NPC를 game.maps의 해당 맵에 주입해, 트리와 퀘스트 카탈로그(buildLuaQuestCatalog)가
  // "배치된 NPC"로 인식하게 한다 — 이후 그 NPC를 퀘스트 기버로 고를 수 있다. (실제 게임 런타임
  // 스폰은 이번 범위 밖: 산출물은 Lua 모듈 미리보기/복사/내보내기로 쓴다.)
  const runGenerateLuaNpc = async (): Promise<void> => {
    const catalog = buildLuaQuestCatalog(game)
    isGenerating = true
    const filesAtStart = currentFiles
    setStatus('Lua NPC 생성 중...')
    render()

    try {
      const npc = await generateLuaNpcJson({
        apiKey: apiKey.trim(),
        userPrompt: promptInput.value,
        catalog
      })
      if (currentFiles !== filesAtStart) {
        return
      }

      // 지금 보고 있는 맵으로 NPC를 고정한다 — LLM이 다른 맵을 고르면 트리(현재 맵 기준)와
      // 라이브(플레이어 옆) 위치가 어긋난다. 현재 맵이 정해져 있으면 그 맵으로 맞춘다.
      if (currentMapId && catalog.scenes.includes(currentMapId)) {
        npc.map_id = currentMapId
      }

      const dryRun = dryRunLuaNpcApply(npc, catalog)
      const issues = createGeneratedLuaNpcValidationIssues(npc, catalog).map(
        (issue) => `${issue.path} - ${issue.message}`
      )
      // 미리보기/복사/내보내기는 editorNPCs.lua 붙여넣기용 항목. 라이브 적용은 bridgePayload로 한다.
      const spawnEntry = renderEditorNpcSpawnEntry(npc)
      const result: GenerationResult = {
        label: npc.name || npc.npc_id,
        preview: spawnEntry,
        issues,
        apply: null,
        exportFileExtension: 'lua',
        // 검증 통과 시: '적용'이 이 페이로드를 패널 iframe(love.js)·네이티브 HTTP 브리지로 보내
        // 실행 중인 게임이 NPC를 플레이어 옆에 라이브 스폰한다(host page editor-bridge.js + 게임
        // pollEditorInbox). 검증 실패면 null이라 적용 버튼이 잠긴다.
        bridgePayload: dryRun.ok ? convertLuaNpcToSpawnBridgePayload(npc, Date.now()) : null
      }
      currentDryRun = dryRun
      currentResult = result
      historyCounter += 1
      history = [{ n: historyCounter, result }, ...history].slice(0, HISTORY_LIMIT)
      sessionTally = {
        generations: sessionTally.generations + 1,
        validatorPasses:
          sessionTally.validatorPasses + (result.issues.length === 0 ? 1 : 0)
      }
      activeBoardTab = 'lua'

      // 검증 통과한 NPC만 카탈로그/트리에 더한다(맵에 같은 id가 없을 때만 — 멱등).
      if (dryRun.ok) {
        const entity = convertLuaNpcToGameEntity(npc)
        const targetMap = game.maps.find((map) => map.id === entity.mapId)
        if (targetMap && !targetMap.entities.some((existing) => existing.id === entity.id)) {
          targetMap.entities.push(entity)
          renderTree()
        }
        setStatus(
          `생성 완료: ${result.label} — 검증 통과, 카탈로그에 추가됨(이제 퀘스트 기버로 쓸 수 있어요).`
        )
      } else {
        setStatus(
          `생성됨 — NPC 무결성 검증 실패. '검증 결과'에서 위치를 확인하고 다시 시도하세요.`
        )
      }
    } catch (error) {
      if (currentFiles !== filesAtStart) {
        return
      }
      setStatus(`NPC 생성 실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isGenerating = false
      render()
    }
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

    // 퀘스트 모드: '이야기 생성'은 1단계(후보 N개)를 만든다. 실제 이벤트는 후보를 골라 2단계에서.
    if (candidateMode) {
      await runGenerateCandidates()
      return
    }

    // NPC 생성 모드(legend-of-lua): 새 NPC를 한 번에 생성→검증→Lua 출력하고, 통과하면
    // game.maps에 주입해 카탈로그/트리가 즉시 인식한다(이후 그 NPC를 퀘스트 기버로 쓸 수 있다).
    if (luaNpcMode) {
      await runGenerateLuaNpc()
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

  // 실행 중인 외부 게임(legend-of-lua)에 라이브 전송하는 브리지 클라이언트(HTTP localhost:17320).
  // 폴링 없이 apply()만 쓴다 — 게임/브리지가 안 떠 있으면 응답이 실패로 와서 상태로 알린다.
  const liveBridge = createGameBridge({
    baseUrl: 'http://localhost:17320',
    onStatusChange: () => {}
  })

  const runApply = (): void => {
    // legend-of-lua NPC 스폰: 패널 iframe(love.js)에 postMessage로 보내고(host page editor-bridge.js가
    // 받아 게임에 전달), 네이티브 게임이 떠 있으면 HTTP 브리지로도 보낸다. 패널은 NPC 코드+브리지가
    // 담긴 재빌드된 love.js 빌드여야 즉시 반영된다(미반영이면 native 또는 editorNPCs.lua 경로 사용).
    const spawnPayload = currentResult?.bridgePayload
    if (spawnPayload?.kind === 'spawn_npc' && currentResult) {
      const appliedResult = currentResult
      iframe.contentWindow?.postMessage({ type: 'editor:apply', payload: spawnPayload }, '*')
      void liveBridge.apply(spawnPayload)
      appliedResults.add(appliedResult)
      appliedCount += 1
      render()
      setStatus(
        '게임에 NPC 스폰 요청을 보냈습니다 — 패널은 재빌드된 love.js 빌드여야 즉시 반영됩니다(아니면 복사→editorNPCs.lua 또는 네이티브).'
      )
      return
    }

    // 브리지 게임(legend-of-lua): 생성한 퀘스트/대사를 실행 중인 게임에 라이브 전송한다.
    const payload = currentResult?.bridgePayload
    if (payload && currentResult) {
      const appliedResult = currentResult
      setStatus('실행 중인 게임에 전송 중...')
      // love.js 프리뷰(iframe)에도 같은 페이로드를 전달한다(핸들러가 있으면 즉시 반영).
      iframe.contentWindow?.postMessage({ type: 'editor:apply', payload }, '*')
      void liveBridge.apply(payload).then((response) => {
        if (response.ok) {
          appliedResults.add(appliedResult)
          appliedCount += 1
          render()
          setStatus('게임에 전송됨 — 실행 중인 legend-of-lua에 적용 요청을 보냈습니다.')
        } else {
          setStatus(
            `전송 실패: ${response.error ?? '게임 브리지에 연결할 수 없습니다(게임 실행 + 브리지 켜짐 확인).'}`
          )
        }
      })
      return
    }

    if (!currentResult?.apply) {
      if (currentResult?.exportFileExtension === 'lua') {
        setStatus('이 결과는 Lua 코드입니다. 게임 브리지가 없으면 복사/내보내기를 사용하세요.')
      }
      return
    }

    // apply()는 localStorage 저장을 동반해 실패할 수 있다. 조용히 죽지 않고 상태로 알린다.
    try {
      currentResult.apply()
      // 결과 보드 "적용 상태"·'오늘 작업' 갱신용 표시 전용 기록.
      appliedResults.add(currentResult)
      appliedCount += 1
      render()
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

    const extension = currentResult.exportFileExtension ?? 'json'
    const fileName = `${currentResult.label || 'generated'}.${extension}`
    const blob = new Blob([currentResult.preview], {
      type: extension === 'lua' ? 'text/plain;charset=utf-8' : 'application/json'
    })
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
      // 새 게임에 맞춰 프리뷰 iframe을 다시 가리키고(예: legend-of-lua면 love.js 빌드), 상단 맵/씬
      // 버튼도 그 게임의 실제 맵으로 다시 구성한다.
      syncPreviewToGame()
      currentMapId = undefined
      renderMapSwitcher()
      selectedEntity = undefined
      currentResult = undefined
      currentAnalysis = undefined
      // 퀘스트 모드 상태도 프로젝트 단위 — 새 게임에 옛 후보/검증이 묻어 나오지 않게 비운다.
      candidateMode = false
      luaNpcMode = false
      candidates = []
      selectedCandidateIndex = undefined
      editingCandidateIndex = undefined
      candidateIteration = 0
      currentDryRun = undefined
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
    // 내 게임으로 복귀 — 프리뷰도 기본 게임 URL로, 맵/씬 버튼도 다시 구성한다.
    syncPreviewToGame()
    currentMapId = undefined
    renderMapSwitcher()
    selectedEntity = undefined
    currentResult = undefined
    currentAnalysis = undefined
    candidateMode = false
    luaNpcMode = false
    candidates = []
    selectedCandidateIndex = undefined
    editingCandidateIndex = undefined
    candidateIteration = 0
    currentDryRun = undefined
    history = []
    historyCounter = 0
    sessionTally = { generations: 0, validatorPasses: 0 }
    expandedGroups.clear()
    renderTree()
    renderAnalysis()
    render()
    setStatus(`내 게임으로 복귀했습니다.${parseErrorNote()}`)
  }

  // 감지된 provider의 모델 목록으로 숨은 select와 게임식 선택 칩을 함께 채운다.
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
    // 제공사 칩(Claude/GPT)은 감지된 쪽만 금색으로 — 키가 정하므로 표시 전용.
    providerChips.anthropic.className = provider === 'anthropic' ? PROVIDER_CHIP_ACTIVE : PROVIDER_CHIP
    providerChips.openai.className = provider === 'openai' ? PROVIDER_CHIP_ACTIVE : PROVIDER_CHIP
    // 모델 칩: 클릭하면 숨은 select에 값을 넣고 change를 쏴서 기존 저장 로직을 그대로 태운다.
    modelChips.replaceChildren(
      ...options.map((modelId) => {
        const chip = el('button', modelId === current ? MODEL_CHIP_ACTIVE : MODEL_CHIP, modelId) as HTMLButtonElement
        chip.type = 'button'
        chip.addEventListener('click', () => {
          modelSelect.value = modelId
          modelSelect.dispatchEvent(new Event('change'))
          // 금색 강조를 새 선택값으로 다시 그린다.
          populateModelSelect(provider)
        })
        return chip
      })
    )
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
      apiKeyStatus.className = 'text-sm font-medium text-[#9d9d9d]'
      apiKeyStatus.textContent = '키를 입력하세요.'
      return
    }
    const seq = ++apiKeyCheckSeq
    apiKeyStatus.className = 'text-sm font-medium text-[#9d9d9d]'
    apiKeyStatus.textContent = '확인 중...'
    const check = await validateApiKey(key)
    // 확인 중 더 최신 입력이 있었으면 이 결과는 버린다(레이스 방지).
    if (seq !== apiKeyCheckSeq) {
      return
    }
    // 장부 종이 위에서 읽히는 진한 포인트 컬러(성공/경고/실패).
    apiKeyStatus.className =
      check.status === 'valid'
        ? 'text-sm font-medium text-[#8fc96a]'
        : check.status === 'invalid'
          ? 'text-sm font-medium text-[#f48771]'
          : 'text-sm font-medium text-[#d9a64f]'
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
    // 진행 상태(요청 작성 중 ↔ 생성 대기)도 입력에 따라 갱신(표시 전용).
    updateComposerSteps()
    updateStepBar()
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
    window.open(previewSrcForGame(), 'game-window', 'width=1280,height=720')
  })
  reloadButton.addEventListener('click', () => {
    previewLoading.style.display = 'flex'
    iframe.src = previewSrcForGame()
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
