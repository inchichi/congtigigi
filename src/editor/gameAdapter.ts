import type { TmxObject } from './tmxObjects'
import type { GameStructureProfile } from './gameStructureProfile'
import { generateEventJsonDraftWithClaude } from './claudeEventJsonGenerator'
import { createGeneratedEventJsonValidationIssues } from './eventJsonSchema'
import { createHolidayDialogueEventSpecFromGeneratedEventJson } from './eventCodeGenerator'
import { savePendingEvent } from './pendingEvents'
import { generateJson } from './llmProvider'
import { createEntityLinesValidationIssues } from './entityLinesValidator'
import type { BridgeApplyMessage } from './gameBridge'
import type { GeneratedEventJson } from './eventJsonSchema'
import type { QuestCandidate } from './questCandidates'

// 게임에 적용하는 방식:
// - 'local-storage': 에디터와 같은 origin 웹게임(my-sample-rpg). apply()가 localStorage로 전달.
// - 'bridge': 별도 프로세스 게임(Love2D legend-of-lua). 실행 중인 게임에 HTTP 브리지로 전송.
// - 'none': 아직 적용 경로가 없는 게임. 생성 미리보기까지만.
export type GameApplyMode = 'local-storage' | 'bridge' | 'none'

// 게임마다 맵/엔티티 규칙·생성·적용이 달라서, 그걸 어댑터로 분리한다. 새 게임 지원 = 새 어댑터 추가.
export type GameEntity = {
  id: string
  name: string
  kind: string
  mapId: string
}

// 재생성(피드백 루프) 맥락: 이전 결과 + 자동 검증 이슈 + 사람 거절 사유 + 반복 횟수.
// 거절된 결과를 사유와 함께 다시 생성할 때만 채워진다.
export type GenerationFeedback = {
  previousOutput: string
  validatorIssues: string[]
  rejectionReason: string
  iteration: number
}

export type GenerationRequest = {
  apiKey: string
  userPrompt: string
  entity?: GameEntity
  profile?: GameStructureProfile
  // LLM 게임 분석에서 얻은 게임 설명(이름·엔진·콘텐츠 모델). 있으면 생성을 그 게임답게 유도한다.
  gameContext?: string
  // 있으면 재생성 모드: 이전 결과를 이 피드백에 맞춰 수정한다(처음부터 새로 쓰지 않음).
  feedback?: GenerationFeedback
  // 퀘스트 2단계: 유저가 고른 자연어 후보. 있으면 그 후보를 이벤트 JSON 생성 프롬프트에 엮는다.
  // 없으면 기존 단일 생성과 완전히 동일하게 동작한다.
  candidate?: QuestCandidate
}

// 재생성 시 모델에 붙이는 수정 지침. 이미 된 부분은 유지하고 거절 사유·검증 문제만 고치게 유도한다.
export const buildFeedbackInstruction = (feedback: GenerationFeedback): string => {
  const issues =
    feedback.validatorIssues.length > 0
      ? `\n\n자동 검증에서 발견된 문제:\n${feedback.validatorIssues
          .map((issue) => `- ${issue}`)
          .join('\n')}`
      : ''

  return (
    `\n\n--- 수정 요청 (반복 ${feedback.iteration}회차) ---\n` +
    `바로 직전 생성 결과:\n${feedback.previousOutput}${issues}\n\n` +
    `사용자가 이 결과를 거절했고, 사유는: "${feedback.rejectionReason}"\n\n` +
    `수정 지침:\n` +
    `- 이미 잘 된 부분은 그대로 유지한다.\n` +
    `- 거절 사유와 위 검증 문제만 고친다.\n` +
    `- 대상 엔티티/대상은 꼭 필요한 경우가 아니면 바꾸지 않는다.\n` +
    `- 검증 제약과 현재 게임 맥락을 따른다.\n` +
    `- 거절 사유가 "전체가 못 쓴다"는 취지가 아니라면 처음부터 새로 쓰지 말고 부분만 수정한다.`
  )
}

export type GenerationResult = {
  label: string
  preview: string
  // 생성과 분리된 결정적 검증(Validator) 결과. 빈 배열이면 통과(이사님 #1: 생성/검증 분리).
  issues: string[]
  // 같은 origin 웹게임(local-storage 적용)용. null이면 이 경로로는 적용하지 않는다.
  apply: (() => void) | null
  // 브리지(별도 프로세스 게임) 적용용 구조화 페이로드. applyMode가 'bridge'인 게임에서 채운다.
  bridgePayload: BridgeApplyMessage | null
  // 이벤트 JSON 게임(my-sample-rpg)에서 채운다. 에디터가 드라이런 검증(dryRunEventApply)에 쓴다.
  eventJson?: GeneratedEventJson
}

export type GameAdapter = {
  id: string
  name: string
  // 열린 폴더의 파일 이름들로 어느 게임인지 판별한다.
  detect: (fileNames: string[]) => boolean
  // 한 맵의 TMX 객체들을 이 게임의 엔티티로 변환한다.
  extractEntities: (mapId: string, objects: TmxObject[]) => GameEntity[]
  // 이 게임에 생성물을 어떻게 적용하는지(UI/적용 라우팅용).
  applyMode: GameApplyMode
  // 이 게임의 love.js 웹 빌드가 앱에 번들돼 있으면 그 URL. 설정에 사용자 입력이 없을 때
  // 이 값을 기본으로 써서, 어느 컴퓨터에서나 별도 설정 없이 패널에서 바로 플레이된다.
  defaultWebBuildUrl?: string
  // 이 게임의 콘텐츠를 LLM으로 생성한다. 결과를 applyMode에 맞는 경로로 게임에 반영한다.
  generate: (request: GenerationRequest) => Promise<GenerationResult>
}

// my-sample-rpg의 맵 object를 에디터 종류(npc/monster/sign/portal/object)로 분류한다.
// 외형(properties.type)의 접두사로 캐릭터류를 가르고, type="portal"은 포털로 본다.
const classifyRpgEntityKind = (object: TmxObject): string => {
  if (object.type === 'portal') {
    return 'portal'
  }
  if (object.type === 'character') {
    const appearance = object.properties.type ?? object.properties.appearanceType ?? ''
    if (appearance.startsWith('character_')) {
      return 'npc'
    }
    if (appearance.startsWith('monster_')) {
      return 'monster'
    }
    if (appearance.startsWith('sign_')) {
      return 'sign'
    }
  }
  return object.type || 'object'
}

export const rpgAdapter: GameAdapter = {
  id: 'my-sample-rpg',
  name: 'My Sample RPG (TS/Pixi)',
  detect: (fileNames) => fileNames.includes('town.tmx'),
  // 맵에 있는 모든 요소를 종류별로 뽑아 에디터 트리에 보여준다(사용자가 무엇을 바꿀지 알 수 있게).
  // 같은 type="character" object여도 외형 접두사로 갈린다: character_*(대화 NPC)·monster_*(몬스터)·
  // sign_*(표지판). type="portal"은 포털. 대화 생성 대상은 NPC뿐이라, profile.npcs(=생성/검증/허용목록)는
  // loadGame이 kind==='npc'로만 좁힌다 — 트리(표시)와 생성 대상(curated)을 분리한다.
  extractEntities: (mapId, objects) =>
    objects.map((object) => {
      const kind = classifyRpgEntityKind(object)
      const fallbackName = `${kind}-${object.id}`
      return {
        id: object.name || fallbackName,
        name: object.properties.displayText || object.name || fallbackName,
        kind,
        mapId
      }
    }),
  applyMode: 'local-storage',
  generate: async ({ apiKey, userPrompt, entity, profile, feedback, candidate }) => {
    if (!profile) {
      throw new Error('이 게임의 구조 프로필이 없습니다.')
    }

    const targetHint = entity
      ? ` 이 이벤트의 대상은 반드시 NPC id="${entity.id}"(${entity.name}, map=${entity.mapId})로 한다.`
      : ''
    // 퀘스트 2단계: 고른 후보가 있으면 그 방향대로 이벤트 JSON을 만들게 엮는다.
    // candidate가 없으면 이 문자열은 빈 값이라 기존 단일 생성과 프롬프트가 동일하다.
    const candidateHint = candidate
      ? `\n\n선택된 퀘스트 후보를 그대로 구현한다:\n- 제목: ${candidate.title}\n- 내용: ${candidate.summary}` +
        (candidate.target_hint ? `\n- 대상 NPC: ${candidate.target_hint}` : '')
      : ''
    const feedbackHint = feedback ? buildFeedbackInstruction(feedback) : ''
    const eventJson = await generateEventJsonDraftWithClaude({
      apiKey,
      userPrompt: `${userPrompt}${candidateHint}${targetHint}${feedbackHint}`,
      profile
    })

    // 생성과 분리된 검증 단계: 필드 + map/npc/item ID 실존성 + 맵-NPC 일치.
    const issues = createGeneratedEventJsonValidationIssues(eventJson, profile).map(
      (issue) => `${issue.path} - ${issue.message}`
    )

    return {
      label: eventJson.event_name,
      preview: JSON.stringify(eventJson, null, 2),
      issues,
      apply: () => {
        const spec = createHolidayDialogueEventSpecFromGeneratedEventJson(eventJson)
        if (spec) {
          savePendingEvent(spec)
        }
      },
      bridgePayload: null,
      // 에디터가 드라이런 검증(dryRunEventApply)에 쓰도록 원본 이벤트 JSON을 노출한다.
      eventJson
    }
  }
}

// 그룹 이름 → 엔티티 종류. 맵마다 그룹 이름의 대소문자·단복수가 제각각(NPCs/npc/Npc...)이라
// 소문자로 정규화해 매칭하고, 모르는 그룹은 그룹 이름 자체를 종류로 쓴다(트리에서 카테고리로 묶임).
// 알 수 없는 그룹은 그룹 이름이 그대로 트리에 영문으로 노출돼 직관성이 떨어진다(예: 'transitions').
// 흔한 그룹 이름을 에디터가 아는 종류(label·icon·카테고리가 붙는)로 정규화해 한눈에 읽히게 한다.
const LEGEND_KIND_BY_GROUP: Record<string, string> = {
  enemies: 'enemy',
  enemy: 'enemy',
  monsters: 'enemy',
  monster: 'enemy',
  npcs: 'npc',
  npc: 'npc',
  characters: 'npc',
  character: 'npc',
  chests: 'chest',
  chest: 'chest',
  loot: 'loot',
  items: 'loot',
  item: 'loot',
  // 맵 전환/포털 류 — 영문 'transitions'가 그대로 보이던 그룹을 '포털'로 정규화한다.
  transitions: 'portal',
  transition: 'portal',
  portals: 'portal',
  portal: 'portal',
  doors: 'portal',
  door: 'portal',
  warps: 'portal',
  warp: 'portal',
  // 지형·장식 류.
  trees: 'tree',
  tree: 'tree',
  props: 'prop',
  prop: 'prop',
  signs: 'sign',
  sign: 'sign',
  buildings: 'building',
  building: 'building'
}

// 엔티티가 아니라 충돌·경계 같은 구조용 도형이 든 그룹은 트리에서 제외한다(NPC는 보이게 하되
// 벽/콜라이더는 빼려는 목적). 그 외 그룹은 이름이 뭐든 엔티티 후보로 본다.
const LEGEND_NON_ENTITY_GROUPS = new Set([
  'walls',
  'wall',
  'collision',
  'collisions',
  'collider',
  'colliders',
  'bounds',
  'boundaries',
  'bound'
])

const legendKindForGroup = (group: string): string => {
  const normalized = group.trim().toLowerCase()
  return LEGEND_KIND_BY_GROUP[normalized] ?? (normalized || 'entity')
}

const isLegendEntityObject = (group: string): boolean =>
  !LEGEND_NON_ENTITY_GROUPS.has(group.trim().toLowerCase())

// 전용 콘텐츠 모델이 없는 게임(legend-of-lua, 미지의 게임)의 공용 생성: 엔티티용 대사/설명.
// 결과의 bridgePayload는 'bridge' 게임에서 실행 중인 게임으로 전송하는 데 쓰인다('none' 게임은 무시).
const generateEntityLines = async (
  gameName: string,
  { apiKey, userPrompt, entity, gameContext, feedback }: GenerationRequest
): Promise<GenerationResult> => {
  const target = entity ? `${entity.kind} "${entity.name}"` : '게임 요소'
  const contextLine = gameContext ? `\n\n게임 정보: ${gameContext}` : ''
  const feedbackLine = feedback ? buildFeedbackInstruction(feedback) : ''
  const generated = await generateJson<{ entity: string; lines: string[] }>({
    apiKey,
    instructions: `${gameName}의 게임 요소에 어울리는 짧은 대사 또는 설명을 1~4줄 생성한다. 게임 안 영문 폰트로 표시되므로 lines는 반드시 영어로 작성한다(사용자 프롬프트가 한국어여도 출력 대사는 영어). 주어진 게임 정보가 있으면 그 분위기에 맞춘다.`,
    input: `${userPrompt}\n\n대상: ${target}${contextLine}${feedbackLine}`,
    schemaName: 'entity_lines',
    schema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        entity: { type: 'string' },
        lines: { type: 'array', items: { type: 'string' } }
      },
      required: ['entity', 'lines']
    }
  })

  // 생성과 분리된 결정적 검증을 rpg 외 게임에도 적용한다(이사님 #1). 빈 배열이면 통과.
  const issues = createEntityLinesValidationIssues(generated, {
    targetEntityName: entity?.name
  })

  return {
    label: generated.entity || (entity?.name ?? '생성 결과'),
    preview: JSON.stringify(generated, null, 2),
    issues,
    apply: null,
    bridgePayload: {
      id: `entity_lines-${Date.now()}`,
      kind: 'entity_lines',
      target: entity
        ? {
            id: entity.id,
            name: entity.name,
            kind: entity.kind,
            mapId: entity.mapId
          }
        : null,
      lines: generated.lines,
      generatedAt: Date.now()
    }
  }
}

export const legendOfLuaAdapter: GameAdapter = {
  id: 'legend-of-lua',
  name: 'Legend of Lua (Love2D)',
  detect: (fileNames) =>
    fileNames.includes('conf.lua') || fileNames.includes('main.lua'),
  // 이름이나 타입이 있는 object layer 요소를 엔티티로 본다(NPC가 특정 그룹/대문자에 묶여 있지
  // 않아도, 이름 대신 type만 있어도 보이게). 벽·콜라이더 같은 구조용 그룹과 무명·무타입은 거른다.
  extractEntities: (mapId, objects) =>
    objects
      .filter(
        (object) =>
          isLegendEntityObject(object.group) &&
          (object.name.length > 0 || object.type.length > 0)
      )
      .map((object) => {
        const kind = legendKindForGroup(object.group)
        return {
          id: `${object.group}-${object.id}`,
          name: object.name || object.type || `${kind}-${object.id}`,
          kind,
          mapId
        }
      }),
  // 실행 중인 Love2D 게임에 HTTP 브리지로 라이브 적용한다(docs/legend-of-lua-bridge-protocol.md).
  applyMode: 'bridge',
  // love.js로 빌드한 웹 버전이 public/legend-of-lua/에 번들돼 있어 Vite가 이 경로로 서빙한다.
  // 기본값이라 별도 프로세스/브리지 없이도 어느 컴퓨터에서나 패널에서 바로 플레이된다.
  defaultWebBuildUrl: '/legend-of-lua/',
  generate: (request) =>
    generateEntityLines('legend-of-lua (Love2D 2D 액션 RPG)', request)
}

// 알려진 어댑터에 안 걸리는 게임의 최후 fallback. 엔티티는 LLM 분석으로 채우고(extractEntities는
// 비어 있음), 생성은 공용 대사 생성을 쓴다.
export const genericAdapter: GameAdapter = {
  id: 'generic',
  name: 'Unknown game (LLM-analyzed)',
  detect: () => true,
  extractEntities: () => [],
  applyMode: 'none',
  generate: (request) => generateEntityLines('이 게임', request)
}

export const GAME_ADAPTERS: GameAdapter[] = [
  legendOfLuaAdapter,
  rpgAdapter,
  genericAdapter
]

export const detectAdapter = (fileNames: string[]): GameAdapter =>
  GAME_ADAPTERS.find((adapter) => adapter.detect(fileNames)) ?? genericAdapter
