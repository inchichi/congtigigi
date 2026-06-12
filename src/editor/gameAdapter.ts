import type { TmxObject } from './tmxObjects'
import type { GameStructureProfile } from './gameStructureProfile'
import { generateEventJsonDraftWithClaude } from './claudeEventJsonGenerator'
import { createGeneratedEventJsonValidationIssues } from './eventJsonSchema'
import { createHolidayDialogueEventSpecFromGeneratedEventJson } from './eventCodeGenerator'
import { savePendingEvent } from './pendingEvents'
import { generateJson } from './llmProvider'
import { createEntityLinesValidationIssues } from './entityLinesValidator'
import type { BridgeApplyMessage } from './gameBridge'

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

export type GenerationRequest = {
  apiKey: string
  userPrompt: string
  entity?: GameEntity
  profile?: GameStructureProfile
  // LLM 게임 분석에서 얻은 게임 설명(이름·엔진·콘텐츠 모델). 있으면 생성을 그 게임답게 유도한다.
  gameContext?: string
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
  generate: async ({ apiKey, userPrompt, entity, profile }) => {
    if (!profile) {
      throw new Error('이 게임의 구조 프로필이 없습니다.')
    }

    const targetHint = entity
      ? ` 이 이벤트의 대상은 반드시 NPC id="${entity.id}"(${entity.name}, map=${entity.mapId})로 한다.`
      : ''
    const eventJson = await generateEventJsonDraftWithClaude({
      apiKey,
      userPrompt: `${userPrompt}${targetHint}`,
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
      bridgePayload: null
    }
  }
}

// 그룹 이름 → 엔티티 종류. 맵마다 그룹 이름의 대소문자·단복수가 제각각(NPCs/npc/Npc...)이라
// 소문자로 정규화해 매칭하고, 모르는 그룹은 그룹 이름 자체를 종류로 쓴다(트리에서 카테고리로 묶임).
const LEGEND_KIND_BY_GROUP: Record<string, string> = {
  enemies: 'enemy',
  enemy: 'enemy',
  npcs: 'npc',
  npc: 'npc',
  characters: 'npc',
  chests: 'chest',
  chest: 'chest',
  loot: 'loot'
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
  { apiKey, userPrompt, entity, gameContext }: GenerationRequest
): Promise<GenerationResult> => {
  const target = entity ? `${entity.kind} "${entity.name}"` : '게임 요소'
  const contextLine = gameContext ? `\n\n게임 정보: ${gameContext}` : ''
  const generated = await generateJson<{ entity: string; lines: string[] }>({
    apiKey,
    instructions: `${gameName}의 게임 요소에 어울리는 짧은 한국어 대사 또는 설명을 1~4줄 생성한다. 주어진 게임 정보가 있으면 그 게임의 분위기에 맞춘다.`,
    input: `${userPrompt}\n\n대상: ${target}${contextLine}`,
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
