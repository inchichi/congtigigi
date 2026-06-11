import type { TmxObject } from './tmxObjects'
import type { GameStructureProfile } from './gameStructureProfile'
import { generateEventJsonDraftWithClaude } from './claudeEventJsonGenerator'
import { createGeneratedEventJsonValidationIssues } from './eventJsonSchema'
import { createHolidayDialogueEventSpecFromGeneratedEventJson } from './eventCodeGenerator'
import { savePendingEvent } from './pendingEvents'
import { generateJson } from './llmProvider'
import { createEntityLinesValidationIssues } from './entityLinesValidator'

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
  // 게임에 적용하는 방법. null이면 이 게임은 아직 적용 미지원(생성 미리보기까지).
  apply: (() => void) | null
}

export type GameAdapter = {
  id: string
  name: string
  // 열린 폴더의 파일 이름들로 어느 게임인지 판별한다.
  detect: (fileNames: string[]) => boolean
  // 한 맵의 TMX 객체들을 이 게임의 엔티티로 변환한다.
  extractEntities: (mapId: string, objects: TmxObject[]) => GameEntity[]
  // 이 게임에 대해 에디터가 생성→적용까지 지원하는지(UI 힌트용).
  supportsApply: boolean
  // 이 게임의 콘텐츠를 LLM으로 생성한다. 결과의 apply()로 게임에 반영한다.
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
  supportsApply: true,
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
      }
    }
  }
}

const LEGEND_KIND_BY_GROUP: Record<string, string> = {
  Enemies: 'enemy',
  NPCs: 'npc',
  Chests: 'chest',
  Loot: 'loot'
}

// 전용 콘텐츠 모델이 없는 게임(legend-of-lua, 미지의 게임)의 공용 생성: 엔티티용 대사/설명.
// 적용은 게임 런타임 연결이 필요해 아직 null(미리보기까지).
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
    apply: null
  }
}

export const legendOfLuaAdapter: GameAdapter = {
  id: 'legend-of-lua',
  name: 'Legend of Lua (Love2D)',
  detect: (fileNames) =>
    fileNames.includes('conf.lua') || fileNames.includes('main.lua'),
  extractEntities: (mapId, objects) =>
    objects
      .filter(
        (object) =>
          LEGEND_KIND_BY_GROUP[object.group] !== undefined && object.name.length > 0
      )
      .map((object) => ({
        id: `${object.group}-${object.id}`,
        name: object.name,
        kind: LEGEND_KIND_BY_GROUP[object.group],
        mapId
      })),
  // Love2D 런타임에 라이브 적용은 아직 미구현(Stage 3). 지금은 엔티티 브라우징·생성까지.
  supportsApply: false,
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
  supportsApply: false,
  generate: (request) => generateEntityLines('이 게임', request)
}

export const GAME_ADAPTERS: GameAdapter[] = [
  legendOfLuaAdapter,
  rpgAdapter,
  genericAdapter
]

export const detectAdapter = (fileNames: string[]): GameAdapter =>
  GAME_ADAPTERS.find((adapter) => adapter.detect(fileNames)) ?? genericAdapter
