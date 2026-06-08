import type { TmxObject } from './tmxObjects'
import type { GameStructureProfile } from './gameStructureProfile'
import { generateEventJsonDraftWithOpenAi } from './openaiEventJsonGenerator'
import { createGeneratedEventJsonValidationIssues } from './eventJsonSchema'
import { createHolidayDialogueEventSpecFromGeneratedEventJson } from './eventCodeGenerator'
import { savePendingEvent } from './pendingEvents'
import { generateJsonWithOpenAi } from './openaiGenerate'

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

export const rpgAdapter: GameAdapter = {
  id: 'my-sample-rpg',
  name: 'My Sample RPG (TS/Pixi)',
  detect: (fileNames) =>
    fileNames.includes('town.tmx') ||
    fileNames.some((name) => name === 'createPixiTiledMapView.ts'),
  extractEntities: (mapId, objects) =>
    objects
      .filter(
        (object) =>
          object.type === 'character' &&
          (object.properties.type !== undefined ||
            object.properties.appearanceType !== undefined)
      )
      .map((object) => ({
        id: object.name || `character-${object.id}`,
        name: object.properties.displayText || object.name || `character-${object.id}`,
        kind: 'npc',
        mapId
      })),
  supportsApply: true,
  generate: async ({ apiKey, userPrompt, entity, profile }) => {
    if (!profile) {
      throw new Error('이 게임의 구조 프로필이 없습니다.')
    }

    const targetHint = entity
      ? ` 이 이벤트의 대상은 반드시 NPC id="${entity.id}"(${entity.name}, map=${entity.mapId})로 한다.`
      : ''
    const eventJson = await generateEventJsonDraftWithOpenAi({
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
  generate: async ({ apiKey, userPrompt, entity }) => {
    const target = entity ? `${entity.kind} "${entity.name}"` : '게임 요소'
    const generated = await generateJsonWithOpenAi<{ entity: string; lines: string[] }>({
      apiKey,
      instructions:
        'legend-of-lua(2D 액션 RPG, Love2D)의 게임 요소에 어울리는 짧은 한국어 대사 또는 설명을 1~4줄 생성한다. 응답에는 JSON만 포함한다.',
      input: `${userPrompt}\n\n대상: ${target}`,
      schemaName: 'legend_entity_lines',
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

    return {
      label: generated.entity || (entity?.name ?? '생성 결과'),
      preview: JSON.stringify(generated, null, 2),
      issues: [],
      // Love2D 런타임 라이브 적용은 Stage 3.
      apply: null
    }
  }
}

export const GAME_ADAPTERS: GameAdapter[] = [legendOfLuaAdapter, rpgAdapter]

export const detectAdapter = (fileNames: string[]): GameAdapter =>
  GAME_ADAPTERS.find((adapter) => adapter.detect(fileNames)) ?? rpgAdapter
