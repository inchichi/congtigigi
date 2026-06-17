import { generateJson } from './llmProvider'
import {
  buildFeedbackInstruction,
  type GenerationFeedback
} from './gameAdapter'
import {
  buildLuaQuestCatalogText,
  type LuaQuestCatalog
} from './luaQuestCatalog'
import {
  LUA_NPC_APPEARANCES,
  LUA_NPC_BEHAVIOR_TYPES,
  type GeneratedLuaNpcJson
} from './luaNpcSchema'

type JsonSchema = Record<string, unknown>

const uniqueStrings = (values: string[]): string[] => [...new Set(values)]

const createStringEnumSchema = (values: string[]): JsonSchema => ({
  type: 'string',
  enum: uniqueStrings(values)
})

// 생성 NPC JSON의 스키마. 맵·외형·행동은 "고를 수 있는 값"으로만 제약해, 게임에 없는 맵이나
// 빌려올 수 없는 외형을 LLM이 지어내지 못하게 한다(퀘스트 생성의 enum 그라운딩과 같은 원리).
const createLuaNpcJsonSchema = (catalog: LuaQuestCatalog): JsonSchema => ({
  type: 'object',
  additionalProperties: false,
  properties: {
    npc_id: { type: 'string', pattern: '^[a-z0-9_]+$' },
    name: { type: 'string', minLength: 1 },
    map_id: createStringEnumSchema(catalog.scenes),
    appearance: { type: 'string', enum: [...LUA_NPC_APPEARANCES] },
    position: {
      type: 'object',
      additionalProperties: false,
      properties: {
        x: { type: 'number', minimum: 0 },
        y: { type: 'number', minimum: 0 }
      },
      required: ['x', 'y']
    },
    dialogue_lines: {
      type: 'array',
      minItems: 1,
      maxItems: 4,
      items: { type: 'string', minLength: 1 }
    },
    behavior: {
      type: 'object',
      additionalProperties: false,
      properties: {
        type: { type: 'string', enum: LUA_NPC_BEHAVIOR_TYPES },
        radius: { type: 'number', minimum: 0 }
      },
      required: ['type', 'radius']
    }
  },
  required: [
    'npc_id',
    'name',
    'map_id',
    'appearance',
    'position',
    'dialogue_lines',
    'behavior'
  ]
})

const createLuaNpcSystemPrompt = (catalog: LuaQuestCatalog): string =>
  [
    'You are creating a brand-new NPC for a Legend of Lua (Love2D) game.',
    'The NPC sprite is borrowed from the my-sample-rpg asset set — pick one appearance id from the allowed list.',
    'Place the NPC on one of the existing maps only. Do not invent map ids.',
    'Choose a fresh npc_id (snake_case) that does not collide with existing entities.',
    'position is in tile coordinates; keep it small and sensible (e.g. near the map center).',
    'behavior.type is "wander" (radius > 0, roams near its home) or "stationary" (radius 0).',
    'dialogue_lines are short interaction lines the NPC says when talked to (1 to 4 lines).',
    `Allowed appearance ids: ${LUA_NPC_APPEARANCES.join(', ')}`,
    `Allowed map ids: ${uniqueStrings(catalog.scenes).join(', ') || '(none)'}`,
    'Return JSON only. The editor renders the Lua module and validates after generation.',
    buildLuaQuestCatalogText(catalog)
  ].join('\n')

// 이미 배치된 엔티티 id를 LLM에 알려, 충돌하지 않는 새 npc_id를 고르도록 유도한다(검증과 별개의 보조).
const buildExistingIdsContext = (catalog: LuaQuestCatalog): string => {
  const ids = [
    ...catalog.npcs.map((entity) => entity.id),
    ...catalog.enemies.map((entity) => entity.id),
    ...catalog.acquirables.map((entity) => entity.id)
  ]
  return ids.length > 0
    ? `\n\nExisting entity ids (avoid colliding with these): ${ids.join(', ')}`
    : ''
}

export const generateLuaNpcJson = ({
  apiKey,
  userPrompt,
  catalog,
  feedback
}: {
  apiKey: string
  userPrompt: string
  catalog: LuaQuestCatalog
  feedback?: GenerationFeedback
}): Promise<GeneratedLuaNpcJson> => {
  const feedbackHint = feedback ? buildFeedbackInstruction(feedback) : ''

  return generateJson<GeneratedLuaNpcJson>({
    apiKey,
    instructions: createLuaNpcSystemPrompt(catalog),
    input: `${userPrompt}${buildExistingIdsContext(catalog)}${feedbackHint}`,
    schemaName: 'generated_lua_npc_json',
    schema: createLuaNpcJsonSchema(catalog)
  })
}
