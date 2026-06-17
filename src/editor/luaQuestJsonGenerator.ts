import { generateJson } from './llmProvider'
import {
  buildFeedbackInstruction,
  type GameEntity,
  type GenerationFeedback
} from './gameAdapter'
import type { QuestCandidate } from './questCandidates'
import {
  buildLuaQuestCatalogText,
  type LuaQuestCatalog
} from './luaQuestCatalog'
import type {
  GeneratedLuaQuestJson,
  GeneratedLuaQuestObjective
} from './luaQuestSchema'

type JsonSchema = Record<string, unknown>

const uniqueStrings = (values: string[]): string[] => [...new Set(values)]

const createStringEnumSchema = (values: string[]): JsonSchema => ({
  type: 'string',
  enum: uniqueStrings(values)
})

const createObjectiveSchema = (
  type: GeneratedLuaQuestObjective['type'],
  targetProperties: Record<string, JsonSchema>,
  requiredTargetFields: string[]
): JsonSchema => ({
  type: 'object',
  additionalProperties: false,
  properties: {
    type: { type: 'string', const: type },
    label: { type: 'string', minLength: 1 },
    required: { type: 'integer', minimum: 1 },
    target: {
      type: 'object',
      additionalProperties: false,
      properties: targetProperties,
      required: requiredTargetFields
    }
  },
  required: ['type', 'label', 'required', 'target']
})

const getQuestGiverNpcIds = (
  catalog: LuaQuestCatalog,
  entity?: GameEntity
): string[] =>
  entity?.kind === 'npc'
    ? uniqueStrings([entity.id])
    : uniqueStrings(catalog.npcs.map((npc) => npc.id))

const createLuaQuestJsonSchema = (
  catalog: LuaQuestCatalog,
  entity?: GameEntity
): JsonSchema => {
  const questGiverNpcIds = getQuestGiverNpcIds(catalog, entity)
  const enemyIds = uniqueStrings(catalog.enemies.map((enemy) => enemy.id))
  const npcIds = uniqueStrings(catalog.npcs.map((npc) => npc.id))
  const acquirableIds = uniqueStrings(catalog.acquirables.map((item) => item.id))
  const sceneIds = uniqueStrings(catalog.scenes)

  return {
    type: 'object',
    additionalProperties: false,
    properties: {
      quest_id: {
        type: 'string',
        pattern: '^[a-z0-9_]+$'
      },
      title: { type: 'string', minLength: 1 },
      giver_npc_entity_id: createStringEnumSchema(questGiverNpcIds),
      request_text: { type: 'string', minLength: 1 },
      guide_text: { type: 'string', minLength: 1 },
      start_dialogue_lines: {
        type: 'array',
        minItems: 1,
        items: { type: 'string', minLength: 1 }
      },
      active_dialogue_lines: {
        type: 'array',
        minItems: 1,
        items: { type: 'string', minLength: 1 }
      },
      completion_dialogue_lines: {
        type: 'array',
        minItems: 1,
        items: { type: 'string', minLength: 1 }
      },
      objectives: {
        type: 'array',
        minItems: 1,
        maxItems: 3,
        items: {
          oneOf: [
            createObjectiveSchema(
              'defeat',
              { entityId: createStringEnumSchema(enemyIds) },
              ['entityId']
            ),
            createObjectiveSchema(
              'talk',
              { entityId: createStringEnumSchema(npcIds) },
              ['entityId']
            ),
            createObjectiveSchema(
              'acquire',
              { entityId: createStringEnumSchema(acquirableIds) },
              ['entityId']
            ),
            createObjectiveSchema(
              'reach',
              { mapId: createStringEnumSchema(sceneIds) },
              ['mapId']
            )
          ]
        }
      },
      rewards: {
        type: 'object',
        additionalProperties: false,
        properties: {
          gold: { type: 'integer', minimum: 0 },
          experience: { type: 'integer', minimum: 0 },
          items: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                label: { type: 'string', minLength: 1 },
                quantity: { type: 'integer', minimum: 1 }
              },
              required: ['label', 'quantity']
            }
          }
        },
        required: ['gold', 'experience', 'items']
      }
    },
    required: [
      'quest_id',
      'title',
      'giver_npc_entity_id',
      'request_text',
      'guide_text',
      'start_dialogue_lines',
      'active_dialogue_lines',
      'completion_dialogue_lines',
      'objectives',
      'rewards'
    ]
  }
}

const createLuaQuestSystemPrompt = (
  catalog: LuaQuestCatalog,
  entity?: GameEntity
): string => {
  const questGiverNpcIds = getQuestGiverNpcIds(catalog, entity)
  const enemyIds = uniqueStrings(catalog.enemies.map((enemy) => enemy.id))
  const npcIds = uniqueStrings(catalog.npcs.map((npc) => npc.id))
  const acquirableIds = uniqueStrings(catalog.acquirables.map((item) => item.id))
  const sceneIds = uniqueStrings(catalog.scenes)

  const selectedNpcLine =
    entity?.kind === 'npc'
      ? `Selected NPC giver: ${entity.id} (${entity.name}). Use this id exactly as giver_npc_entity_id.`
      : undefined

  return [
    'You are writing quest JSON for a Legend of Lua game.',
    'Use only the exact ids from the provided catalog.',
    'Do not invent new giver NPC ids, enemy ids, acquirable ids, or scene ids.',
    'Return JSON only. The editor will render the Lua module after validation.',
    `Quest giver NPC ids: ${questGiverNpcIds.join(', ') || '(none)'}`,
    `Enemy ids for defeat: ${enemyIds.join(', ') || '(none)'}`,
    `NPC ids for talk: ${npcIds.join(', ') || '(none)'}`,
    `Acquirable ids for acquire: ${acquirableIds.join(', ') || '(none)'}`,
    `Scene ids for reach: ${sceneIds.join(', ') || '(none)'}`,
    'Use 1 to 3 objectives only.',
    'Reward items use free-form labels, not ids.',
    ...(selectedNpcLine ? [selectedNpcLine] : []),
    buildLuaQuestCatalogText(catalog)
  ].join('\n')
}

const createQuestTargetHint = (
  candidate?: QuestCandidate,
  entity?: GameEntity
): string => {
  const parts: string[] = []

  if (candidate) {
    parts.push(
      `Selected candidate:\n- title: ${candidate.title}\n- summary: ${candidate.summary}` +
        (candidate.target_hint ? `\n- target npc hint: ${candidate.target_hint}` : '')
    )
  }

  if (entity) {
    parts.push(`Selected entity: ${entity.id} (${entity.name}) on map ${entity.mapId}.`)
  }

  return parts.length > 0 ? `\n\n${parts.join('\n\n')}` : ''
}

export const generateLuaQuestJson = ({
  apiKey,
  userPrompt,
  catalog,
  candidate,
  feedback,
  entity
}: {
  apiKey: string
  userPrompt: string
  catalog: LuaQuestCatalog
  candidate?: QuestCandidate
  feedback?: GenerationFeedback
  entity?: GameEntity
}): Promise<GeneratedLuaQuestJson> => {
  const feedbackHint = feedback ? buildFeedbackInstruction(feedback) : ''

  return generateJson<GeneratedLuaQuestJson>({
    apiKey,
    instructions: createLuaQuestSystemPrompt(catalog, entity),
    input: `${userPrompt}${createQuestTargetHint(candidate, entity)}${feedbackHint}`,
    schemaName: 'generated_lua_quest_json',
    schema: createLuaQuestJsonSchema(catalog, entity)
  })
}
