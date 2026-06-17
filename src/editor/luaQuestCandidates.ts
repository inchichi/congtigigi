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

export type GenerateLuaQuestCandidatesInput = {
  apiKey: string
  userPrompt: string
  catalog: LuaQuestCatalog
  entity?: GameEntity
  gameContext?: string
  feedback?: GenerationFeedback
}

export const LUA_QUEST_CANDIDATE_COUNT = 3

const uniqueStrings = (values: string[]): string[] => [...new Set(values)]

const getQuestGiverNpcIds = (
  catalog: LuaQuestCatalog,
  entity?: GameEntity
): string[] =>
  entity?.kind === 'npc'
    ? uniqueStrings([entity.id])
    : uniqueStrings(catalog.npcs.map((npc) => npc.id))

const buildGroundingContext = (catalog: LuaQuestCatalog): string =>
  `\n\nAvailable NPCs: ${catalog.npcs.map((npc) => `${npc.id}(${npc.name}, map=${npc.mapId})`).join(', ') || '(none)'}\n` +
  `Available enemies: ${catalog.enemies.map((enemy) => `${enemy.id}(${enemy.name}, map=${enemy.mapId})`).join(', ') || '(none)'}\n` +
  `Available acquirables: ${catalog.acquirables.map((item) => `${item.id}(${item.name}, map=${item.mapId})`).join(', ') || '(none)'}\n` +
  `Available scenes: ${catalog.scenes.join(', ') || '(none)'}`

const normalizeCandidate = (candidate: Partial<QuestCandidate>): QuestCandidate => ({
  title: (candidate.title ?? '').trim(),
  summary: (candidate.summary ?? '').trim(),
  target_hint: (candidate.target_hint ?? '').trim()
})

const createLuaQuestCandidatesSchema = (
  catalog: LuaQuestCatalog,
  entity?: GameEntity
): object => {
  const targetHintEnum = ['', ...getQuestGiverNpcIds(catalog, entity)]

  return {
    type: 'object',
    additionalProperties: false,
    properties: {
      candidates: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            title: { type: 'string' },
            summary: { type: 'string' },
            target_hint: {
              type: 'string',
              enum: targetHintEnum
            }
          },
          required: ['title', 'summary', 'target_hint']
        }
      }
    },
    required: ['candidates']
  }
}

const createLuaQuestCandidatePrompt = (
  catalog: LuaQuestCatalog,
  entity?: GameEntity
): string => {
  const selectedNpcLine =
    entity?.kind === 'npc'
      ? `Selected NPC giver: ${entity.id} (${entity.name}). Use this id exactly as target_hint when relevant.`
      : undefined

  return [
    `You are writing quest candidate ideas for a Legend of Lua game.`,
    'Use only exact ids from the provided catalog.',
    'Do not invent new NPC ids, enemy ids, acquirable ids, or scene ids.',
    `Candidate count: ${LUA_QUEST_CANDIDATE_COUNT}.`,
    'Each candidate must be a short natural-language summary, not structured JSON.',
    'target_hint should be an NPC id when the idea clearly belongs to a specific giver; otherwise leave it blank.',
    ...(selectedNpcLine ? [selectedNpcLine] : []),
    buildLuaQuestCatalogText(catalog)
  ].join('\n')
}

export const generateLuaQuestCandidates = async ({
  apiKey,
  userPrompt,
  catalog,
  entity,
  gameContext,
  feedback
}: GenerateLuaQuestCandidatesInput): Promise<QuestCandidate[]> => {
  const targetLine = entity
    ? `\n\nSelected entity: ${entity.id} (${entity.name}) on map ${entity.mapId}.`
    : ''
  const contextLine = gameContext ? `\n\nGame context: ${gameContext}` : ''
  const feedbackLine = feedback ? buildFeedbackInstruction(feedback) : ''

  const generated = await generateJson<{ candidates: QuestCandidate[] }>({
    apiKey,
    instructions: createLuaQuestCandidatePrompt(catalog, entity),
    input: `${userPrompt}${targetLine}${contextLine}${buildGroundingContext(catalog)}${feedbackLine}`,
    schemaName: 'lua_quest_candidates',
    schema: createLuaQuestCandidatesSchema(catalog, entity)
  })

  return (generated.candidates ?? [])
    .map(normalizeCandidate)
    .filter((candidate) => candidate.title.length > 0 || candidate.summary.length > 0)
}
