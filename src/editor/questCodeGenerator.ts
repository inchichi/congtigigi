import type {
  QuestDefinition,
  QuestObjectiveDefinition,
  QuestItemReward
} from '../games/my-sample-rpg/questLog'
import type { GameStructureProfile } from './gameStructureProfile'
import { QUEST_GIVER_NPCS } from './questGenerationCatalog'
import type { GeneratedQuestJson } from './questJsonSchema'

const resolveGiverName = (
  giverNpcId: string,
  profile: GameStructureProfile
): string =>
  QUEST_GIVER_NPCS.find((giver) => giver.npcId === giverNpcId)?.label ??
  profile.npcs.find((npc) => npc.id === giverNpcId)?.name ??
  giverNpcId

const resolveItemLabel = (
  itemId: string,
  profile: GameStructureProfile
): string => profile.items.find((item) => item.id === itemId)?.name ?? itemId

export const convertGeneratedQuestToDefinition = (
  quest: GeneratedQuestJson,
  profile: GameStructureProfile
): QuestDefinition => {
  const giverName = resolveGiverName(quest.giver_npc_id, profile)

  const objectives: QuestObjectiveDefinition[] = quest.objectives.map(
    (objective, index) => {
      const target: QuestObjectiveDefinition['target'] = {}
      if (objective.target.sceneId) target.sceneId = objective.target.sceneId
      if (objective.target.appearanceType) {
        target.appearanceType = objective.target.appearanceType
      }
      if (objective.target.itemId) target.itemId = objective.target.itemId
      if (objective.target.shopId) target.shopId = objective.target.shopId
      if (objective.target.npcId) target.npcId = objective.target.npcId

      return {
        id: `${quest.quest_id}_objective_${index + 1}`,
        label: objective.label,
        required: objective.required,
        type: objective.type,
        target
      }
    }
  )

  const items: QuestItemReward[] = quest.rewards.items.map((item) => ({
    id: item.item_id,
    label: resolveItemLabel(item.item_id, profile),
    quantity: item.quantity
  }))

  return {
    id: quest.quest_id,
    regionName: quest.region,
    giverNpcId: quest.giver_npc_id,
    giverName,
    title: quest.title,
    trackerLabel: quest.title,
    turnInTrackerText: `${quest.title}: ${giverName}에게 돌아가기`,
    prerequisiteQuestIds: [],
    requestText: quest.request_text,
    guideText: quest.guide_text,
    startDialogueLines: quest.start_dialogue_lines,
    activeDialogueLines: quest.active_dialogue_lines,
    completionDialogueLines: quest.completion_dialogue_lines,
    objectives,
    rewards: {
      gold: quest.rewards.gold,
      experience: quest.rewards.experience,
      items
    }
  }
}
