// questLog 순수 로직을 Lua로 실행하는 래퍼.
// QUEST_DEFINITIONS는 TS 소유 — 초기화 시 quest_log_init()으로 Lua에 주입.
// init-with-data 패턴: runModule 후 callJson('quest_log_init', QUEST_DEFINITIONS) 호출.

import questLogLuaSource from '../assets/lua/quest-log.lua?raw'

import {
  QUEST_DEFINITIONS,
  FIRST_SLIME_HUNT_QUEST_ID,
  POTION_SURVIVAL_BASICS_QUEST_ID,
  PIG_TROUBLE_QUEST_ID,
  BLACKSMITH_PREPARATION_QUEST_ID,
  CAVE_ENTRANCE_INVESTIGATION_QUEST_ID,
  SLIME_BOSS_SHADOW_QUEST_ID,
  FINAL_SUPPLIES_QUEST_ID,
  PIG_BOSS_THREAT_QUEST_ID,
  FIRST_SLIME_HUNT_OBJECTIVE_ID,
  FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS,
  FIRST_SLIME_HUNT_REWARD_GOLD,
  FIRST_SLIME_HUNT_REWARD_EXPERIENCE,
  WIZARD_NPC_ID,
  POTION_MERCHANT_NPC_ID,
  BLACKSMITH_NPC_ID,
  QUEST_GIVER_NPC_IDS,
  type QuestLogState,
  type QuestProgress,
  type QuestDefinition,
  type QuestNpcBadgeKind,
  type NpcQuestInteraction,
  type CompleteQuestResult,
  type QuestTrackerItem,
  type QuestTextFormatContext
} from '../questLog'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type QuestLogLua = {
  createInitialQuestLog: () => QuestLogState
  getQuestProgress: (questLog: QuestLogState, questId: string) => QuestProgress
  getQuestDefinition: (questId: string) => QuestDefinition
  startQuest: (questLog: QuestLogState, questId: string) => QuestLogState
  recordQuestObjectiveProgress: (
    questLog: QuestLogState,
    questId: string,
    objectiveId: string,
    amount?: number
  ) => QuestLogState
  recordMonsterDefeatQuestProgress: (
    questLog: QuestLogState,
    target: { sceneId: string; appearanceType: string }
  ) => QuestLogState
  recordItemUseQuestProgress: (
    questLog: QuestLogState,
    itemId: string
  ) => QuestLogState
  recordShopOpenQuestProgress: (
    questLog: QuestLogState,
    shopId: string
  ) => QuestLogState
  recordSceneEnterQuestProgress: (
    questLog: QuestLogState,
    sceneId: string
  ) => QuestLogState
  recordTalkQuestProgress: (
    questLog: QuestLogState,
    npcId: string
  ) => QuestLogState
  setQuestTrackerVisible: (
    questLog: QuestLogState,
    questId: string,
    trackerVisible: boolean
  ) => QuestLogState
  hideVisibleQuestTrackers: (questLog: QuestLogState) => QuestLogState
  abandonQuest: (questLog: QuestLogState, questId: string) => QuestLogState
  completeQuest: (
    questLog: QuestLogState,
    questId: string
  ) => CompleteQuestResult
  getVisibleQuestTrackers: (questLog: QuestLogState) => QuestTrackerItem[]
  getQuestNpcBadgeKind: (
    questLog: QuestLogState,
    questId: string
  ) => QuestNpcBadgeKind | undefined
  getQuestNpcBadgeKindForNpc: (
    questLog: QuestLogState,
    npcId: string
  ) => QuestNpcBadgeKind | undefined
  getNextQuestInteractionForNpc: (
    questLog: QuestLogState,
    npcId: string
  ) => NpcQuestInteraction | undefined
  formatQuestText: (text: string, context: QuestTextFormatContext) => string
  formatQuestTextLines: (
    lines: string[],
    context: QuestTextFormatContext
  ) => string[]
  close: () => void
}

// Re-export constants so callers can get them from one place.
export {
  QUEST_DEFINITIONS,
  FIRST_SLIME_HUNT_QUEST_ID,
  POTION_SURVIVAL_BASICS_QUEST_ID,
  PIG_TROUBLE_QUEST_ID,
  BLACKSMITH_PREPARATION_QUEST_ID,
  CAVE_ENTRANCE_INVESTIGATION_QUEST_ID,
  SLIME_BOSS_SHADOW_QUEST_ID,
  FINAL_SUPPLIES_QUEST_ID,
  PIG_BOSS_THREAT_QUEST_ID,
  FIRST_SLIME_HUNT_OBJECTIVE_ID,
  FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS,
  FIRST_SLIME_HUNT_REWARD_GOLD,
  FIRST_SLIME_HUNT_REWARD_EXPERIENCE,
  WIZARD_NPC_ID,
  POTION_MERCHANT_NPC_ID,
  BLACKSMITH_NPC_ID,
  QUEST_GIVER_NPC_IDS
}

export const createQuestLogLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<QuestLogLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(questLogLuaSource, '@quest-log.lua')
  // init-with-data: QUEST_DEFINITIONS를 Lua에 주입
  host.callJson('quest_log_init', QUEST_DEFINITIONS)

  return {
    createInitialQuestLog: (): QuestLogState =>
      host.callJson<QuestLogState>('quest_log_create_initial'),

    getQuestProgress: (
      questLog: QuestLogState,
      questId: string
    ): QuestProgress =>
      host.callJson<QuestProgress>('quest_log_get_progress', questLog, questId),

    getQuestDefinition: (questId: string): QuestDefinition =>
      host.callJson<QuestDefinition>('quest_log_get_definition', questId),

    startQuest: (questLog: QuestLogState, questId: string): QuestLogState =>
      host.callJson<QuestLogState>('quest_log_start', questLog, questId),

    recordQuestObjectiveProgress: (
      questLog: QuestLogState,
      questId: string,
      objectiveId: string,
      amount = 1
    ): QuestLogState =>
      host.callJson<QuestLogState>(
        'quest_log_record_objective',
        questLog,
        questId,
        objectiveId,
        amount
      ),

    recordMonsterDefeatQuestProgress: (
      questLog: QuestLogState,
      target: { sceneId: string; appearanceType: string }
    ): QuestLogState =>
      host.callJson<QuestLogState>(
        'quest_log_record_monster_defeat',
        questLog,
        target.sceneId,
        target.appearanceType
      ),

    recordItemUseQuestProgress: (
      questLog: QuestLogState,
      itemId: string
    ): QuestLogState =>
      host.callJson<QuestLogState>(
        'quest_log_record_item_use',
        questLog,
        itemId
      ),

    recordShopOpenQuestProgress: (
      questLog: QuestLogState,
      shopId: string
    ): QuestLogState =>
      host.callJson<QuestLogState>(
        'quest_log_record_shop_open',
        questLog,
        shopId
      ),

    recordSceneEnterQuestProgress: (
      questLog: QuestLogState,
      sceneId: string
    ): QuestLogState =>
      host.callJson<QuestLogState>(
        'quest_log_record_scene_enter',
        questLog,
        sceneId
      ),

    recordTalkQuestProgress: (
      questLog: QuestLogState,
      npcId: string
    ): QuestLogState =>
      host.callJson<QuestLogState>('quest_log_record_talk', questLog, npcId),

    setQuestTrackerVisible: (
      questLog: QuestLogState,
      questId: string,
      trackerVisible: boolean
    ): QuestLogState =>
      host.callJson<QuestLogState>(
        'quest_log_set_tracker_visible',
        questLog,
        questId,
        trackerVisible
      ),

    hideVisibleQuestTrackers: (questLog: QuestLogState): QuestLogState =>
      host.callJson<QuestLogState>('quest_log_hide_trackers', questLog),

    abandonQuest: (questLog: QuestLogState, questId: string): QuestLogState =>
      host.callJson<QuestLogState>('quest_log_abandon', questLog, questId),

    completeQuest: (
      questLog: QuestLogState,
      questId: string
    ): CompleteQuestResult =>
      host.callJson<CompleteQuestResult>(
        'quest_log_complete',
        questLog,
        questId
      ),

    getVisibleQuestTrackers: (questLog: QuestLogState): QuestTrackerItem[] =>
      host.callJson<QuestTrackerItem[]>(
        'quest_log_get_visible_trackers',
        questLog
      ),

    getQuestNpcBadgeKind: (
      questLog: QuestLogState,
      questId: string
    ): QuestNpcBadgeKind | undefined => {
      const result = host.callJson<QuestNpcBadgeKind | null>(
        'quest_log_get_npc_badge',
        questLog,
        questId
      )

      return result === null ? undefined : result
    },

    getQuestNpcBadgeKindForNpc: (
      questLog: QuestLogState,
      npcId: string
    ): QuestNpcBadgeKind | undefined => {
      const result = host.callJson<QuestNpcBadgeKind | null>(
        'quest_log_get_npc_badge_for_npc',
        questLog,
        npcId
      )

      return result === null ? undefined : result
    },

    getNextQuestInteractionForNpc: (
      questLog: QuestLogState,
      npcId: string
    ): NpcQuestInteraction | undefined => {
      const result = host.callJson<NpcQuestInteraction | null>(
        'quest_log_get_next_interaction_for_npc',
        questLog,
        npcId
      )

      return result === null ? undefined : result
    },

    formatQuestText: (
      text: string,
      context: QuestTextFormatContext
    ): string =>
      host.callJson<string>(
        'quest_log_format_text',
        text,
        context.playerName
      ),

    formatQuestTextLines: (
      lines: string[],
      context: QuestTextFormatContext
    ): string[] =>
      host.callJson<string[]>(
        'quest_log_format_text_lines',
        lines,
        context.playerName
      ),

    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
