import { describe, expect, it } from 'vitest'

import {
  BLACKSMITH_PREPARATION_QUEST_ID,
  FIRST_SLIME_HUNT_OBJECTIVE_ID,
  FIRST_SLIME_HUNT_QUEST_ID,
  FINAL_SUPPLIES_QUEST_ID,
  PIG_BOSS_THREAT_QUEST_ID,
  PIG_TROUBLE_QUEST_ID,
  POTION_SURVIVAL_BASICS_QUEST_ID,
  QUEST_DEFINITIONS,
  SLIME_BOSS_SHADOW_QUEST_ID,
  abandonQuest,
  completeQuest,
  createInitialQuestLog,
  formatQuestText,
  getNextQuestInteractionForNpc,
  getQuestNpcBadgeKindForNpc,
  getQuestProgress,
  getVisibleQuestTrackers,
  hideVisibleQuestTrackers,
  recordItemUseQuestProgress,
  recordMonsterDefeatQuestProgress,
  recordQuestObjectiveProgress,
  recordSceneEnterQuestProgress,
  recordShopOpenQuestProgress,
  recordTalkQuestProgress,
  setQuestTrackerVisible,
  startQuest
} from './questLog'
import { grantPlayerExperience } from './playerExperience'
import {
  PLAYER_JOB_PROMOTION_LEVEL,
  createInitialPlayerProfile
} from './playerProfile'

const WIZARD_NPC_ID = 'wizard'
const POTION_MERCHANT_NPC_ID = 'potion_merchant'

const completeQuestById = (questLog = createInitialQuestLog(), questId: string) =>
  completeQuest(
    recordQuestObjectiveProgress(
      startQuest(questLog, questId),
      questId,
      FIRST_SLIME_HUNT_OBJECTIVE_ID,
      99
    ),
    questId
  ).nextQuestLog

describe('questLog', () => {
  it('registers the full beginner quest arc with stable q001-q008 ids', () => {
    expect(QUEST_DEFINITIONS.map((definition) => definition.id)).toEqual([
      FIRST_SLIME_HUNT_QUEST_ID,
      POTION_SURVIVAL_BASICS_QUEST_ID,
      PIG_TROUBLE_QUEST_ID,
      BLACKSMITH_PREPARATION_QUEST_ID,
      'q005-investigate-cave-entrance',
      SLIME_BOSS_SHADOW_QUEST_ID,
      FINAL_SUPPLIES_QUEST_ID,
      PIG_BOSS_THREAT_QUEST_ID
    ])
    expect(QUEST_DEFINITIONS.every((definition) => definition.regionName === '티르코네일 마을')).toBe(true)
    expect(JSON.stringify(QUEST_DEFINITIONS)).not.toContain('준수')
  })

  it('grants enough quest experience to reach level 10 after the arc', () => {
    const questExperience = QUEST_DEFINITIONS.reduce(
      (totalExperience, definition) => totalExperience + definition.rewards.experience,
      0
    )

    const result = grantPlayerExperience(
      createInitialPlayerProfile(),
      questExperience
    )

    expect(result.nextProfile.level).toBe(PLAYER_JOB_PROMOTION_LEVEL)
    expect(result.nextProfile.experience.current).toBe(0)
  })

  it('starts only unlocked quests and exposes the next NPC quest interaction', () => {
    let questLog = createInitialQuestLog()

    expect(
      getNextQuestInteractionForNpc(questLog, WIZARD_NPC_ID)
    ).toMatchObject({
      action: 'start',
      questId: FIRST_SLIME_HUNT_QUEST_ID
    })
    expect(startQuest(questLog, POTION_SURVIVAL_BASICS_QUEST_ID)).toBe(questLog)

    questLog = startQuest(questLog, FIRST_SLIME_HUNT_QUEST_ID)
    questLog = recordQuestObjectiveProgress(
      questLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID,
      3
    )
    questLog = completeQuest(questLog, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog

    expect(
      getNextQuestInteractionForNpc(questLog, POTION_MERCHANT_NPC_ID)
    ).toMatchObject({
      action: 'start',
      questId: POTION_SURVIVAL_BASICS_QUEST_ID
    })
  })

  it('tracks the first slime hunt and toggles the tracker independently', () => {
    let questLog = startQuest(createInitialQuestLog(), FIRST_SLIME_HUNT_QUEST_ID)

    // 트래커 아이템은 활성 목표(objective)도 함께 담는다 — questId/text만 확인한다.
    expect(getVisibleQuestTrackers(questLog)).toMatchObject([
      {
        questId: FIRST_SLIME_HUNT_QUEST_ID,
        text: '첫 사냥: 말캉이 처치 0/3'
      }
    ])

    questLog = recordMonsterDefeatQuestProgress(questLog, {
      sceneId: 'hunting-ground',
      appearanceType: 'monster_slime'
    })
    questLog = setQuestTrackerVisible(questLog, FIRST_SLIME_HUNT_QUEST_ID, false)

    expect(getVisibleQuestTrackers(questLog)).toEqual([])
    expect(
      getQuestProgress(questLog, FIRST_SLIME_HUNT_QUEST_ID).objectives[
        FIRST_SLIME_HUNT_OBJECTIVE_ID
      ]
    ).toBe(1)

    questLog = hideVisibleQuestTrackers(
      setQuestTrackerVisible(questLog, FIRST_SLIME_HUNT_QUEST_ID, true)
    )

    expect(getQuestProgress(questLog, FIRST_SLIME_HUNT_QUEST_ID).status).toBe(
      'active'
    )
  })

  it('records item-use, shop-open, scene-enter, talk, and boss objectives', () => {
    let questLog = completeQuestById(undefined, FIRST_SLIME_HUNT_QUEST_ID)

    questLog = startQuest(questLog, POTION_SURVIVAL_BASICS_QUEST_ID)
    questLog = recordItemUseQuestProgress(questLog, 'health-potion')
    expect(
      getQuestProgress(questLog, POTION_SURVIVAL_BASICS_QUEST_ID).status
    ).toBe('ready-to-turn-in')
    questLog = completeQuest(questLog, POTION_SURVIVAL_BASICS_QUEST_ID).nextQuestLog

    questLog = startQuest(questLog, PIG_TROUBLE_QUEST_ID)
    questLog = recordMonsterDefeatQuestProgress(questLog, {
      sceneId: 'hunting-ground',
      appearanceType: 'monster_pig'
    })
    questLog = recordMonsterDefeatQuestProgress(questLog, {
      sceneId: 'hunting-ground',
      appearanceType: 'monster_pig'
    })
    expect(getQuestProgress(questLog, PIG_TROUBLE_QUEST_ID).status).toBe(
      'ready-to-turn-in'
    )
    questLog = completeQuest(questLog, PIG_TROUBLE_QUEST_ID).nextQuestLog

    questLog = startQuest(questLog, BLACKSMITH_PREPARATION_QUEST_ID)
    questLog = recordShopOpenQuestProgress(questLog, 'blacksmith')
    expect(
      getQuestProgress(questLog, BLACKSMITH_PREPARATION_QUEST_ID).status
    ).toBe('ready-to-turn-in')
    questLog = completeQuest(questLog, BLACKSMITH_PREPARATION_QUEST_ID)
      .nextQuestLog

    questLog = startQuest(questLog, 'q005-investigate-cave-entrance')
    questLog = recordSceneEnterQuestProgress(questLog, 'cave')
    expect(
      getQuestProgress(questLog, 'q005-investigate-cave-entrance').status
    ).toBe('ready-to-turn-in')
    questLog = completeQuest(questLog, 'q005-investigate-cave-entrance')
      .nextQuestLog

    questLog = startQuest(questLog, SLIME_BOSS_SHADOW_QUEST_ID)
    questLog = recordMonsterDefeatQuestProgress(questLog, {
      sceneId: 'cave',
      appearanceType: 'monster_slime'
    })
    expect(getQuestProgress(questLog, SLIME_BOSS_SHADOW_QUEST_ID).status).toBe(
      'ready-to-turn-in'
    )
    questLog = completeQuest(questLog, SLIME_BOSS_SHADOW_QUEST_ID).nextQuestLog

    questLog = startQuest(questLog, FINAL_SUPPLIES_QUEST_ID)
    questLog = recordTalkQuestProgress(questLog, POTION_MERCHANT_NPC_ID)
    expect(getQuestProgress(questLog, FINAL_SUPPLIES_QUEST_ID).status).toBe(
      'ready-to-turn-in'
    )
    questLog = completeQuest(questLog, FINAL_SUPPLIES_QUEST_ID).nextQuestLog

    questLog = startQuest(questLog, PIG_BOSS_THREAT_QUEST_ID)
    questLog = recordMonsterDefeatQuestProgress(questLog, {
      sceneId: 'cave',
      appearanceType: 'monster_pig'
    })
    expect(getQuestProgress(questLog, PIG_BOSS_THREAT_QUEST_ID).status).toBe(
      'ready-to-turn-in'
    )
  })

  it('returns item rewards and grants all rewards once', () => {
    let questLog = completeQuestById(undefined, FIRST_SLIME_HUNT_QUEST_ID)

    questLog = startQuest(questLog, POTION_SURVIVAL_BASICS_QUEST_ID)
    questLog = recordItemUseQuestProgress(questLog, 'health-potion')

    const completedResult = completeQuest(
      questLog,
      POTION_SURVIVAL_BASICS_QUEST_ID
    )
    const repeatedResult = completeQuest(
      completedResult.nextQuestLog,
      POTION_SURVIVAL_BASICS_QUEST_ID
    )

    expect(completedResult).toMatchObject({
      didComplete: true,
      goldReward: 0,
      experienceReward: 108,
      itemRewards: [
        {
          id: 'health-potion',
          label: '체력 회복 포션',
          quantity: 5
        },
        {
          id: 'mana-potion',
          label: '마나 회복 포션',
          quantity: 3
        }
      ]
    })
    expect(repeatedResult).toMatchObject({
      didComplete: false,
      goldReward: 0,
      experienceReward: 0,
      itemRewards: []
    })
  })

  it('returns npc badge state for unlocked and ready quests', () => {
    let questLog = createInitialQuestLog()

    expect(getQuestNpcBadgeKindForNpc(questLog, WIZARD_NPC_ID)).toBe('new')
    expect(
      getQuestNpcBadgeKindForNpc(questLog, POTION_MERCHANT_NPC_ID)
    ).toBeUndefined()

    questLog = startQuest(questLog, FIRST_SLIME_HUNT_QUEST_ID)
    questLog = recordQuestObjectiveProgress(
      questLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID,
      3
    )

    expect(getQuestNpcBadgeKindForNpc(questLog, WIZARD_NPC_ID)).toBe('finish')

    questLog = completeQuest(questLog, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog

    expect(getQuestNpcBadgeKindForNpc(questLog, POTION_MERCHANT_NPC_ID)).toBe(
      'new'
    )
  })

  it('abandons only the selected quest and resets its objectives', () => {
    const activeQuestLog = recordQuestObjectiveProgress(
      startQuest(createInitialQuestLog(), FIRST_SLIME_HUNT_QUEST_ID),
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID
    )
    const abandonedQuestLog = abandonQuest(
      activeQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID
    )
    const quest = getQuestProgress(
      abandonedQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID
    )

    expect(quest).toMatchObject({
      status: 'not-started',
      trackerVisible: false
    })
    expect(quest.objectives[FIRST_SLIME_HUNT_OBJECTIVE_ID]).toBe(0)
    expect(getVisibleQuestTrackers(abandonedQuestLog)).toEqual([])
  })

  it('formats quest dialogue with the current player name placeholder', () => {
    expect(formatQuestText('잘했다, {playerName}.', { playerName: '루아' })).toBe(
      '잘했다, 루아.'
    )
  })
})
