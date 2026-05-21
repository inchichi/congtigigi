import { describe, expect, it } from 'vitest'

import {
  FIRST_SLIME_HUNT_OBJECTIVE_ID,
  FIRST_SLIME_HUNT_QUEST_ID,
  abandonQuest,
  completeQuest,
  createInitialQuestLog,
  getQuestNpcBadgeKind,
  getQuestProgress,
  getVisibleQuestTrackers,
  hideVisibleQuestTrackers,
  recordQuestObjectiveProgress,
  setQuestTrackerVisible,
  startQuest
} from './questLog'

describe('questLog', () => {
  it('starts the first slime hunt from the wizard with tracker visible', () => {
    const questLog = startQuest(
      createInitialQuestLog(),
      FIRST_SLIME_HUNT_QUEST_ID
    )
    const quest = getQuestProgress(questLog, FIRST_SLIME_HUNT_QUEST_ID)

    expect(quest.status).toBe('active')
    expect(quest.trackerVisible).toBe(true)
    expect(getVisibleQuestTrackers(questLog)).toEqual([
      {
        questId: FIRST_SLIME_HUNT_QUEST_ID,
        text: '첫 사냥: 말캉이 처치 0/3'
      }
    ])
  })

  it('tracks slime defeats and becomes ready to turn in at three defeats', () => {
    let questLog = startQuest(
      createInitialQuestLog(),
      FIRST_SLIME_HUNT_QUEST_ID
    )

    questLog = recordQuestObjectiveProgress(
      questLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID
    )
    questLog = recordQuestObjectiveProgress(
      questLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID
    )
    questLog = recordQuestObjectiveProgress(
      questLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID
    )
    questLog = recordQuestObjectiveProgress(
      questLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID
    )

    const quest = getQuestProgress(questLog, FIRST_SLIME_HUNT_QUEST_ID)

    expect(quest.status).toBe('ready-to-turn-in')
    expect(quest.objectives[FIRST_SLIME_HUNT_OBJECTIVE_ID]).toBe(3)
    expect(getVisibleQuestTrackers(questLog)).toEqual([
      {
        questId: FIRST_SLIME_HUNT_QUEST_ID,
        text: '첫 사냥: 마법사에게 돌아가기'
      }
    ])
  })

  it('toggles tracker visibility without changing quest progress', () => {
    const activeQuestLog = startQuest(
      createInitialQuestLog(),
      FIRST_SLIME_HUNT_QUEST_ID
    )
    const hiddenQuestLog = setQuestTrackerVisible(
      activeQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      false
    )
    const visibleQuestLog = setQuestTrackerVisible(
      hiddenQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      true
    )

    expect(getVisibleQuestTrackers(hiddenQuestLog)).toEqual([])
    expect(
      getQuestProgress(hiddenQuestLog, FIRST_SLIME_HUNT_QUEST_ID).status
    ).toBe('active')
    expect(getVisibleQuestTrackers(visibleQuestLog)).toEqual([
      {
        questId: FIRST_SLIME_HUNT_QUEST_ID,
        text: '첫 사냥: 말캉이 처치 0/3'
      }
    ])
  })

  it('hides visible trackers from the tracker close button without changing quest progress', () => {
    const activeQuestLog = startQuest(
      createInitialQuestLog(),
      FIRST_SLIME_HUNT_QUEST_ID
    )
    const hiddenQuestLog = hideVisibleQuestTrackers(activeQuestLog)

    expect(getVisibleQuestTrackers(hiddenQuestLog)).toEqual([])
    expect(
      getQuestProgress(hiddenQuestLog, FIRST_SLIME_HUNT_QUEST_ID).status
    ).toBe('active')
  })

  it('abandons only the selected quest and removes it from the tracker', () => {
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

  it('returns npc badge kinds for quest state', () => {
    const notStartedQuestLog = createInitialQuestLog()
    const activeQuestLog = startQuest(
      notStartedQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID
    )
    let readyQuestLog = activeQuestLog

    readyQuestLog = recordQuestObjectiveProgress(
      readyQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID,
      3
    )

    expect(
      getQuestNpcBadgeKind(notStartedQuestLog, FIRST_SLIME_HUNT_QUEST_ID)
    ).toBe('new')
    expect(
      getQuestNpcBadgeKind(activeQuestLog, FIRST_SLIME_HUNT_QUEST_ID)
    ).toBeUndefined()
    expect(
      getQuestNpcBadgeKind(readyQuestLog, FIRST_SLIME_HUNT_QUEST_ID)
    ).toBe('finish')
    expect(
      getQuestNpcBadgeKind(
        completeQuest(readyQuestLog, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog,
        FIRST_SLIME_HUNT_QUEST_ID
      )
    ).toBeUndefined()
  })

  it('completes once and returns the reward once', () => {
    const activeQuestLog = startQuest(
      createInitialQuestLog(),
      FIRST_SLIME_HUNT_QUEST_ID
    )
    const readyQuestLog = recordQuestObjectiveProgress(
      activeQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID,
      FIRST_SLIME_HUNT_OBJECTIVE_ID,
      3
    )
    const completedResult = completeQuest(
      readyQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID
    )
    const repeatedResult = completeQuest(
      completedResult.nextQuestLog,
      FIRST_SLIME_HUNT_QUEST_ID
    )

    expect(completedResult).toMatchObject({
      didComplete: true,
      goldReward: 100,
      experienceReward: 60
    })
    expect(
      getQuestProgress(completedResult.nextQuestLog, FIRST_SLIME_HUNT_QUEST_ID)
    ).toMatchObject({
      status: 'completed',
      trackerVisible: false
    })
    expect(repeatedResult).toMatchObject({
      didComplete: false,
      goldReward: 0,
      experienceReward: 0
    })
  })
})
