import { describe, expect, it } from 'vitest'

import {
  FIRST_SLIME_HUNT_REWARD_EXPERIENCE,
  FIRST_SLIME_HUNT_REWARD_GOLD,
  completeFirstSlimeHuntQuest,
  createInitialFirstSlimeHuntQuest,
  getFirstSlimeHuntQuestNpcBadgeKind,
  getFirstSlimeHuntTrackerText,
  recordFirstSlimeHuntSlimeDefeat,
  startFirstSlimeHuntQuest
} from './firstSlimeHuntQuest'

describe('firstSlimeHuntQuest', () => {
  it('starts as not started', () => {
    expect(createInitialFirstSlimeHuntQuest()).toMatchObject({
      status: 'not-started',
      slimeDefeats: 0
    })
  })

  it('starts the quest when the wizard gives it', () => {
    const quest = startFirstSlimeHuntQuest(createInitialFirstSlimeHuntQuest())

    expect(quest.status).toBe('active')
  })

  it('returns npc badge kinds for quest state', () => {
    expect(
      getFirstSlimeHuntQuestNpcBadgeKind(createInitialFirstSlimeHuntQuest())
    ).toBe('new')
    expect(
      getFirstSlimeHuntQuestNpcBadgeKind({
        ...createInitialFirstSlimeHuntQuest(),
        status: 'active'
      })
    ).toBeUndefined()
    expect(
      getFirstSlimeHuntQuestNpcBadgeKind({
        ...createInitialFirstSlimeHuntQuest(),
        status: 'ready-to-turn-in'
      })
    ).toBe('finish')
    expect(
      getFirstSlimeHuntQuestNpcBadgeKind({
        ...createInitialFirstSlimeHuntQuest(),
        status: 'completed'
      })
    ).toBeUndefined()
  })

  it('tracks slime defeats while active', () => {
    const quest = recordFirstSlimeHuntSlimeDefeat(
      startFirstSlimeHuntQuest(createInitialFirstSlimeHuntQuest())
    )

    expect(quest).toMatchObject({
      status: 'active',
      slimeDefeats: 1
    })
    expect(getFirstSlimeHuntTrackerText(quest)).toBe(
      '첫 사냥: 말캉이 처치 1 / 3'
    )
  })

  it('becomes ready to turn in after three slime defeats', () => {
    let quest = startFirstSlimeHuntQuest(createInitialFirstSlimeHuntQuest())

    quest = recordFirstSlimeHuntSlimeDefeat(quest)
    quest = recordFirstSlimeHuntSlimeDefeat(quest)
    quest = recordFirstSlimeHuntSlimeDefeat(quest)
    quest = recordFirstSlimeHuntSlimeDefeat(quest)

    expect(quest).toMatchObject({
      status: 'ready-to-turn-in',
      slimeDefeats: 3
    })
    expect(getFirstSlimeHuntTrackerText(quest)).toBe(
      '첫 사냥: 마법사에게 돌아가기'
    )
  })

  it('completes once and returns the reward', () => {
    const readyQuest = {
      ...createInitialFirstSlimeHuntQuest(),
      status: 'ready-to-turn-in' as const,
      slimeDefeats: 3
    }
    const completedResult = completeFirstSlimeHuntQuest(readyQuest)
    const repeatedResult = completeFirstSlimeHuntQuest(
      completedResult.nextQuest
    )

    expect(completedResult).toMatchObject({
      didComplete: true,
      goldReward: FIRST_SLIME_HUNT_REWARD_GOLD,
      experienceReward: FIRST_SLIME_HUNT_REWARD_EXPERIENCE,
      nextQuest: {
        status: 'completed'
      }
    })
    expect(repeatedResult).toMatchObject({
      didComplete: false,
      goldReward: 0,
      experienceReward: 0
    })
  })
})
