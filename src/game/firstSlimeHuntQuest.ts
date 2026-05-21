export const FIRST_SLIME_HUNT_QUEST_ID = 'first-slime-hunt'
export const FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS = 3
export const FIRST_SLIME_HUNT_REWARD_GOLD = 100
export const FIRST_SLIME_HUNT_REWARD_EXPERIENCE = 60

export type FirstSlimeHuntQuestStatus =
  | 'not-started'
  | 'active'
  | 'ready-to-turn-in'
  | 'completed'

export type FirstSlimeHuntQuestState = {
  id: typeof FIRST_SLIME_HUNT_QUEST_ID
  status: FirstSlimeHuntQuestStatus
  slimeDefeats: number
}

export type CompleteFirstSlimeHuntQuestResult = {
  nextQuest: FirstSlimeHuntQuestState
  didComplete: boolean
  goldReward: number
  experienceReward: number
}

export const createInitialFirstSlimeHuntQuest = (): FirstSlimeHuntQuestState => ({
  id: FIRST_SLIME_HUNT_QUEST_ID,
  status: 'not-started',
  slimeDefeats: 0
})

export const startFirstSlimeHuntQuest = (
  quest: FirstSlimeHuntQuestState
): FirstSlimeHuntQuestState =>
  quest.status === 'not-started'
    ? {
        ...quest,
        status: 'active'
      }
    : quest

export const recordFirstSlimeHuntSlimeDefeat = (
  quest: FirstSlimeHuntQuestState
): FirstSlimeHuntQuestState => {
  if (quest.status !== 'active') {
    return quest
  }

  const slimeDefeats = Math.min(
    FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS,
    quest.slimeDefeats + 1
  )

  return {
    ...quest,
    slimeDefeats,
    status:
      slimeDefeats >= FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS
        ? 'ready-to-turn-in'
        : 'active'
  }
}

export const completeFirstSlimeHuntQuest = (
  quest: FirstSlimeHuntQuestState
): CompleteFirstSlimeHuntQuestResult => {
  if (quest.status !== 'ready-to-turn-in') {
    return {
      nextQuest: quest,
      didComplete: false,
      goldReward: 0,
      experienceReward: 0
    }
  }

  return {
    nextQuest: {
      ...quest,
      status: 'completed'
    },
    didComplete: true,
    goldReward: FIRST_SLIME_HUNT_REWARD_GOLD,
    experienceReward: FIRST_SLIME_HUNT_REWARD_EXPERIENCE
  }
}

export const getFirstSlimeHuntTrackerText = (
  quest: FirstSlimeHuntQuestState
): string | undefined => {
  switch (quest.status) {
    case 'active':
      return `첫 사냥: 말캉이 처치 ${quest.slimeDefeats} / ${FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS}`
    case 'ready-to-turn-in':
      return '첫 사냥: 마법사에게 돌아가기'
    case 'not-started':
    case 'completed':
      return undefined
  }
}
