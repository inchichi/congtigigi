export const FIRST_SLIME_HUNT_QUEST_ID = 'first-slime-hunt'
export const FIRST_SLIME_HUNT_OBJECTIVE_ID = 'slime-defeats'
export const FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS = 3
export const FIRST_SLIME_HUNT_REWARD_GOLD = 100
export const FIRST_SLIME_HUNT_REWARD_EXPERIENCE = 60

export type QuestStatus =
  | 'not-started'
  | 'active'
  | 'ready-to-turn-in'
  | 'completed'

export type QuestObjectiveDefinition = {
  id: string
  label: string
  required: number
}

export type QuestDefinition = {
  id: string
  regionName: string
  giverName: string
  title: string
  requestText: string
  guideText: string
  objectives: QuestObjectiveDefinition[]
  rewardGold: number
  rewardExperience: number
}

export type QuestProgress = {
  id: string
  status: QuestStatus
  objectives: Record<string, number>
  trackerVisible: boolean
}

export type QuestLogState = {
  progressByQuestId: Record<string, QuestProgress>
}

export type QuestNpcBadgeKind = 'new' | 'finish'

export type QuestTrackerItem = {
  questId: string
  text: string
}

export type CompleteQuestResult = {
  nextQuestLog: QuestLogState
  didComplete: boolean
  goldReward: number
  experienceReward: number
}

export const QUEST_DEFINITIONS: QuestDefinition[] = [
  {
    id: FIRST_SLIME_HUNT_QUEST_ID,
    regionName: '티르코네일 마을',
    giverName: '마법사',
    title: '첫 사냥: 말캉이 처치',
    requestText: '마을의 마법사가 퀘스트를 의뢰했다.',
    guideText:
      '마을 오른쪽 "사냥터로 가는 길"로 포탈을 타 말캉이 3마리를 잡고 오자.',
    objectives: [
      {
        id: FIRST_SLIME_HUNT_OBJECTIVE_ID,
        label: '말캉이',
        required: FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS
      }
    ],
    rewardGold: FIRST_SLIME_HUNT_REWARD_GOLD,
    rewardExperience: FIRST_SLIME_HUNT_REWARD_EXPERIENCE
  }
]

const QUEST_DEFINITION_BY_ID = Object.fromEntries(
  QUEST_DEFINITIONS.map((definition) => [definition.id, definition])
) as Record<string, QuestDefinition>

export const createInitialQuestLog = (): QuestLogState => ({
  progressByQuestId: Object.fromEntries(
    QUEST_DEFINITIONS.map((definition) => [
      definition.id,
      createInitialQuestProgress(definition)
    ])
  ) as Record<string, QuestProgress>
})

export const getQuestDefinition = (questId: string): QuestDefinition =>
  QUEST_DEFINITION_BY_ID[questId]

export const getQuestProgress = (
  questLog: QuestLogState,
  questId: string
): QuestProgress => questLog.progressByQuestId[questId]

export const startQuest = (
  questLog: QuestLogState,
  questId: string
): QuestLogState => {
  const quest = getQuestProgress(questLog, questId)

  if (quest.status !== 'not-started') {
    return questLog
  }

  return updateQuestProgress(questLog, {
    ...quest,
    status: 'active',
    trackerVisible: true
  })
}

export const recordQuestObjectiveProgress = (
  questLog: QuestLogState,
  questId: string,
  objectiveId: string,
  amount = 1
): QuestLogState => {
  const quest = getQuestProgress(questLog, questId)

  if (quest.status !== 'active') {
    return questLog
  }

  const definition = getQuestDefinition(questId)
  const objective = definition.objectives.find(
    (candidate) => candidate.id === objectiveId
  )

  if (!objective) {
    return questLog
  }

  const current = quest.objectives[objectiveId] ?? 0
  const nextCurrent = Math.min(objective.required, current + amount)
  const nextQuest = {
    ...quest,
    objectives: {
      ...quest.objectives,
      [objectiveId]: nextCurrent
    }
  }

  return updateQuestProgress(questLog, {
    ...nextQuest,
    status: areQuestObjectivesComplete(definition, nextQuest)
      ? 'ready-to-turn-in'
      : 'active'
  })
}

export const setQuestTrackerVisible = (
  questLog: QuestLogState,
  questId: string,
  trackerVisible: boolean
): QuestLogState => {
  const quest = getQuestProgress(questLog, questId)

  if (
    trackerVisible &&
    quest.status !== 'active' &&
    quest.status !== 'ready-to-turn-in'
  ) {
    return questLog
  }

  if (quest.trackerVisible === trackerVisible) {
    return questLog
  }

  return updateQuestProgress(questLog, {
    ...quest,
    trackerVisible
  })
}

export const hideVisibleQuestTrackers = (
  questLog: QuestLogState
): QuestLogState => ({
  progressByQuestId: Object.fromEntries(
    Object.entries(questLog.progressByQuestId).map(([questId, quest]) => [
      questId,
      {
        ...quest,
        trackerVisible: false
      }
    ])
  ) as Record<string, QuestProgress>
})

export const abandonQuest = (
  questLog: QuestLogState,
  questId: string
): QuestLogState =>
  updateQuestProgress(
    questLog,
    createInitialQuestProgress(getQuestDefinition(questId))
  )

export const completeQuest = (
  questLog: QuestLogState,
  questId: string
): CompleteQuestResult => {
  const quest = getQuestProgress(questLog, questId)
  const definition = getQuestDefinition(questId)

  if (quest.status !== 'ready-to-turn-in') {
    return {
      nextQuestLog: questLog,
      didComplete: false,
      goldReward: 0,
      experienceReward: 0
    }
  }

  return {
    nextQuestLog: updateQuestProgress(questLog, {
      ...quest,
      status: 'completed',
      trackerVisible: false
    }),
    didComplete: true,
    goldReward: definition.rewardGold,
    experienceReward: definition.rewardExperience
  }
}

export const getVisibleQuestTrackers = (
  questLog: QuestLogState
): QuestTrackerItem[] =>
  QUEST_DEFINITIONS.flatMap((definition) => {
    const quest = getQuestProgress(questLog, definition.id)

    if (!quest.trackerVisible) {
      return []
    }

    switch (quest.status) {
      case 'active':
        const objective = definition.objectives[0]

        return [
          {
            questId: definition.id,
            text: `${definition.title} ${quest.objectives[objective.id]}/${objective.required}`
          }
        ]
      case 'ready-to-turn-in':
        return [
          {
            questId: definition.id,
            text: '첫 사냥: 마법사에게 돌아가기'
          }
        ]
      case 'not-started':
      case 'completed':
        return []
    }
  })

export const getQuestNpcBadgeKind = (
  questLog: QuestLogState,
  questId: string
): QuestNpcBadgeKind | undefined => {
  switch (getQuestProgress(questLog, questId).status) {
    case 'not-started':
      return 'new'
    case 'ready-to-turn-in':
      return 'finish'
    case 'active':
    case 'completed':
      return undefined
  }
}

const createInitialQuestProgress = (
  definition: QuestDefinition
): QuestProgress => ({
  id: definition.id,
  status: 'not-started',
  objectives: Object.fromEntries(
    definition.objectives.map((objective) => [objective.id, 0])
  ) as Record<string, number>,
  trackerVisible: false
})

const updateQuestProgress = (
  questLog: QuestLogState,
  quest: QuestProgress
): QuestLogState => ({
  progressByQuestId: {
    ...questLog.progressByQuestId,
    [quest.id]: quest
  }
})

const areQuestObjectivesComplete = (
  definition: QuestDefinition,
  quest: QuestProgress
): boolean =>
  definition.objectives.every(
    (objective) => (quest.objectives[objective.id] ?? 0) >= objective.required
  )
