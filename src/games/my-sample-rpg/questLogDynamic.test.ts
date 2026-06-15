import { afterEach, describe, expect, it } from 'vitest'

import {
  clearDynamicQuestDefinitions,
  completeQuest,
  createInitialQuestLog,
  ensureQuestProgressEntries,
  getAllQuestDefinitions,
  getQuestProgress,
  recordMonsterDefeatQuestProgress,
  registerDynamicQuestDefinitions,
  startQuest,
  type QuestDefinition
} from './questLog'

const DYNAMIC_QUEST_ID = 'dynamic_test_hunt'

const dynamicQuest = (): QuestDefinition => ({
  id: DYNAMIC_QUEST_ID,
  regionName: '테스트 지역',
  giverNpcId: 'wizard',
  giverName: '마법사',
  title: '동적 사냥 퀘스트',
  trackerLabel: '동적 사냥 퀘스트',
  prerequisiteQuestIds: [],
  requestText: '테스트 의뢰',
  guideText: '사냥터에서 슬라임을 잡아라',
  startDialogueLines: ['시작 대사'],
  activeDialogueLines: ['진행 대사'],
  completionDialogueLines: ['완료 대사'],
  objectives: [
    {
      id: 'obj_1',
      label: '슬라임 1마리 처치',
      required: 1,
      type: 'monster-defeat',
      target: { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
    }
  ],
  rewards: { gold: 50, experience: 30, items: [] }
})

afterEach(() => {
  clearDynamicQuestDefinitions()
})

describe('dynamic quest registry', () => {
  it('registers a dynamic quest and runs it through the same engine', () => {
    registerDynamicQuestDefinitions([dynamicQuest()])
    let log = ensureQuestProgressEntries(createInitialQuestLog())

    expect(getQuestProgress(log, DYNAMIC_QUEST_ID).status).toBe('not-started')

    log = startQuest(log, DYNAMIC_QUEST_ID)
    expect(getQuestProgress(log, DYNAMIC_QUEST_ID).status).toBe('active')

    log = recordMonsterDefeatQuestProgress(log, {
      sceneId: 'hunting-ground',
      appearanceType: 'monster_slime'
    })
    expect(getQuestProgress(log, DYNAMIC_QUEST_ID).status).toBe('ready-to-turn-in')

    const result = completeQuest(log, DYNAMIC_QUEST_ID)
    expect(result.didComplete).toBe(true)
    expect(result.goldReward).toBe(50)
    expect(getQuestProgress(result.nextQuestLog, DYNAMIC_QUEST_ID).status).toBe('completed')
  })

  it('appears in getAllQuestDefinitions after registration only', () => {
    const before = getAllQuestDefinitions().length
    registerDynamicQuestDefinitions([dynamicQuest()])
    expect(getAllQuestDefinitions().length).toBe(before + 1)
  })

  it('ignores a dynamic id that collides with a static quest id', () => {
    const staticId = getAllQuestDefinitions()[0].id
    const before = getAllQuestDefinitions().length
    registerDynamicQuestDefinitions([{ ...dynamicQuest(), id: staticId }])
    expect(getAllQuestDefinitions().length).toBe(before)
  })

  it('ensureQuestProgressEntries does not overwrite existing progress', () => {
    registerDynamicQuestDefinitions([dynamicQuest()])
    let log = ensureQuestProgressEntries(createInitialQuestLog())
    log = startQuest(log, DYNAMIC_QUEST_ID)
    // 다시 호출해도 active 진행 상태를 not-started로 되돌리지 않는다.
    const after = ensureQuestProgressEntries(log)
    expect(getQuestProgress(after, DYNAMIC_QUEST_ID).status).toBe('active')
  })
})
