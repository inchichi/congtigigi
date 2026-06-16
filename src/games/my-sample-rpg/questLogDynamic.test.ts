import { afterEach, describe, expect, it } from 'vitest'

import {
  clearDynamicQuestDefinitions,
  completeQuest,
  createInitialQuestLog,
  ensureQuestProgressEntries,
  getAllQuestDefinitions,
  getNextQuestInteractionForNpc,
  getQuestNpcBadgeKindForNpc,
  getQuestProgress,
  recordItemAcquireQuestProgress,
  recordMonsterDefeatQuestProgress,
  recordTalkQuestProgress,
  registerDynamicQuestDefinitions,
  startQuest,
  setQuestDefinitionVisibilityFilter,
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

  it('prioritizes a newly registered dynamic quest over the static wizard quest', () => {
    registerDynamicQuestDefinitions([dynamicQuest()])
    const log = ensureQuestProgressEntries(createInitialQuestLog())

    expect(getNextQuestInteractionForNpc(log, 'wizard')).toMatchObject({
      questId: DYNAMIC_QUEST_ID,
      action: 'start'
    })
  })

  it('appears in getAllQuestDefinitions after registration only', () => {
    const before = getAllQuestDefinitions().length
    registerDynamicQuestDefinitions([dynamicQuest()])
    expect(getAllQuestDefinitions().length).toBe(before + 1)
  })

  it('limits preview visibility to the latest pending quest snapshot', () => {
    const potionQuest: QuestDefinition = {
      ...dynamicQuest(),
      id: 'dynamic_potion_merchant',
      giverNpcId: 'potion_merchant',
      giverName: '물약상인',
      title: '물약상인 돕기',
      trackerLabel: '물약상인 돕기'
    }

    registerDynamicQuestDefinitions([potionQuest])
    setQuestDefinitionVisibilityFilter([potionQuest.id])

    const log = ensureQuestProgressEntries(createInitialQuestLog())

    expect(getQuestNpcBadgeKindForNpc(log, 'wizard')).toBeUndefined()
    expect(getNextQuestInteractionForNpc(log, 'wizard')).toBeUndefined()
    expect(getQuestNpcBadgeKindForNpc(log, 'potion_merchant')).toBe('new')
    expect(getNextQuestInteractionForNpc(log, 'potion_merchant')).toMatchObject(
      {
        questId: potionQuest.id,
        action: 'start'
      }
    )
  })

  it('shows a quest badge for a town resident giver', () => {
    const villagerQuest: QuestDefinition = {
      ...dynamicQuest(),
      id: 'dynamic_villager_quest',
      giverNpcId: 'villager_1',
      giverName: '마을 주민',
      title: '마을 주민의 부탁',
      trackerLabel: '마을 주민의 부탁'
    }

    registerDynamicQuestDefinitions([villagerQuest])
    setQuestDefinitionVisibilityFilter([villagerQuest.id])

    const log = ensureQuestProgressEntries(createInitialQuestLog())

    expect(getQuestNpcBadgeKindForNpc(log, 'wizard')).toBeUndefined()
    expect(getQuestNpcBadgeKindForNpc(log, 'villager_1')).toBe('new')
    expect(getNextQuestInteractionForNpc(log, 'villager_1')).toMatchObject({
      questId: villagerQuest.id,
      action: 'start'
    })
  })

  it('shows a quest badge and interaction for an active talk target npc', () => {
    const talkQuest: QuestDefinition = {
      ...dynamicQuest(),
      id: 'dynamic_talk_target_quest',
      giverNpcId: 'wizard',
      giverName: '마법사',
      title: '주민에게 단서 묻기',
      trackerLabel: '주민에게 단서 묻기',
      objectives: [
        {
          id: 'talk_target',
          label: '마을 주민에게 묻기',
          required: 1,
          type: 'talk',
          target: { npcId: 'villager_1' }
        }
      ]
    }

    registerDynamicQuestDefinitions([talkQuest])
    setQuestDefinitionVisibilityFilter([talkQuest.id])

    let log = ensureQuestProgressEntries(createInitialQuestLog())
    log = startQuest(log, talkQuest.id)

    expect(getQuestNpcBadgeKindForNpc(log, 'villager_1')).toBe('new')
    expect(getNextQuestInteractionForNpc(log, 'villager_1')).toMatchObject({
      questId: talkQuest.id,
      action: 'active'
    })

    log = recordTalkQuestProgress(log, 'villager_1')
    expect(getQuestProgress(log, talkQuest.id).status).toBe('ready-to-turn-in')
  })

  it('ignores a dynamic id that collides with a static quest id', () => {
    const staticId = getAllQuestDefinitions()[0].id
    const before = getAllQuestDefinitions().length
    registerDynamicQuestDefinitions([{ ...dynamicQuest(), id: staticId }])
    expect(getAllQuestDefinitions().length).toBe(before)
  })

  it('progresses an item-acquire objective when the item is acquired', () => {
    registerDynamicQuestDefinitions([
      {
        ...dynamicQuest(),
        id: 'dynamic_acquire',
        objectives: [
          {
            id: 'obj_acquire',
            label: '강철 검 획득',
            required: 1,
            type: 'item-acquire',
            target: { itemId: 'iron-sword' }
          }
        ]
      }
    ])
    let log = ensureQuestProgressEntries(createInitialQuestLog())
    log = startQuest(log, 'dynamic_acquire')

    // 다른 아이템 획득은 진행시키지 않는다.
    log = recordItemAcquireQuestProgress(log, 'health-potion')
    expect(getQuestProgress(log, 'dynamic_acquire').status).toBe('active')

    log = recordItemAcquireQuestProgress(log, 'iron-sword')
    expect(getQuestProgress(log, 'dynamic_acquire').status).toBe('ready-to-turn-in')
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
