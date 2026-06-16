import { describe, expect, it } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { convertGeneratedQuestToDefinition } from './questCodeGenerator'
import type { GeneratedQuestJson } from './questJsonSchema'

const profile = CURRENT_GAME_PROJECT_PROFILE

const quest = (): GeneratedQuestJson => ({
  quest_id: 'hunt_forest_slimes',
  title: '숲의 슬라임 사냥',
  giver_npc_id: 'wizard',
  region: '티르코네일',
  request_text: '슬라임을 잡아줘',
  guide_text: '사냥터로 가라',
  start_dialogue_lines: ['시작'],
  active_dialogue_lines: ['진행'],
  completion_dialogue_lines: ['완료'],
  objectives: [
    {
      type: 'monster-defeat',
      label: '슬라임 3마리 처치',
      required: 3,
      target: { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
    }
  ],
  rewards: { gold: 100, experience: 50, items: [{ item_id: 'health-potion', quantity: 2 }] }
})

describe('convertGeneratedQuestToDefinition', () => {
  it('maps a generated quest to a runtime QuestDefinition', () => {
    const definition = convertGeneratedQuestToDefinition(quest(), profile)

    expect(definition.id).toBe('hunt_forest_slimes')
    expect(definition.giverNpcId).toBe('wizard')
    expect(definition.giverName).toBe('마법사')
    expect(definition.prerequisiteQuestIds).toEqual([])
    expect(definition.objectives).toHaveLength(1)
    expect(definition.objectives[0]).toMatchObject({
      type: 'monster-defeat',
      required: 3,
      target: { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
    })
    // 목표 id는 quest_id 기반으로 안정적으로 생성된다.
    expect(definition.objectives[0].id).toBe('hunt_forest_slimes_objective_1')
    // 보상 아이템 라벨은 프로필에서 해결된다.
    expect(definition.rewards.items[0]).toMatchObject({
      id: 'health-potion',
      quantity: 2
    })
    expect(definition.rewards.items[0].label.length).toBeGreaterThan(0)
  })

  it('drops empty target fields', () => {
    const talkQuest: GeneratedQuestJson = {
      ...quest(),
      objectives: [
        {
          type: 'talk',
          label: '상인과 대화',
          required: 1,
          target: { npcId: 'potion_merchant' }
        }
      ]
    }
    const definition = convertGeneratedQuestToDefinition(talkQuest, profile)
    expect(definition.objectives[0].target).toEqual({ npcId: 'potion_merchant' })
  })
})
