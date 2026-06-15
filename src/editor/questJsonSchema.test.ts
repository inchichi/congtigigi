import { describe, expect, it } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import {
  createGeneratedQuestValidationIssues,
  isGeneratedQuestValid,
  type GeneratedQuestJson
} from './questJsonSchema'

const profile = CURRENT_GAME_PROJECT_PROFILE

const validQuest = (): GeneratedQuestJson => ({
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
  rewards: { gold: 100, experience: 50, items: [{ item_id: 'health-potion', quantity: 1 }] }
})

const paths = (quest: GeneratedQuestJson): string[] =>
  createGeneratedQuestValidationIssues(quest, profile).map((issue) => issue.path)

describe('createGeneratedQuestValidationIssues', () => {
  it('accepts a quest grounded in real catalog/profile values', () => {
    expect(createGeneratedQuestValidationIssues(validQuest(), profile)).toEqual([])
    expect(isGeneratedQuestValid(validQuest(), profile)).toBe(true)
  })

  it('flags a non-snake_case quest_id', () => {
    expect(paths({ ...validQuest(), quest_id: 'Hunt Slimes' })).toContain('quest_id')
  })

  it('flags a giver that is not a wired quest giver', () => {
    expect(paths({ ...validQuest(), giver_npc_id: 'santa' })).toContain('giver_npc_id')
  })

  it('flags a monster-defeat objective targeting an unknown monster', () => {
    const quest = {
      ...validQuest(),
      objectives: [
        {
          type: 'monster-defeat' as const,
          label: '용 처치',
          required: 1,
          target: { sceneId: 'hunting-ground', appearanceType: 'monster_dragon' }
        }
      ]
    }
    expect(paths(quest)).toContain('objectives[0].target.appearanceType')
  })

  it('flags a monster-defeat objective in a scene with no monsters', () => {
    const quest = {
      ...validQuest(),
      objectives: [
        {
          type: 'monster-defeat' as const,
          label: '마을 슬라임',
          required: 1,
          target: { sceneId: 'town', appearanceType: 'monster_slime' }
        }
      ]
    }
    expect(paths(quest)).toContain('objectives[0].target.sceneId')
  })

  it('flags required less than 1', () => {
    const quest = {
      ...validQuest(),
      objectives: [
        {
          type: 'monster-defeat' as const,
          label: '슬라임',
          required: 0,
          target: { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
        }
      ]
    }
    expect(paths(quest)).toContain('objectives[0].required')
  })

  it('does NOT block an unknown reward item (runtime tolerates)', () => {
    const quest = {
      ...validQuest(),
      rewards: { gold: 0, experience: 0, items: [{ item_id: 'battle-axe', quantity: 1 }] }
    }
    expect(isGeneratedQuestValid(quest, profile)).toBe(true)
  })
})
