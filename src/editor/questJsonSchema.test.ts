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
  title: 'Forest Slime Hunt',
  giver_npc_id: 'wizard',
  region: 'Town',
  request_text: 'Please help me.',
  guide_text: 'Go now.',
  start_dialogue_lines: ['Start'],
  active_dialogue_lines: ['Active'],
  completion_dialogue_lines: ['Done'],
  objectives: [
    {
      type: 'monster-defeat',
      label: 'Defeat slimes',
      required: 3,
      target: { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
    }
  ],
  rewards: { gold: 100, experience: 50, items: [{ item_id: 'health-potion', quantity: 1 }] }
})

const paths = (
  quest: GeneratedQuestJson,
  context?: Parameters<typeof createGeneratedQuestValidationIssues>[2]
): string[] => createGeneratedQuestValidationIssues(quest, profile, context).map((issue) => issue.path)

describe('createGeneratedQuestValidationIssues', () => {
  it('accepts a quest grounded in real catalog/profile values', () => {
    expect(createGeneratedQuestValidationIssues(validQuest(), profile)).toEqual([])
    expect(isGeneratedQuestValid(validQuest(), profile)).toBe(true)
  })

  it('flags a non-snake_case quest_id', () => {
    expect(paths({ ...validQuest(), quest_id: 'Hunt Slimes' })).toContain('quest_id')
  })

  it('flags an unknown giver id', () => {
    expect(paths({ ...validQuest(), giver_npc_id: 'ghost' })).toContain('giver_npc_id')
  })

  it('accepts santa when santa is the selected npc', () => {
    expect(
      createGeneratedQuestValidationIssues(
        { ...validQuest(), giver_npc_id: 'santa' },
        profile,
        { selectedEntityId: 'santa' }
      )
    ).toEqual([])
  })

  it('flags a giver that does not match the selected npc', () => {
    expect(paths({ ...validQuest(), giver_npc_id: 'wizard' }, { selectedEntityId: 'santa' })).toContain(
      'giver_npc_id'
    )
  })

  it('flags a monster-defeat objective targeting an unknown monster', () => {
    const quest = {
      ...validQuest(),
      objectives: [
        {
          type: 'monster-defeat' as const,
          label: 'Bad target',
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
          label: 'Town slime',
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
          label: 'Bad count',
          required: 0,
          target: { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
        }
      ]
    }
    expect(paths(quest)).toContain('objectives[0].required')
  })

  it('accepts an item-acquire objective targeting a monster-drop item', () => {
    const quest = {
      ...validQuest(),
      objectives: [
        {
          type: 'item-acquire' as const,
          label: 'Iron sword',
          required: 1,
          target: { itemId: 'iron-sword' }
        }
      ]
    }
    expect(isGeneratedQuestValid(quest, profile)).toBe(true)
  })

  it('flags an item-acquire objective targeting a non-monster-drop item', () => {
    const quest = {
      ...validQuest(),
      objectives: [
        {
          type: 'item-acquire' as const,
          label: 'Smith charm',
          required: 1,
          target: { itemId: 'smith-charm' }
        }
      ]
    }
    expect(paths(quest)).toContain('objectives[0].target.itemId')
  })

  it('does NOT block an unknown reward item (runtime tolerates)', () => {
    const quest = {
      ...validQuest(),
      rewards: { gold: 0, experience: 0, items: [{ item_id: 'battle-axe', quantity: 1 }] }
    }
    expect(isGeneratedQuestValid(quest, profile)).toBe(true)
  })
})
