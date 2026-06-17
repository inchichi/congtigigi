import { describe, expect, it } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { dryRunQuestApply } from './dryRunQuestApply'
import type { GeneratedQuestJson } from './questJsonSchema'

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

const stepStatus = (report: ReturnType<typeof dryRunQuestApply>, id: string) =>
  report.steps.find((step) => step.id === id)?.status

describe('dryRunQuestApply', () => {
  it('passes a fully valid quest', () => {
    const report = dryRunQuestApply(validQuest(), profile)
    expect(report.ok).toBe(true)
    expect(report.steps.every((step) => step.status !== 'fail')).toBe(true)
    expect(report.jsonIssues).toEqual([])
  })

  it('fails the objective step when the monster scene has no monsters', () => {
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
    const report = dryRunQuestApply(quest, profile)
    expect(stepStatus(report, 'objective_1')).toBe('fail')
    expect(report.ok).toBe(false)
  })

  it('fails resolve_giver when the selected npc does not match', () => {
    const report = dryRunQuestApply(validQuest(), profile, { selectedEntityId: 'santa' })
    expect(stepStatus(report, 'resolve_giver')).toBe('fail')
    expect(report.ok).toBe(false)
  })

  it('fails resolve_giver for an unknown npc', () => {
    const report = dryRunQuestApply({ ...validQuest(), giver_npc_id: 'ghost' }, profile)
    expect(stepStatus(report, 'resolve_giver')).toBe('fail')
    expect(report.ok).toBe(false)
  })

  it('warns (not fails) on an unknown reward item, keeping ok=true', () => {
    const quest = {
      ...validQuest(),
      rewards: { gold: 0, experience: 0, items: [{ item_id: 'battle-axe', quantity: 1 }] }
    }
    const report = dryRunQuestApply(quest, profile)
    expect(stepStatus(report, 'reward_items')).toBe('warn')
    expect(report.ok).toBe(true)
  })

  it('passes an item-acquire objective for a monster-drop item', () => {
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
    const report = dryRunQuestApply(quest, profile)
    expect(stepStatus(report, 'objective_1')).toBe('ok')
    expect(report.ok).toBe(true)
  })

  it('fails an item-acquire objective for a non-monster-drop item', () => {
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
    const report = dryRunQuestApply(quest, profile)
    expect(stepStatus(report, 'objective_1')).toBe('fail')
    expect(report.ok).toBe(false)
  })

  it('does not mutate the profile (snapshot discarded)', () => {
    const before = JSON.parse(JSON.stringify(profile))
    dryRunQuestApply(validQuest(), profile)
    expect(profile).toEqual(before)
  })
})
