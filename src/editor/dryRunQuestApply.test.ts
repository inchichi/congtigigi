import { describe, expect, it } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { dryRunQuestApply } from './dryRunQuestApply'
import type { GeneratedQuestJson } from './questJsonSchema'

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
          label: '마을 슬라임',
          required: 1,
          target: { sceneId: 'town', appearanceType: 'monster_slime' }
        }
      ]
    }
    const report = dryRunQuestApply(quest, profile)
    expect(stepStatus(report, 'objective_1')).toBe('fail')
    expect(report.ok).toBe(false)
  })

  it('fails resolve_giver for a non-giver npc', () => {
    const report = dryRunQuestApply({ ...validQuest(), giver_npc_id: 'santa' }, profile)
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

  it('does not mutate the profile (snapshot discarded)', () => {
    const before = JSON.parse(JSON.stringify(profile))
    dryRunQuestApply(validQuest(), profile)
    expect(profile).toEqual(before)
  })
})
