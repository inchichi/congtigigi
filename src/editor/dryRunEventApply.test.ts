import { describe, expect, it } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { dryRunEventApply } from './dryRunEventApply'
import type { GeneratedEventJson } from './eventJsonSchema'

const profile = CURRENT_GAME_PROJECT_PROFILE

// 프로필에 실재하는 id(town / blacksmith / health-potion)로 구성한 유효 기준 이벤트.
const validEvent = (): GeneratedEventJson => ({
  event_id: 'test_event',
  event_name: 'test_event',
  description: '테스트 이벤트',
  trigger: {
    type: 'talk',
    target: 'blacksmith',
    condition: 'player_interacts_with_npc'
  },
  location: { map_id: 'town', x: 400, y: 516 },
  npc: { id: 'blacksmith', name: '대장장이', dialogue_id: 'blacksmith_dialogue' },
  dialogue: [{ speaker: '대장장이', text: '안녕하시오.' }],
  reward: { item_id: 'health-potion', amount: 1 },
  duration: {
    start: '2026-01-01T00:00:00.000Z',
    end: '2026-01-08T00:00:00.000Z'
  }
})

describe('dryRunEventApply', () => {
  it('passes a fully valid event with no fail steps', () => {
    const report = dryRunEventApply(validEvent(), profile)

    expect(report.ok).toBe(true)
    expect(report.failedStepId).toBeUndefined()
    expect(report.steps.every((step) => step.status !== 'fail')).toBe(true)
    expect(report.jsonIssues).toEqual([])
  })

  it('reports steps in the fixed apply order', () => {
    const report = dryRunEventApply(validEvent(), profile)

    expect(report.steps.map((step) => step.id)).toEqual([
      'convert',
      'resolve_npc',
      'resolve_map',
      'dialogue',
      'reward_item',
      'spec_validate'
    ])
  })

  it('fails at convert when there is no usable dialogue', () => {
    const report = dryRunEventApply({ ...validEvent(), dialogue: [] }, profile)

    expect(report.ok).toBe(false)
    expect(report.failedStepId).toBe('convert')
  })

  it('fails resolve_map when the npc map and event map differ', () => {
    const otherMap = profile.maps.find((map) => map.id !== 'town')
    expect(otherMap).toBeDefined()

    const report = dryRunEventApply(
      { ...validEvent(), location: { map_id: otherMap?.id ?? 'cave', x: 0, y: 0 } },
      profile
    )

    expect(report.steps.find((step) => step.id === 'resolve_map')?.status).toBe('fail')
    expect(report.ok).toBe(false)
  })

  it('warns (not fails) on an unknown reward item, keeping ok=true', () => {
    const report = dryRunEventApply(
      { ...validEvent(), reward: { item_id: 'excalibur', amount: 1 } },
      profile
    )

    expect(report.steps.find((step) => step.id === 'reward_item')?.status).toBe('warn')
    expect(report.ok).toBe(true)
  })

  it('fails spec_validate when event_name is not lowercase_underscore', () => {
    const report = dryRunEventApply({ ...validEvent(), event_name: 'Bad Name' }, profile)

    expect(report.steps.find((step) => step.id === 'spec_validate')?.status).toBe('fail')
    expect(report.ok).toBe(false)
  })

  it('does not mutate the profile (snapshot is discarded = rollback)', () => {
    const before = JSON.parse(JSON.stringify(profile))
    dryRunEventApply(validEvent(), profile)
    expect(profile).toEqual(before)
  })
})
