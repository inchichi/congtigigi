import { describe, expect, it } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import {
  createGeneratedEventJsonValidationIssues,
  isGeneratedEventJsonValid,
  type GeneratedEventJson
} from './eventJsonSchema'

const profile = CURRENT_GAME_PROJECT_PROFILE

// 프로필에 실재하는 id(town / blacksmith / health-potion)로 구성한 유효한 기준 이벤트.
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

const paths = (event: GeneratedEventJson): string[] =>
  createGeneratedEventJsonValidationIssues(event, profile).map((issue) => issue.path)

describe('createGeneratedEventJsonValidationIssues', () => {
  it('accepts an event that only references existing assets', () => {
    expect(createGeneratedEventJsonValidationIssues(validEvent(), profile)).toEqual([])
    expect(isGeneratedEventJsonValid(validEvent(), profile)).toBe(true)
  })

  it('flags a non-existent npc id', () => {
    const event = { ...validEvent(), npc: { id: 'ghost', name: '유령', dialogue_id: 'ghost_dialogue' } }

    expect(paths(event)).toContain('npc.id')
  })

  it('flags a non-existent map id', () => {
    const event = { ...validEvent(), location: { map_id: 'sky_castle', x: 0, y: 0 } }

    expect(paths(event)).toContain('location.map_id')
  })

  it('flags a non-existent reward item id', () => {
    const event = { ...validEvent(), reward: { item_id: 'excalibur', amount: 1 } }

    expect(paths(event)).toContain('reward.item_id')
  })

  it('keeps the id issue but tags it when requires_new_asset is set', () => {
    const event: GeneratedEventJson = {
      ...validEvent(),
      npc: { id: 'ghost', name: '유령', dialogue_id: 'ghost_dialogue' },
      requires_new_asset: true,
      requires_new_asset_reasons: ['새 NPC 유령이 필요하다.']
    }

    const issue = createGeneratedEventJsonValidationIssues(event, profile).find(
      (candidate) => candidate.path === 'npc.id'
    )

    expect(issue?.message).toContain('새 자산 필요로 표시됨')
  })

  it('requires at least one reason when requires_new_asset is true', () => {
    const event: GeneratedEventJson = {
      ...validEvent(),
      requires_new_asset: true,
      requires_new_asset_reasons: []
    }

    expect(paths(event)).toContain('requires_new_asset_reasons')
  })

  it('flags an empty dialogue list', () => {
    const event = { ...validEvent(), dialogue: [] }

    expect(paths(event)).toContain('dialogue')
  })

  it('flags a non-snake_case event_id', () => {
    const event = { ...validEvent(), event_id: 'Test Event' }

    expect(paths(event)).toContain('event_id')
  })

  it('flags a non-positive reward amount', () => {
    const event = { ...validEvent(), reward: { item_id: 'health-potion', amount: 0 } }

    expect(paths(event)).toContain('reward.amount')
  })

  it('flags an npc whose home map differs from location.map_id', () => {
    // blacksmith lives in 'town'; placing the event in the valid 'cave' map is a mismatch
    // that would pass independent id checks but fail applyEventDraft.
    const event = { ...validEvent(), location: { map_id: 'cave', x: 0, y: 0 } }

    expect(paths(event)).toContain('npc.id')
  })
})
