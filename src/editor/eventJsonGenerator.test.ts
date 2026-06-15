import { describe, expect, it } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { generateMockEventJsonDraft } from './eventJsonGenerator'
import {
  createGeneratedEventJsonValidationIssues,
  isGeneratedEventJsonValid
} from './eventJsonSchema'

describe('eventJsonGenerator', () => {
  it('creates a christmas draft that uses existing project assets', () => {
    const { eventJson, reasoning } = generateMockEventJsonDraft(
      '크리스마스 이벤트를 만들어줘. 산타 NPC가 등장하고 대화하면 선물을 주도록 해줘.',
      CURRENT_GAME_PROJECT_PROFILE
    )

    expect(eventJson.event_id).toBe('christmas_event')
    expect(eventJson.location.map_id).toBe('town')
    expect(eventJson.npc.id).toBe('santa')
    expect(eventJson.requires_new_asset).toBeFalsy()
    expect(reasoning.length).toBeGreaterThan(0)
    expect(isGeneratedEventJsonValid(eventJson, CURRENT_GAME_PROJECT_PROFILE)).toBe(
      true
    )
    expect(
      createGeneratedEventJsonValidationIssues(
        eventJson,
        CURRENT_GAME_PROJECT_PROFILE
      )
    ).toEqual([])
  })
})
