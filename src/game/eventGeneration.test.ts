import { describe, expect, it } from 'vitest'

import {
  HOLIDAY_DIALOGUE_EVENT_TYPE,
  type HolidayDialogueEventSpec,
  createHolidayDialogueEventValidationErrors,
  createTiledNpcEventObject,
  isHolidayDialogueEventValid
} from './eventGeneration'

const createValidHolidayDialogueEvent = (): HolidayDialogueEventSpec => ({
  event_name: 'christmas_event',
  event_type: HOLIDAY_DIALOGUE_EVENT_TYPE,
  npc: {
    id: 'santa',
    display_name: '산타',
    appearance_type: 'character_villager_brown_tunic'
  },
  trigger: {
    type: 'talk' as const,
    target_scene: 'town'
  },
  dialogue: {
    opening_lines: ['메리 크리스마스!', '오늘은 특별한 날이란다.'],
    active_lines: [],
    completion_lines: []
  },
  reward: {
    type: 'item' as const,
    id: 'gift',
    count: 1
  },
  duration: 2.5
})

describe('eventGeneration', () => {
  it('accepts a valid holiday dialogue event', () => {
    const event = createValidHolidayDialogueEvent()

    expect(isHolidayDialogueEventValid(event)).toBe(true)
    expect(createHolidayDialogueEventValidationErrors(event)).toEqual([])
  })

  it('rejects invalid event data', () => {
    const event = createValidHolidayDialogueEvent()

    event.event_name = 'Christmas Event'
    event.dialogue.opening_lines = []
    event.reward = {
      type: 'item',
      id: '',
      count: 0
    }

    expect(createHolidayDialogueEventValidationErrors(event)).toEqual([
      'event_name must use lowercase letters, numbers, and underscores only',
      'dialogue.opening_lines must contain at least one line',
      'reward.id must not be empty when reward.type is "item"',
      'reward.count must be a positive finite number when reward.type is "item"'
    ])
  })

  it('converts a holiday dialogue event into Tiled NPC properties', () => {
    const event = createValidHolidayDialogueEvent()

    expect(createTiledNpcEventObject(event)).toEqual({
      name: 'santa',
      type: 'character_villager_brown_tunic',
      properties: {
        displayText: '산타',
        'controller.scriptId': 'reply-with-message',
        'controller.dialogueLines': [
          '메리 크리스마스!',
          '오늘은 특별한 날이란다.'
        ],
        'controller.messageDurationSeconds': 2.5
      }
    })
  })
})
