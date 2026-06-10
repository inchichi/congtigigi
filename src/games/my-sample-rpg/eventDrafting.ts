import {
  HOLIDAY_DIALOGUE_EVENT_TYPE,
  TALK_EVENT_TRIGGER_TYPE,
  type HolidayDialogueEventSpec
} from './eventGeneration'

export const createHolidayDialogueEventDraftFromText = (
  prompt: string
): HolidayDialogueEventSpec | undefined => {
  const normalizedPrompt = prompt.toLowerCase()

  if (includesAny(normalizedPrompt, ['크리스마스', 'christmas', 'xmas'])) {
    return {
      event_name: 'christmas_event',
      event_type: HOLIDAY_DIALOGUE_EVENT_TYPE,
      npc: {
        id: 'santa',
        display_name: '산타',
        appearance_type: 'character_villager_brown_tunic'
      },
      trigger: {
        type: TALK_EVENT_TRIGGER_TYPE,
        target_scene: 'town'
      },
      dialogue: {
        opening_lines: ['메리 크리스마스!', '오늘은 특별한 날이란다.'],
        active_lines: [],
        completion_lines: []
      },
      reward: {
        type: 'item',
        id: 'gift',
        count: 1
      },
      duration: 7
    }
  }

  if (includesAny(normalizedPrompt, ['할로윈', 'halloween'])) {
    return {
      event_name: 'halloween_event',
      event_type: HOLIDAY_DIALOGUE_EVENT_TYPE,
      npc: {
        id: 'halloween_npc',
        display_name: '할로윈 NPC',
        appearance_type: 'character_commoner_tan_tunic'
      },
      trigger: {
        type: TALK_EVENT_TRIGGER_TYPE,
        target_scene: 'town'
      },
      dialogue: {
        opening_lines: ['할로윈이야!', '사탕을 받아가!'],
        active_lines: [],
        completion_lines: []
      },
      reward: {
        type: 'item',
        id: 'candy',
        count: 3
      },
      duration: 7
    }
  }

  return undefined
}

const includesAny = (value: string, keywords: string[]): boolean =>
  keywords.some((keyword) => value.includes(keyword))
