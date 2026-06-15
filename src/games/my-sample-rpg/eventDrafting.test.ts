import { describe, expect, it } from 'vitest'

import { createHolidayDialogueEventDraftFromText } from './eventDrafting'

describe('eventDrafting', () => {
  it('creates a christmas event draft from natural language', () => {
    const draft = createHolidayDialogueEventDraftFromText(
      '크리스마스 이벤트를 만들어줘. 산타 NPC가 등장하고 대화하면 선물을 주도록 해줘.'
    )

    expect(draft).toMatchObject({
      event_name: 'christmas_event',
      npc: {
        id: 'santa',
        display_name: '산타'
      },
      dialogue: {
        opening_lines: ['메리 크리스마스!', '오늘은 특별한 날이란다.']
      }
    })
  })

  it('creates a halloween event draft from natural language', () => {
    const draft = createHolidayDialogueEventDraftFromText(
      '할로윈 이벤트를 만들어줘.'
    )

    expect(draft).toMatchObject({
      event_name: 'halloween_event',
      npc: {
        id: 'halloween_npc',
        display_name: '할로윈 NPC'
      },
      reward: {
        type: 'item',
        id: 'candy',
        count: 3
      }
    })
  })

  it('returns undefined for unsupported prompts', () => {
    expect(
      createHolidayDialogueEventDraftFromText('마을 안내 NPC를 만들어줘.')
    ).toBeUndefined()
  })
})
