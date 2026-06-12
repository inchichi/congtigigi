import { describe, expect, it } from 'vitest'

import {
  buildScenarioEditorUserPrompt,
  extractScenarioEditorResponseText,
  parseScenarioEditorResult
} from './scenarioEditorProtocol'

describe('scenario editor protocol', () => {
  it('builds a prompt that preserves the scenario and requests JSON only', () => {
    const prompt = buildScenarioEditorUserPrompt(
      '마을 외곽에서 슬라임이 늘어난다.',
      ['마법사', '대장장이', '물약상인']
    )

    expect(prompt).toContain('마을 외곽에서 슬라임이 늘어난다.')
    expect(prompt).toContain('2 to 4 quests')
    expect(prompt).toContain('NPC dialogue entries')
    expect(prompt).toContain('Available NPC roster:')
    expect(prompt).toContain('- 마법사')
    expect(prompt).toContain('- 대장장이')
    expect(prompt).toContain('- 물약상인')
    expect(prompt).toContain('Use only these NPC names')
  })

  it('extracts assistant text from common OpenAI-style response shapes', () => {
    expect(
      extractScenarioEditorResponseText({
        choices: [
          {
            message: {
              content: '```json\n{"summary":"OK"}\n```'
            }
          }
        ]
      })
    ).toBe('```json\n{"summary":"OK"}\n```')

    expect(
      extractScenarioEditorResponseText({
        output_text: '결과 텍스트'
      })
    ).toBe('결과 텍스트')

    expect(
      extractScenarioEditorResponseText({
        output: [
          {
            content: [{ text: '첫 줄' }, { text: '둘째 줄' }]
          }
        ]
      })
    ).toBe('첫 줄\n둘째 줄')
  })

  it('parses a JSON quest draft from wrapped model text', () => {
    const responseText = [
      'Here is the draft:',
      '```json',
      '{',
      '  "summary": "마을을 구하는 이야기",',
      '  "quests": [',
      '    {',
      '      "title": "슬라임 조사",',
      '      "giver": "촌장",',
      '      "goal": "우물 주변의 슬라임을 정리한다",',
      '      "objectives": ["우물 조사", "슬라임 3마리 처치"],',
      '      "reward": "100 골드"',
      '    },',
      '    {',
      '      "title": "우물 정화",',
      '      "giver": "마법사",',
      '      "goal": "우물에 남은 이상한 기운을 확인한다",',
      '      "objectives": ["우물 주변 조사", "마법 흔적 기록"],',
      '      "reward": "마나 포션 2개"',
      '    }',
      '  ],',
      '  "npc_dialogues": [',
      '    {',
      '      "npc": "촌장",',
      '      "context": "퀘스트 시작",',
      '      "lines": ["마을을 좀 도와주게.", "우물 쪽이 수상하네."]',
      '    },',
      '    {',
      '      "npc": "마법사",',
      '      "context": "조사 진행",',
      '      "lines": ["마법 반응이 남아 있어.", "좀 더 가까이 확인해 보자."]',
      '    }',
      '  ]',
      '}',
      '```'
    ].join('\n')

    expect(parseScenarioEditorResult(responseText)).toEqual({
      summary: '마을을 구하는 이야기',
      quests: [
        {
          title: '슬라임 조사',
          giver: '촌장',
          goal: '우물 주변의 슬라임을 정리한다',
          objectives: ['우물 조사', '슬라임 3마리 처치'],
          reward: '100 골드'
        },
        {
          title: '우물 정화',
          giver: '마법사',
          goal: '우물에 남은 이상한 기운을 확인한다',
          objectives: ['우물 주변 조사', '마법 흔적 기록'],
          reward: '마나 포션 2개'
        }
      ],
      npcDialogues: [
        {
          npc: '촌장',
          context: '퀘스트 시작',
          lines: ['마을을 좀 도와주게.', '우물 쪽이 수상하네.']
        },
        {
          npc: '마법사',
          context: '조사 진행',
          lines: ['마법 반응이 남아 있어.', '좀 더 가까이 확인해 보자.']
        }
      ]
    })
  })

  it('rejects drafts that do not meet the minimum quest and dialogue counts', () => {
    const responseText = JSON.stringify({
      summary: '마을을 구하는 이야기',
      quests: [
        {
          title: '슬라임 조사',
          giver: '촌장',
          goal: '우물 주변의 슬라임을 정리한다',
          objectives: ['우물 조사', '슬라임 3마리 처치'],
          reward: '100 골드'
        }
      ],
      npc_dialogues: [
        {
          npc: '촌장',
          context: '퀘스트 시작',
          lines: ['마을을 좀 도와주게.', '우물 쪽이 수상하네.']
        }
      ]
    })

    expect(parseScenarioEditorResult(responseText)).toBeUndefined()
  })

  it('rejects drafts without a summary even when the quest and dialogue counts are sufficient', () => {
    const responseText = JSON.stringify({
      quests: [
        {
          title: '슬라임 조사',
          giver: '촌장',
          goal: '우물 주변의 슬라임을 정리한다',
          objectives: ['우물 조사', '슬라임 3마리 처치'],
          reward: '100 골드'
        },
        {
          title: '우물 정화',
          giver: '마법사',
          goal: '우물에 남은 이상한 기운을 확인한다',
          objectives: ['우물 주변 조사', '마법 흔적 기록'],
          reward: '마나 포션 2개'
        }
      ],
      npc_dialogues: [
        {
          npc: '촌장',
          context: '퀘스트 시작',
          lines: ['마을을 좀 도와주게.', '우물 쪽이 수상하네.']
        },
        {
          npc: '마법사',
          context: '조사 진행',
          lines: ['마법 반응이 남아 있어.', '좀 더 가까이 확인해 보자.']
        }
      ]
    })

    expect(parseScenarioEditorResult(responseText)).toBeUndefined()
  })
})
