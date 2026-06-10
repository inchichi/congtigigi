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
      '    }',
      '  ],',
      '  "npc_dialogues": [',
      '    {',
      '      "npc": "촌장",',
      '      "context": "퀘스트 시작",',
      '      "lines": ["마을을 좀 도와주게.", "우물 쪽이 수상하네."]',
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
        }
      ],
      npcDialogues: [
        {
          npc: '촌장',
          context: '퀘스트 시작',
          lines: ['마을을 좀 도와주게.', '우물 쪽이 수상하네.']
        }
      ]
    })
  })
})
