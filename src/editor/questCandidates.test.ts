import { describe, expect, it, vi, beforeEach } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'

// generateJson(LLM 호출)을 모킹해 네트워크 없이 정규화/접지/피드백 동작만 검증한다.
vi.mock('./llmProvider', () => ({
  generateJson: vi.fn()
}))

import { generateJson } from './llmProvider'
import { generateQuestCandidates } from './questCandidates'

const profile = CURRENT_GAME_PROJECT_PROFILE
const mockGenerateJson = vi.mocked(generateJson)

beforeEach(() => {
  mockGenerateJson.mockReset()
})

describe('generateQuestCandidates', () => {
  it('trims/normalizes candidates and grounds the prompt in profile ids', async () => {
    mockGenerateJson.mockResolvedValue({
      candidates: [
        { title: ' 대장장이의 부탁 ', summary: ' 검을 찾아줘 ', target_hint: ' blacksmith ' },
        { title: '마법사의 경고', summary: '숲을 조심해', target_hint: '' },
        { title: '잃어버린 물약', summary: '물약을 회수', target_hint: '' }
      ]
    })

    const result = await generateQuestCandidates({
      apiKey: 'sk-ant-test',
      userPrompt: '마을에 새 퀘스트',
      profile
    })

    expect(result).toHaveLength(3)
    expect(result[0]).toEqual({
      title: '대장장이의 부탁',
      summary: '검을 찾아줘',
      target_hint: 'blacksmith'
    })

    const args = mockGenerateJson.mock.calls[0][0]
    expect(args.schemaName).toBe('quest_candidates')
    // 접지: profile의 실재 NPC/아이템 id가 프롬프트 입력에 들어가야 한다.
    expect(args.input).toContain('blacksmith')
    expect(args.input).toContain('health-potion')
  })

  it('returns fewer than 3 without throwing and drops fully-empty candidates', async () => {
    mockGenerateJson.mockResolvedValue({
      candidates: [
        { title: '쓸만한 후보', summary: '내용 있음', target_hint: '' },
        { title: '', summary: '', target_hint: '' }
      ]
    })

    const result = await generateQuestCandidates({
      apiKey: 'sk-ant-test',
      userPrompt: 'p',
      profile
    })

    expect(result).toHaveLength(1)
    expect(result[0].title).toBe('쓸만한 후보')
  })

  it('returns an empty array when the model returns none', async () => {
    mockGenerateJson.mockResolvedValue({ candidates: [] })

    const result = await generateQuestCandidates({
      apiKey: 'sk-ant-test',
      userPrompt: 'p',
      profile
    })

    expect(result).toEqual([])
  })

  it('weaves the feedback instruction into the prompt when regenerating', async () => {
    mockGenerateJson.mockResolvedValue({ candidates: [] })

    await generateQuestCandidates({
      apiKey: 'sk-ant-test',
      userPrompt: 'p',
      profile,
      feedback: {
        previousOutput: '이전 후보',
        validatorIssues: [],
        rejectionReason: '다른 방향으로',
        iteration: 2
      }
    })

    const args = mockGenerateJson.mock.calls[0][0]
    expect(args.input).toContain('수정 요청')
    expect(args.input).toContain('다른 방향으로')
  })
})
