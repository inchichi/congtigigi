import { describe, expect, it } from 'vitest'

import { buildFeedbackInstruction } from './gameAdapter'

describe('buildFeedbackInstruction (피드백 루프 재생성 지침)', () => {
  it('이전 출력·거절 사유·반복 회차를 지침에 담고, 부분 수정을 유도한다', () => {
    const text = buildFeedbackInstruction({
      previousOutput: '{"lines":["안녕"]}',
      validatorIssues: [],
      rejectionReason: '너무 장황함',
      iteration: 2
    })

    expect(text).toContain('반복 2회차')
    expect(text).toContain('{"lines":["안녕"]}')
    expect(text).toContain('너무 장황함')
    // 처음부터 새로 쓰지 말고 부분만 고치라는 핵심 지침이 들어가야 한다.
    expect(text).toContain('부분만 수정')
  })

  it('검증 이슈가 있으면 목록으로 포함한다', () => {
    const text = buildFeedbackInstruction({
      previousOutput: '{}',
      validatorIssues: ['npc.id - 존재하지 않는 NPC', 'trigger - 맵 불일치'],
      rejectionReason: '구조 오류',
      iteration: 3
    })

    expect(text).toContain('자동 검증에서 발견된 문제')
    expect(text).toContain('- npc.id - 존재하지 않는 NPC')
    expect(text).toContain('- trigger - 맵 불일치')
  })

  it('검증 이슈가 없으면 그 섹션을 넣지 않는다', () => {
    const text = buildFeedbackInstruction({
      previousOutput: '{}',
      validatorIssues: [],
      rejectionReason: '톤이 안 맞음',
      iteration: 1
    })

    expect(text).not.toContain('자동 검증에서 발견된 문제')
  })
})
