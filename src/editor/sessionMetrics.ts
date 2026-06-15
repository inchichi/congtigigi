import { createEvaluationAcceptanceStats, type EventEvaluation } from './eventEvaluator'

// 회의의 두 평가 개념을 한 줄 지표로 묶는다: 자동 Validator 통과율 + 사람 이진평가 수용률.
// 점수 스케일링이 아니라 비율만 쓴다(이사님 피드백). 모두 순수 함수라 테스트가 쉽다.

// 사람 수용률 목표(회의: 60~70%+). meetsAcceptanceGoal 표시에 쓴다.
export const ACCEPTANCE_GOAL = 0.6

export type SessionGenerationTally = {
  // 이번 세션에 생성한 총 횟수.
  generations: number
  // 그중 Validator를 통과(issues 0건)한 횟수.
  validatorPasses: number
}

export type SessionMetrics = {
  generations: number
  validatorPassRate: number
  acceptanceRate: number
  acceptanceTotal: number
  meetsAcceptanceGoal: boolean
}

export const buildSessionMetrics = (
  tally: SessionGenerationTally,
  evaluations: EventEvaluation[]
): SessionMetrics => {
  const acceptance = createEvaluationAcceptanceStats(evaluations)

  return {
    generations: tally.generations,
    validatorPassRate:
      tally.generations === 0 ? 0 : tally.validatorPasses / tally.generations,
    acceptanceRate: acceptance.acceptance_rate,
    acceptanceTotal: acceptance.total,
    // 평가가 하나도 없으면 목표 달성 여부는 의미가 없으므로 false로 둔다.
    meetsAcceptanceGoal:
      acceptance.total > 0 && acceptance.acceptance_rate >= ACCEPTANCE_GOAL
  }
}
