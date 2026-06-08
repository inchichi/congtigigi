// Evaluator: 생성과 분리된 별도 "품질 평가" 단계. Validator(자동 필드/ID 검증)와 달리
// 사람이 생성 결과를 보고 acceptable/not 이진 판정만 남긴다. 점수 스케일링이 아니라
// 단일 지표 acceptance_rate = accepted / total 만 사용한다(목표 60~70%+).

export type EventEvaluationVerdict = 'acceptable' | 'not_acceptable'

export type EventEvaluation = {
  event_id: string
  event_name: string
  verdict: EventEvaluationVerdict
  reason: string
  evaluated_at: number
}

export type EvaluationAcceptanceStats = {
  total: number
  accepted: number
  acceptance_rate: number
}

const STORAGE_KEY = 'my-sample-rpg:event-evaluations'

export const createEvaluationAcceptanceStats = (
  evaluations: EventEvaluation[]
): EvaluationAcceptanceStats => {
  const total = evaluations.length
  const accepted = evaluations.filter(
    (evaluation) => evaluation.verdict === 'acceptable'
  ).length

  return {
    total,
    accepted,
    acceptance_rate: total === 0 ? 0 : accepted / total
  }
}

export const loadEventEvaluations = (): EventEvaluation[] => {
  const raw = window.localStorage.getItem(STORAGE_KEY)

  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? (parsed as EventEvaluation[]) : []
  } catch {
    return []
  }
}

export const appendEventEvaluation = (
  evaluation: EventEvaluation
): EventEvaluation[] => {
  const next = [...loadEventEvaluations(), evaluation]
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  return next
}
