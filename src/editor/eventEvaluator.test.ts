import { describe, expect, it } from 'vitest'

import {
  createEvaluationAcceptanceStats,
  type EventEvaluation
} from './eventEvaluator'

const evaluation = (
  verdict: EventEvaluation['verdict'],
  id: string
): EventEvaluation => ({
  event_id: id,
  event_name: id,
  verdict,
  reason: '',
  evaluated_at: 0
})

describe('createEvaluationAcceptanceStats', () => {
  it('returns a zero rate when there are no evaluations', () => {
    const stats = createEvaluationAcceptanceStats([])

    expect(stats).toEqual({ total: 0, accepted: 0, acceptance_rate: 0 })
  })

  it('computes acceptance_rate as accepted over total', () => {
    const stats = createEvaluationAcceptanceStats([
      evaluation('acceptable', 'a'),
      evaluation('not_acceptable', 'b'),
      evaluation('acceptable', 'c'),
      evaluation('acceptable', 'd')
    ])

    expect(stats.total).toBe(4)
    expect(stats.accepted).toBe(3)
    expect(stats.acceptance_rate).toBeCloseTo(0.75)
  })

  it('reports a full rate when every evaluation is acceptable', () => {
    const stats = createEvaluationAcceptanceStats([
      evaluation('acceptable', 'a'),
      evaluation('acceptable', 'b')
    ])

    expect(stats.acceptance_rate).toBe(1)
  })
})
