import { describe, expect, it } from 'vitest'

import { buildSessionMetrics, ACCEPTANCE_GOAL } from './sessionMetrics'
import type { EventEvaluation } from './eventEvaluator'

const evaluation = (verdict: EventEvaluation['verdict']): EventEvaluation => ({
  event_id: 'e',
  event_name: 'e',
  verdict,
  reason: '',
  evaluated_at: 0
})

describe('buildSessionMetrics', () => {
  it('reports zeros with no generations and no evaluations', () => {
    const metrics = buildSessionMetrics({ generations: 0, validatorPasses: 0 }, [])
    expect(metrics).toEqual({
      generations: 0,
      validatorPassRate: 0,
      acceptanceRate: 0,
      acceptanceTotal: 0,
      meetsAcceptanceGoal: false
    })
  })

  it('computes validator pass rate over generations', () => {
    const metrics = buildSessionMetrics({ generations: 4, validatorPasses: 3 }, [])
    expect(metrics.validatorPassRate).toBe(0.75)
    expect(metrics.generations).toBe(4)
  })

  it('computes acceptance rate from evaluations and goal status', () => {
    const metrics = buildSessionMetrics({ generations: 2, validatorPasses: 2 }, [
      evaluation('acceptable'),
      evaluation('acceptable'),
      evaluation('not_acceptable')
    ])
    expect(metrics.acceptanceRate).toBeCloseTo(2 / 3)
    expect(metrics.acceptanceTotal).toBe(3)
    expect(metrics.meetsAcceptanceGoal).toBe(2 / 3 >= ACCEPTANCE_GOAL)
  })

  it('does not claim the goal is met when there are no evaluations', () => {
    const metrics = buildSessionMetrics({ generations: 5, validatorPasses: 5 }, [])
    expect(metrics.meetsAcceptanceGoal).toBe(false)
  })

  it('marks the goal unmet when acceptance is below the threshold', () => {
    const metrics = buildSessionMetrics({ generations: 4, validatorPasses: 4 }, [
      evaluation('acceptable'),
      evaluation('not_acceptable'),
      evaluation('not_acceptable')
    ])
    expect(metrics.acceptanceRate).toBeCloseTo(1 / 3)
    expect(metrics.meetsAcceptanceGoal).toBe(false)
  })
})
