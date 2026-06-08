import { describe, expect, it } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { calculateGameUnderstandingScore } from './gusCalculator'

describe('gusCalculator', () => {
  it('calculates a passing score for the current project profile', () => {
    const report = calculateGameUnderstandingScore(CURRENT_GAME_PROJECT_PROFILE)

    expect(report.gus_score).toBeGreaterThanOrEqual(75)
    expect(report.status).toBe('passed')
    expect(report.details.qa_correctness).toBeGreaterThan(0)
    expect(report.details.dependency_f1).toBeGreaterThan(0)
  })
})
