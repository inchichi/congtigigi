import { describe, expect, it } from 'vitest'

import {
  getMonsterExperienceDropAmount,
  getMonsterGoldDropAmount
} from './monsterRewards'

describe('getMonsterGoldDropAmount', () => {
  it('returns a beginner gold drop amount for level 3 monsters', () => {
    expect(getMonsterGoldDropAmount(3)).toBe(22)
  })

  it('clamps monster level to at least 1', () => {
    expect(getMonsterGoldDropAmount(0)).toBe(14)
  })
})

describe('getMonsterExperienceDropAmount', () => {
  it('returns a beginner experience drop amount for level 3 monsters', () => {
    expect(getMonsterExperienceDropAmount(3)).toBe(30)
  })

  it('clamps monster level to at least 1', () => {
    expect(getMonsterExperienceDropAmount(0)).toBe(18)
  })
})
