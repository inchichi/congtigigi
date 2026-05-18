import { describe, expect, it } from 'vitest'

import { getMonsterGoldDropAmount } from './monsterRewards'

describe('getMonsterGoldDropAmount', () => {
  it('returns a beginner gold drop amount for level 3 monsters', () => {
    expect(getMonsterGoldDropAmount(3)).toBe(22)
  })

  it('clamps monster level to at least 1', () => {
    expect(getMonsterGoldDropAmount(0)).toBe(14)
  })
})
