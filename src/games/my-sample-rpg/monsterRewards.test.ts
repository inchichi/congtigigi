import { describe, expect, it } from 'vitest'

import {
  getMonsterExperienceDropAmount,
  getMonsterGoldDropAmount,
  getMonsterSkillPointDropAmount
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

describe('getMonsterSkillPointDropAmount', () => {
  it('returns a skill point amount that matches the monster level', () => {
    expect(getMonsterSkillPointDropAmount(3)).toBe(3)
  })

  it('clamps monster level to at least 1', () => {
    expect(getMonsterSkillPointDropAmount(0)).toBe(1)
  })
})
