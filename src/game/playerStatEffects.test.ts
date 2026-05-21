import { describe, expect, it } from 'vitest'

import { createInitialPlayerProfile } from './playerProfile'
import {
  getPlayerEvadeChance,
  getPlayerMovementSpeedTilesPerSecond,
  getPlayerPhysicalAttackPower,
  shouldPlayerEvadeDamage
} from './playerStatEffects'

describe('getPlayerPhysicalAttackPower', () => {
  it('uses strength as the physical damage base', () => {
    expect(getPlayerPhysicalAttackPower(createInitialPlayerProfile())).toBe(5)
  })
})

describe('getPlayerMovementSpeedTilesPerSecond', () => {
  it('gets faster when agility goes up', () => {
    const profile = createInitialPlayerProfile()
    const fasterProfile = {
      ...profile,
      stats: {
        ...profile.stats,
        agility: profile.stats.agility + 2
      }
    }

    expect(getPlayerMovementSpeedTilesPerSecond(fasterProfile)).toBeGreaterThan(
      getPlayerMovementSpeedTilesPerSecond(profile)
    )
  })
})

describe('getPlayerEvadeChance', () => {
  it('gets higher when luck goes up', () => {
    const profile = createInitialPlayerProfile()
    const luckierProfile = {
      ...profile,
      stats: {
        ...profile.stats,
        luck: profile.stats.luck + 5
      }
    }

    expect(getPlayerEvadeChance(luckierProfile)).toBeGreaterThan(
      getPlayerEvadeChance(profile)
    )
  })
})

describe('shouldPlayerEvadeDamage', () => {
  it('uses the provided random value', () => {
    const profile = createInitialPlayerProfile()

    expect(shouldPlayerEvadeDamage(profile, 0)).toBe(true)
    expect(shouldPlayerEvadeDamage(profile, 1)).toBe(false)
  })
})
