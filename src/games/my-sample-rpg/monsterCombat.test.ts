import { describe, expect, it } from 'vitest'

import {
  applyMonsterDamage,
  createMonsterCombatState,
  isMonsterDefeated
} from './monsterCombat'

describe('monsterCombat', () => {
  it('creates beginner monster combat state from monster level', () => {
    expect(createMonsterCombatState(3)).toEqual({
      maxHp: 16,
      currentHp: 16,
      contactDamage: 2
    })
  })

  it('applies boss multipliers to monster combat state', () => {
    expect(
      createMonsterCombatState(3, {
        hpMultiplier: 2,
        damageMultiplier: 2
      })
    ).toEqual({
      maxHp: 32,
      currentHp: 32,
      contactDamage: 4
    })
  })

  it('clamps monster damage to zero and keeps defeated state stable', () => {
    const initialState = createMonsterCombatState(1)
    const damagedState = applyMonsterDamage(initialState, 99)

    expect(damagedState.currentHp).toBe(0)
    expect(isMonsterDefeated(damagedState)).toBe(true)
    expect(applyMonsterDamage(damagedState, 5)).toBe(damagedState)
  })
})
