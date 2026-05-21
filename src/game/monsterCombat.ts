export type MonsterCombatState = {
  maxHp: number
  currentHp: number
  contactDamage: number
}

type CreateMonsterCombatStateOptions = {
  hpMultiplier?: number
  damageMultiplier?: number
}

const MONSTER_BASE_HP = 10
const MONSTER_HP_PER_LEVEL = 2

export const createMonsterCombatState = (
  monsterLevel = 1,
  options: CreateMonsterCombatStateOptions = {}
): MonsterCombatState => {
  const level = Math.max(1, Math.floor(monsterLevel))
  const hpMultiplier = Math.max(1, options.hpMultiplier ?? 1)
  const damageMultiplier = Math.max(1, options.damageMultiplier ?? 1)
  const maxHp = (MONSTER_BASE_HP + level * MONSTER_HP_PER_LEVEL) * hpMultiplier

  return {
    maxHp,
    currentHp: maxHp,
    contactDamage:
      Math.max(1, Math.ceil(level / 2)) * damageMultiplier
  }
}

export const applyMonsterDamage = (
  combatState: MonsterCombatState,
  damage: number
): MonsterCombatState => {
  const nextDamage = Math.max(0, Math.floor(damage))

  if (nextDamage === 0 || combatState.currentHp === 0) {
    return combatState
  }

  const nextCurrentHp = Math.max(0, combatState.currentHp - nextDamage)

  if (nextCurrentHp === combatState.currentHp) {
    return combatState
  }

  return {
    ...combatState,
    currentHp: nextCurrentHp
  }
}

export const isMonsterDefeated = (
  combatState: MonsterCombatState
): boolean => combatState.currentHp <= 0
