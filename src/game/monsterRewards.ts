const MONSTER_GOLD_DROP_BASE = 10
const MONSTER_GOLD_DROP_PER_LEVEL = 4
const MONSTER_EXPERIENCE_DROP_BASE = 12
const MONSTER_EXPERIENCE_DROP_PER_LEVEL = 6

export const getMonsterGoldDropAmount = (
  monsterLevel: number
): number => {
  const level = Math.max(1, Math.floor(monsterLevel))

  return MONSTER_GOLD_DROP_BASE + level * MONSTER_GOLD_DROP_PER_LEVEL
}

export const getMonsterExperienceDropAmount = (
  monsterLevel: number
): number => {
  const level = Math.max(1, Math.floor(monsterLevel))

  return (
    MONSTER_EXPERIENCE_DROP_BASE +
    level * MONSTER_EXPERIENCE_DROP_PER_LEVEL
  )
}
