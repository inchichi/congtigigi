const MONSTER_GOLD_DROP_BASE = 10
const MONSTER_GOLD_DROP_PER_LEVEL = 4

export const getMonsterGoldDropAmount = (
  monsterLevel: number
): number => {
  const level = Math.max(1, Math.floor(monsterLevel))

  return MONSTER_GOLD_DROP_BASE + level * MONSTER_GOLD_DROP_PER_LEVEL
}
