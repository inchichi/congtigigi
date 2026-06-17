import {
  getMonsterGoldDropAmount as getMonsterGoldDropAmountTs,
  getMonsterExperienceDropAmount as getMonsterExperienceDropAmountTs,
  getMonsterSkillPointDropAmount as getMonsterSkillPointDropAmountTs
} from './monsterRewards'

// 게임 규칙(몬스터 보상)을 "Lua 코드"로 옮긴 것. 호스트(TS)는 이 규칙을 직접 계산하지 않고
// Lua 로직 모듈을 실행해 결과만 받는다(Lua=규칙, TS=호출/적용). monsterRewards.ts(TS)는
// 동등성(golden) 비교 기준이자, Lua 실행이 불가능할 때의 폴백으로만 남는다.
export const MONSTER_REWARDS_LUA = `
local function monster_level(value)
  return math.max(1, math.floor(value))
end

function monster_gold_drop(level)
  return 10 + monster_level(level) * 4
end

function monster_experience_drop(level)
  return 12 + monster_level(level) * 6
end

function monster_skill_point_drop(level)
  return 0 + monster_level(level)
end
`

export type MonsterRewardsRules = {
  getMonsterGoldDropAmount: (monsterLevel: number) => number
  getMonsterExperienceDropAmount: (monsterLevel: number) => number
  getMonsterSkillPointDropAmount: (monsterLevel: number) => number
}

type LoadDataModule = (source: string) => unknown

// Lua 로 규칙을 평가한다. 실패하거나 유한수가 아니면 TS 폴백으로 안전하게 떨어진다.
const evaluateRule = (
  loadDataModule: LoadDataModule,
  expression: string,
  fallback: number
): number => {
  try {
    const result = loadDataModule(`${MONSTER_REWARDS_LUA}\nreturn ${expression}`)
    return typeof result === 'number' && Number.isFinite(result) ? result : fallback
  } catch {
    return fallback
  }
}

// 몬스터 보상 규칙을 Lua 로 실행하는 구현. TS 함수와 동일한 시그니처라 호출부를 그대로 교체한다.
export const createLuaMonsterRewards = (
  loadDataModule: LoadDataModule
): MonsterRewardsRules => ({
  getMonsterGoldDropAmount: (monsterLevel) =>
    evaluateRule(
      loadDataModule,
      `monster_gold_drop(${monsterLevel})`,
      getMonsterGoldDropAmountTs(monsterLevel)
    ),
  getMonsterExperienceDropAmount: (monsterLevel) =>
    evaluateRule(
      loadDataModule,
      `monster_experience_drop(${monsterLevel})`,
      getMonsterExperienceDropAmountTs(monsterLevel)
    ),
  getMonsterSkillPointDropAmount: (monsterLevel) =>
    evaluateRule(
      loadDataModule,
      `monster_skill_point_drop(${monsterLevel})`,
      getMonsterSkillPointDropAmountTs(monsterLevel)
    )
})
