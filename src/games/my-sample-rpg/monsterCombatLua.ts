import {
  createMonsterCombatState as createMonsterCombatStateTs,
  type MonsterCombatState
} from './monsterCombat'
import { evaluateLuaValue, type LoadLuaDataModule } from './luaRuleEvaluation'

// 몬스터 전투 "상태 생성" 규칙(레벨/배수 → maxHp·contactDamage)을 Lua 코드로 옮긴 것.
// 데미지 적용/패배 판정은 매 히트·매 프레임 호출되는 핫패스라 TS(호스트)에 남긴다 —
// Lua 는 가끔 호출되는 "규칙 계산"만 맡는다. 상수는 monsterCombat.ts 와 일치(golden 패리티).
export const MONSTER_COMBAT_LUA = `
function create_monster_combat_state(level, hp_mult, damage_mult)
  local l = math.max(1, math.floor(level))
  local hp_multiplier = math.max(1, hp_mult)
  local damage_multiplier = math.max(1, damage_mult)
  local max_hp = (10 + l * 2) * hp_multiplier
  return {
    maxHp = max_hp,
    currentHp = max_hp,
    contactDamage = math.max(1, math.ceil(l / 2)) * damage_multiplier
  }
end
`

type MonsterCombatStateOptions = {
  hpMultiplier?: number
  damageMultiplier?: number
}

const isMonsterCombatState = (value: unknown): value is MonsterCombatState =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as Record<string, unknown>).maxHp === 'number' &&
  typeof (value as Record<string, unknown>).currentHp === 'number' &&
  typeof (value as Record<string, unknown>).contactDamage === 'number'

export const createLuaMonsterCombat = (loadDataModule: LoadLuaDataModule) => ({
  createMonsterCombatState: (
    monsterLevel = 1,
    options: MonsterCombatStateOptions = {}
  ): MonsterCombatState => {
    const hpMultiplier = options.hpMultiplier ?? 1
    const damageMultiplier = options.damageMultiplier ?? 1
    try {
      const result = evaluateLuaValue(
        loadDataModule,
        MONSTER_COMBAT_LUA,
        `create_monster_combat_state(${monsterLevel}, ${hpMultiplier}, ${damageMultiplier})`
      )
      return isMonsterCombatState(result)
        ? result
        : createMonsterCombatStateTs(monsterLevel, options)
    } catch {
      return createMonsterCombatStateTs(monsterLevel, options)
    }
  }
})
