import {
  getPlayerPhysicalAttackPower as getPlayerPhysicalAttackPowerTs,
  getPlayerMovementSpeedTilesPerSecond as getPlayerMovementSpeedTilesPerSecondTs,
  getPlayerEvadeChance as getPlayerEvadeChanceTs
} from './playerStatEffects'
import type { PlayerProfile } from './playerProfile'
import { evaluateLuaNumber, type LoadLuaDataModule } from './luaRuleEvaluation'

// 플레이어 스탯 효과 규칙(공격력·이동속도·회피)을 Lua 코드로 옮긴 것. 상수는 playerStatEffects.ts 와
// 일치하며(이동속도 기본 8 = DEFAULT_CHARACTER_MOVE_SPEED), golden 패리티 테스트로 보장된다.
export const PLAYER_STAT_EFFECTS_LUA = `
local function clamp(value, min_value, max_value)
  return math.min(max_value, math.max(min_value, value))
end

function player_attack_power(strength)
  return math.max(1, math.floor(strength))
end

function player_move_speed(agility)
  return clamp(8 + (agility - 4) * 0.35, 4, 12)
end

function player_evade_chance(luck)
  return clamp(0.04 + luck * 0.015, 0, 0.35)
end
`

type StatProfile = Pick<PlayerProfile, 'stats'>

export type PlayerStatEffectRules = {
  getPlayerPhysicalAttackPower: (profile: StatProfile) => number
  getPlayerMovementSpeedTilesPerSecond: (profile: StatProfile) => number
  getPlayerEvadeChance: (profile: StatProfile) => number
  shouldPlayerEvadeDamage: (profile: StatProfile, randomValue?: number) => boolean
}

export const createLuaPlayerStatEffects = (
  loadDataModule: LoadLuaDataModule
): PlayerStatEffectRules => {
  const getPlayerEvadeChance = (profile: StatProfile): number =>
    evaluateLuaNumber(
      loadDataModule,
      PLAYER_STAT_EFFECTS_LUA,
      `player_evade_chance(${profile.stats.luck})`,
      getPlayerEvadeChanceTs(profile)
    )

  return {
    getPlayerPhysicalAttackPower: (profile) =>
      evaluateLuaNumber(
        loadDataModule,
        PLAYER_STAT_EFFECTS_LUA,
        `player_attack_power(${profile.stats.strength})`,
        getPlayerPhysicalAttackPowerTs(profile)
      ),
    getPlayerMovementSpeedTilesPerSecond: (profile) =>
      evaluateLuaNumber(
        loadDataModule,
        PLAYER_STAT_EFFECTS_LUA,
        `player_move_speed(${profile.stats.agility})`,
        getPlayerMovementSpeedTilesPerSecondTs(profile)
      ),
    getPlayerEvadeChance,
    shouldPlayerEvadeDamage: (profile, randomValue = Math.random()) =>
      randomValue < getPlayerEvadeChance(profile)
  }
}
