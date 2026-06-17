// playerStatEffects의 파생 스탯 계산을 Lua로 실행하는 래퍼(객체 입력 → callJson 마샬링).
// shouldPlayerEvadeDamage는 난수값 인자 비교라 회피 확률만 Lua에서 받고 비교는 TS에서 한다.

import playerStatEffectsSource from '../assets/lua/player-stat-effects.lua?raw'

import { DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND } from '../characterState'
import type { PlayerProfile } from '../playerProfile'
import {
  PLAYER_AGILITY_MOVE_SPEED_BONUS_PER_POINT,
  PLAYER_MIN_MOVE_SPEED_TILES_PER_SECOND,
  PLAYER_MAX_MOVE_SPEED_TILES_PER_SECOND,
  PLAYER_LUCK_EVADE_BASE_CHANCE,
  PLAYER_LUCK_EVADE_BONUS_PER_POINT,
  PLAYER_MAX_EVADE_CHANCE
} from '../playerStatEffects'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

type StatsProfile = Pick<PlayerProfile, 'stats'>

export type PlayerStatEffectsLua = {
  getPlayerPhysicalAttackPower: (profile: StatsProfile) => number
  getPlayerMovementSpeedTilesPerSecond: (profile: StatsProfile) => number
  getPlayerEvadeChance: (profile: StatsProfile) => number
  shouldPlayerEvadeDamage: (profile: StatsProfile, randomValue: number) => boolean
  close: () => void
}

export const createPlayerStatEffectsLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerStatEffectsLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerStatEffectsSource, '@player-stat-effects.lua')

  const evadeChance = (profile: StatsProfile): number =>
    host.callJson<number>(
      'player_evade_chance',
      profile,
      PLAYER_LUCK_EVADE_BASE_CHANCE,
      PLAYER_LUCK_EVADE_BONUS_PER_POINT,
      PLAYER_MAX_EVADE_CHANCE
    )

  return {
    getPlayerPhysicalAttackPower: (profile: StatsProfile): number =>
      host.callJson<number>('player_physical_attack_power', profile),
    getPlayerMovementSpeedTilesPerSecond: (profile: StatsProfile): number =>
      host.callJson<number>(
        'player_movement_speed',
        profile,
        DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND,
        PLAYER_AGILITY_MOVE_SPEED_BONUS_PER_POINT,
        PLAYER_MIN_MOVE_SPEED_TILES_PER_SECOND,
        PLAYER_MAX_MOVE_SPEED_TILES_PER_SECOND
      ),
    getPlayerEvadeChance: evadeChance,
    shouldPlayerEvadeDamage: (profile: StatsProfile, randomValue: number): boolean =>
      randomValue < evadeChance(profile),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
