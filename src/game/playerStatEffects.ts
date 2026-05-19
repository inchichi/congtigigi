import { DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND } from './characterState'
import type { PlayerProfile } from './playerProfile'

export const PLAYER_AGILITY_MOVE_SPEED_BONUS_PER_POINT = 0.35
export const PLAYER_MIN_MOVE_SPEED_TILES_PER_SECOND = 4
export const PLAYER_MAX_MOVE_SPEED_TILES_PER_SECOND = 12
export const PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT = 2
export const PLAYER_LUCK_EVADE_BASE_CHANCE = 0.04
export const PLAYER_LUCK_EVADE_BONUS_PER_POINT = 0.015
export const PLAYER_MAX_EVADE_CHANCE = 0.35

export const getPlayerPhysicalAttackPower = (
  profile: Pick<PlayerProfile, 'stats'>
): number => Math.max(1, Math.floor(profile.stats.strength))

export const getPlayerMovementSpeedTilesPerSecond = (
  profile: Pick<PlayerProfile, 'stats'>
): number =>
  clamp(
    DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND +
      (profile.stats.agility - 4) * PLAYER_AGILITY_MOVE_SPEED_BONUS_PER_POINT,
    PLAYER_MIN_MOVE_SPEED_TILES_PER_SECOND,
    PLAYER_MAX_MOVE_SPEED_TILES_PER_SECOND
  )

export const getPlayerEvadeChance = (
  profile: Pick<PlayerProfile, 'stats'>
): number =>
  clamp(
    PLAYER_LUCK_EVADE_BASE_CHANCE +
      profile.stats.luck * PLAYER_LUCK_EVADE_BONUS_PER_POINT,
    0,
    PLAYER_MAX_EVADE_CHANCE
  )

export const shouldPlayerEvadeDamage = (
  profile: Pick<PlayerProfile, 'stats'>,
  randomValue = Math.random()
): boolean => randomValue < getPlayerEvadeChance(profile)

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))
