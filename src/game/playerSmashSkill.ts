import type { CharacterMoveDirection } from './characterState'

export const PLAYER_SMASH_SKILL_ID = 'smash'
export const PLAYER_SMASH_SKILL_SEGMENT_COUNT = 3
export const PLAYER_SMASH_SKILL_FIRST_SEGMENT_DISTANCE_PIXELS = 22
export const PLAYER_SMASH_SKILL_SEGMENT_SPACING_PIXELS = 14
export const PLAYER_SMASH_SKILL_LAUNCH_OFFSET_PIXELS = 28
export const PLAYER_SMASH_SKILL_TRAVEL_DISTANCE_PIXELS = 96
export const PLAYER_SMASH_SKILL_SEGMENT_STAGGER_MILLISECONDS = 45
export const PLAYER_SMASH_SKILL_SEGMENT_DURATION_MILLISECONDS = 360
export const PLAYER_SMASH_SKILL_COOLDOWN_MILLISECONDS = 300
export const PLAYER_SMASH_SKILL_EFFECT_ANIMATION_SPEED = 0.38
export const PLAYER_SMASH_SKILL_EFFECT_BASE_SCALE_X = 0.3
export const PLAYER_SMASH_SKILL_EFFECT_BASE_SCALE_Y = 0.26
export const PLAYER_SMASH_SKILL_EFFECT_SCALE_STEP = 0.025
export const PLAYER_SMASH_SKILL_EFFECT_SCALE_GROWTH = 0.08

export type PlayerSmashSkillSegmentPlacement = {
  x: number
  y: number
  scaleX: number
  scaleY: number
  alpha: number
}

export const getPlayerSmashSkillDirectionVector = (
  facing: CharacterMoveDirection
): {
  x: number
  y: number
} => {
  switch (facing) {
    case 'up':
      return { x: 0, y: -1 }
    case 'down':
      return { x: 0, y: 1 }
    case 'left':
      return { x: -1, y: 0 }
    case 'right':
      return { x: 1, y: 0 }
    default:
      return { x: 0, y: 0 }
  }
}

export const getPlayerSmashSkillSegmentDistancePixels = (
  segmentIndex: number,
  progress = 0
): number =>
  PLAYER_SMASH_SKILL_LAUNCH_OFFSET_PIXELS +
  PLAYER_SMASH_SKILL_FIRST_SEGMENT_DISTANCE_PIXELS +
  segmentIndex * PLAYER_SMASH_SKILL_SEGMENT_SPACING_PIXELS +
  (1 - (1 - Math.min(1, Math.max(0, progress))) ** 2) *
    PLAYER_SMASH_SKILL_TRAVEL_DISTANCE_PIXELS

export const getPlayerSmashSkillSegmentPlacement = ({
  characterCenterX,
  characterCenterY,
  facing,
  segmentIndex,
  progress = 0
}: {
  characterCenterX: number
  characterCenterY: number
  facing: CharacterMoveDirection
  segmentIndex: number
  progress?: number
}): PlayerSmashSkillSegmentPlacement => {
  const direction = getPlayerSmashSkillDirectionVector(facing)
  const normalizedProgress = Math.min(1, Math.max(0, progress))
  const distance = getPlayerSmashSkillSegmentDistancePixels(
    segmentIndex,
    normalizedProgress
  )
  const pulseProgress = Math.sin(normalizedProgress * Math.PI)
  const growthMultiplier =
    1 + pulseProgress * PLAYER_SMASH_SKILL_EFFECT_SCALE_GROWTH
  const baseScaleX =
    PLAYER_SMASH_SKILL_EFFECT_BASE_SCALE_X +
    segmentIndex * PLAYER_SMASH_SKILL_EFFECT_SCALE_STEP
  const baseScaleY =
    PLAYER_SMASH_SKILL_EFFECT_BASE_SCALE_Y +
    segmentIndex * PLAYER_SMASH_SKILL_EFFECT_SCALE_STEP
  const fadeOutStartProgress = 0.78
  const alpha =
    normalizedProgress < fadeOutStartProgress
      ? 1
      : Math.max(
          0,
          1 -
            (normalizedProgress - fadeOutStartProgress) /
              (1 - fadeOutStartProgress)
        )

  return {
    x: characterCenterX + direction.x * distance,
    y: characterCenterY + direction.y * distance,
    scaleX: baseScaleX * growthMultiplier,
    scaleY: baseScaleY * growthMultiplier,
    alpha
  }
}
