import type { CharacterMoveDirection } from './characterState'

export const PLAYER_ROLL_DURATION_MILLISECONDS = 260
export const PLAYER_ROLL_COOLDOWN_MILLISECONDS = 120
export const PLAYER_ROLL_DISTANCE_TILES = 3

export type PlayerRollState = {
  vector: PlayerRollVector
  startedAtMilliseconds: number
  previousDistanceTiles: number
}

export type PlayerRollVector = {
  x: number
  y: number
}

export type PlayerRollVisualState = {
  offsetX: number
  offsetY: number
  rotation: number
  scaleX: number
  scaleY: number
}

export const getPlayerRollDirectionVector = (
  direction: CharacterMoveDirection
): PlayerRollVector => {
  switch (direction) {
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

export const normalizePlayerRollVector = (
  vector: PlayerRollVector
): PlayerRollVector | undefined => {
  if (vector.x === 0 && vector.y === 0) {
    return undefined
  }

  const length = Math.hypot(vector.x, vector.y)

  return {
    x: vector.x / length,
    y: vector.y / length
  }
}

export const getPlayerRollProgress = ({
  nowMilliseconds,
  startedAtMilliseconds
}: {
  nowMilliseconds: number
  startedAtMilliseconds: number
}): number =>
  Math.min(
    1,
    Math.max(
      0,
      (nowMilliseconds - startedAtMilliseconds) /
        PLAYER_ROLL_DURATION_MILLISECONDS
    )
  )

export const getPlayerRollDistanceTiles = (progress: number): number => {
  const normalizedProgress = Math.min(1, Math.max(0, progress))

  return PLAYER_ROLL_DISTANCE_TILES * (1 - (1 - normalizedProgress) ** 3)
}

export const getPlayerRollVisualState = ({
  vector,
  progress
}: {
  vector: PlayerRollVector
  progress: number
}): PlayerRollVisualState => {
  const normalizedProgress = Math.min(1, Math.max(0, progress))
  const directionSign = vector.x < 0 || (vector.x === 0 && vector.y < 0) ? -1 : 1
  const liftAmount = Math.sin(normalizedProgress * Math.PI)

  return {
    offsetX: 0,
    offsetY: -Math.abs(liftAmount) * 4,
    rotation: directionSign * normalizedProgress * Math.PI * 2,
    scaleX: 1,
    scaleY: 1
  }
}
