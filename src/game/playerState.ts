export const HERO_TILE_LOCAL_ID = 98

export type PlayerMoveDirection = 'up' | 'down' | 'left' | 'right'

export type PlayerState = {
  tileLocalId: number
  position: {
    x: number
    y: number
  }
}

type CreateInitialPlayerStateInput = {
  mapWidth: number
  mapHeight: number
}

type MovePlayerStateInput = {
  player: PlayerState
  direction: PlayerMoveDirection
  mapWidth: number
  mapHeight: number
}

export const createInitialPlayerState = ({
  mapWidth,
  mapHeight
}: CreateInitialPlayerStateInput): PlayerState => ({
  tileLocalId: HERO_TILE_LOCAL_ID,
  position: {
    x: Math.floor(mapWidth / 2),
    y: Math.floor(mapHeight / 2)
  }
})

export const getPlayerMoveDirectionFromKey = (
  key: string
): PlayerMoveDirection | undefined => {
  switch (key) {
    case 'ArrowUp':
      return 'up'
    case 'ArrowDown':
      return 'down'
    case 'ArrowLeft':
      return 'left'
    case 'ArrowRight':
      return 'right'
    default:
      return undefined
  }
}

export const movePlayerState = ({
  player,
  direction,
  mapWidth,
  mapHeight
}: MovePlayerStateInput): PlayerState => {
  const nextPosition = {
    x: player.position.x,
    y: player.position.y
  }

  switch (direction) {
    case 'up':
      nextPosition.y -= 1
      break
    case 'down':
      nextPosition.y += 1
      break
    case 'left':
      nextPosition.x -= 1
      break
    case 'right':
      nextPosition.x += 1
      break
  }

  const clampedPosition = {
    x: clamp(nextPosition.x, 0, mapWidth - 1),
    y: clamp(nextPosition.y, 0, mapHeight - 1)
  }

  if (
    clampedPosition.x === player.position.x &&
    clampedPosition.y === player.position.y
  ) {
    return player
  }

  return {
    ...player,
    position: clampedPosition
  }
}

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(Math.max(value, minimum), maximum)
