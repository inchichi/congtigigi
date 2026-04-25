export const PLAYER_CHARACTER_ID = 'player'
export const PLAYER_CHARACTER_APPEARANCE_TYPE =
  'character_adventurer_brown_hair'
export const DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND = 8

export type CharacterMoveDirection = 'up' | 'down' | 'left' | 'right'
export type CharacterAction = 'interact'
export type CharacterControllerIntent = {
  movement?: {
    x: number
    y: number
  }
  actions?: CharacterAction[]
}

export type CharacterState = {
  id: string
  appearanceType: string
  position: {
    x: number
    y: number
  }
  facing: CharacterMoveDirection
  collisionSize: {
    width: number
    height: number
  }
  blocksMovement: boolean
  controller: CharacterController
}

export type CharacterController =
  | KeyboardCharacterController
  | LuaCharacterController
  | NpcCharacterController

export type KeyboardCharacterController = {
  kind: 'keyboard'
  moveSpeedTilesPerSecond: number
}

export type NpcCharacterController = {
  kind: 'npc'
  behavior: 'idle'
  moveSpeedTilesPerSecond: number
}

export type LuaCharacterControllerConfigValue =
  | boolean
  | number
  | string
  | boolean[]
  | number[]
  | string[]

export type LuaCharacterControllerConfig = Record<
  string,
  LuaCharacterControllerConfigValue
>

export type LuaCharacterController = {
  kind: 'lua'
  scriptId: string
  radiusInTiles: number
  moveSpeedTilesPerSecond: number
  config: LuaCharacterControllerConfig
}

type CreateInitialPlayerCharacterInput = {
  mapWidth: number
  mapHeight: number
}

type CreateNpcCharacterInput = {
  id: string
  appearanceType: string
  position: {
    x: number
    y: number
  }
  facing?: CharacterMoveDirection
  collisionSize: {
    width: number
    height: number
  }
  blocksMovement?: boolean
  controller?: CharacterController
}

type MoveCharacterStateInput = {
  character: CharacterState
  delta: {
    x: number
    y: number
  }
  mapWidth: number
  mapHeight: number
}

type GetCharacterControllerIntentInput = {
  character: CharacterState
  deltaMilliseconds: number
  pressedDirections?: ReadonlySet<CharacterMoveDirection>
  triggeredActions?: ReadonlySet<CharacterAction>
}

export const createKeyboardCharacterController = ({
  moveSpeedTilesPerSecond = DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND
}: {
  moveSpeedTilesPerSecond?: number
} = {}): KeyboardCharacterController => ({
  kind: 'keyboard',
  moveSpeedTilesPerSecond
})

export const createIdleNpcCharacterController = ({
  moveSpeedTilesPerSecond = DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND
}: {
  moveSpeedTilesPerSecond?: number
} = {}): NpcCharacterController => ({
  kind: 'npc',
  behavior: 'idle',
  moveSpeedTilesPerSecond
})

export const createLuaCharacterController = ({
  scriptId,
  radiusInTiles,
  config = {},
  moveSpeedTilesPerSecond = DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND
}: {
  scriptId: string
  radiusInTiles: number
  config?: LuaCharacterControllerConfig
  moveSpeedTilesPerSecond?: number
}): LuaCharacterController => ({
  kind: 'lua',
  scriptId,
  radiusInTiles,
  moveSpeedTilesPerSecond,
  config
})

export const createInitialPlayerCharacter = ({
  mapWidth,
  mapHeight
}: CreateInitialPlayerCharacterInput): CharacterState => ({
  id: PLAYER_CHARACTER_ID,
  appearanceType: PLAYER_CHARACTER_APPEARANCE_TYPE,
  position: {
    x: Math.floor(mapWidth / 2),
    y: Math.floor(mapHeight / 2)
  },
  facing: 'down',
  collisionSize: {
    width: 1,
    height: 1
  },
  blocksMovement: true,
  controller: createKeyboardCharacterController()
})

export const createNpcCharacter = ({
  id,
  appearanceType,
  position,
  facing = 'down',
  collisionSize,
  blocksMovement = true,
  controller = createIdleNpcCharacterController()
}: CreateNpcCharacterInput): CharacterState => ({
  id,
  appearanceType,
  position,
  facing,
  collisionSize,
  blocksMovement,
  controller
})

export const getCharacterMoveDirectionFromKey = (
  key: string
): CharacterMoveDirection | undefined => {
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

export const getCharacterActionFromKey = (
  key: string
): CharacterAction | undefined => {
  switch (key) {
    case 'Enter':
    case ' ':
    case 'Space':
    case 'Spacebar':
      return 'interact'
    default:
      return undefined
  }
}

export const getCharacterControllerIntent = ({
  character,
  deltaMilliseconds,
  pressedDirections = new Set<CharacterMoveDirection>(),
  triggeredActions = new Set<CharacterAction>()
}: GetCharacterControllerIntentInput): CharacterControllerIntent | undefined => {
  switch (character.controller.kind) {
    case 'keyboard':
      return createCharacterControllerIntent({
        movement: getMovementDeltaFromDirection({
          direction: getNormalizedMovementVector(pressedDirections),
          moveSpeedTilesPerSecond: character.controller.moveSpeedTilesPerSecond,
          deltaMilliseconds
        }),
        actions: triggeredActions.has('interact') ? ['interact'] : undefined
      })
    case 'npc':
      return createCharacterControllerIntent({
        movement: getNpcControllerDelta(character.controller)
      })
    case 'lua':
      return undefined
  }
}

export const moveCharacterState = ({
  character,
  delta,
  mapWidth,
  mapHeight
}: MoveCharacterStateInput): CharacterState => {
  const nextFacing = getFacingFromDelta(character.facing, delta)
  const clampedPosition = {
    x: clamp(
      character.position.x + delta.x,
      0,
      mapWidth - character.collisionSize.width
    ),
    y: clamp(
      character.position.y + delta.y,
      0,
      mapHeight - character.collisionSize.height
    )
  }

  if (
    clampedPosition.x === character.position.x &&
    clampedPosition.y === character.position.y &&
    nextFacing === character.facing
  ) {
    return character
  }

  return {
    ...character,
    position: clampedPosition,
    facing: nextFacing
  }
}

const getMovementDeltaFromDirection = ({
  direction,
  moveSpeedTilesPerSecond,
  deltaMilliseconds
}: {
  direction: { x: number; y: number } | undefined
  moveSpeedTilesPerSecond: number
  deltaMilliseconds: number
}): { x: number; y: number } | undefined => {
  if (!direction) {
    return undefined
  }

  const distanceInTiles =
    (moveSpeedTilesPerSecond * deltaMilliseconds) / 1000

  return {
    x: direction.x * distanceInTiles,
    y: direction.y * distanceInTiles
  }
}

const getNpcControllerDelta = (
  controller: NpcCharacterController
): { x: number; y: number } | undefined => {
  switch (controller.behavior) {
    case 'idle':
      return undefined
  }
}

const createCharacterControllerIntent = ({
  movement,
  actions
}: CharacterControllerIntent): CharacterControllerIntent | undefined => {
  if (!movement && (!actions || actions.length === 0)) {
    return undefined
  }

  return {
    movement,
    actions
  }
}

const getFacingFromDelta = (
  currentFacing: CharacterMoveDirection,
  delta: { x: number; y: number }
): CharacterMoveDirection => {
  if (delta.x === 0 && delta.y === 0) {
    return currentFacing
  }

  if (Math.abs(delta.x) >= Math.abs(delta.y) && delta.x !== 0) {
    return delta.x > 0 ? 'right' : 'left'
  }

  if (delta.y !== 0) {
    return delta.y > 0 ? 'down' : 'up'
  }

  return currentFacing
}

const getNormalizedMovementVector = (
  pressedDirections: ReadonlySet<CharacterMoveDirection>
): { x: number; y: number } | undefined => {
  let x = 0
  let y = 0

  if (pressedDirections.has('left')) {
    x -= 1
  }

  if (pressedDirections.has('right')) {
    x += 1
  }

  if (pressedDirections.has('up')) {
    y -= 1
  }

  if (pressedDirections.has('down')) {
    y += 1
  }

  if (x === 0 && y === 0) {
    return undefined
  }

  const magnitude = Math.hypot(x, y)

  return {
    x: x / magnitude,
    y: y / magnitude
  }
}

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(Math.max(value, minimum), maximum)
