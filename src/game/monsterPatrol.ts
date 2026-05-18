import type { CharacterState } from './characterState'

export type MonsterPatrolState = {
  homePosition: {
    x: number
    y: number
  }
  targetPosition: {
    x: number
    y: number
  }
  pauseUntilMilliseconds: number
}

export type StepMonsterPatrolInput = {
  character: CharacterState
  patrolState: MonsterPatrolState
  deltaMilliseconds: number
  nowMilliseconds: number
  mapWidth: number
  mapHeight: number
  radiusInTiles?: number
  speedTilesPerSecond?: number
  random?: () => number
}

export type StepMonsterPatrolResult = {
  patrolState: MonsterPatrolState
  movement?: {
    x: number
    y: number
  }
}

const DEFAULT_MONSTER_PATROL_RADIUS_TILES = 4
const DEFAULT_MONSTER_PATROL_SPEED_TILES_PER_SECOND = 2.4
const DEFAULT_MONSTER_PATROL_PAUSE_MIN_MILLISECONDS = 250
const DEFAULT_MONSTER_PATROL_PAUSE_MAX_MILLISECONDS = 700
const MONSTER_PATROL_ARRIVAL_EPSILON = 0.03

export const createMonsterPatrolState = (
  character: CharacterState
): MonsterPatrolState => ({
  homePosition: {
    x: character.position.x,
    y: character.position.y
  },
  targetPosition: {
    x: character.position.x,
    y: character.position.y
  },
  pauseUntilMilliseconds: 0
})

export const stepMonsterPatrol = ({
  character,
  patrolState,
  deltaMilliseconds,
  nowMilliseconds,
  mapWidth,
  mapHeight,
  radiusInTiles = DEFAULT_MONSTER_PATROL_RADIUS_TILES,
  speedTilesPerSecond = DEFAULT_MONSTER_PATROL_SPEED_TILES_PER_SECOND,
  random = Math.random
}: StepMonsterPatrolInput): StepMonsterPatrolResult => {
  if (nowMilliseconds < patrolState.pauseUntilMilliseconds) {
    return {
      patrolState
    }
  }

  let nextTargetPosition = patrolState.targetPosition
  const currentTargetDistance = getDistanceBetweenPoints(
    character.position,
    nextTargetPosition
  )

  if (currentTargetDistance <= MONSTER_PATROL_ARRIVAL_EPSILON) {
    nextTargetPosition = choosePatrolTarget({
      homePosition: patrolState.homePosition,
      radiusInTiles,
      mapWidth,
      mapHeight,
      collisionWidth: character.collisionSize.width,
      collisionHeight: character.collisionSize.height,
      random
    })
  }

  const remainingDistance = getDistanceBetweenPoints(
    character.position,
    nextTargetPosition
  )

  if (remainingDistance <= MONSTER_PATROL_ARRIVAL_EPSILON) {
    return {
      patrolState: {
        ...patrolState,
        targetPosition: nextTargetPosition,
        pauseUntilMilliseconds:
          nowMilliseconds +
          choosePatrolPauseDurationMilliseconds({ random })
      }
    }
  }

  const stepDistance =
    (speedTilesPerSecond * deltaMilliseconds) / 1000

  if (stepDistance <= 0) {
    return {
      patrolState: {
        ...patrolState,
        targetPosition: nextTargetPosition
      }
    }
  }

  const movementDistance = Math.min(stepDistance, remainingDistance)
  const direction = normalizeVector({
    x: nextTargetPosition.x - character.position.x,
    y: nextTargetPosition.y - character.position.y
  })

  return {
    patrolState: {
      ...patrolState,
      targetPosition: nextTargetPosition,
      pauseUntilMilliseconds:
        movementDistance + MONSTER_PATROL_ARRIVAL_EPSILON >= remainingDistance
          ? nowMilliseconds +
            choosePatrolPauseDurationMilliseconds({ random })
          : patrolState.pauseUntilMilliseconds
    },
    movement: {
      x: direction.x * movementDistance,
      y: direction.y * movementDistance
    }
  }
}

const choosePatrolTarget = ({
  homePosition,
  radiusInTiles,
  mapWidth,
  mapHeight,
  collisionWidth,
  collisionHeight,
  random
}: {
  homePosition: {
    x: number
    y: number
  }
  radiusInTiles: number
  mapWidth: number
  mapHeight: number
  collisionWidth: number
  collisionHeight: number
  random: () => number
}): {
  x: number
  y: number
} => {
  const angle = random() * Math.PI * 2
  const distance = Math.max(
    radiusInTiles * 0.35,
    Math.sqrt(random()) * radiusInTiles
  )
  const maxX = Math.max(0, mapWidth - collisionWidth)
  const maxY = Math.max(0, mapHeight - collisionHeight)

  return {
    x: clamp(
      homePosition.x + Math.cos(angle) * distance,
      0,
      maxX
    ),
    y: clamp(
      homePosition.y + Math.sin(angle) * distance,
      0,
      maxY
    )
  }
}

const choosePatrolPauseDurationMilliseconds = ({
  random
}: {
  random: () => number
}): number => {
  const span =
    DEFAULT_MONSTER_PATROL_PAUSE_MAX_MILLISECONDS -
    DEFAULT_MONSTER_PATROL_PAUSE_MIN_MILLISECONDS

  return (
    DEFAULT_MONSTER_PATROL_PAUSE_MIN_MILLISECONDS +
    Math.round(random() * span)
  )
}

const normalizeVector = ({
  x,
  y
}: {
  x: number
  y: number
}): {
  x: number
  y: number
} => {
  const length = Math.hypot(x, y)

  if (length === 0) {
    return {
      x: 0,
      y: 0
    }
  }

  return {
    x: x / length,
    y: y / length
  }
}

const getDistanceBetweenPoints = (
  left: {
    x: number
    y: number
  },
  right: {
    x: number
    y: number
  }
): number => Math.hypot(right.x - left.x, right.y - left.y)

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(value, max))
