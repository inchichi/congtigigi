import { describe, expect, it } from 'vitest'

import { createNpcCharacter } from './characterState'
import {
  createMonsterPatrolState,
  stepMonsterPatrol
} from './monsterPatrol'

const createRandomSequence = (values: number[]) => {
  let index = 0

  return () => {
    const value = values[index] ?? values[values.length - 1] ?? 0.5

    index += 1

    return value
  }
}

describe('createMonsterPatrolState', () => {
  it('uses the current position as the patrol home and target', () => {
    const character = createNpcCharacter({
      id: 'monster',
      appearanceType: 'monster_pig',
      level: 3,
      position: {
        x: 10,
        y: 8
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })

    expect(createMonsterPatrolState(character)).toEqual({
      homePosition: {
        x: 10,
        y: 8
      },
      targetPosition: {
        x: 10,
        y: 8
      },
      pauseUntilMilliseconds: 0
    })
  })
})

describe('stepMonsterPatrol', () => {
  it('keeps the monster paused until the delay expires', () => {
    const character = createNpcCharacter({
      id: 'monster',
      appearanceType: 'monster_pig',
      level: 3,
      position: {
        x: 10,
        y: 8
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })

    expect(
      stepMonsterPatrol({
        character,
        patrolState: {
          homePosition: {
            x: 10,
            y: 8
          },
          targetPosition: {
            x: 12,
            y: 8
          },
          pauseUntilMilliseconds: 500
        },
        deltaMilliseconds: 16,
        nowMilliseconds: 200,
        mapWidth: 32,
        mapHeight: 20
      })
    ).toEqual({
      patrolState: {
        homePosition: {
          x: 10,
          y: 8
        },
        targetPosition: {
          x: 12,
          y: 8
        },
        pauseUntilMilliseconds: 500
      }
    })
  })

  it('moves toward a new target inside the patrol radius', () => {
    const character = createNpcCharacter({
      id: 'monster',
      appearanceType: 'monster_pig',
      level: 3,
      position: {
        x: 10,
        y: 8
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })

    const nextStep = stepMonsterPatrol({
      character,
      patrolState: createMonsterPatrolState(character),
      deltaMilliseconds: 250,
      nowMilliseconds: 1000,
      mapWidth: 32,
      mapHeight: 20,
      random: createRandomSequence([0.25, 0.5, 0.75])
    })

    expect(nextStep.movement).toBeDefined()
    expect(
      Math.hypot(
        nextStep.patrolState.targetPosition.x -
          nextStep.patrolState.homePosition.x,
        nextStep.patrolState.targetPosition.y -
          nextStep.patrolState.homePosition.y
      )
    ).toBeLessThanOrEqual(4)
  })
})
