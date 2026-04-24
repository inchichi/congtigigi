import { describe, expect, it } from 'vitest'

import {
  DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND,
  PLAYER_CHARACTER_APPEARANCE_TYPE,
  PLAYER_CHARACTER_ID,
  createIdleNpcCharacterController,
  createInitialPlayerCharacter,
  createKeyboardCharacterController,
  createLuaCharacterController,
  createNpcCharacter,
  getCharacterControllerDelta,
  getCharacterMoveDirectionFromKey,
  moveCharacterState
} from './characterState'

describe('createInitialPlayerCharacter', () => {
  it('starts the player at the floored center of the map', () => {
    expect(
      createInitialPlayerCharacter({
        mapWidth: 32,
        mapHeight: 20
      })
    ).toEqual({
      id: PLAYER_CHARACTER_ID,
      appearanceType: PLAYER_CHARACTER_APPEARANCE_TYPE,
      position: {
        x: 16,
        y: 10
      },
      collisionSize: {
        width: 1,
        height: 1
      },
      blocksMovement: true,
      controller: createKeyboardCharacterController()
    })
  })
})

describe('getCharacterMoveDirectionFromKey', () => {
  it('maps arrow keys to movement directions', () => {
    expect(getCharacterMoveDirectionFromKey('ArrowUp')).toBe('up')
    expect(getCharacterMoveDirectionFromKey('ArrowDown')).toBe('down')
    expect(getCharacterMoveDirectionFromKey('ArrowLeft')).toBe('left')
    expect(getCharacterMoveDirectionFromKey('ArrowRight')).toBe('right')
  })

  it('ignores unrelated keys', () => {
    expect(getCharacterMoveDirectionFromKey('KeyW')).toBeUndefined()
  })
})

describe('getCharacterControllerDelta', () => {
  it('moves keyboard-controlled characters with a continuous delta', () => {
    expect(
      getCharacterControllerDelta({
        character: createInitialPlayerCharacter({
          mapWidth: 32,
          mapHeight: 20
        }),
        deltaMilliseconds: 250,
        pressedDirections: new Set(['left', 'down'])
      })
    ).toEqual({
      x:
        (-DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND / 4) /
        Math.SQRT2,
      y:
        (DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND / 4) /
        Math.SQRT2
    })
  })

  it('keeps idle npc controllers still', () => {
    expect(
      getCharacterControllerDelta({
        character: createNpcCharacter({
          id: 'blacksmith',
          appearanceType: 'character_bearded_apron_man',
          position: {
            x: 10,
            y: 12
          },
          collisionSize: {
            width: 1,
            height: 1
          }
        }),
        deltaMilliseconds: 250
      })
    ).toBeUndefined()
  })

  it('delegates lua-controlled characters to an external controller runtime', () => {
    expect(
      getCharacterControllerDelta({
        character: {
          ...createNpcCharacter({
            id: 'villager_1',
            appearanceType: 'character_commoner_tan_tunic',
            position: {
              x: 10,
              y: 12
            },
            collisionSize: {
              width: 1,
              height: 1
            }
          }),
          controller: createLuaCharacterController({
            scriptId: 'wander-near-home',
            radiusInTiles: 2,
            moveSpeedTilesPerSecond: 4
          })
        },
        deltaMilliseconds: 250,
        resolveLuaControllerDirection: () => ({
          x: 0,
          y: 1
        })
      })
    ).toEqual({
      x: 0,
      y: 1
    })
  })
})

describe('moveCharacterState', () => {
  it('moves a character by a continuous delta', () => {
    expect(
      moveCharacterState({
        character: createNpcCharacter({
          id: 'villager',
          appearanceType: 'character_commoner_tan_tunic',
          position: {
            x: 16,
            y: 10
          },
          collisionSize: {
            width: 1,
            height: 1
          },
          controller: createIdleNpcCharacterController()
        }),
        delta: {
          x: -2,
          y: 1
        },
        mapWidth: 32,
        mapHeight: 20
      })
    ).toMatchObject({
      position: {
        x: 14,
        y: 11
      }
    })
  })

  it('clamps the full character footprint to the map bounds', () => {
    expect(
      moveCharacterState({
        character: createNpcCharacter({
          id: 'guard',
          appearanceType: 'character_knight_closed_helmet',
          position: {
            x: 31,
            y: 19
          },
          collisionSize: {
            width: 1.5,
            height: 2
          }
        }),
        delta: {
          x: 1,
          y: 1
        },
        mapWidth: 32,
        mapHeight: 20
      })
    ).toMatchObject({
      position: {
        x: 30.5,
        y: 18
      }
    })
  })
})
