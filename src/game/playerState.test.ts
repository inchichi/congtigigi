import { describe, expect, it } from 'vitest'

import {
  HERO_TILE_LOCAL_ID,
  createInitialPlayerState,
  getPlayerMoveDirectionFromKey,
  movePlayerState
} from './playerState'

describe('createInitialPlayerState', () => {
  it('starts the player at the floored center of the map', () => {
    expect(
      createInitialPlayerState({
        mapWidth: 32,
        mapHeight: 20
      })
    ).toEqual({
      tileLocalId: HERO_TILE_LOCAL_ID,
      position: {
        x: 16,
        y: 10
      }
    })
  })
})

describe('getPlayerMoveDirectionFromKey', () => {
  it('maps arrow keys to movement directions', () => {
    expect(getPlayerMoveDirectionFromKey('ArrowUp')).toBe('up')
    expect(getPlayerMoveDirectionFromKey('ArrowDown')).toBe('down')
    expect(getPlayerMoveDirectionFromKey('ArrowLeft')).toBe('left')
    expect(getPlayerMoveDirectionFromKey('ArrowRight')).toBe('right')
  })

  it('ignores unrelated keys', () => {
    expect(getPlayerMoveDirectionFromKey('KeyW')).toBeUndefined()
  })
})

describe('movePlayerState', () => {
  it('moves the player one tile in the requested direction', () => {
    expect(
      movePlayerState({
        player: {
          tileLocalId: HERO_TILE_LOCAL_ID,
          position: {
            x: 16,
            y: 10
          }
        },
        direction: 'left',
        mapWidth: 32,
        mapHeight: 20
      })
    ).toEqual({
      tileLocalId: HERO_TILE_LOCAL_ID,
      position: {
        x: 15,
        y: 10
      }
    })
  })

  it('clamps the player to the map bounds', () => {
    expect(
      movePlayerState({
        player: {
          tileLocalId: HERO_TILE_LOCAL_ID,
          position: {
            x: 0,
            y: 0
          }
        },
        direction: 'up',
        mapWidth: 32,
        mapHeight: 20
      })
    ).toEqual({
      tileLocalId: HERO_TILE_LOCAL_ID,
      position: {
        x: 0,
        y: 0
      }
    })
  })
})
