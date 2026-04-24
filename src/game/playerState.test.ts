import { describe, expect, it } from 'vitest'

import {
  HERO_TILE_LOCAL_ID,
  PLAYER_MOVE_SPEED_TILES_PER_SECOND,
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
  it('moves the player by a continuous delta', () => {
    expect(
      movePlayerState({
        player: {
          tileLocalId: HERO_TILE_LOCAL_ID,
          position: {
            x: 16,
            y: 10
          }
        },
        delta: {
          x: -PLAYER_MOVE_SPEED_TILES_PER_SECOND / 4,
          y: PLAYER_MOVE_SPEED_TILES_PER_SECOND / 8
        },
        mapWidth: 32,
        mapHeight: 20
      })
    ).toEqual({
      tileLocalId: HERO_TILE_LOCAL_ID,
      position: {
        x: 14,
        y: 11
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
        delta: {
          x: -1,
          y: -1
        },
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

  it('keeps the full player footprint inside the map bounds', () => {
    expect(
      movePlayerState({
        player: {
          tileLocalId: HERO_TILE_LOCAL_ID,
          position: {
            x: 31,
            y: 19
          }
        },
        delta: {
          x: 1,
          y: 1
        },
        mapWidth: 32,
        mapHeight: 20,
        playerWidth: 1,
        playerHeight: 1
      })
    ).toEqual({
      tileLocalId: HERO_TILE_LOCAL_ID,
      position: {
        x: 31,
        y: 19
      }
    })
  })
})
