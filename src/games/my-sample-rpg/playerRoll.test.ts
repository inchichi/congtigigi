import { describe, expect, it } from 'vitest'

import {
  PLAYER_ROLL_DISTANCE_TILES,
  PLAYER_ROLL_DURATION_MILLISECONDS,
  getPlayerRollDirectionVector,
  getPlayerRollDistanceTiles,
  getPlayerRollProgress,
  getPlayerRollVisualState,
  normalizePlayerRollVector
} from './playerRoll'

describe('playerRoll', () => {
  it('resolves movement vectors by direction', () => {
    expect(getPlayerRollDirectionVector('up')).toEqual({ x: 0, y: -1 })
    expect(getPlayerRollDirectionVector('down')).toEqual({ x: 0, y: 1 })
    expect(getPlayerRollDirectionVector('left')).toEqual({ x: -1, y: 0 })
    expect(getPlayerRollDirectionVector('right')).toEqual({ x: 1, y: 0 })
  })

  it('normalizes diagonal movement vectors', () => {
    const vector = normalizePlayerRollVector({ x: 1, y: 1 })

    expect(vector?.x).toBeCloseTo(Math.SQRT1_2)
    expect(vector?.y).toBeCloseTo(Math.SQRT1_2)
    expect(normalizePlayerRollVector({ x: 0, y: 0 })).toBeUndefined()
  })

  it('clamps progress to the roll duration', () => {
    expect(
      getPlayerRollProgress({
        nowMilliseconds: 50,
        startedAtMilliseconds: 100
      })
    ).toBe(0)
    expect(
      getPlayerRollProgress({
        nowMilliseconds: 100 + PLAYER_ROLL_DURATION_MILLISECONDS,
        startedAtMilliseconds: 100
      })
    ).toBe(1)
  })

  it('eases distance to the configured roll distance', () => {
    expect(getPlayerRollDistanceTiles(0)).toBe(0)
    expect(getPlayerRollDistanceTiles(1)).toBe(PLAYER_ROLL_DISTANCE_TILES)
    expect(getPlayerRollDistanceTiles(0.5)).toBeGreaterThan(
      PLAYER_ROLL_DISTANCE_TILES * 0.5
    )
  })

  it('rotates one full turn toward the roll direction', () => {
    expect(
      getPlayerRollVisualState({
        vector: { x: 1, y: 0 },
        progress: 1
      }).rotation
    ).toBeCloseTo(Math.PI * 2)
    expect(
      getPlayerRollVisualState({
        vector: { x: -1, y: 0 },
        progress: 1
      }).rotation
    ).toBeCloseTo(-Math.PI * 2)
  })
})
