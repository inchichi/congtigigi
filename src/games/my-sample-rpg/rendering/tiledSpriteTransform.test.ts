import { describe, expect, it } from 'vitest'

import {
  getSpriteTransformForTile,
  hasTileTransform
} from './tiledSpriteTransform'

describe('hasTileTransform', () => {
  it('detects whether a tile uses Tiled flip flags', () => {
    expect(
      hasTileTransform({
        x: 0,
        y: 0,
        gid: 1,
        localId: 0,
        flipHorizontally: false,
        flipVertically: false,
        flipDiagonally: false
      })
    ).toBe(false)

    expect(
      hasTileTransform({
        x: 0,
        y: 0,
        gid: 1,
        localId: 0,
        flipHorizontally: false,
        flipVertically: true,
        flipDiagonally: true
      })
    ).toBe(true)
  })
})

describe('getSpriteTransformForTile', () => {
  it('rotates the v+d combination used by the 50-52 wall tiles', () => {
    expect(
      getSpriteTransformForTile({
        x: 0,
        y: 0,
        gid: 51,
        localId: 50,
        flipHorizontally: false,
        flipVertically: true,
        flipDiagonally: true
      })
    ).toEqual({
      rotation: -Math.PI / 2,
      scaleX: 1,
      scaleY: 1
    })
  })

  it('rotates the h+d combination in the opposite direction', () => {
    expect(
      getSpriteTransformForTile({
        x: 0,
        y: 0,
        gid: 50,
        localId: 49,
        flipHorizontally: true,
        flipVertically: false,
        flipDiagonally: true
      })
    ).toEqual({
      rotation: Math.PI / 2,
      scaleX: 1,
      scaleY: 1
    })
  })

  it('preserves diagonal reflections without losing the mirror sign', () => {
    expect(
      getSpriteTransformForTile({
        x: 0,
        y: 0,
        gid: 50,
        localId: 49,
        flipHorizontally: true,
        flipVertically: true,
        flipDiagonally: true
      })
    ).toEqual({
      rotation: -Math.PI / 2,
      scaleX: 1,
      scaleY: -1
    })
  })
})
