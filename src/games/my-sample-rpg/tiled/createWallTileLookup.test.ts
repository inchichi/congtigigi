import { describe, expect, it } from 'vitest'

import { createWallTileLookup, isWallTileAt } from './createWallTileLookup'
import type { ParsedTiledMap } from './parseTiledMap'

describe('createWallTileLookup', () => {
  it('marks coordinates occupied by the object layer', () => {
    const map: ParsedTiledMap = {
      width: 2,
      height: 2,
      tileWidth: 16,
      tileHeight: 16,
      pixelWidth: 32,
      pixelHeight: 32,
      layers: [
        {
          id: 1,
          name: 'Ground',
          width: 2,
          height: 2,
          opacity: 1,
          visible: true,
          tiles: [
            {
              x: 0,
              y: 0,
              gid: 1,
              localId: 0,
              flipHorizontally: false,
              flipVertically: false,
              flipDiagonally: false
            },
            {
              x: 0,
              y: 0,
              gid: 2,
              localId: 1,
              flipHorizontally: false,
              flipVertically: false,
              flipDiagonally: false
            }
          ]
        },
        {
          id: 2,
          name: 'object',
          width: 2,
          height: 2,
          opacity: 1,
          visible: true,
          tiles: [
            {
              x: 1,
              y: 0,
              gid: 2,
              localId: 1,
              flipHorizontally: false,
              flipVertically: false,
              flipDiagonally: false
            }
          ]
        }
      ],
      tilesets: [
        {
          firstGid: 1,
          source: '../tilesets/test.tsx',
          name: 'test',
          tileWidth: 16,
          tileHeight: 16,
          spacing: 0,
          margin: 0,
          tileCount: 2,
          columns: 2,
          image: {
            source: 'test.png',
            width: 32,
            height: 16
          },
          tileTypes: {},
          tileProperties: {}
        }
      ],
      eventLayers: []
    }

    const wallTiles = createWallTileLookup(map)

    expect(isWallTileAt(wallTiles, 0, 0)).toBe(false)
    expect(isWallTileAt(wallTiles, 1, 0)).toBe(true)
    expect(isWallTileAt(wallTiles, 0, 1)).toBe(false)
  })
})
