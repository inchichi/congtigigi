import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import { decodeTiledGlobalId, parseTiledMap } from './parseTiledMap'

const townMapXml = readFileSync(
  new URL('../../assets/maps/town.tmx', import.meta.url),
  'utf8'
)

const townTilesetXml = readFileSync(
  new URL('../../assets/tilesets/town-32.tsx', import.meta.url),
  'utf8'
)

describe('decodeTiledGlobalId', () => {
  it('clears transform bits and returns flip flags', () => {
    const tile = decodeTiledGlobalId(14 | 0x80000000 | 0x40000000 | 0x20000000)

    expect(tile).toEqual({
      gid: 14,
      flipHorizontally: true,
      flipVertically: true,
      flipDiagonally: true
    })
  })
})

describe('parseTiledMap', () => {
  it('parses the town TMX map and external TSX tilesets', () => {
    const map = parseTiledMap({
      mapXml: townMapXml,
      externalTilesets: {
        '../tilesets/town-32.tsx': townTilesetXml
      }
    })

    expect(map.width).toBe(50)
    expect(map.height).toBe(50)
    expect(map.pixelWidth).toBe(1600)
    expect(map.pixelHeight).toBe(1600)
    expect(map.layers.map((layer) => layer.name)).toEqual([
      'ground',
      'shadow_lower',
      'object',
      'shadow_upper',
      'object_upper',
      'deco',
      'roof'
    ])
    expect(map.tilesets).toHaveLength(1)
    expect(map.tilesets[0]).toMatchObject({
      firstGid: 1,
      source: '../tilesets/town-32.tsx',
      name: 'town-32',
      tileWidth: 32,
      tileHeight: 32,
      tileCount: 560,
      columns: 8,
      image: {
        source: 'town-32.png',
        width: 256,
        height: 2240
      },
      tileProperties: {}
    })
    expect(map.layers[0].tiles[0]).toMatchObject({
      x: 0,
      y: 0,
      gid: 55,
      localId: 54,
      flipHorizontally: false,
      flipVertically: false,
      flipDiagonally: false
    })
    expect(map.layers[2].tiles[0]).toMatchObject({
      x: 17,
      y: 3,
      gid: 82,
      localId: 81,
      flipHorizontally: false,
      flipVertically: false,
      flipDiagonally: false
    })
  })
})
