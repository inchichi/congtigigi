import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import { decodeTiledGlobalId, parseTiledMap } from './parseTiledMap'

const sampleMapXml = readFileSync(
  new URL('../../assets/maps/sample-map.tmx', import.meta.url),
  'utf8'
)

const tinyDungeonTilesetXml = readFileSync(
  new URL('../../assets/tilesets/tiny-dungeon-16.tsx', import.meta.url),
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
  it('parses the sample TMX map and external TSX tileset', () => {
    const map = parseTiledMap({
      mapXml: sampleMapXml,
      externalTilesets: {
        '../tilesets/tiny-dungeon-16.tsx': tinyDungeonTilesetXml
      }
    })

    expect(map.width).toBe(32)
    expect(map.height).toBe(20)
    expect(map.pixelWidth).toBe(512)
    expect(map.pixelHeight).toBe(320)
    expect(map.layers.map((layer) => layer.name)).toEqual([
      'Dungeon',
      'Objects',
      'Carts'
    ])
    expect(map.tilesets[0]).toMatchObject({
      firstGid: 1,
      source: '../tilesets/tiny-dungeon-16.tsx',
      name: 'tiny-dungeon-16',
      tileWidth: 16,
      tileHeight: 16,
      tileCount: 132,
      columns: 12,
      image: {
        source: 'tiny-dungeon-16.png',
        width: 192,
        height: 176
      },
      tileProperties: {
        0: {
          wall: true
        },
        4: {
          wall: true
        },
        27: {
          wall: true
        },
        59: {
          wall: true
        }
      }
    })
    expect(map.layers[0].tiles[0]).toMatchObject({
      x: 0,
      y: 0,
      gid: 14,
      localId: 13,
      flipHorizontally: false,
      flipVertically: false,
      flipDiagonally: false
    })
    expect(map.layers[0].tiles[1]).toMatchObject({
      x: 1,
      y: 0,
      gid: 51,
      localId: 50,
      flipHorizontally: false,
      flipVertically: true,
      flipDiagonally: true
    })
  })
})
