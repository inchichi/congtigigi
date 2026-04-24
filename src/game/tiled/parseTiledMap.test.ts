import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

import {
  decodeTiledGlobalId,
  parseTiledMap,
  parseTiledTileset
} from './parseTiledMap'

const townMapXml = readFileSync(
  new URL('../../assets/maps/town.tmx', import.meta.url),
  'utf8'
)

const townTilesetXml = readFileSync(
  new URL('../../assets/tilesets/town-32.tsx', import.meta.url),
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
    expect(map.eventLayers.map((layer) => layer.name)).toEqual(['characters'])
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
      tileTypes: {},
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
    expect(map.eventLayers[0]).toMatchObject({
      id: 8,
      name: 'characters',
      opacity: 1,
      visible: true
    })
    expect(map.eventLayers[0].events[0]).toMatchObject({
      id: 1,
      name: 'blacksmith',
      className: 'character',
      width: 0,
      height: 0,
      visible: true,
      properties: {
        type: 'character_bearded_apron_man'
      },
      appearanceType: 'character_bearded_apron_man'
    })
    expect(map.eventLayers[0].events[0].x).toBeGreaterThan(0)
    expect(map.eventLayers[0].events[0].y).toBeGreaterThan(0)
  })

  it('parses tileset tile types for character appearance lookup', () => {
    const tileset = parseTiledTileset({
      firstGid: 1,
      source: '../tilesets/tiny-dungeon-16.tsx',
      tilesetXml: tinyDungeonTilesetXml
    })

    expect(tileset.tileTypes[98]).toBe('character_adventurer_brown_hair')
    expect(tileset.tileTypes[110]).toBe('monster_crab_red')
    expect(tileset.tileTypes[124]).toBe('monster_scorpion_gray')
  })

  it('parses event layers and character appearance from object groups', () => {
    const map = parseTiledMap({
      mapXml: `<?xml version="1.0" encoding="UTF-8"?>
<map version="1.10" tiledversion="1.12.1" orientation="orthogonal" renderorder="right-down" width="1" height="1" tilewidth="32" tileheight="32" infinite="0">
  <tileset firstgid="1" source="../tilesets/town-32.tsx"/>
  <layer id="1" name="ground" width="1" height="1">
    <data encoding="csv">0</data>
  </layer>
  <objectgroup id="2" name="characters">
    <object id="7" name="blacksmith" type="character" x="400.5" y="496.25" width="24" height="28">
      <properties>
        <property name="type" value="character_bearded_apron_man"/>
      </properties>
    </object>
  </objectgroup>
</map>`,
      externalTilesets: {
        '../tilesets/town-32.tsx': townTilesetXml
      }
    })

    expect(map.eventLayers).toHaveLength(1)
    expect(map.eventLayers[0]).toMatchObject({
      id: 2,
      name: 'characters',
      opacity: 1,
      visible: true
    })
    expect(map.eventLayers[0].events[0]).toMatchObject({
      id: 7,
      name: 'blacksmith',
      className: 'character',
      x: 400.5,
      y: 496.25,
      width: 24,
      height: 28,
      visible: true,
      properties: {
        type: 'character_bearded_apron_man'
      },
      appearanceType: 'character_bearded_apron_man'
    })
  })
})
