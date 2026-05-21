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

const huntingGroundMapXml = readFileSync(
  new URL('../../assets/maps/hunting-ground.tmx', import.meta.url),
  'utf8'
)

const caveMapXml = readFileSync(
  new URL('../../assets/maps/cave.tmx', import.meta.url),
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
    expect(map.eventLayers.map((layer) => layer.name)).toEqual([
      'characters',
      'portals'
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
        type: 'character_bearded_apron_man',
        'controller.scriptId': 'reply-with-message',
        'controller.messageDurationSeconds': 2.8,
        'controller.dialogueLines': [
          '힘세고 강한 아침, 만일 내게 물어보면 I AM 대장장이.',
          '나중에는 YOU 에게 무기도 팔게 될 거야.'
        ]
      },
      appearanceType: 'character_bearded_apron_man'
    })
    expect(map.eventLayers[0].events[0].x).toBeGreaterThan(0)
    expect(map.eventLayers[0].events[0].y).toBeGreaterThan(0)
    expect(map.eventLayers[1]).toMatchObject({
      id: 9,
      name: 'portals',
      opacity: 1,
      visible: true
    })
    expect(map.eventLayers[1].events[0]).toMatchObject({
      id: 8,
      name: 'east_gate',
      className: 'portal',
      x: 1568,
      y: 544,
      width: 32,
      height: 64,
      visible: true,
      properties: {
        appearanceType: 'stairs_stone_step_base_00',
        targetFacing: 'left',
        targetSceneId: 'hunting-ground',
        targetSpawnTileX: 2,
        targetSpawnTileY: 10
      }
    })
  })

  it('parses the beginner hunting ground TMX map', () => {
    const map = parseTiledMap({
      mapXml: huntingGroundMapXml,
      externalTilesets: {
        '../tilesets/town-32.tsx': townTilesetXml
      }
    })

    expect(map.width).toBe(50)
    expect(map.height).toBe(50)
    expect(map.pixelWidth).toBe(1600)
    expect(map.pixelHeight).toBe(1600)
    expect(map.layers).toHaveLength(1)
    expect(map.layers[0].name).toBe('ground')
    expect(map.layers[0].tiles[0]).toMatchObject({
      x: 0,
      y: 0,
      gid: 517,
      localId: 516,
      flipHorizontally: false,
      flipVertically: false,
      flipDiagonally: false
    })
    expect(map.eventLayers.map((layer) => layer.name)).toEqual([
      'characters',
      'portals'
    ])
    expect(map.eventLayers[0].events).toHaveLength(5)
    expect(map.eventLayers[0].events).toMatchObject([
      {
        id: 5,
        name: '꿀꿀이-1',
        className: 'character',
        x: 224,
        y: 160,
        width: 32,
        height: 32,
        visible: true,
        properties: {
          blocksMovement: true,
          'monster.level': 3,
          type: 'monster_pig'
        },
        appearanceType: 'monster_pig'
      },
      {
        id: 6,
        name: '말캉이-1',
        className: 'character',
        x: 416,
        y: 224,
        width: 32,
        height: 32,
        visible: true,
        properties: {
          blocksMovement: true,
          'monster.level': 1,
          type: 'monster_slime'
        },
        appearanceType: 'monster_slime'
      },
      {
        id: 8,
        name: '꿀꿀이-2',
        className: 'character',
        x: 832,
        y: 160,
        width: 32,
        height: 32,
        visible: true,
        properties: {
          blocksMovement: true,
          'monster.level': 3,
          type: 'monster_pig'
        },
        appearanceType: 'monster_pig'
      },
      {
        id: 9,
        name: '말캉이-2',
        className: 'character',
        x: 512,
        y: 512,
        width: 32,
        height: 32,
        visible: true,
        properties: {
          blocksMovement: true,
          'monster.level': 1,
          type: 'monster_slime'
        },
        appearanceType: 'monster_slime'
      },
      {
        id: 11,
        name: 'cave_entrance_sign',
        className: 'character',
        x: 1440,
        y: 384,
        width: 32,
        height: 32,
        visible: true,
        properties: {
          blocksMovement: false,
          type: 'sign_inn',
          displayText: '동굴입구'
        },
        appearanceType: 'sign_inn'
      }
    ])
    expect(map.eventLayers[1].events[0]).toMatchObject({
      id: 4,
      name: 'return_gate',
      className: 'portal',
      width: 32,
      height: 64,
      properties: {
        appearanceType: 'stairs_stone_step_base_00',
        targetFacing: 'right',
        targetSceneId: 'town',
        targetSpawnTileX: 46,
        targetSpawnTileY: 17
      }
    })
    expect(map.eventLayers[1].events[1]).toMatchObject({
      id: 10,
      name: 'cave_entrance',
      className: 'portal',
      x: 1568,
      y: 320,
      width: 32,
      height: 64,
      visible: true,
      properties: {
        appearanceType: 'cave_entrance',
        targetFacing: 'left',
        targetSceneId: 'cave',
        targetSpawnTileX: 2,
        targetSpawnTileY: 10
      }
    })
  })

  it('parses the cave TMX map', () => {
    const map = parseTiledMap({
      mapXml: caveMapXml,
      externalTilesets: {
        '../tilesets/town-32.tsx': townTilesetXml
      }
    })

    expect(map.width).toBe(34)
    expect(map.height).toBe(34)
    expect(map.pixelWidth).toBe(1088)
    expect(map.pixelHeight).toBe(1088)
    expect(map.layers).toHaveLength(1)
    expect(map.eventLayers.map((layer) => layer.name)).toEqual([
      'characters',
      'portals'
    ])
    expect(map.eventLayers[0].events).toHaveLength(2)
    expect(map.eventLayers[0].events).toMatchObject([
      {
        id: 5,
        name: '꿀꿀이-보스',
        className: 'character',
        width: 32,
        height: 32,
        visible: true,
        properties: {
          blocksMovement: true,
          'monster.level': 3,
          type: 'monster_pig'
        },
        appearanceType: 'monster_pig'
      },
      {
        id: 6,
        name: '말캉이-보스',
        className: 'character',
        width: 32,
        height: 32,
        visible: true,
        properties: {
          blocksMovement: true,
          'monster.level': 1,
          type: 'monster_slime'
        },
        appearanceType: 'monster_slime'
      }
    ])
    expect(map.eventLayers[1].events[0]).toMatchObject({
      id: 4,
      name: 'return_gate',
      className: 'portal',
      width: 32,
      height: 64,
      properties: {
        appearanceType: 'stairs_stone_step_base_00',
        targetFacing: 'right',
        targetSceneId: 'hunting-ground',
        targetSpawnTileX: 43,
        targetSpawnTileY: 10
      }
    })
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

  it('parses list properties as typed primitive arrays', () => {
    const map = parseTiledMap({
      mapXml: `<?xml version="1.0" encoding="UTF-8"?>
<map version="1.10" tiledversion="1.12.1" orientation="orthogonal" renderorder="right-down" width="1" height="1" tilewidth="32" tileheight="32" infinite="0">
  <tileset firstgid="1" source="../tilesets/town-32.tsx"/>
  <layer id="1" name="ground" width="1" height="1">
    <data encoding="csv">0</data>
  </layer>
  <objectgroup id="2" name="characters">
    <object id="7" name="blacksmith" type="character" x="32" y="32">
      <properties>
        <property name="type" value="character_bearded_apron_man"/>
        <property name="controller.dialogueLines" type="list">
          <item value="Need any tools?"/>
          <item value="Best steel in town."/>
        </property>
        <property name="controller.patrolDelaysSeconds" type="list" propertytype="float">
          <item value="0.5"/>
          <item value="1.25"/>
        </property>
        <property name="controller.patrolFlags" type="list" propertytype="bool">
          <item value="true"/>
          <item value="false"/>
        </property>
      </properties>
    </object>
  </objectgroup>
</map>`,
      externalTilesets: {
        '../tilesets/town-32.tsx': townTilesetXml
      }
    })

    expect(map.eventLayers[0].events[0].properties['controller.dialogueLines']).toEqual([
      'Need any tools?',
      'Best steel in town.'
    ])
    expect(
      map.eventLayers[0].events[0].properties['controller.patrolDelaysSeconds']
    ).toEqual([0.5, 1.25])
    expect(map.eventLayers[0].events[0].properties['controller.patrolFlags']).toEqual([
      true,
      false
    ])
  })

})
