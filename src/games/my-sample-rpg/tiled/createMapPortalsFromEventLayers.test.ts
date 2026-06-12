import { describe, expect, it } from 'vitest'

import { createMapPortalsFromEventLayers } from './createMapPortalsFromEventLayers'
import { parseTiledMap } from './parseTiledMap'

const townTilesetXml = `<?xml version="1.0" encoding="UTF-8"?>
<tileset version="1.10" tiledversion="1.11.2" name="town-32" tilewidth="32" tileheight="32" tilecount="1" columns="1">
 <image source="town-32.png" width="32" height="32"/>
</tileset>`

describe('createMapPortalsFromEventLayers', () => {
  it('converts portal events into map portal state', () => {
    const map = parseTiledMap({
      mapXml: `<?xml version="1.0" encoding="UTF-8"?>
<map version="1.10" tiledversion="1.12.1" orientation="orthogonal" renderorder="right-down" width="4" height="4" tilewidth="32" tileheight="32" infinite="0">
  <tileset firstgid="1" source="../tilesets/town-32.tsx"/>
  <layer id="1" name="ground" width="4" height="4">
    <data encoding="csv">
0,0,0,0,
0,0,0,0,
0,0,0,0,
0,0,0,0
    </data>
  </layer>
  <objectgroup id="2" name="portals">
    <object id="7" name="east_gate" type="portal" x="64" y="96" width="32" height="64">
      <properties>
        <property name="appearanceType" value="stairs_stone_step_base_00"/>
        <property name="targetSceneId" value="hunting-ground"/>
        <property name="targetSpawnTileX" type="int" value="2"/>
        <property name="targetSpawnTileY" type="int" value="5"/>
        <property name="targetFacing" value="left"/>
      </properties>
    </object>
  </objectgroup>
</map>`,
      externalTilesets: {
        '../tilesets/town-32.tsx': townTilesetXml
      }
    })

    expect(createMapPortalsFromEventLayers({ map })).toEqual([
      {
        id: 'east_gate',
        appearanceType: 'stairs_stone_step_base_00',
        position: {
          x: 2,
          y: 3
        },
        collisionSize: {
          width: 1,
          height: 2
        },
        targetSceneId: 'hunting-ground',
        targetSpawn: {
          x: 2,
          y: 5
        },
        targetFacing: 'left'
      }
    ])
  })
})
