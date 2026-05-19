import { describe, expect, it } from 'vitest'

import { createNpcCharactersFromEventLayers } from './createNpcCharactersFromEventLayers'
import { parseTiledMap } from './parseTiledMap'

const townTilesetXml = `<?xml version="1.0" encoding="UTF-8"?>
<tileset version="1.10" tiledversion="1.11.2" name="town-32" tilewidth="32" tileheight="32" tilecount="1" columns="1">
 <image source="town-32.png" width="32" height="32"/>
</tileset>`

describe('createNpcCharactersFromEventLayers', () => {
  it('converts character events into shared npc character state', () => {
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
  <objectgroup id="2" name="characters">
    <object id="7" name="blacksmith" type="character" x="80" y="96">
      <properties>
        <property name="type" value="character_bearded_apron_man"/>
        <property name="controller.scriptId" value="reply-with-message"/>
        <property name="controller.dialogueLines" type="list">
          <item value="Need any tools?"/>
          <item value="Best steel in town."/>
        </property>
        <property name="controller.messageDurationSeconds" type="float" value="3.5"/>
      </properties>
    </object>
    <object id="8" name="ghost" type="character" x="112" y="128" width="16" height="32">
      <properties>
        <property name="type" value="character_wizard_purple"/>
        <property name="blocksMovement" type="bool" value="false"/>
        <property name="controller.scriptId" value="wander-near-home"/>
        <property name="controller.radiusInTiles" type="int" value="3"/>
        <property name="controller.moveSpeedTilesPerSecond" type="float" value="1.5"/>
      </properties>
    </object>
    <object id="9" name="꿀꿀이" type="character" x="176" y="160" width="32" height="32">
      <properties>
        <property name="type" value="monster_pig"/>
        <property name="blocksMovement" type="bool" value="true"/>
        <property name="monster.level" type="int" value="3"/>
      </properties>
    </object>
    <object id="10" name="hunting_path_sign" type="character" x="64" y="96" width="32" height="32">
      <properties>
        <property name="type" value="sign_inn"/>
        <property name="blocksMovement" type="bool" value="false"/>
        <property name="displayText" value="사냥터로 가는 길"/>
      </properties>
    </object>
  </objectgroup>
</map>`,
      externalTilesets: {
        '../tilesets/town-32.tsx': townTilesetXml
      }
    })

    expect(
      createNpcCharactersFromEventLayers({
        map,
        defaultPixelWidth: 32,
        defaultPixelHeight: 32
      })
    ).toEqual([
      {
        id: 'blacksmith',
        appearanceType: 'character_bearded_apron_man',
        position: {
          x: 2,
          y: 2
        },
        facing: 'down',
        collisionSize: {
          width: 1,
          height: 1
        },
        blocksMovement: true,
        controller: {
          kind: 'lua',
          scriptId: 'reply-with-message',
          radiusInTiles: 0,
          moveSpeedTilesPerSecond: 8,
          config: {
            dialogueLines: ['Need any tools?', 'Best steel in town.'],
            messageDurationSeconds: 3.5
          }
        }
      },
      {
        id: 'ghost',
        appearanceType: 'character_wizard_purple',
        position: {
          x: 3.25,
          y: 3
        },
        facing: 'down',
        collisionSize: {
          width: 0.5,
          height: 1
        },
        blocksMovement: false,
        controller: {
          kind: 'lua',
          scriptId: 'wander-near-home',
          radiusInTiles: 3,
          moveSpeedTilesPerSecond: 1.5,
          config: {}
        }
      },
      {
        id: '꿀꿀이',
        appearanceType: 'monster_pig',
        level: 3,
        position: {
          x: 5,
          y: 4
        },
        facing: 'down',
        collisionSize: {
          width: 1,
          height: 1
        },
        blocksMovement: true,
        controller: {
          kind: 'npc',
          behavior: 'idle',
          moveSpeedTilesPerSecond: 8
        }
      },
      {
        id: 'hunting_path_sign',
        appearanceType: 'sign_inn',
        displayText: '사냥터로 가는 길',
        position: {
          x: 1.5,
          y: 2
        },
        facing: 'down',
        collisionSize: {
          width: 1,
          height: 1
        },
        blocksMovement: false,
        controller: {
          kind: 'npc',
          behavior: 'idle',
          moveSpeedTilesPerSecond: 8
        }
      }
    ])
  })
})
