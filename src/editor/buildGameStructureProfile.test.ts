import { describe, expect, it } from 'vitest'

import { parseTiledMap } from '../game/tiled/parseTiledMap'
import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import {
  buildGameStructureProfile,
  extractNpcsFromParsedMap,
  type EditorMapSource
} from './buildGameStructureProfile'

const tilesetXml = `<?xml version="1.0" encoding="UTF-8"?>
<tileset version="1.10" tiledversion="1.11.2" name="town-32" tilewidth="32" tileheight="32" tilecount="1" columns="1">
 <image source="town-32.png" width="32" height="32"/>
</tileset>`

const mapXml = `<?xml version="1.0" encoding="UTF-8"?>
<map version="1.10" tiledversion="1.12.1" orientation="orthogonal" renderorder="right-down" width="4" height="4" tilewidth="32" tileheight="32" infinite="0">
  <tileset firstgid="1" source="../tilesets/town-32.tsx"/>
  <layer id="1" name="ground" width="4" height="4">
    <data encoding="csv">0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0</data>
  </layer>
  <objectgroup id="2" name="characters">
    <object id="1" name="blacksmith" type="character" x="80" y="96">
      <properties>
        <property name="type" value="character_bearded_apron_man"/>
        <property name="displayText" value="대장장이"/>
      </properties>
    </object>
    <object id="2" name="villager_9" type="character" x="112" y="128">
      <properties>
        <property name="type" value="character_villager_brown"/>
      </properties>
    </object>
    <object id="3" name="east_gate" type="portal" x="160" y="160" width="32" height="64"/>
  </objectgroup>
</map>`

const source: EditorMapSource = {
  id: 'town',
  name: 'Town',
  file: 'src/assets/maps/town.tmx',
  map: parseTiledMap({
    mapXml,
    externalTilesets: { '../tilesets/town-32.tsx': tilesetXml }
  })
}

describe('extractNpcsFromParsedMap', () => {
  it('uses displayText as the npc name when present', () => {
    expect(extractNpcsFromParsedMap(source)).toContainEqual({
      id: 'blacksmith',
      name: '대장장이',
      map: 'town',
      file: 'src/assets/maps/town.tmx'
    })
  })

  it('falls back to the object name when displayText is absent', () => {
    expect(extractNpcsFromParsedMap(source)).toContainEqual({
      id: 'villager_9',
      name: 'villager_9',
      map: 'town',
      file: 'src/assets/maps/town.tmx'
    })
  })

  it('ignores non-character objects such as portals', () => {
    expect(extractNpcsFromParsedMap(source).some((npc) => npc.id === 'east_gate')).toBe(
      false
    )
  })
})

describe('buildGameStructureProfile', () => {
  it('replaces maps and npcs from the parsed sources', () => {
    const profile = buildGameStructureProfile([source], CURRENT_GAME_PROJECT_PROFILE)

    expect(profile.maps).toEqual([
      { id: 'town', name: 'Town', file: 'src/assets/maps/town.tmx' }
    ])
    expect(profile.npcs.map((npc) => npc.id)).toEqual(['blacksmith', 'villager_9'])
  })

  it('keeps base items and systems that are not in the tmx', () => {
    const profile = buildGameStructureProfile([source], CURRENT_GAME_PROJECT_PROFILE)

    expect(profile.items).toEqual(CURRENT_GAME_PROJECT_PROFILE.items)
    expect(profile.event_system).toEqual(CURRENT_GAME_PROJECT_PROFILE.event_system)
  })
})
