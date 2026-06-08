import { describe, expect, it } from 'vitest'

import { extractTmxObjects } from './tmxObjects'
import { detectAdapter, legendOfLuaAdapter, rpgAdapter } from './gameAdapter'

const rpgTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map>
  <objectgroup name="characters">
    <object id="1" name="blacksmith" type="character">
      <properties>
        <property name="type" value="character_bearded"/>
        <property name="displayText" value="대장장이"/>
      </properties>
    </object>
    <object id="2" name="east_gate" type="portal"/>
  </objectgroup>
</map>`

// 임베드 타일셋(source 없음) + 다른 object group 규칙. 관용 파서가 타일셋 없이도 객체를 읽어야 한다.
const legendTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map>
  <tileset firstgid="1" name="overworld" tilewidth="16" tileheight="16" tilecount="10" columns="5"/>
  <objectgroup id="6" name="Enemies">
    <object id="105" name="slime" type="small"/>
  </objectgroup>
  <objectgroup id="3" name="Walls">
    <object id="200" name="wall_1"/>
  </objectgroup>
</map>`

describe('extractTmxObjects', () => {
  it('reads objects without requiring tilesets (embedded tileset ok)', () => {
    const objects = extractTmxObjects(legendTmx)

    expect(objects.map((object) => object.name)).toEqual(['slime', 'wall_1'])
    expect(objects[0].group).toBe('Enemies')
  })
})

describe('rpgAdapter', () => {
  it('extracts character objects as npc entities, ignoring portals', () => {
    const entities = rpgAdapter.extractEntities('town', extractTmxObjects(rpgTmx))

    expect(entities).toEqual([
      { id: 'blacksmith', name: '대장장이', kind: 'npc', mapId: 'town' }
    ])
  })
})

describe('legendOfLuaAdapter', () => {
  it('extracts entities from its own object groups (Enemies), ignoring Walls', () => {
    const entities = legendOfLuaAdapter.extractEntities('test', extractTmxObjects(legendTmx))

    expect(entities).toEqual([
      { id: 'Enemies-105', name: 'slime', kind: 'enemy', mapId: 'test' }
    ])
  })
})

describe('detectAdapter', () => {
  it('detects legend-of-lua by its Love2D signature files', () => {
    expect(detectAdapter(['conf.lua', 'main.lua', 'test.tmx']).id).toBe('legend-of-lua')
  })

  it('falls back to the rpg adapter otherwise', () => {
    expect(detectAdapter(['town.tmx', 'cave.tmx']).id).toBe('my-sample-rpg')
  })
})
