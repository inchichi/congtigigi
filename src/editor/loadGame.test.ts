import { describe, expect, it } from 'vitest'

import { loadGame, type GameFile } from './loadGame'

const rpgTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map>
  <objectgroup name="characters">
    <object id="1" name="blacksmith" type="character">
      <properties>
        <property name="type" value="character_bearded"/>
        <property name="displayText" value="대장장이"/>
      </properties>
    </object>
  </objectgroup>
</map>`

const legendTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map>
  <tileset firstgid="1" name="overworld" tilewidth="16" tileheight="16" tilecount="10" columns="5"/>
  <objectgroup name="Enemies">
    <object id="105" name="slime" type="small"/>
  </objectgroup>
</map>`

const file = (name: string, text: string): GameFile => ({ name, path: name, text })

describe('loadGame', () => {
  it('loads my-sample-rpg with npc entities and a generation profile', () => {
    const game = loadGame([file('town.tmx', rpgTmx)])

    expect(game.adapter.id).toBe('my-sample-rpg')
    expect(game.maps[0].entities).toEqual([
      { id: 'blacksmith', name: '대장장이', kind: 'npc', mapId: 'town' }
    ])
    expect(game.profile?.npcs.map((npc) => npc.id)).toEqual(['blacksmith'])
  })

  it('loads legend-of-lua entities but no generation profile (apply unsupported yet)', () => {
    const game = loadGame([file('conf.lua', ''), file('test.tmx', legendTmx)])

    expect(game.adapter.id).toBe('legend-of-lua')
    expect(game.adapter.supportsApply).toBe(false)
    expect(game.maps[0].entities).toEqual([
      { id: 'Enemies-105', name: 'slime', kind: 'enemy', mapId: 'test' }
    ])
    expect(game.profile).toBeUndefined()
  })
})
