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

  it('shows all map elements in the tree but limits the generation profile to dialogue NPCs', () => {
    // rpg 어댑터는 town.tmx 존재로 감지되므로 파일명은 town.tmx로 두되, 내용은 NPC 없이
    // 몬스터·표지판·포털만 있는 맵(사냥터/동굴 같은)으로 둔다.
    const monsterMapTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map>
  <layer name="ground" width="1" height="1"><data>1</data></layer>
  <objectgroup name="characters">
    <object id="5" name="꿀꿀이-1" type="character">
      <properties><property name="type" value="monster_pig"/></properties>
    </object>
    <object id="11" name="cave_entrance_sign" type="character">
      <properties>
        <property name="type" value="sign_inn"/>
        <property name="displayText" value="동굴입구"/>
      </properties>
    </object>
  </objectgroup>
  <objectgroup name="portals">
    <object id="10" name="cave_entrance" type="portal"/>
  </objectgroup>
</map>`
    const game = loadGame([file('town.tmx', monsterMapTmx)])

    // 트리(표시)에는 몬스터·표지판·포털이 모두 들어온다 — 사용자가 맵에 뭐가 있는지 본다.
    expect(game.maps[0].entities.map((entity) => entity.kind)).toEqual([
      'monster',
      'sign',
      'portal'
    ])
    // "그라운드 맵" 같은 타일 레이어도 표시용으로 읽힌다.
    expect(game.maps[0].layers).toEqual(['ground'])
    // 하지만 대화 생성 대상(profile.npcs)에는 NPC가 아닌 것들이 새어 들어가면 안 된다 — 여긴 NPC 0개.
    expect(game.profile?.npcs).toEqual([])
  })

  it('recognizes tile-drawn structures by category and skips ones already covered by objects', () => {
    // 지붕 2×1(건물)과 나무 1타일이 타일 레이어에 그려져 있고, 지붕 자리에는 이미 building
    // object가 등록돼 있다 → 지붕 군집은 중복이라 빠지고, 나무만 타일 엔티티로 들어온다.
    const tiledTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map width="2" height="2" tilewidth="16" tileheight="16">
  <tileset firstgid="1" name="t" tilewidth="16" tileheight="16" tilecount="2" columns="2">
    <tile id="0" type="roof_blue_shingle"/>
    <tile id="1" type="tree_large_canopy"/>
  </tileset>
  <layer id="1" name="ground" width="2" height="2"><data encoding="csv">1,1,2,0</data></layer>
  <objectgroup name="buildings">
    <object id="1" name="house" type="building" x="0" y="0" width="32" height="16"/>
  </objectgroup>
</map>`
    const game = loadGame([file('town.tmx', tiledTmx)])

    expect(game.maps[0].entities).toEqual([
      { id: 'house', name: 'house', kind: 'building', mapId: 'town' },
      { id: 'tile:tree:0,1', name: '(0, 1)', kind: 'tree', mapId: 'town' }
    ])
  })

  it('resolves external tilesets from the opened files and reads class= objects (Tiled 1.9+)', () => {
    // 에디터 기본 로드(editorPage)와 같은 모양: 맵은 ../tilesets/… 로 외부 타일셋을 참조하고,
    // 타일셋 텍스트는 별도 파일로 같이 들어온다. object는 type= 대신 class= 를 쓴다.
    const mapTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map width="1" height="1" tilewidth="16" tileheight="16">
  <tileset firstgid="1" source="../tilesets/props.tsx"/>
  <layer id="1" name="ground" width="1" height="1"><data encoding="csv">1</data></layer>
  <objectgroup name="characters">
    <object id="1" name="greeter" class="character" x="8" y="8">
      <properties><property name="type" value="character_villager"/></properties>
    </object>
  </objectgroup>
</map>`
    const tilesetTsx = `<?xml version="1.0"?>
<tileset name="props" tilewidth="16" tileheight="16" tilecount="1" columns="1">
  <tile id="0" type="fountain_basin_left"/>
</tileset>`
    const game = loadGame([
      { name: 'town.tmx', path: 'assets/maps/town.tmx', text: mapTmx },
      { name: 'props.tsx', path: 'assets/tilesets/props.tsx', text: tilesetTsx }
    ])

    expect(game.maps[0].entities).toEqual([
      { id: 'greeter', name: 'greeter', kind: 'npc', mapId: 'town' },
      { id: 'tile:fountain:0,0', name: '(0, 0)', kind: 'fountain', mapId: 'town' }
    ])
  })

  it('does not throw when a map fails to parse — loads it empty and records the path', () => {
    const game = loadGame([
      file('town.tmx', rpgTmx),
      file('broken.tmx', 'not valid xml at all')
    ])

    expect(game.parseErrors).toEqual(['broken.tmx'])
    // 멀쩡한 맵은 그대로, 깨진 맵은 엔티티 0개로 들어온다(에디터가 통째로 죽지 않음).
    expect(game.maps.map((map) => map.id)).toEqual(['town', 'broken'])
    expect(game.maps[1].entities).toEqual([])
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
