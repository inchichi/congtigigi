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
    <object id="3" name="hunting_path_sign" type="character">
      <properties>
        <property name="type" value="sign_inn"/>
        <property name="displayText" value="사냥터로 가는 길"/>
      </properties>
    </object>
    <object id="4" name="꿀꿀이" type="character">
      <properties>
        <property name="type" value="monster_pig"/>
        <property name="monster.level" type="int" value="3"/>
      </properties>
    </object>
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

  it('throws on a fatally unparseable TMX so callers can distinguish it from an empty map', () => {
    expect(() => extractTmxObjects('')).toThrow(/TMX 파싱 실패/u)
    expect(() => extractTmxObjects('   ')).toThrow(/TMX 파싱 실패/u)
    expect(() => extractTmxObjects('not xml at all')).toThrow(/TMX 파싱 실패/u)
    // 잘린(truncated) 맵: 루트 <map>은 생기지만 구조 에러가 나고 콘텐츠가 날아간다 — 실패로 봐야 한다.
    expect(() =>
      extractTmxObjects('<map><objectgroup name="c"><object id="1" name="a" type="character"/><object ')
    ).toThrow(/TMX 파싱 실패/u)
    // TMX 루트(<map>)가 아닌 문서도 실패로 본다.
    expect(() => extractTmxObjects('<html><body/></html>')).toThrow(/TMX 파싱 실패/u)
  })

  it('still parses a usable map whose attribute has a recoverable stray entity ref', () => {
    // &foo;는 xmldom의 error 레벨을 내지만 문서는 멀쩡히 나온다 — 버리지 말고 파싱해야 한다.
    const objects = extractTmxObjects(
      '<map><objectgroup name="c"><object id="1" name="Tom &foo; J" type="character"/></objectgroup></map>'
    )
    expect(objects).toHaveLength(1)
    expect(objects[0].id).toBe('1')
  })
})

describe('rpgAdapter', () => {
  it('extracts every map element with its kind (npc/portal/sign/monster)', () => {
    const entities = rpgAdapter.extractEntities('town', extractTmxObjects(rpgTmx))

    // 트리에 다 보여줘야 사용자가 무엇을 바꿀지 안다 — 외형/타입으로 종류만 가른다. displayText가
    // 있으면 표시 이름으로, 없으면 object name으로 떨어진다. (대화 생성 대상 한정은 loadGame에서.)
    expect(entities).toEqual([
      { id: 'blacksmith', name: '대장장이', kind: 'npc', mapId: 'town' },
      { id: 'east_gate', name: 'east_gate', kind: 'portal', mapId: 'town' },
      { id: 'hunting_path_sign', name: '사냥터로 가는 길', kind: 'sign', mapId: 'town' },
      { id: '꿀꿀이', name: '꿀꿀이', kind: 'monster', mapId: 'town' }
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
