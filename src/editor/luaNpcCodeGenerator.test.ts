import { describe, expect, it } from 'vitest'

import {
  convertLuaNpcToGameEntity,
  convertLuaNpcToSpawnBridgePayload,
  renderEditorNpcSpawnEntry,
  renderGeneratedLuaNpcModule
} from './luaNpcCodeGenerator'
import type { GeneratedLuaNpcJson } from './luaNpcSchema'

const npc: GeneratedLuaNpcJson = {
  npc_id: 'herb_seller',
  name: '약초 상인',
  map_id: 'town',
  appearance: 'character_villager_brown_tunic',
  position: { x: 8, y: 6 },
  dialogue_lines: ['약초가 필요한가?', '오늘은 특별히 싸게 주지.'],
  behavior: { type: 'wander', radius: 3 }
}

describe('renderGeneratedLuaNpcModule', () => {
  it('renders a requireable Lua table with id/map/appearance/dialogue/behavior', () => {
    const module = renderGeneratedLuaNpcModule(npc)

    expect(module).toContain('return {')
    expect(module).toContain('entity_id = "NPCs-herb_seller"')
    expect(module).toContain('map = "town"')
    expect(module).toContain('appearance = "character_villager_brown_tunic"')
    expect(module).toContain('position = { x = 8, y = 6 }')
    expect(module).toContain('"약초가 필요한가?"')
    expect(module).toContain('behavior = { type = "wander", radius = 3 }')
  })
})

describe('renderEditorNpcSpawnEntry', () => {
  it('renders an editorNPCs.lua entry that spawns near the player with wander radius', () => {
    const entry = renderEditorNpcSpawnEntry(npc)

    expect(entry).toContain('editorNPCs["town"]')
    expect(entry).toContain('name = "herb_seller"')
    // 위치는 플레이어 옆 — 절대 좌표 대신 nearPlayer; wander radius 3 타일 -> 48 픽셀
    expect(entry).toContain('nearPlayer = true')
    expect(entry).toContain('appearance = "character_villager_brown_tunic"')
    expect(entry).toContain('radius = 48')
    expect(entry).toContain('"약초가 필요한가?"')
  })

  it('uses radius 0 for stationary behavior', () => {
    const entry = renderEditorNpcSpawnEntry({
      ...npc,
      behavior: { type: 'stationary', radius: 0 }
    })
    expect(entry).toContain('radius = 0')
  })
})

describe('convertLuaNpcToSpawnBridgePayload', () => {
  it('builds a spawn_npc bridge message with near-player + pixel radius', () => {
    const payload = convertLuaNpcToSpawnBridgePayload(npc, 1718200000000)

    expect(payload.kind).toBe('spawn_npc')
    expect(payload.id).toBe('spawn_npc-herb_seller-1718200000000')
    // 게임이 loadstring으로 로드할 한 줄 Lua 테이블
    expect(payload.lua).toBe(
      '{ name="herb_seller", map="town", appearance="character_villager_brown_tunic", dialogue={"약초가 필요한가?", "오늘은 특별히 싸게 주지."}, radius=48, nearPlayer=true }'
    )
    expect(payload.npc).toEqual({
      name: 'herb_seller',
      map: 'town',
      appearance: 'character_villager_brown_tunic',
      dialogue: ['약초가 필요한가?', '오늘은 특별히 싸게 주지.'],
      radius: 48,
      nearPlayer: true
    })
  })

  it('uses radius 0 for stationary behavior', () => {
    const payload = convertLuaNpcToSpawnBridgePayload(
      { ...npc, behavior: { type: 'stationary', radius: 0 } },
      1
    )
    expect(payload.npc.radius).toBe(0)
  })
})

describe('convertLuaNpcToGameEntity', () => {
  it('produces an npc-kind entity so the catalog and tree recognize it', () => {
    expect(convertLuaNpcToGameEntity(npc)).toEqual({
      id: 'NPCs-herb_seller',
      name: '약초 상인',
      kind: 'npc',
      mapId: 'town'
    })
  })
})
