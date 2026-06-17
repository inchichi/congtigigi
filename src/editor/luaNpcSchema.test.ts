import { describe, expect, it } from 'vitest'

import type { LuaQuestCatalog } from './luaQuestCatalog'
import {
  createGeneratedLuaNpcValidationIssues,
  isGeneratedLuaNpcValid,
  luaNpcEntityId,
  type GeneratedLuaNpcJson
} from './luaNpcSchema'

const catalog: LuaQuestCatalog = {
  enemies: [{ id: 'Enemies-101', name: 'bat', mapId: 'testCave' }],
  npcs: [{ id: 'NPCs-3', name: 'elder', mapId: 'town' }],
  acquirables: [{ id: 'Chests-12', name: 'chest', mapId: 'testCave' }],
  scenes: ['town', 'testCave']
}

const validNpc: GeneratedLuaNpcJson = {
  npc_id: 'herb_seller',
  name: '약초 상인',
  map_id: 'town',
  appearance: 'character_villager_brown_tunic',
  position: { x: 8, y: 6 },
  dialogue_lines: ['약초가 필요한가?'],
  behavior: { type: 'wander', radius: 3 }
}

describe('luaNpcEntityId', () => {
  it('follows the placed-NPC group id rule so the catalog treats it as an NPC', () => {
    expect(luaNpcEntityId(validNpc)).toBe('NPCs-herb_seller')
  })
})

describe('createGeneratedLuaNpcValidationIssues', () => {
  it('passes a well-formed new NPC on an existing map', () => {
    expect(createGeneratedLuaNpcValidationIssues(validNpc, catalog)).toEqual([])
    expect(isGeneratedLuaNpcValid(validNpc, catalog)).toBe(true)
  })

  it('flags a map that does not exist in the game', () => {
    const issues = createGeneratedLuaNpcValidationIssues(
      { ...validNpc, map_id: 'nowhere' },
      catalog
    )
    expect(issues.map((issue) => issue.path)).toContain('map_id')
  })

  it('flags an appearance outside the borrowed asset list', () => {
    const issues = createGeneratedLuaNpcValidationIssues(
      { ...validNpc, appearance: 'character_made_up' },
      catalog
    )
    expect(issues.map((issue) => issue.path)).toContain('appearance')
  })

  it('rejects wander behavior with a non-positive radius', () => {
    const issues = createGeneratedLuaNpcValidationIssues(
      { ...validNpc, behavior: { type: 'wander', radius: 0 } },
      catalog
    )
    expect(issues.map((issue) => issue.path)).toContain('behavior.radius')
  })

  it('requires at least one dialogue line', () => {
    const issues = createGeneratedLuaNpcValidationIssues(
      { ...validNpc, dialogue_lines: [] },
      catalog
    )
    expect(issues.map((issue) => issue.path)).toContain('dialogue_lines')
  })

  it('rejects an npc_id that collides with an already-placed entity', () => {
    const colliding: GeneratedLuaNpcJson = { ...validNpc, npc_id: '3' }
    const withExistingId: LuaQuestCatalog = {
      ...catalog,
      npcs: [{ id: 'NPCs-3', name: 'elder', mapId: 'town' }]
    }
    const issues = createGeneratedLuaNpcValidationIssues(colliding, withExistingId)
    expect(issues.map((issue) => issue.path)).toContain('npc_id')
  })
})
