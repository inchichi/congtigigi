import { describe, expect, it } from 'vitest'

import type { LuaQuestCatalog } from './luaQuestCatalog'
import { dryRunLuaNpcApply } from './dryRunLuaNpcApply'
import type { GeneratedLuaNpcJson } from './luaNpcSchema'

const catalog: LuaQuestCatalog = {
  enemies: [{ id: 'Enemies-101', name: 'bat', mapId: 'testCave' }],
  npcs: [],
  acquirables: [],
  scenes: ['town', 'testCave']
}

const npc: GeneratedLuaNpcJson = {
  npc_id: 'herb_seller',
  name: '약초 상인',
  map_id: 'town',
  appearance: 'character_villager_brown_tunic',
  position: { x: 8, y: 6 },
  dialogue_lines: ['약초가 필요한가?'],
  behavior: { type: 'wander', radius: 3 }
}

describe('dryRunLuaNpcApply', () => {
  it('passes for a valid new NPC and reports every step ok', () => {
    const report = dryRunLuaNpcApply(npc, catalog)

    expect(report.ok).toBe(true)
    expect(report.failedStepId).toBeUndefined()
    expect(report.steps.every((step) => step.status === 'ok')).toBe(true)
    expect(report.jsonIssues).toEqual([])
  })

  it('fails on a missing map and points at the map step', () => {
    const report = dryRunLuaNpcApply({ ...npc, map_id: 'nowhere' }, catalog)

    expect(report.ok).toBe(false)
    expect(report.failedStepId).toBe('resolve_map')
  })

  it('fails when the new id collides with an existing entity', () => {
    const colliding: LuaQuestCatalog = {
      ...catalog,
      npcs: [{ id: 'NPCs-herb_seller', name: '기존', mapId: 'town' }]
    }
    const report = dryRunLuaNpcApply(npc, colliding)

    expect(report.ok).toBe(false)
    expect(report.steps.find((step) => step.id === 'id_collision')?.status).toBe('fail')
  })
})
