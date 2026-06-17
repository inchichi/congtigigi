import { describe, expect, it } from 'vitest'

import type { LoadedGame } from './loadGame'
import { buildLuaQuestCatalog, buildLuaQuestCatalogText } from './luaQuestCatalog'

const game = {
  adapter: {
    id: 'legend-of-lua',
    name: 'Legend of Lua',
    detect: () => true,
    extractEntities: () => [],
    applyMode: 'bridge',
    generate: async () => {
      throw new Error('not used')
    }
  },
  maps: [
    {
      id: 'town',
      name: 'town',
      file: 'town.tmx',
      layers: ['ground'],
      entities: [
        { id: 'enemy-1', name: 'slime', kind: 'enemy', mapId: 'town' },
        { id: 'npc-1', name: 'wizard', kind: 'npc', mapId: 'town' },
        { id: 'loot-1', name: 'chest', kind: 'loot', mapId: 'town' }
      ]
    },
    {
      id: 'cave',
      name: 'cave',
      file: 'cave.tmx',
      layers: ['ground'],
      entities: [
        { id: 'enemy-2', name: 'bat', kind: 'enemy', mapId: 'cave' },
        { id: 'chest-1', name: 'ore', kind: 'chest', mapId: 'cave' }
      ]
    }
  ],
  parseErrors: []
} as LoadedGame

describe('buildLuaQuestCatalog', () => {
  it('collects enemies, npcs, acquirables, and scenes from loaded maps', () => {
    const catalog = buildLuaQuestCatalog(game)

    expect(catalog.enemies.map((entity) => entity.id)).toEqual(['enemy-1', 'enemy-2'])
    expect(catalog.npcs.map((entity) => entity.id)).toEqual(['npc-1'])
    expect(catalog.acquirables.map((entity) => entity.id)).toEqual(['loot-1', 'chest-1'])
    expect(catalog.scenes).toEqual(['town', 'cave'])
  })

  it('renders a prompt-friendly catalog summary', () => {
    const catalog = buildLuaQuestCatalog(game)

    expect(buildLuaQuestCatalogText(catalog)).toContain('enemy-1(slime, map=town)')
    expect(buildLuaQuestCatalogText(catalog)).toContain('cave')
  })
})
