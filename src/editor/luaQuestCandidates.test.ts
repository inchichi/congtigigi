import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./llmProvider', () => ({
  generateJson: vi.fn()
}))

import { generateJson } from './llmProvider'
import type { LuaQuestCatalog } from './luaQuestCatalog'
import { generateLuaQuestCandidates } from './luaQuestCandidates'

const mockGenerateJson = vi.mocked(generateJson)

const catalog: LuaQuestCatalog = {
  enemies: [{ id: 'enemy-1', name: 'slime', mapId: 'cave' }],
  npcs: [{ id: 'npc-1', name: 'wizard', mapId: 'town' }],
  acquirables: [{ id: 'loot-1', name: 'charm', mapId: 'cave' }],
  scenes: ['town', 'cave']
}

beforeEach(() => {
  mockGenerateJson.mockReset()
})

describe('generateLuaQuestCandidates', () => {
  it('grounds candidate generation in catalog npc ids', async () => {
    mockGenerateJson.mockResolvedValue({
      candidates: [
        { title: 'Wizard Task', summary: 'A short idea.', target_hint: 'npc-1' }
      ]
    })

    const result = await generateLuaQuestCandidates({
      apiKey: 'sk-ant-test',
      userPrompt: 'Make candidates.',
      catalog,
      entity: { id: 'npc-1', name: 'wizard', kind: 'npc', mapId: 'town' }
    })

    expect(result).toEqual([
      { title: 'Wizard Task', summary: 'A short idea.', target_hint: 'npc-1' }
    ])

    const args = mockGenerateJson.mock.calls[0][0]
    expect(args.schemaName).toBe('lua_quest_candidates')
    expect(
      (args.schema as {
        properties?: {
          candidates?: {
            items?: {
              properties?: {
                target_hint?: { enum?: string[] }
              }
            }
          }
        }
      }).properties?.candidates?.items?.properties?.target_hint?.enum
    ).toEqual(['', 'npc-1'])
    expect(args.instructions).toContain('Selected NPC giver: npc-1 (wizard)')
    expect(args.input).toContain('Make candidates.')
  })
})
