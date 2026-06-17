import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('./llmProvider', () => ({
  generateJson: vi.fn()
}))

import { generateJson } from './llmProvider'
import type { LuaQuestCatalog } from './luaQuestCatalog'
import { generateLuaQuestJson } from './luaQuestJsonGenerator'

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

describe('generateLuaQuestJson', () => {
  it('grounds the prompt in catalog ids and selected npc giver ids', async () => {
    mockGenerateJson.mockResolvedValue({
      quest_id: 'wizard_charm_hunt',
      title: 'Wizard Charm Hunt',
      giver_npc_entity_id: 'npc-1',
      request_text: 'Find the charm.',
      guide_text: 'Go to the cave.',
      start_dialogue_lines: ['Please help me.'],
      active_dialogue_lines: ['Still working on it.'],
      completion_dialogue_lines: ['Thank you.'],
      objectives: [
        {
          type: 'defeat',
          label: 'Defeat the slime',
          required: 3,
          target: { entityId: 'enemy-1' }
        }
      ],
      rewards: {
        gold: 100,
        experience: 25,
        items: [{ label: 'Smelly Charm', quantity: 1 }]
      }
    })

    const result = await generateLuaQuestJson({
      apiKey: 'sk-ant-test',
      userPrompt: 'Make a quest.',
      catalog,
      entity: { id: 'npc-1', name: 'wizard', kind: 'npc', mapId: 'town' }
    })

    expect(result.title).toBe('Wizard Charm Hunt')

    const args = mockGenerateJson.mock.calls[0][0]
    expect(args.schemaName).toBe('generated_lua_quest_json')
    expect(
      (args.schema as {
        properties?: {
          giver_npc_entity_id?: { enum?: string[] }
        }
      }).properties?.giver_npc_entity_id?.enum
    ).toEqual(['npc-1'])
    expect(args.input).toContain('Make a quest.')
    expect(args.instructions).toContain('enemy-1(slime, map=cave)')
  })
})
