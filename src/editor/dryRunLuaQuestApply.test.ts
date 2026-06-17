import { describe, expect, it } from 'vitest'

import type { LuaQuestCatalog } from './luaQuestCatalog'
import { dryRunLuaQuestApply } from './dryRunLuaQuestApply'
import type { GeneratedLuaQuestJson } from './luaQuestSchema'

const catalog: LuaQuestCatalog = {
  enemies: [{ id: 'enemy-1', name: 'slime', mapId: 'cave' }],
  npcs: [{ id: 'npc-1', name: 'wizard', mapId: 'town' }],
  acquirables: [{ id: 'loot-1', name: 'charm', mapId: 'cave' }],
  scenes: ['town', 'cave']
}

const validQuest: GeneratedLuaQuestJson = {
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
}

describe('dryRunLuaQuestApply', () => {
  it('passes when the quest is grounded in the catalog', () => {
    const report = dryRunLuaQuestApply(validQuest, catalog, {
      selectedEntityId: 'npc-1'
    })

    expect(report.ok).toBe(true)
    expect(report.failedStepId).toBeUndefined()
    expect(report.jsonIssues).toEqual([])
  })

  it('fails when the quest points at missing catalog entities', () => {
    const report = dryRunLuaQuestApply(
      {
        ...validQuest,
        giver_npc_entity_id: 'missing',
        objectives: [
          {
            type: 'reach',
            label: 'Go somewhere else',
            required: 1,
            target: { mapId: 'missing-map' }
          }
        ]
      },
      catalog
    )

    expect(report.ok).toBe(false)
    expect(report.jsonIssues).toContain(
      'giver_npc_entity_id - 이 게임의 NPC 엔티티가 아니다: missing'
    )
    expect(report.jsonIssues).toContain(
      'objectives[0].target.mapId - 이 게임의 맵이 아니다: missing-map'
    )
  })
})
