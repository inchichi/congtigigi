import { describe, expect, it } from 'vitest'

import {
  renderGeneratedLuaQuestModule,
  convertLuaQuestToBridgePayload
} from './luaQuestCodeGenerator'
import type { GeneratedLuaQuestJson } from './luaQuestSchema'

const quest: GeneratedLuaQuestJson = {
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
    },
    {
      type: 'reach',
      label: 'Reach the cave',
      required: 1,
      target: { mapId: 'cave' }
    }
  ],
  rewards: {
    gold: 100,
    experience: 25,
    items: [{ label: 'Smelly Charm', quantity: 1 }]
  }
}

describe('renderGeneratedLuaQuestModule', () => {
  it('renders a Lua module that can be exported directly', () => {
    const code = renderGeneratedLuaQuestModule(quest)

    expect(code).toContain('return {')
    expect(code).toContain('quest_id = "wizard_charm_hunt"')
    expect(code).toContain('giver_npc_entity_id = "npc-1"')
    expect(code).toContain('start_dialogue_lines = {')
    expect(code).toContain('target = {')
    expect(code).toContain('entityId = "enemy-1"')
    expect(code).toContain('mapId = "cave"')
    expect(code).toContain('label = "Smelly Charm"')
  })
})

describe('convertLuaQuestToBridgePayload', () => {
  it('builds a kind:"quest" bridge message mirroring the quest data', () => {
    const message = convertLuaQuestToBridgePayload(quest)

    expect(message.kind).toBe('quest')
    expect(message.quest.quest_id).toBe('wizard_charm_hunt')
    expect(message.quest.giver_entity_id).toBe('npc-1')
    expect(message.quest.dialogue.start).toEqual(['Please help me.'])
    expect(message.quest.objectives).toEqual([
      { type: 'defeat', label: 'Defeat the slime', required: 3, target: { entityId: 'enemy-1' } },
      { type: 'reach', label: 'Reach the cave', required: 1, target: { mapId: 'cave' } }
    ])
    expect(message.quest.rewards.items).toEqual([{ label: 'Smelly Charm', quantity: 1 }])
  })
})
