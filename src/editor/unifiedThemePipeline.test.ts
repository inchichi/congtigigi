import { describe, expect, it } from 'vitest'
import type { GameStructureProfile } from './gameStructureProfile'
import {
  createUnifiedThemePlanSchema,
  createUnifiedThemeValidationIssues,
  isStylableAsset,
  type StyleCatalog,
  type UnifiedThemePlan
} from './unifiedThemePipeline'

const profile: GameStructureProfile = {
  game_title: 'Test RPG',
  engine: 'PixiJS',
  maps: [{ id: 'town', name: 'Town', file: 'town.tmx' }],
  npcs: [{ id: 'wizard', name: 'Wizard', map: 'town', file: 'town.tmx' }],
  items: [{ id: 'mana_potion', name: 'Mana Potion', file: 'items.ts' }],
  events: [],
  dialogue_system: { file: 'dialogue.ts', format: 'json' },
  event_system: { file: 'events.ts', register_function: 'register' },
  modifiable_files: []
}

const catalog: StyleCatalog = {
  assets: [{ id: 'src/games/my-sample-rpg/assets/portraits/wizard.png', label: 'wizard.png' }],
  objects: [{ id: 'wizard_tower', label: 'wizard tower' }]
}

describe('unifiedThemePipeline', () => {
  it('does not expose runtime animation sheets as FLUX targets', () => {
    expect(isStylableAsset('src/games/my-sample-rpg/assets/monsters/monster-pig-sheet.png')).toBe(false)
    expect(isStylableAsset('src/games/my-sample-rpg/assets/monsters/pig-motion.png')).toBe(false)
    expect(isStylableAsset('src/games/my-sample-rpg/assets/tilesets/town-32.png')).toBe(false)
    expect(isStylableAsset('src/games/my-sample-rpg/assets/weapons/weapon-sheet.png')).toBe(false)
    expect(isStylableAsset('src/games/my-sample-rpg/assets/portraits/wizard.png')).toBe(true)
  })

  it('grounds the Qwen schema in exact quest and FLUX target ids', () => {
    const schema = createUnifiedThemePlanSchema(profile, catalog)
    const properties = (schema as { properties: Record<string, unknown> }).properties
    const styleTargets = properties.style_targets as {
      items: { properties: { target_ref: { enum: string[] } } }
    }

    expect(styleTargets.items.properties.target_ref.enum).toEqual([
      'asset:src/games/my-sample-rpg/assets/portraits/wizard.png',
      'object:wizard_tower'
    ])
    expect((properties.schema_version as { const: number }).const).toBe(1)
    expect((properties.game_id as { const: string }).const).toBe('my-sample-rpg')
  })

  it('rejects a style target that is not in the catalog', () => {
    const plan: UnifiedThemePlan = {
      schema_version: 1,
      game_id: 'my-sample-rpg',
      theme: 'Moonlit winter',
      art_direction: { style: 'pixel art', mood: 'quiet', palette: 'blue and silver' },
      quest: {
        quest_id: 'moonlit_winter',
        title: 'Moonlit Winter',
        giver_npc_id: 'wizard',
        region: 'Town',
        request_text: 'Find the missing moon shard.',
        guide_text: 'Search the town.',
        start_dialogue_lines: ['Please find the shard.'],
        active_dialogue_lines: ['The shard is still missing.'],
        completion_dialogue_lines: ['You found it.'],
        objectives: [{ type: 'talk', label: 'Talk to the wizard', required: 1, target: { npcId: 'wizard' } }],
        rewards: { gold: 10, experience: 5, items: [] }
      },
      style_targets: [{ target_ref: 'asset:missing.png', prompt: 'Make it wintry.', alpha: 0.8 }]
    }

    const issues = createUnifiedThemeValidationIssues(plan, profile, catalog)
    expect(issues.some((issue) => issue.path === 'style_targets[0].target_ref')).toBe(true)
  })
})
