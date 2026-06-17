import { beforeEach, describe, expect, it, vi } from 'vitest'

import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'

vi.mock('./llmProvider', () => ({
  generateJson: vi.fn()
}))

import { generateJson } from './llmProvider'
import { generateQuestJson } from './questJsonGenerator'
import type { GeneratedQuestJson } from './questJsonSchema'

const profile = CURRENT_GAME_PROJECT_PROFILE
const mockGenerateJson = vi.mocked(generateJson)

const questDraft = {
  quest_id: 'weapon_upgrade_baseline',
  title: 'Weapon Upgrade Baseline',
  giver_npc_id: 'blacksmith',
  region: 'Town',
  request_text: 'Please help me with a weapon upgrade.',
  guide_text: 'Go to the blacksmith.',
  start_dialogue_lines: ['Hello there.'],
  active_dialogue_lines: ['Come back when you are ready.'],
  completion_dialogue_lines: ['Thank you.'],
  objectives: [
    {
      type: 'talk',
      label: 'Talk to the blacksmith',
      required: 1,
      target: { npcId: 'blacksmith' }
    }
  ],
  rewards: {
    gold: 0,
    experience: 0,
    items: []
  }
} satisfies GeneratedQuestJson

beforeEach(() => {
  mockGenerateJson.mockReset()
})

describe('generateQuestJson', () => {
  it('grounds quest generation to existing profile and catalog ids', async () => {
    mockGenerateJson.mockResolvedValue(questDraft)

    await generateQuestJson({
      apiKey: 'sk-ant-test',
      userPrompt: '무기를 강화하는 퀘스트를 생성해줘',
      profile
    })

    const args = mockGenerateJson.mock.calls[0][0]
    const schema = args.schema as {
      properties?: {
        giver_npc_id?: { enum?: string[] }
        region?: { enum?: string[] }
        objectives?: {
          items?: {
            oneOf?: Array<{
              properties?: {
                type?: { const?: string }
                target?: {
                  properties?: {
                    sceneId?: { enum?: string[] }
                    appearanceType?: { enum?: string[] }
                    itemId?: { enum?: string[] }
                    shopId?: { enum?: string[] }
                    npcId?: { enum?: string[] }
                  }
                }
              }
            }>
          }
        }
        rewards?: {
          properties?: {
            items?: {
              items?: {
                properties?: {
                  item_id?: { enum?: string[] }
                }
              }
            }
          }
        }
      }
    }

    expect(args.schemaName).toBe('generated_quest_json')
    expect(args.instructions).toContain('Use only the exact ids from the provided profile and catalog.')
    expect(schema.properties?.giver_npc_id?.enum).toEqual(
      profile.npcs.map((npc) => npc.id)
    )
    expect(schema.properties?.region?.enum).toEqual(['Town', 'Hunting Ground', 'Cave'])

    const objectiveBranches = schema.properties?.objectives?.items?.oneOf ?? []
    const monsterBranch = objectiveBranches.find(
      (branch) => branch.properties?.type?.const === 'monster-defeat'
    )
    const itemUseBranch = objectiveBranches.find((branch) => branch.properties?.type?.const === 'item-use')
    const itemAcquireBranch = objectiveBranches.find(
      (branch) => branch.properties?.type?.const === 'item-acquire'
    )
    const sceneEnterBranch = objectiveBranches.find(
      (branch) => branch.properties?.type?.const === 'scene-enter'
    )
    const talkBranch = objectiveBranches.find((branch) => branch.properties?.type?.const === 'talk')

    expect(monsterBranch?.properties?.target?.properties?.sceneId?.enum).toEqual([
      'hunting-ground',
      'cave'
    ])
    expect(monsterBranch?.properties?.target?.properties?.appearanceType?.enum).toEqual([
      'monster_slime',
      'monster_pig'
    ])
    expect(itemUseBranch?.properties?.target?.properties?.itemId?.enum).toEqual([
      'health-potion',
      'mana-potion',
      'basic-sword',
      'iron-sword'
    ])
    expect(itemAcquireBranch?.properties?.target?.properties?.itemId?.enum).toContain('iron-sword')
    expect(itemAcquireBranch?.properties?.target?.properties?.itemId?.enum).not.toContain(
      'smith-charm'
    )
    expect(sceneEnterBranch?.properties?.target?.properties?.sceneId?.enum).toEqual([
      'hunting-ground',
      'cave'
    ])
    expect(sceneEnterBranch?.properties?.target?.properties?.sceneId?.enum).not.toContain('town')
    expect(talkBranch?.properties?.target?.properties?.npcId?.enum).toEqual(
      profile.npcs.map((npc) => npc.id)
    )

    expect(schema.properties?.rewards?.properties?.items?.items?.properties?.item_id?.enum).toEqual(
      profile.items.map((item) => item.id)
    )
  })

  it('locks giver when a specific NPC is selected', async () => {
    mockGenerateJson.mockResolvedValue(questDraft)
    const santaEntity = profile.npcs.find((npc) => npc.id === 'santa')

    await generateQuestJson({
      apiKey: 'sk-ant-test',
      userPrompt: '산타 퀘스트를 만들어줘',
      profile,
      entity: santaEntity
        ? {
            id: santaEntity.id,
            name: santaEntity.name,
            kind: 'npc',
            mapId: santaEntity.map
          }
        : undefined
    })

    const args = mockGenerateJson.mock.calls[0][0]
    const schema = args.schema as {
      properties?: {
        giver_npc_id?: { enum?: string[] }
        objectives?: {
          items?: {
            oneOf?: Array<{
              properties?: {
                type?: { const?: string }
                target?: {
                  properties?: {
                    sceneId?: { enum?: string[] }
                  }
                }
              }
            }>
          }
        }
      }
    }

    expect(schema.properties?.giver_npc_id?.enum).toEqual(['santa'])
    expect(args.instructions).toContain('Selected NPC giver: santa (산타). Use this id exactly as giver_npc_id.')
    expect(args.instructions).not.toContain('Do not use this id for scene-enter objectives.')
  })
})
