import { describe, expect, it } from 'vitest'

import {
  BLACKSMITH_NPC_ID,
  POTION_MERCHANT_NPC_ID,
  QUEST_DEFINITIONS,
  QUEST_GIVER_NPC_IDS,
  WIZARD_NPC_ID
} from '../questLog'
import {
  QUEST_GIVER_PORTRAIT_FRAME_BY_NPC_ID,
  getQuestGiverPortraitFrame
} from './createQuestLogOverlay'

describe('quest log overlay portraits', () => {
  it('uses the matching portrait frame for every quest giver', () => {
    expect(getQuestGiverPortraitFrame(WIZARD_NPC_ID)).toEqual({
      x: 0,
      y: 112,
      width: 16,
      height: 16
    })
    expect(getQuestGiverPortraitFrame(BLACKSMITH_NPC_ID)).toEqual({
      x: 32,
      y: 112,
      width: 16,
      height: 16
    })
    expect(getQuestGiverPortraitFrame(POTION_MERCHANT_NPC_ID)).toEqual({
      x: 48,
      y: 128,
      width: 16,
      height: 16
    })
  })

  it('has portrait frames for all NPCs that give quests', () => {
    for (const npcId of QUEST_GIVER_NPC_IDS) {
      expect(QUEST_GIVER_PORTRAIT_FRAME_BY_NPC_ID[npcId]).toBeDefined()
    }
  })

  it('keeps every quest definition covered by a portrait frame', () => {
    for (const definition of QUEST_DEFINITIONS) {
      expect(QUEST_GIVER_PORTRAIT_FRAME_BY_NPC_ID[definition.giverNpcId]).toBeDefined()
    }
  })
})
