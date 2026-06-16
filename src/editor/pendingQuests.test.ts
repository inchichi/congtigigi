import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { QuestDefinition } from '../games/my-sample-rpg/questLog'

vi.mock('./safeStorage', () => {
  const storage = new Map<string, string>()

  return {
    readLocalStorage: (key: string) => storage.get(key) ?? null,
    writeLocalStorage: (key: string, value: string) => {
      storage.set(key, value)
      return true
    }
  }
})

import {
  clearPendingQuests,
  loadPendingQuests,
  replacePendingQuests,
  savePendingQuest
} from './pendingQuests'

const makeQuest = (id: string, title: string): QuestDefinition => ({
  id,
  regionName: 'town',
  giverNpcId: 'wizard',
  giverName: 'Wizard',
  title,
  trackerLabel: title,
  prerequisiteQuestIds: [],
  requestText: 'Request',
  guideText: 'Guide',
  startDialogueLines: ['Start'],
  activeDialogueLines: ['Active'],
  completionDialogueLines: ['Complete'],
  objectives: [
    {
      id: `${id}_objective_1`,
      label: 'Talk',
      required: 1,
      type: 'talk',
      target: {
        npcId: 'wizard'
      }
    }
  ],
  rewards: {
    gold: 0,
    experience: 0,
    items: []
  }
})

beforeEach(() => {
  replacePendingQuests([])
})

describe('pendingQuests', () => {
  it('replaces the pending quest list with the latest quest', () => {
    const questA = makeQuest('quest-a', 'Quest A')
    const questB = makeQuest('quest-b', 'Quest B')

    savePendingQuest(questA)
    replacePendingQuests([questB])

    expect(loadPendingQuests()).toEqual([questB])
  })

  it('can clear pending quests entirely', () => {
    savePendingQuest(makeQuest('quest-a', 'Quest A'))

    clearPendingQuests()

    expect(loadPendingQuests()).toEqual([])
  })
})
