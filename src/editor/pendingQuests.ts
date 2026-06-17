import type { QuestDefinition } from '../games/my-sample-rpg/questLog'
import { readLocalStorage, writeLocalStorage } from './safeStorage'

export const PENDING_QUESTS_STORAGE_KEY = 'my-sample-rpg:pending-quests'

export const loadPendingQuests = (): QuestDefinition[] => {
  const raw = readLocalStorage(PENDING_QUESTS_STORAGE_KEY)

  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? (parsed as QuestDefinition[]) : []
  } catch {
    return []
  }
}

export const savePendingQuest = (quest: QuestDefinition): QuestDefinition[] => {
  const next = [
    quest,
    ...loadPendingQuests().filter((existing) => existing.id !== quest.id)
  ]

  if (!writeLocalStorage(PENDING_QUESTS_STORAGE_KEY, JSON.stringify(next))) {
    throw new Error('Failed to save the pending quest snapshot.')
  }

  return next
}

export const replacePendingQuests = (
  quests: QuestDefinition[]
): QuestDefinition[] => {
  if (!writeLocalStorage(PENDING_QUESTS_STORAGE_KEY, JSON.stringify(quests))) {
    throw new Error('Failed to replace the pending quest snapshot.')
  }

  return quests
}

export const clearPendingQuests = (): void => {
  replacePendingQuests([])
}
