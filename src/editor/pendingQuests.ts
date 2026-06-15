import type { QuestDefinition } from '../games/my-sample-rpg/questLog'
import { readLocalStorage, writeLocalStorage } from './safeStorage'

// 에디터가 생성한 "진짜 퀘스트"(목표 포함 런타임 QuestDefinition)를 같은 origin localStorage로
// 게임에 전달한다. 게임은 부팅(및 storage 변경) 시 읽어 동적 퀘스트로 등록한다. pendingEvents와
// 같은 패턴 — 대사 이벤트와 별개 채널이다.
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
    ...loadPendingQuests().filter((existing) => existing.id !== quest.id),
    quest
  ]
  // 저장 실패(저장소 차단/용량 초과)는 호출부(runApply)가 사용자에게 알릴 수 있도록 명시적으로 알린다.
  if (!writeLocalStorage(PENDING_QUESTS_STORAGE_KEY, JSON.stringify(next))) {
    throw new Error('생성한 퀘스트를 저장하지 못했습니다 (브라우저 저장소 차단 또는 용량 초과).')
  }
  return next
}
