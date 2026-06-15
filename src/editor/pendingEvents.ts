import type { HolidayDialogueEventSpec } from '../games/my-sample-rpg/eventGeneration'
import { readLocalStorage, writeLocalStorage } from './safeStorage'

// 에디터(별도 페이지)와 게임은 별개 page지만 같은 origin이라 localStorage를 공유한다.
// 에디터가 생성한 이벤트를 여기에 쌓아두면, 게임이 씬을 띄울 때 읽어서 적용한다.
export const PENDING_EVENTS_STORAGE_KEY = 'my-sample-rpg:pending-events'

export const loadPendingEvents = (): HolidayDialogueEventSpec[] => {
  const raw = readLocalStorage(PENDING_EVENTS_STORAGE_KEY)

  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? (parsed as HolidayDialogueEventSpec[]) : []
  } catch {
    return []
  }
}

export const savePendingEvent = (
  event: HolidayDialogueEventSpec
): HolidayDialogueEventSpec[] => {
  const next = [
    ...loadPendingEvents().filter(
      (existing) => existing.event_name !== event.event_name
    ),
    event
  ]
  // 저장 실패(저장소 차단/용량 초과)는 호출부(runApply)가 사용자에게 알릴 수 있도록 명시적으로 알린다.
  if (!writeLocalStorage(PENDING_EVENTS_STORAGE_KEY, JSON.stringify(next))) {
    throw new Error('생성한 이벤트를 저장하지 못했습니다 (브라우저 저장소 차단 또는 용량 초과).')
  }
  return next
}

// 에디터가 생성한 Lua 컨트롤러 코드도 같은 origin localStorage로 게임에 전달한다.
export const PENDING_LUA_SCRIPTS_STORAGE_KEY = 'my-sample-rpg:pending-lua-scripts'

export type PendingLuaScript = {
  target_character_id: string
  source: string
}

export const loadPendingLuaScripts = (): PendingLuaScript[] => {
  const raw = readLocalStorage(PENDING_LUA_SCRIPTS_STORAGE_KEY)

  if (!raw) {
    return []
  }

  try {
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? (parsed as PendingLuaScript[]) : []
  } catch {
    return []
  }
}

export const savePendingLuaScript = (
  script: PendingLuaScript
): PendingLuaScript[] => {
  const next = [
    ...loadPendingLuaScripts().filter(
      (existing) => existing.target_character_id !== script.target_character_id
    ),
    script
  ]
  if (!writeLocalStorage(PENDING_LUA_SCRIPTS_STORAGE_KEY, JSON.stringify(next))) {
    throw new Error('생성한 Lua 스크립트를 저장하지 못했습니다 (브라우저 저장소 차단 또는 용량 초과).')
  }
  return next
}
