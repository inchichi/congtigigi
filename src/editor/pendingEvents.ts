import type { HolidayDialogueEventSpec } from '../game/eventGeneration'

// 에디터(별도 페이지)와 게임은 별개 page지만 같은 origin이라 localStorage를 공유한다.
// 에디터가 생성한 이벤트를 여기에 쌓아두면, 게임이 씬을 띄울 때 읽어서 적용한다.
export const PENDING_EVENTS_STORAGE_KEY = 'my-sample-rpg:pending-events'

export const loadPendingEvents = (): HolidayDialogueEventSpec[] => {
  const raw = window.localStorage.getItem(PENDING_EVENTS_STORAGE_KEY)

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
  window.localStorage.setItem(PENDING_EVENTS_STORAGE_KEY, JSON.stringify(next))
  return next
}
