import { readLocalStorage, writeLocalStorage } from './safeStorage'

export const THEME_WORK_LOG_STORAGE_KEY = 'my-sample-rpg:theme-work-log'

// 로그는 브라우저 localStorage에 남으므로 무한히 쌓이지 않게 최근 항목만 유지한다.
const MAX_LOG_ENTRIES = 50

export type ThemeWorkLogEntry = {
  id: string
  created_at: string
  user_prompt: string
  giver_npc_id: string | null
  theme: string
  art_direction: { style: string; mood: string; palette: string }
  quest_summary: {
    quest_id: string
    title: string
    giver_npc_id: string
    objective_label: string
  }
  style_targets: Array<{ target_ref: string; prompt: string; alpha: number }>
  applied_targets: string[]
  explanations: { direction: string; quest: string; styles: string }
  // 신규 보상 아이템이 함께 생성된 적용에만 존재한다.
  reward_item?: { item_id: string; label: string; icon_path: string }
  // 부분 실패가 있었던 적용에만 존재한다.
  failed_targets?: string[]
}

export const loadThemeWorkLog = (): ThemeWorkLogEntry[] => {
  const raw = readLocalStorage(THEME_WORK_LOG_STORAGE_KEY)
  if (!raw) {
    return []
  }
  try {
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? (parsed as ThemeWorkLogEntry[]) : []
  } catch {
    return []
  }
}

export const appendThemeWorkLog = (entry: ThemeWorkLogEntry): ThemeWorkLogEntry[] => {
  const next = [entry, ...loadThemeWorkLog()].slice(0, MAX_LOG_ENTRIES)
  writeLocalStorage(THEME_WORK_LOG_STORAGE_KEY, JSON.stringify(next))
  return next
}

export const clearThemeWorkLog = (): void => {
  writeLocalStorage(THEME_WORK_LOG_STORAGE_KEY, JSON.stringify([]))
}
