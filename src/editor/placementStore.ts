// 마우스로 맵에 배치한 에셋(타일/오브젝트)의 영구 저장소.
// 에디터(별도 페이지)와 게임(iframe)은 같은 origin이라 localStorage를 공유한다 —
// 에디터가 배치 모드를 켜고, 게임이 캔버스 클릭으로 실제 배치를 만들어 여기에 저장한다.
// 게임은 부팅 시 읽어 그리고, 다른 컨텍스트(에디터)의 변경은 'storage' 이벤트로 라이브 반영한다.
// 맵별로 분리 저장(맵을 바꿔도 각 맵의 배치가 유지됨).

import { readLocalStorage, writeLocalStorage } from './safeStorage'

export const PENDING_PLACEMENTS_STORAGE_KEY = 'my-sample-rpg:pending-placements'

// 배치 1건. kind에 따라 타일(타일셋 로컬 id) 또는 오브젝트(누끼 PNG url)로 그린다.
export type PlacedItem = {
  id: string
  kind: 'tile' | 'object'
  col: number
  row: number
  label?: string
  // kind === 'tile'
  tilesetSource?: string
  tileId?: number
  // kind === 'object'
  imageUrl?: string
}

// 배치 모드에서 게임으로 보내는 "지금 놓을 항목" 템플릿(좌표는 클릭 시 채운다).
export type PlacementTemplate = Omit<PlacedItem, 'id' | 'col' | 'row'>

type PlacementsByMap = Record<string, PlacedItem[]>

const loadAll = (): PlacementsByMap => {
  const raw = readLocalStorage(PENDING_PLACEMENTS_STORAGE_KEY)
  if (!raw) {
    return {}
  }
  try {
    const parsed = JSON.parse(raw) as unknown
    return parsed && typeof parsed === 'object' ? (parsed as PlacementsByMap) : {}
  } catch {
    return {}
  }
}

const persist = (all: PlacementsByMap): void => {
  if (!writeLocalStorage(PENDING_PLACEMENTS_STORAGE_KEY, JSON.stringify(all))) {
    throw new Error('배치를 저장하지 못했습니다 (브라우저 저장소 차단 또는 용량 초과).')
  }
}

export const loadPlacementsForMap = (mapId: string): PlacedItem[] => {
  const items = loadAll()[mapId]
  return Array.isArray(items) ? items : []
}

// 고유 id — 같은 칸에 여러 번 배치해도 구분되도록 시각 + 난수.
const newId = (): string =>
  `p_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`

export const addPlacement = (
  mapId: string,
  template: PlacementTemplate,
  col: number,
  row: number
): PlacedItem => {
  const all = loadAll()
  const item: PlacedItem = { ...template, id: newId(), col, row }
  all[mapId] = [...(all[mapId] ?? []), item]
  persist(all)
  return item
}

export const removePlacement = (mapId: string, id: string): PlacedItem[] => {
  const all = loadAll()
  const next = (all[mapId] ?? []).filter((item) => item.id !== id)
  all[mapId] = next
  persist(all)
  return next
}

export const clearPlacementsForMap = (mapId: string): void => {
  const all = loadAll()
  delete all[mapId]
  persist(all)
}
