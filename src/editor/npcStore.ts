// 마우스로 맵에 수기 배치한 NPC의 영구 저장소(타일/오브젝트 배치와 별개).
// 배치 에셋(placementStore)은 정적 스프라이트지만, NPC는 게임의 CharacterState로 스폰되는
// 기능적 엔티티다(이동 차단 + Enter 상호작용 대사). 그래서 별도 저장소/스폰 경로를 둔다.
// 에디터와 게임은 같은 origin이라 localStorage를 공유한다 — 에디터가 NPC 템플릿을 보내고,
// 게임이 캔버스 클릭으로 실제 NPC를 만들어 여기에 저장한 뒤 부팅/‘storage’ 이벤트로 반영한다.
// 맵별로 분리 저장(맵을 바꿔도 각 맵의 NPC가 유지됨).

import { readLocalStorage, writeLocalStorage } from './safeStorage'

export const PENDING_NPCS_STORAGE_KEY = 'my-sample-rpg:pending-npcs'

// 수기 배치 NPC 1건. appearanceType은 tiny-dungeon-16의 캐릭터 타일 type(예: character_wizard_purple).
// name은 머리 위 라벨, dialogueLines는 Enter 상호작용 시 순환 표시할 대사.
export type PlacedNpc = {
  id: string
  appearanceType: string
  col: number
  row: number
  name?: string
  dialogueLines?: string[]
}

// 배치 모드에서 게임으로 보내는 "지금 놓을 NPC" 템플릿(좌표는 클릭 시 채운다).
export type NpcTemplate = Omit<PlacedNpc, 'id' | 'col' | 'row'>

// 에디터 팔레트 → 게임 postMessage(editor:placement-template)로 흐르는 NPC 와이어 템플릿.
// 타일/오브젝트 배치 템플릿과 구분하기 위한 kind 식별자 + 배너 표시용 label을 추가로 싣는다.
// (게임은 저장 시 NpcTemplate 필드만 추려 저장하므로 kind/label은 저장소에 남지 않는다.)
export type NpcWireTemplate = NpcTemplate & { kind: 'npc'; label?: string }

type NpcsByMap = Record<string, PlacedNpc[]>

const loadAll = (): NpcsByMap => {
  const raw = readLocalStorage(PENDING_NPCS_STORAGE_KEY)
  if (!raw) {
    return {}
  }
  try {
    const parsed = JSON.parse(raw) as unknown
    return parsed && typeof parsed === 'object' ? (parsed as NpcsByMap) : {}
  } catch {
    return {}
  }
}

const persist = (all: NpcsByMap): void => {
  if (!writeLocalStorage(PENDING_NPCS_STORAGE_KEY, JSON.stringify(all))) {
    throw new Error('NPC를 저장하지 못했습니다 (브라우저 저장소 차단 또는 용량 초과).')
  }
}

export const loadNpcsForMap = (mapId: string): PlacedNpc[] => {
  const items = loadAll()[mapId]
  return Array.isArray(items) ? items : []
}

// 고유 id — 같은 칸에 여러 NPC를 놓아도 구분되도록 시각 + 난수.
const newId = (): string =>
  `n_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`

export const addNpc = (
  mapId: string,
  template: NpcTemplate,
  col: number,
  row: number
): PlacedNpc => {
  const all = loadAll()
  const npc: PlacedNpc = { ...template, id: newId(), col, row }
  all[mapId] = [...(all[mapId] ?? []), npc]
  persist(all)
  return npc
}

export const removeNpc = (mapId: string, id: string): PlacedNpc[] => {
  const all = loadAll()
  const next = (all[mapId] ?? []).filter((npc) => npc.id !== id)
  all[mapId] = next
  persist(all)
  return next
}

export const clearNpcsForMap = (mapId: string): void => {
  const all = loadAll()
  delete all[mapId]
  persist(all)
}
