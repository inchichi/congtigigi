import type {
  PlayerEquipment,
  PlayerEquipmentItem,
  PlayerEquipmentSlot,
  PlayerEquipmentSlotId
} from './playerEquipment'
import type {
  PlayerInventory,
  PlayerInventoryItem,
  PlayerInventorySlot
} from './playerInventory'
import type {
  PlayerExperience,
  PlayerProfile,
  PlayerResource,
  PlayerSkillSlot,
  PlayerStatBlock
} from './playerProfile'
import type {
  PlayerQuickslotAssignment,
  PlayerQuickslots
} from './playerQuickslots'
import type {
  PlayerSkillSlotAssignment,
  PlayerSkillSlots
} from './playerSkillSlots'

export const PLAYER_SAVE_STATE_STORAGE_KEY = 'my-sample-rpg:player-save-state'

// 저장 포맷이 바뀔 때마다 올린다. 버전이 다른 저장본은 normalize 단계에서 폐기되어
// 게임이 깨끗한 초기 상태로 시작한다. (구버전 데이터로 인한 런타임 크래시 방지)
export const PLAYER_SAVE_STATE_VERSION = 1

// 영속화 대상인 핵심 플레이어 상태 묶음. 모두 순수 데이터(함수/클래스 인스턴스 없음)라
// 그대로 JSON 직렬화된다. 월드 상태(상점 재고·퀘스트 로그)는 v1 범위에서 제외한다.
export type PlayerSaveStateInput = {
  profile: PlayerProfile
  equipment: PlayerEquipment
  inventory: PlayerInventory
  quickslots: PlayerQuickslots
  skillSlots: PlayerSkillSlots
}

export type PlayerSaveState = PlayerSaveStateInput & {
  version: number
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const isString = (value: unknown): value is string => typeof value === 'string'

const normalizeResource = (value: unknown): PlayerResource | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  if (!isFiniteNumber(value.current) || !isFiniteNumber(value.max)) {
    return undefined
  }

  return { current: value.current, max: value.max }
}

const normalizeExperience = (value: unknown): PlayerExperience | undefined => {
  if (!isRecord(value) || !isFiniteNumber(value.current)) {
    return undefined
  }

  return { current: value.current }
}

const normalizeStatBlock = (value: unknown): PlayerStatBlock | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  const { strength, agility, intelligence, luck } = value

  if (
    !isFiniteNumber(strength) ||
    !isFiniteNumber(agility) ||
    !isFiniteNumber(intelligence) ||
    !isFiniteNumber(luck)
  ) {
    return undefined
  }

  return { strength, agility, intelligence, luck }
}

const normalizeSkillSlot = (value: unknown): PlayerSkillSlot | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  if (
    !isString(value.hotkey) ||
    !isString(value.label) ||
    !isString(value.description) ||
    !isFiniteNumber(value.level) ||
    !isFiniteNumber(value.maxLevel)
  ) {
    return undefined
  }

  return {
    hotkey: value.hotkey,
    label: value.label,
    description: value.description,
    level: value.level,
    maxLevel: value.maxLevel
  }
}

const normalizeProfile = (value: unknown): PlayerProfile | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  if (
    !isString(value.name) ||
    !isString(value.job) ||
    !isFiniteNumber(value.level) ||
    !isFiniteNumber(value.statPoints) ||
    !isFiniteNumber(value.availableSkillPoints) ||
    !isFiniteNumber(value.totalSkillPointsEarned)
  ) {
    return undefined
  }

  const experience = normalizeExperience(value.experience)
  const hp = normalizeResource(value.hp)
  const mp = normalizeResource(value.mp)
  const stats = normalizeStatBlock(value.stats)

  if (!experience || !hp || !mp || !stats || !Array.isArray(value.skills)) {
    return undefined
  }

  // 스킬 슬롯은 고정 구성이라 하나라도 손상되면 저장본 전체를 폐기한다.
  const skills: PlayerSkillSlot[] = []

  for (const rawSkill of value.skills) {
    const skill = normalizeSkillSlot(rawSkill)

    if (!skill) {
      return undefined
    }

    skills.push(skill)
  }

  return {
    name: value.name,
    job: value.job,
    level: value.level,
    experience,
    statPoints: value.statPoints,
    availableSkillPoints: value.availableSkillPoints,
    totalSkillPointsEarned: value.totalSkillPointsEarned,
    hp,
    mp,
    stats,
    skills
  }
}

const normalizeInventoryItem = (
  value: unknown
): PlayerInventoryItem | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  if (
    !isString(value.id) ||
    !isString(value.label) ||
    !isFiniteNumber(value.quantity)
  ) {
    return undefined
  }

  return { id: value.id, label: value.label, quantity: value.quantity }
}

const normalizeInventory = (value: unknown): PlayerInventory | undefined => {
  if (!isRecord(value) || !isFiniteNumber(value.gold) || !Array.isArray(value.slots)) {
    return undefined
  }

  // JSON에서 빈 슬롯(undefined)은 null로 직렬화된다. 손상된 아이템도 빈 슬롯으로 관대하게 처리한다.
  const slots: PlayerInventorySlot[] = value.slots.map((rawSlot) =>
    rawSlot == null ? undefined : normalizeInventoryItem(rawSlot)
  )

  return { gold: value.gold, slots }
}

const normalizeEquipmentItem = (
  value: unknown
): PlayerEquipmentItem | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  if (
    !isString(value.id) ||
    !isString(value.label) ||
    !isString(value.description) ||
    !isFiniteNumber(value.level)
  ) {
    return undefined
  }

  return {
    id: value.id,
    label: value.label,
    level: value.level,
    description: value.description
  }
}

const normalizeEquipmentSlot = (
  value: unknown
): PlayerEquipmentSlot | undefined => {
  if (!isRecord(value) || !isString(value.id) || !isString(value.label)) {
    return undefined
  }

  const item =
    value.item == null ? undefined : normalizeEquipmentItem(value.item)

  return {
    id: value.id as PlayerEquipmentSlotId,
    label: value.label,
    item
  }
}

const normalizeEquipment = (value: unknown): PlayerEquipment | undefined => {
  if (
    !isRecord(value) ||
    !isString(value.setName) ||
    !isFiniteNumber(value.level) ||
    !Array.isArray(value.slots)
  ) {
    return undefined
  }

  // 장비 슬롯은 고정 구성(무기/옷/모자/신발/장신구)이라 하나라도 손상되면 폐기한다.
  const slots: PlayerEquipmentSlot[] = []

  for (const rawSlot of value.slots) {
    const slot = normalizeEquipmentSlot(rawSlot)

    if (!slot) {
      return undefined
    }

    slots.push(slot)
  }

  return { setName: value.setName, level: value.level, slots }
}

const normalizeQuickslotAssignment = (
  value: unknown
): PlayerQuickslotAssignment | undefined => {
  if (!isRecord(value) || !isFiniteNumber(value.inventorySlotIndex)) {
    return undefined
  }

  return { inventorySlotIndex: value.inventorySlotIndex }
}

const normalizeQuickslots = (value: unknown): PlayerQuickslots | undefined => {
  if (!isRecord(value) || !Array.isArray(value.slots)) {
    return undefined
  }

  const slots = value.slots.map((rawSlot) =>
    rawSlot == null ? undefined : normalizeQuickslotAssignment(rawSlot)
  )

  return { slots }
}

const normalizeSkillSlotAssignment = (
  value: unknown
): PlayerSkillSlotAssignment | undefined => {
  if (!isRecord(value) || !isString(value.skillId) || value.skillId.length === 0) {
    return undefined
  }

  return { skillId: value.skillId }
}

const normalizeSkillSlots = (value: unknown): PlayerSkillSlots | undefined => {
  if (!isRecord(value) || !Array.isArray(value.slots)) {
    return undefined
  }

  const slots = value.slots.map((rawSlot) =>
    rawSlot == null ? undefined : normalizeSkillSlotAssignment(rawSlot)
  )

  return { slots }
}

export const serializePlayerSaveState = (
  input: PlayerSaveStateInput
): string => {
  const saveState: PlayerSaveState = {
    version: PLAYER_SAVE_STATE_VERSION,
    ...input
  }

  return JSON.stringify(saveState)
}

// 저장된(신뢰할 수 없는) 값을 검증·정규화한다. 버전 불일치나 구조 손상이면 undefined를 돌려
// 호출부가 깨끗한 초기 상태로 시작하게 한다.
export const normalizeStoredPlayerSaveState = (
  value: unknown
): PlayerSaveState | undefined => {
  if (!isRecord(value) || value.version !== PLAYER_SAVE_STATE_VERSION) {
    return undefined
  }

  const profile = normalizeProfile(value.profile)
  const equipment = normalizeEquipment(value.equipment)
  const inventory = normalizeInventory(value.inventory)
  const quickslots = normalizeQuickslots(value.quickslots)
  const skillSlots = normalizeSkillSlots(value.skillSlots)

  if (!profile || !equipment || !inventory || !quickslots || !skillSlots) {
    return undefined
  }

  return {
    version: PLAYER_SAVE_STATE_VERSION,
    profile,
    equipment,
    inventory,
    quickslots,
    skillSlots
  }
}

// 직렬화 문자열을 파싱해 검증된 저장 상태로 되돌린다. 파싱 실패/손상 시 undefined.
export const parseStoredPlayerSaveState = (
  raw: string | null | undefined
): PlayerSaveState | undefined => {
  if (!raw) {
    return undefined
  }

  try {
    return normalizeStoredPlayerSaveState(JSON.parse(raw))
  } catch {
    return undefined
  }
}
