export type PlayerInventoryItem = {
  id: string
  label: string
  quantity: number
}

export type PlayerInventorySlot = PlayerInventoryItem | undefined

export type PlayerInventory = {
  gold: number
  slots: PlayerInventorySlot[]
}

const DEFAULT_PLAYER_INVENTORY_SLOT_COUNT = 30
const DEFAULT_PLAYER_INVENTORY_GOLD = 1000
const DEFAULT_PLAYER_INVENTORY_STARTER_ITEMS: PlayerInventoryItem[] = [
  {
    id: 'health-potion',
    label: '체력 회복 포션',
    quantity: 30
  },
  {
    id: 'mana-potion',
    label: '마나 회복 포션',
    quantity: 30
  }
]

type CreateInitialPlayerInventoryInput = {
  slotCount?: number
  gold?: number
}

type SetPlayerInventorySlotInput = {
  inventory: PlayerInventory
  slotIndex: number
  item: PlayerInventoryItem
}

type ClearPlayerInventorySlotInput = {
  inventory: PlayerInventory
  slotIndex: number
}

export const createInitialPlayerInventory = ({
  slotCount,
  gold = DEFAULT_PLAYER_INVENTORY_GOLD
}: CreateInitialPlayerInventoryInput = {}): PlayerInventory => {
  const resolvedSlotCount = slotCount ?? DEFAULT_PLAYER_INVENTORY_SLOT_COUNT
  const starterItems = slotCount === undefined ? DEFAULT_PLAYER_INVENTORY_STARTER_ITEMS : []

  return {
    gold,
    slots: Array.from({ length: resolvedSlotCount }, (_, index) =>
      starterItems[index] ? { ...starterItems[index] } : undefined
    )
  }
}

export const setPlayerInventorySlot = ({
  inventory,
  slotIndex,
  item
}: SetPlayerInventorySlotInput): PlayerInventory => {
  assertPlayerInventorySlotIndex(inventory, slotIndex)

  const slots = [...inventory.slots]

  slots[slotIndex] = item

  return {
    gold: inventory.gold,
    slots
  }
}

export const clearPlayerInventorySlot = ({
  inventory,
  slotIndex
}: ClearPlayerInventorySlotInput): PlayerInventory => {
  assertPlayerInventorySlotIndex(inventory, slotIndex)

  const slots = [...inventory.slots]

  slots[slotIndex] = undefined

  return {
    gold: inventory.gold,
    slots
  }
}

export const getPlayerInventoryFilledSlotCount = (
  inventory: PlayerInventory
): number => inventory.slots.filter(isPlayerInventoryItem).length

export const findFirstEmptyPlayerInventorySlotIndex = (
  inventory: PlayerInventory
): number | undefined => {
  const slotIndex = inventory.slots.findIndex((slot) => slot === undefined)

  return slotIndex < 0 ? undefined : slotIndex
}

const assertPlayerInventorySlotIndex = (
  inventory: PlayerInventory,
  slotIndex: number
) => {
  if (slotIndex < 0 || slotIndex >= inventory.slots.length) {
    throw new Error(`Invalid inventory slot index ${slotIndex}`)
  }
}

const isPlayerInventoryItem = (
  slot: PlayerInventorySlot
): slot is PlayerInventoryItem => slot !== undefined
