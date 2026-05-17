export type PlayerInventoryItem = {
  id: string
  label: string
  quantity: number
}

export type PlayerInventorySlot = PlayerInventoryItem | undefined

export type PlayerInventory = {
  slots: PlayerInventorySlot[]
}

const DEFAULT_PLAYER_INVENTORY_SLOT_COUNT = 12

type CreateInitialPlayerInventoryInput = {
  slotCount?: number
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
  slotCount = DEFAULT_PLAYER_INVENTORY_SLOT_COUNT
}: CreateInitialPlayerInventoryInput = {}): PlayerInventory => ({
  slots: Array.from({ length: slotCount }, () => undefined)
})

export const setPlayerInventorySlot = ({
  inventory,
  slotIndex,
  item
}: SetPlayerInventorySlotInput): PlayerInventory => {
  assertPlayerInventorySlotIndex(inventory, slotIndex)

  const slots = [...inventory.slots]

  slots[slotIndex] = item

  return {
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
