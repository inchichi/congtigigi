export type PlayerQuickslotAssignment = {
  inventorySlotIndex: number
}

export type PlayerQuickslots = {
  slots: Array<PlayerQuickslotAssignment | undefined>
}

const DEFAULT_PLAYER_QUICKSLOT_COUNT = 6

type CreateInitialPlayerQuickslotsInput = {
  slotCount?: number
}

type SetPlayerQuickslotAssignmentInput = {
  quickslots: PlayerQuickslots
  quickslotIndex: number
  inventorySlotIndex: number
}

type ClearPlayerQuickslotAssignmentInput = {
  quickslots: PlayerQuickslots
  quickslotIndex: number
}

export const createInitialPlayerQuickslots = ({
  slotCount = DEFAULT_PLAYER_QUICKSLOT_COUNT
}: CreateInitialPlayerQuickslotsInput = {}): PlayerQuickslots => ({
  slots: Array.from({ length: slotCount }, () => undefined)
})

export const setPlayerQuickslotAssignment = ({
  quickslots,
  quickslotIndex,
  inventorySlotIndex
}: SetPlayerQuickslotAssignmentInput): PlayerQuickslots => {
  assertPlayerQuickslotIndex(quickslots, quickslotIndex)

  if (inventorySlotIndex < 0) {
    throw new Error(`Invalid inventory slot index ${inventorySlotIndex}`)
  }

  const slots = quickslots.slots.map((slot, index) =>
    index !== quickslotIndex && slot?.inventorySlotIndex === inventorySlotIndex
      ? undefined
      : slot
  )

  slots[quickslotIndex] = {
    inventorySlotIndex
  }

  return {
    slots
  }
}

export const clearPlayerQuickslotAssignment = ({
  quickslots,
  quickslotIndex
}: ClearPlayerQuickslotAssignmentInput): PlayerQuickslots => {
  assertPlayerQuickslotIndex(quickslots, quickslotIndex)

  const slots = [...quickslots.slots]

  slots[quickslotIndex] = undefined

  return {
    slots
  }
}

export const findPlayerQuickslotIndexByInventorySlotIndex = (
  quickslots: PlayerQuickslots,
  inventorySlotIndex: number
): number | undefined => {
  const quickslotIndex = quickslots.slots.findIndex(
    (slot) => slot?.inventorySlotIndex === inventorySlotIndex
  )

  return quickslotIndex < 0 ? undefined : quickslotIndex
}

export const getPlayerQuickslotAssignment = (
  quickslots: PlayerQuickslots,
  quickslotIndex: number
): PlayerQuickslotAssignment | undefined => quickslots.slots[quickslotIndex]

const assertPlayerQuickslotIndex = (
  quickslots: PlayerQuickslots,
  quickslotIndex: number
) => {
  if (quickslotIndex < 0 || quickslotIndex >= quickslots.slots.length) {
    throw new Error(`Invalid quickslot index ${quickslotIndex}`)
  }
}
