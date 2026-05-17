import {
  clearPlayerEquipmentSlot,
  createPlayerEquipmentItemFromDefinition,
  getPlayerEquipmentItemDefinitionById,
  setPlayerEquipmentSlot,
  type PlayerEquipment,
  type PlayerEquipmentItem,
  type PlayerEquipmentSlotId
} from './playerEquipment'
import {
  clearPlayerInventorySlot,
  findFirstEmptyPlayerInventorySlotIndex,
  setPlayerInventorySlot,
  type PlayerInventory,
  type PlayerInventoryItem
} from './playerInventory'

export type PlayerLoadoutState = {
  equipment: PlayerEquipment
  inventory: PlayerInventory
}

type UnequipPlayerEquipmentSlotInput = PlayerLoadoutState & {
  slotId: PlayerEquipmentSlotId
}

type EquipPlayerInventorySlotInput = PlayerLoadoutState & {
  slotIndex: number
}

const createPlayerInventoryItemFromEquipmentItem = (
  item: PlayerEquipmentItem
): PlayerInventoryItem => ({
  id: item.id,
  label: item.label,
  quantity: 1
})

export const unequipPlayerEquipmentSlot = ({
  equipment,
  inventory,
  slotId
}: UnequipPlayerEquipmentSlotInput): PlayerLoadoutState | undefined => {
  const equipmentSlot = equipment.slots.find((slot) => slot.id === slotId)

  if (!equipmentSlot?.item) {
    return undefined
  }

  const emptySlotIndex = findFirstEmptyPlayerInventorySlotIndex(inventory)

  if (emptySlotIndex === undefined) {
    return undefined
  }

  const nextInventory = setPlayerInventorySlot({
    inventory,
    slotIndex: emptySlotIndex,
    item: createPlayerInventoryItemFromEquipmentItem(equipmentSlot.item)
  })

  return {
    equipment: clearPlayerEquipmentSlot({ equipment, slotId }),
    inventory: nextInventory
  }
}

export const equipPlayerInventorySlot = ({
  equipment,
  inventory,
  slotIndex
}: EquipPlayerInventorySlotInput): PlayerLoadoutState | undefined => {
  const item = inventory.slots[slotIndex]

  if (!item) {
    return undefined
  }

  const definition = getPlayerEquipmentItemDefinitionById(item.id)

  if (!definition) {
    return undefined
  }

  const equipmentSlot = equipment.slots.find(
    (slot) => slot.id === definition.slotId
  )

  if (!equipmentSlot) {
    return undefined
  }

  const equippedItem = equipmentSlot.item
  const nextEquipment = setPlayerEquipmentSlot({
    equipment,
    slotId: definition.slotId,
    item: createPlayerEquipmentItemFromDefinition(definition)
  })
  let nextInventory = clearPlayerInventorySlot({
    inventory,
    slotIndex
  })

  if (equippedItem) {
    nextInventory = setPlayerInventorySlot({
      inventory: nextInventory,
      slotIndex,
      item: createPlayerInventoryItemFromEquipmentItem(equippedItem)
    })
  }

  return {
    equipment: nextEquipment,
    inventory: nextInventory
  }
}
