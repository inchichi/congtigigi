export type PlayerEquipmentSlotId = 'weapon' | 'armor' | 'boots' | 'accessory'

export type PlayerEquipmentIconKey =
  | 'tiny-dungeon-weapon'
  | 'tiny-knight-gray-helmet'
  | 'tiny-knight-open-helmet'
  | 'town-crate-sword-right'
  | 'ui-circle-beige'
  | 'ui-check-beige'

export type PlayerEquipmentIcon = {
  key: PlayerEquipmentIconKey
  scale: number
}

export type PlayerEquipmentItem = {
  id: string
  label: string
  level: number
  description: string
}

export type PlayerEquipmentItemDefinition = PlayerEquipmentItem & {
  slotId: PlayerEquipmentSlotId
  icon: PlayerEquipmentIcon
  price: number
}

export type PlayerEquipmentSlot = {
  id: PlayerEquipmentSlotId
  label: string
  item: PlayerEquipmentItem | undefined
}

export type PlayerEquipment = {
  setName: string
  level: number
  slots: PlayerEquipmentSlot[]
}

const EQUIPMENT_SLOT_LABEL_BY_ID: Record<PlayerEquipmentSlotId, string> = {
  weapon: '무기',
  armor: '옷',
  boots: '신발',
  accessory: '장신구'
}

const EQUIPMENT_SLOT_IDS: PlayerEquipmentSlotId[] = [
  'weapon',
  'armor',
  'boots',
  'accessory'
]

const PLAYER_EQUIPMENT_ITEM_DEFINITIONS: PlayerEquipmentItemDefinition[] = [
  {
    id: 'basic-sword',
    slotId: 'weapon',
    label: '기본 무기',
    level: 1,
    description: '초보용 근접 무기',
    price: 120,
    icon: {
      key: 'tiny-dungeon-weapon',
      scale: 1.6
    }
  },
  {
    id: 'basic-armor',
    slotId: 'armor',
    label: '기본 옷',
    level: 1,
    description: '초보용 옷',
    price: 100,
    icon: {
      key: 'tiny-knight-gray-helmet',
      scale: 1.4
    }
  },
  {
    id: 'basic-boots',
    slotId: 'boots',
    label: '기본 신발',
    level: 1,
    description: '초보용 신발',
    price: 80,
    icon: {
      key: 'ui-circle-beige',
      scale: 1.2
    }
  },
  {
    id: 'basic-charm',
    slotId: 'accessory',
    label: '기본 장신구',
    level: 1,
    description: '초보용 장신구',
    price: 60,
    icon: {
      key: 'ui-check-beige',
      scale: 1.2
    }
  },
  {
    id: 'bronze-sword',
    slotId: 'weapon',
    label: '청동 검',
    level: 2,
    description: '대장장이가 만든 단단한 검',
    price: 320,
    icon: {
      key: 'town-crate-sword-right',
      scale: 1.5
    }
  },
  {
    id: 'iron-armor',
    slotId: 'armor',
    label: '철 옷',
    level: 2,
    description: '초보용을 넘어선 철제 옷',
    price: 260,
    icon: {
      key: 'tiny-knight-open-helmet',
      scale: 1.4
    }
  },
  {
    id: 'leather-boots',
    slotId: 'boots',
    label: '가죽 신발',
    level: 2,
    description: '가볍고 단단한 가죽 신발',
    price: 180,
    icon: {
      key: 'ui-circle-beige',
      scale: 1.2
    }
  },
  {
    id: 'smith-charm',
    slotId: 'accessory',
    label: '수호 부적',
    level: 2,
    description: '대장장이가 준 작은 보호 부적',
    price: 140,
    icon: {
      key: 'ui-check-beige',
      scale: 1.2
    }
  }
]

const PLAYER_EQUIPMENT_ITEM_DEFINITION_BY_ID = new Map(
  PLAYER_EQUIPMENT_ITEM_DEFINITIONS.map((definition) => [
    definition.id,
    definition
  ])
)
const PLAYER_EQUIPMENT_ITEM_DEFINITION_BY_SLOT_ID = new Map<
  PlayerEquipmentSlotId,
  PlayerEquipmentItemDefinition
>()

for (const definition of PLAYER_EQUIPMENT_ITEM_DEFINITIONS) {
  if (!PLAYER_EQUIPMENT_ITEM_DEFINITION_BY_SLOT_ID.has(definition.slotId)) {
    PLAYER_EQUIPMENT_ITEM_DEFINITION_BY_SLOT_ID.set(definition.slotId, definition)
  }
}

export const createInitialPlayerEquipment = (): PlayerEquipment => ({
  setName: '기본 장비',
  level: 1,
  slots: EQUIPMENT_SLOT_IDS.map((slotId) => {
    const itemDefinition = getPlayerEquipmentItemDefinitionBySlotId(slotId)

    if (!itemDefinition) {
      throw new Error(`Missing equipment item definition for slot ${slotId}`)
    }

    return {
      id: slotId,
      label: EQUIPMENT_SLOT_LABEL_BY_ID[slotId],
      item: createPlayerEquipmentItemFromDefinition(itemDefinition)
    }
  })
})

export const getPlayerEquipmentSlotLabelById = (
  slotId: PlayerEquipmentSlotId
): string => EQUIPMENT_SLOT_LABEL_BY_ID[slotId]

export const createPlayerEquipmentItemFromDefinition = (
  definition: PlayerEquipmentItemDefinition
): PlayerEquipmentItem => ({
  id: definition.id,
  label: definition.label,
  level: definition.level,
  description: definition.description
})

export const getPlayerEquipmentItemDefinitionById = (
  itemId: string
): PlayerEquipmentItemDefinition | undefined =>
  PLAYER_EQUIPMENT_ITEM_DEFINITION_BY_ID.get(itemId)

export const getPlayerEquipmentItemDefinitionBySlotId = (
  slotId: PlayerEquipmentSlotId
): PlayerEquipmentItemDefinition | undefined =>
  PLAYER_EQUIPMENT_ITEM_DEFINITION_BY_SLOT_ID.get(slotId)

export const getPlayerEquipmentSlotById = (
  equipment: PlayerEquipment,
  slotId: PlayerEquipmentSlotId
): PlayerEquipmentSlot | undefined =>
  equipment.slots.find((slot) => slot.id === slotId)

export const findPlayerEquipmentSlotIndexById = (
  equipment: PlayerEquipment,
  slotId: PlayerEquipmentSlotId
): number => equipment.slots.findIndex((slot) => slot.id === slotId)

export const setPlayerEquipmentSlot = ({
  equipment,
  slotId,
  item
}: {
  equipment: PlayerEquipment
  slotId: PlayerEquipmentSlotId
  item: PlayerEquipmentItem
}): PlayerEquipment => {
  const slotIndex = findPlayerEquipmentSlotIndexById(equipment, slotId)

  if (slotIndex < 0) {
    throw new Error(`Invalid equipment slot id ${slotId}`)
  }

  const slots = [...equipment.slots]
  const slot = slots[slotIndex]

  slots[slotIndex] = {
    ...slot,
    item
  }

  return {
    ...equipment,
    slots
  }
}

export const clearPlayerEquipmentSlot = ({
  equipment,
  slotId
}: {
  equipment: PlayerEquipment
  slotId: PlayerEquipmentSlotId
}): PlayerEquipment => {
  const slotIndex = findPlayerEquipmentSlotIndexById(equipment, slotId)

  if (slotIndex < 0) {
    throw new Error(`Invalid equipment slot id ${slotId}`)
  }

  const slots = [...equipment.slots]
  const slot = slots[slotIndex]

  slots[slotIndex] = {
    ...slot,
    item: undefined
  }

  return {
    ...equipment,
    slots
  }
}
