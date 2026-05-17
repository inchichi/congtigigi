import { describe, expect, it } from 'vitest'

import {
  createInitialPlayerEquipment,
  setPlayerEquipmentSlot
} from './playerEquipment'
import {
  createInitialPlayerInventory,
  setPlayerInventorySlot
} from './playerInventory'
import {
  equipPlayerInventorySlot,
  unequipPlayerEquipmentSlot
} from './playerLoadout'

describe('unequipPlayerEquipmentSlot', () => {
  it('moves the equipped item into the first empty inventory slot', () => {
    const equipment = createInitialPlayerEquipment()
    const inventory = createInitialPlayerInventory({ slotCount: 3 })

    const nextState = unequipPlayerEquipmentSlot({
      equipment,
      inventory,
      slotId: 'weapon'
    })

    expect(nextState).toEqual({
      equipment: {
        ...equipment,
        slots: [
          {
            id: 'weapon',
            label: '무기',
            item: undefined
          },
          equipment.slots[1],
          equipment.slots[2],
          equipment.slots[3]
        ]
      },
      inventory: {
        gold: 1000,
        slots: [
          {
            id: 'basic-sword',
            label: '기본 무기',
            quantity: 1
          },
          undefined,
          undefined
        ]
      }
    })
  })
})

describe('equipPlayerInventorySlot', () => {
  it('equips a matching item and swaps back the previous equipment', () => {
    const equipment = setPlayerEquipmentSlot({
      equipment: createInitialPlayerEquipment(),
      slotId: 'weapon',
      item: {
        id: 'basic-armor',
        label: '기본 옷',
        level: 1,
        description: '초보용 옷'
      }
    })
    const inventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2 }),
      slotIndex: 0,
      item: {
        id: 'basic-sword',
        label: '기본 무기',
        quantity: 1
      }
    })

    const nextState = equipPlayerInventorySlot({
      equipment,
      inventory,
      slotIndex: 0
    })

    expect(nextState).toEqual({
      equipment: {
        ...equipment,
        slots: [
          {
            id: 'weapon',
            label: '무기',
            item: {
              id: 'basic-sword',
              label: '기본 무기',
              level: 1,
              description: '초보용 근접 무기'
            }
          },
          equipment.slots[1],
          equipment.slots[2],
          equipment.slots[3]
        ]
      },
      inventory: {
        gold: 1000,
        slots: [
          {
            id: 'basic-armor',
            label: '기본 옷',
            quantity: 1
          },
          undefined
        ]
      }
    })
  })
})
