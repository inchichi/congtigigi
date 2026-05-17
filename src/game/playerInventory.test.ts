import { describe, expect, it } from 'vitest'

import {
  clearPlayerInventorySlot,
  createInitialPlayerInventory,
  findFirstEmptyPlayerInventorySlotIndex,
  getPlayerInventoryFilledSlotCount,
  setPlayerInventorySlot
} from './playerInventory'

describe('createInitialPlayerInventory', () => {
  it('creates 12 empty slots by default', () => {
    expect(createInitialPlayerInventory()).toEqual({
      gold: 1000,
      slots: Array.from({ length: 12 }, () => undefined)
    })
  })

  it('supports custom slot counts', () => {
    expect(createInitialPlayerInventory({ slotCount: 4 })).toEqual({
      gold: 1000,
      slots: Array.from({ length: 4 }, () => undefined)
    })
  })

  it('supports custom starting gold', () => {
    expect(createInitialPlayerInventory({ gold: 250 })).toEqual({
      gold: 250,
      slots: Array.from({ length: 12 }, () => undefined)
    })
  })
})

describe('setPlayerInventorySlot', () => {
  it('returns a new inventory with the selected slot filled', () => {
    const inventory = createInitialPlayerInventory({ slotCount: 3 })

    const nextInventory = setPlayerInventorySlot({
      inventory,
      slotIndex: 1,
      item: {
        id: 'potion',
        label: 'Potion',
        quantity: 2
      }
    })

    expect(inventory.slots[1]).toBeUndefined()
    expect(nextInventory.slots[1]).toEqual({
      id: 'potion',
      label: 'Potion',
      quantity: 2
    })
    expect(getPlayerInventoryFilledSlotCount(nextInventory)).toBe(1)
  })

  it('rejects invalid slot indices', () => {
    expect(() =>
      setPlayerInventorySlot({
        inventory: createInitialPlayerInventory({ slotCount: 1 }),
        slotIndex: 1,
        item: {
          id: 'potion',
          label: 'Potion',
          quantity: 1
        }
      })
    ).toThrow('Invalid inventory slot index 1')
  })
})

describe('clearPlayerInventorySlot', () => {
  it('returns a new inventory with the selected slot cleared', () => {
    const inventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2 }),
      slotIndex: 0,
      item: {
        id: 'key',
        label: 'Rusty Key',
        quantity: 1
      }
    })

    const nextInventory = clearPlayerInventorySlot({
      inventory,
      slotIndex: 0
    })

    expect(nextInventory.slots[0]).toBeUndefined()
    expect(getPlayerInventoryFilledSlotCount(nextInventory)).toBe(0)
  })
})

describe('findFirstEmptyPlayerInventorySlotIndex', () => {
  it('returns the first empty slot index', () => {
    const inventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 3 }),
      slotIndex: 1,
      item: {
        id: 'potion',
        label: 'Potion',
        quantity: 1
      }
    })

    expect(findFirstEmptyPlayerInventorySlotIndex(inventory)).toBe(0)
  })

  it('returns undefined when the inventory is full', () => {
    const inventory = {
      gold: 1000,
      slots: [
        {
          id: 'potion',
          label: 'Potion',
          quantity: 1
        },
        {
          id: 'key',
          label: 'Rusty Key',
          quantity: 1
        }
      ]
    }

    expect(findFirstEmptyPlayerInventorySlotIndex(inventory)).toBeUndefined()
  })
})
