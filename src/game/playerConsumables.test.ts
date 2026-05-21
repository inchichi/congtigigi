import { describe, expect, it } from 'vitest'

import { createInitialPlayerInventory, setPlayerInventorySlot } from './playerInventory'
import { createInitialPlayerProfile } from './playerProfile'
import { usePlayerInventoryConsumable } from './playerConsumables'

describe('usePlayerInventoryConsumable', () => {
  it('restores hp and consumes one health potion', () => {
    const profile = {
      ...createInitialPlayerProfile(),
      hp: {
        current: 13,
        max: 24
      }
    }
    const inventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 3 }),
      slotIndex: 2,
      item: {
        id: 'health-potion',
        label: '체력 회복 포션',
        quantity: 2
      }
    })

    const nextState = usePlayerInventoryConsumable({
      profile,
      inventory,
      slotIndex: 2
    })

    expect(nextState).toEqual({
      profile: {
        ...profile,
        hp: {
          current: 23,
          max: 24
        }
      },
      inventory: {
        gold: 1000,
        slots: [
          undefined,
          undefined,
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 1
          }
        ]
      }
    })
  })

  it('restores mp and clears the slot when the last mana potion is used', () => {
    const profile = {
      ...createInitialPlayerProfile(),
      mp: {
        current: 3,
        max: 12
      }
    }
    const inventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2 }),
      slotIndex: 1,
      item: {
        id: 'mana-potion',
        label: '마나 회복 포션',
        quantity: 1
      }
    })

    const nextState = usePlayerInventoryConsumable({
      profile,
      inventory,
      slotIndex: 1
    })

    expect(nextState).toEqual({
      profile: {
        ...profile,
        mp: {
          current: 12,
          max: 12
        }
      },
      inventory: {
        gold: 1000,
        slots: [undefined, undefined]
      }
    })
  })

  it('ignores non-consumable items', () => {
    const inventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 1 }),
      slotIndex: 0,
      item: {
        id: 'key',
        label: 'Rusty Key',
        quantity: 1
      }
    })

    expect(
      usePlayerInventoryConsumable({
        profile: createInitialPlayerProfile(),
        inventory,
        slotIndex: 0
      })
    ).toBeUndefined()
  })
})
