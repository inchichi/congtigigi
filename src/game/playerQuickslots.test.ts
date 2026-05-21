import { describe, expect, it } from 'vitest'

import {
  clearPlayerQuickslotAssignment,
  createInitialPlayerQuickslots,
  findPlayerQuickslotIndexByInventorySlotIndex,
  setPlayerQuickslotAssignment
} from './playerQuickslots'

describe('createInitialPlayerQuickslots', () => {
  it('creates six empty quickslots by default', () => {
    expect(createInitialPlayerQuickslots()).toEqual({
      slots: Array.from({ length: 6 }, () => undefined)
    })
  })

  it('supports custom slot counts', () => {
    expect(createInitialPlayerQuickslots({ slotCount: 3 })).toEqual({
      slots: Array.from({ length: 3 }, () => undefined)
    })
  })
})

describe('setPlayerQuickslotAssignment', () => {
  it('assigns an inventory slot to a quickslot', () => {
    const quickslots = createInitialPlayerQuickslots()

    expect(
      setPlayerQuickslotAssignment({
        quickslots,
        quickslotIndex: 2,
        inventorySlotIndex: 5
      })
    ).toEqual({
      slots: [undefined, undefined, { inventorySlotIndex: 5 }, undefined, undefined, undefined]
    })
  })

  it('moves an inventory slot assignment instead of duplicating it', () => {
    const quickslots = setPlayerQuickslotAssignment({
      quickslots: createInitialPlayerQuickslots(),
      quickslotIndex: 1,
      inventorySlotIndex: 3
    })

    expect(
      setPlayerQuickslotAssignment({
        quickslots,
        quickslotIndex: 4,
        inventorySlotIndex: 3
      })
    ).toEqual({
      slots: [undefined, undefined, undefined, undefined, { inventorySlotIndex: 3 }, undefined]
    })
  })

  it('rejects invalid quickslot indices', () => {
    expect(() =>
      setPlayerQuickslotAssignment({
        quickslots: createInitialPlayerQuickslots({ slotCount: 1 }),
        quickslotIndex: 1,
        inventorySlotIndex: 0
      })
    ).toThrow('Invalid quickslot index 1')
  })
})

describe('clearPlayerQuickslotAssignment', () => {
  it('clears the selected quickslot', () => {
    const quickslots = setPlayerQuickslotAssignment({
      quickslots: createInitialPlayerQuickslots(),
      quickslotIndex: 0,
      inventorySlotIndex: 4
    })

    expect(
      clearPlayerQuickslotAssignment({
        quickslots,
        quickslotIndex: 0
      })
    ).toEqual({
      slots: [undefined, undefined, undefined, undefined, undefined, undefined]
    })
  })
})

describe('findPlayerQuickslotIndexByInventorySlotIndex', () => {
  it('returns the assigned quickslot index', () => {
    const quickslots = setPlayerQuickslotAssignment({
      quickslots: createInitialPlayerQuickslots(),
      quickslotIndex: 5,
      inventorySlotIndex: 2
    })

    expect(findPlayerQuickslotIndexByInventorySlotIndex(quickslots, 2)).toBe(5)
  })

  it('returns undefined when the inventory slot is not assigned', () => {
    expect(
      findPlayerQuickslotIndexByInventorySlotIndex(
        createInitialPlayerQuickslots(),
        2
      )
    ).toBeUndefined()
  })
})
