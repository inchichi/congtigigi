import { describe, expect, it } from 'vitest'

import {
  createInitialPlayerInventory,
  setPlayerInventorySlot
} from './playerInventory'
import {
  buyPotionShopItem,
  createInitialPotionInventory,
  sellPotionShopItem
} from './potionShop'

describe('createInitialPotionInventory', () => {
  it('creates stocked potion merchant inventory with gold', () => {
    expect(createInitialPotionInventory({ slotCount: 4, gold: 4200 })).toEqual({
      gold: 4200,
      slots: [
        {
          id: 'health-potion',
          label: '체력 회복 포션',
          quantity: 999_999
        },
        {
          id: 'mana-potion',
          label: '마나 회복 포션',
          quantity: 999_999
        },
        undefined,
        undefined
      ]
    })
  })
})

describe('buyPotionShopItem', () => {
  it('moves a requested potion quantity from the merchant inventory into the player inventory', () => {
    const playerInventory = createInitialPlayerInventory({ slotCount: 2, gold: 1000 })
    const merchantInventory = createInitialPotionInventory({
      slotCount: 4,
      gold: 5000
    })

    const result = buyPotionShopItem({
      playerInventory,
      merchantInventory,
      merchantSlotIndex: 0,
      quantity: 12
    })

    expect(result).toEqual({
      ok: true,
      playerInventory: {
        gold: 880,
        slots: [
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 12
          },
          undefined
        ]
      },
      merchantInventory: {
        gold: 5120,
        slots: [
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 999_987
          },
          {
            id: 'mana-potion',
            label: '마나 회복 포션',
            quantity: 999_999
          },
          undefined,
          undefined
        ]
      },
      message: '12개를 구매했습니다.'
    })
  })

  it('stacks purchased items into an existing slot when the player inventory is full', () => {
    const playerInventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 1, gold: 1000 }),
      slotIndex: 0,
      item: {
        id: 'health-potion',
        label: '체력 회복 포션',
        quantity: 5
      }
    })
    const merchantInventory = createInitialPotionInventory({
      slotCount: 4,
      gold: 5000
    })

    const result = buyPotionShopItem({
      playerInventory,
      merchantInventory,
      merchantSlotIndex: 0,
      quantity: 3
    })

    expect(result).toEqual({
      ok: true,
      playerInventory: {
        gold: 970,
        slots: [
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 8
          }
        ]
      },
      merchantInventory: {
        gold: 5030,
        slots: [
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 999_996
          },
          {
            id: 'mana-potion',
            label: '마나 회복 포션',
            quantity: 999_999
          },
          undefined,
          undefined
        ]
      },
      message: '3개를 구매했습니다.'
    })
  })

  it('rejects purchases when the player cannot afford the item', () => {
    const playerInventory = createInitialPlayerInventory({ slotCount: 2, gold: 5 })
    const merchantInventory = createInitialPotionInventory({
      slotCount: 4,
      gold: 5000
    })

    const result = buyPotionShopItem({
      playerInventory,
      merchantInventory,
      merchantSlotIndex: 0
    })

    expect(result).toEqual({
      ok: false,
      playerInventory,
      merchantInventory,
      message: '돈이 부족합니다.'
    })
  })
})

describe('sellPotionShopItem', () => {
  it('sells a requested potion quantity for gold without changing merchant stock', () => {
    const playerInventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2, gold: 1000 }),
      slotIndex: 0,
      item: {
        id: 'mana-potion',
        label: '마나 회복 포션',
        quantity: 30
      }
    })
    const merchantInventory = createInitialPotionInventory({
      slotCount: 4,
      gold: 5000
    })

    const result = sellPotionShopItem({
      playerInventory,
      merchantInventory,
      playerSlotIndex: 0,
      quantity: 5
    })

    expect(result).toEqual({
      ok: true,
      playerInventory: {
        gold: 1035,
        slots: [
          {
            id: 'mana-potion',
            label: '마나 회복 포션',
            quantity: 25
          },
          undefined
        ]
      },
      merchantInventory: {
        gold: 4965,
        slots: [
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 999_999
          },
          {
            id: 'mana-potion',
            label: '마나 회복 포션',
            quantity: 999_999
          },
          undefined,
          undefined
        ]
      },
      message: '5개를 판매했습니다.'
    })
  })

  it('sells equipment items for gold as well', () => {
    const playerInventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2, gold: 1000 }),
      slotIndex: 0,
      item: {
        id: 'bronze-sword',
        label: '청동 검',
        quantity: 1
      }
    })
    const merchantInventory = createInitialPotionInventory({
      slotCount: 4,
      gold: 5000
    })

    const result = sellPotionShopItem({
      playerInventory,
      merchantInventory,
      playerSlotIndex: 0
    })

    expect(result).toEqual({
      ok: true,
      playerInventory: {
        gold: 1160,
        slots: [undefined, undefined]
      },
      merchantInventory: {
        gold: 4840,
        slots: [
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 999_999
          },
          {
            id: 'mana-potion',
            label: '마나 회복 포션',
            quantity: 999_999
          },
          undefined,
          undefined
        ]
      },
      message: '1개를 판매했습니다.'
    })
  })

  it('rejects sales for unknown items', () => {
    const playerInventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2, gold: 1000 }),
      slotIndex: 0,
      item: {
        id: 'unknown-item',
        label: 'Unknown Item',
        quantity: 1
      }
    })
    const merchantInventory = createInitialPotionInventory({
      slotCount: 4,
      gold: 5000
    })

    const result = sellPotionShopItem({
      playerInventory,
      merchantInventory,
      playerSlotIndex: 0
    })

    expect(result).toEqual({
      ok: false,
      playerInventory,
      merchantInventory,
      message: '이 상점에서는 매입하지 않는 아이템입니다.'
    })
  })
})
