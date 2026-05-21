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
          quantity: 30
        },
        {
          id: 'mana-potion',
          label: '마나 회복 포션',
          quantity: 30
        },
        undefined,
        undefined
      ]
    })
  })
})

describe('buyPotionShopItem', () => {
  it('moves a potion stack from the merchant inventory into the player inventory', () => {
    const playerInventory = createInitialPlayerInventory({ slotCount: 2, gold: 1000 })
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
      ok: true,
      playerInventory: {
        gold: 990,
        slots: [
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 30
          },
          undefined
        ]
      },
      merchantInventory: {
        gold: 5010,
        slots: [
          undefined,
          {
            id: 'mana-potion',
            label: '마나 회복 포션',
            quantity: 30
          },
          undefined,
          undefined
        ]
      },
      message: '구매했습니다.'
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
  it('sells a potion stack for gold without changing merchant stock', () => {
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
      playerSlotIndex: 0
    })

    expect(result).toEqual({
      ok: true,
      playerInventory: {
        gold: 1007,
        slots: [undefined, undefined]
      },
      merchantInventory: {
        gold: 4993,
        slots: [
          {
            id: 'health-potion',
            label: '체력 회복 포션',
            quantity: 30
          },
          {
            id: 'mana-potion',
            label: '마나 회복 포션',
            quantity: 30
          },
          undefined,
          undefined
        ]
      },
      message: '판매했습니다.'
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
            quantity: 30
          },
          {
            id: 'mana-potion',
            label: '마나 회복 포션',
            quantity: 30
          },
          undefined,
          undefined
        ]
      },
      message: '판매했습니다.'
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
