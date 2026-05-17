import { describe, expect, it } from 'vitest'

import {
  createInitialPlayerInventory,
  setPlayerInventorySlot
} from './playerInventory'
import {
  buyBlacksmithShopItem,
  createInitialBlacksmithInventory,
  sellBlacksmithShopItem
} from './blacksmithShop'

describe('createInitialBlacksmithInventory', () => {
  it('creates stocked merchant inventory with gold', () => {
    expect(createInitialBlacksmithInventory({ slotCount: 6, gold: 4200 })).toEqual(
      {
        gold: 4200,
        slots: [
          {
            id: 'bronze-sword',
            label: '청동 검',
            quantity: 1
          },
          {
            id: 'iron-armor',
            label: '철 옷',
            quantity: 1
          },
          {
            id: 'leather-boots',
            label: '가죽 신발',
            quantity: 1
          },
          {
            id: 'smith-charm',
            label: '수호 부적',
            quantity: 1
          },
          undefined,
          undefined
        ]
      }
    )
  })
})

describe('buyBlacksmithShopItem', () => {
  it('moves an item from the merchant inventory into the player inventory', () => {
    const playerInventory = createInitialPlayerInventory({ slotCount: 2, gold: 1000 })
    const merchantInventory = createInitialBlacksmithInventory({
      slotCount: 6,
      gold: 5000
    })

    const result = buyBlacksmithShopItem({
      playerInventory,
      merchantInventory,
      merchantSlotIndex: 0
    })

    expect(result).toEqual({
      ok: true,
      playerInventory: {
        gold: 680,
        slots: [
          {
            id: 'bronze-sword',
            label: '청동 검',
            quantity: 1
          },
          undefined
        ]
      },
      merchantInventory: {
        gold: 5320,
        slots: [
          undefined,
          {
            id: 'iron-armor',
            label: '철 옷',
            quantity: 1
          },
          {
            id: 'leather-boots',
            label: '가죽 신발',
            quantity: 1
          },
          {
            id: 'smith-charm',
            label: '수호 부적',
            quantity: 1
          },
          undefined,
          undefined
        ]
      },
      message: '구매했습니다.'
    })
  })

  it('rejects purchases when the player cannot afford the item', () => {
    const playerInventory = createInitialPlayerInventory({ slotCount: 2, gold: 100 })
    const merchantInventory = createInitialBlacksmithInventory({
      slotCount: 6,
      gold: 5000
    })

    const result = buyBlacksmithShopItem({
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

  it('rejects purchases when the player inventory is full', () => {
    const playerInventory = {
      gold: 1000,
      slots: [
        {
          id: 'potion',
          label: 'Potion',
          quantity: 1
        }
      ]
    }
    const merchantInventory = createInitialBlacksmithInventory({
      slotCount: 4,
      gold: 5000
    })

    const result = buyBlacksmithShopItem({
      playerInventory,
      merchantInventory,
      merchantSlotIndex: 0
    })

    expect(result).toEqual({
      ok: false,
      playerInventory,
      merchantInventory,
      message: '인벤토리가 가득 찼습니다.'
    })
  })
})

describe('sellBlacksmithShopItem', () => {
  it('moves an item from the player inventory into the merchant inventory', () => {
    const playerInventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2, gold: 1000 }),
      slotIndex: 0,
      item: {
        id: 'bronze-sword',
        label: '청동 검',
        quantity: 1
      }
    })
    const merchantInventory = createInitialBlacksmithInventory({
      slotCount: 6,
      gold: 5000
    })

    const result = sellBlacksmithShopItem({
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
            id: 'bronze-sword',
            label: '청동 검',
            quantity: 1
          },
          {
            id: 'iron-armor',
            label: '철 옷',
            quantity: 1
          },
          {
            id: 'leather-boots',
            label: '가죽 신발',
            quantity: 1
          },
          {
            id: 'smith-charm',
            label: '수호 부적',
            quantity: 1
          },
          {
            id: 'bronze-sword',
            label: '청동 검',
            quantity: 1
          },
          undefined
        ]
      },
      message: '판매했습니다.'
    })
  })

  it('rejects sales for items the blacksmith does not buy', () => {
    const playerInventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2, gold: 1000 }),
      slotIndex: 0,
      item: {
        id: 'potion',
        label: 'Potion',
        quantity: 1
      }
    })
    const merchantInventory = createInitialBlacksmithInventory({
      slotCount: 6,
      gold: 5000
    })

    const result = sellBlacksmithShopItem({
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
