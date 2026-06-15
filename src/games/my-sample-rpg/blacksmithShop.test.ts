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
import { getPlayerEquipmentItemDefinitionById } from './playerEquipment'

const BLACKSMITH_STOCK_ITEM_IDS = [
  'basic-sword',
  'basic-armor',
  'basic-boots',
  'basic-charm',
  'bronze-sword',
  'iron-sword',
  'battle-axe',
  'long-spear',
  'quick-dagger',
  'spiked-mace',
  'magic-staff',
  'iron-armor',
  'Leather_Armor',
  'Leather_Helmet',
  'Chain_Armor',
  'Chain_Helmet',
  'Iron_Armor',
  'Iron_Helmet',
  'leather-boots',
  'smith-charm'
] as const

const createExpectedStockSlots = (itemIds: readonly string[]) =>
  itemIds.map((itemId) => {
    const definition = getPlayerEquipmentItemDefinitionById(itemId)

    if (!definition) {
      throw new Error(`Missing expected equipment definition for ${itemId}`)
    }

    return {
      id: definition.id,
      label: definition.label,
      quantity: 1
    }
  })

const createExpectedMerchantInventory = (
  gold: number,
  itemIds: readonly string[] = BLACKSMITH_STOCK_ITEM_IDS
) => ({
  gold,
  slots: createExpectedStockSlots(itemIds)
})

describe('createInitialBlacksmithInventory', () => {
  it('creates stocked merchant inventory with gold', () => {
    expect(createInitialBlacksmithInventory({ gold: 4200 })).toEqual(
      createExpectedMerchantInventory(4200)
    )
  })
})

describe('buyBlacksmithShopItem', () => {
  it('moves an item from the merchant inventory into the player inventory', () => {
    const playerInventory = createInitialPlayerInventory({ slotCount: 2, gold: 1000 })
    const merchantInventory = createInitialBlacksmithInventory({ gold: 5000 })

    const result = buyBlacksmithShopItem({
      playerInventory,
      merchantInventory,
      merchantSlotIndex: 0
    })

    expect(result).toEqual({
      ok: true,
      playerInventory: {
        gold: 880,
        slots: [
          {
            id: 'basic-sword',
            label: '기본 무기',
            quantity: 1
          },
          undefined
        ]
      },
      merchantInventory: {
        gold: 5120,
        slots: [
          undefined,
          ...createExpectedStockSlots(BLACKSMITH_STOCK_ITEM_IDS.slice(1))
        ]
      },
      message: '구매했습니다.'
    })
  })

  it('rejects purchases when the player cannot afford the item', () => {
    const playerInventory = createInitialPlayerInventory({ slotCount: 2, gold: 100 })
    const merchantInventory = createInitialBlacksmithInventory({ gold: 5000 })

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
    const merchantInventory = createInitialBlacksmithInventory({ gold: 5000 })

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
  it('converts a potion sale into gold without changing merchant stock', () => {
    const playerInventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2, gold: 1000 }),
      slotIndex: 0,
      item: {
        id: 'health-potion',
        label: '체력 회복 포션',
        quantity: 1
      }
    })
    const merchantInventory = createInitialBlacksmithInventory({ gold: 5000 })

    const result = sellBlacksmithShopItem({
      playerInventory,
      merchantInventory,
      playerSlotIndex: 0
    })

    expect(result).toEqual({
      ok: true,
      playerInventory: {
        gold: 1005,
        slots: [undefined, undefined]
      },
      merchantInventory: {
        gold: 4995,
        slots: createExpectedMerchantInventory(4995).slots
      },
      message: '판매했습니다.'
    })
  })

  it('rejects sales for unknown items', () => {
    const playerInventory = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2, gold: 1000 }),
      slotIndex: 0,
      item: {
        id: 'potion',
        label: 'Potion',
        quantity: 1
      }
    })
    const merchantInventory = createInitialBlacksmithInventory({ gold: 5000 })

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
