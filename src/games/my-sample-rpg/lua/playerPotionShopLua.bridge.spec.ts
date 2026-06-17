import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialPlayerInventory,
  setPlayerInventorySlot,
  type PlayerInventory,
  type PlayerInventoryItem
} from '../playerInventory'
import {
  createInitialPotionInventory,
  getPotionShopItemDefinitionById,
  getPotionShopBuyPriceById,
  getPotionShopSellPriceById,
  buyPotionShopItem,
  sellPotionShopItem
} from '../potionShop'
import {
  createPlayerPotionShopLua,
  type PlayerPotionShopLua
} from './playerPotionShopLua'

// potionShop(빈 슬롯 = 배열 null 구멍 + 인벤토리 골드/슬롯 조작)을 callJson + json_null 센티넬로
// 변환한 결과가 원본 TS와 동등한지 실제 WASM으로 검증한다(player-inventory.lua 의존 포함).
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const HEALTH_POTION: PlayerInventoryItem = {
  id: 'health-potion',
  label: '체력 회복 포션',
  quantity: 5
}
const MANA_POTION: PlayerInventoryItem = {
  id: 'mana-potion',
  label: '마나 회복 포션',
  quantity: 3
}
const SWORD: PlayerInventoryItem = {
  id: 'iron-sword',
  label: '강철 검',
  quantity: 2
}
const UNKNOWN_ITEM: PlayerInventoryItem = {
  id: 'mystery-rock',
  label: '수수께끼 돌',
  quantity: 4
}

// 슬롯 배열로 인벤토리를 만든다(빈 슬롯은 undefined).
const inventoryFromSlots = (
  gold: number,
  slots: (PlayerInventoryItem | undefined)[]
): PlayerInventory => {
  let inventory = createInitialPlayerInventory({ slotCount: slots.length, gold })

  for (const [slotIndex, item] of slots.entries()) {
    if (item) {
      inventory = setPlayerInventorySlot({ inventory, slotIndex, item })
    }
  }

  return inventory
}

describe('playerPotionShopLua (real wasm)', () => {
  let lua: PlayerPotionShopLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS potionShop across create/price/buy/sell incl edge cases', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerPotionShopLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // ── createInitialPotionInventory ──
    expect(lua.createInitialPotionInventory()).toEqual(createInitialPotionInventory())
    for (const slotCount of [0, 1, 2, 3, 5, 12, 20]) {
      expect(lua.createInitialPotionInventory({ slotCount })).toEqual(
        createInitialPotionInventory({ slotCount })
      )
    }
    expect(lua.createInitialPotionInventory({ gold: 50 })).toEqual(
      createInitialPotionInventory({ gold: 50 })
    )
    expect(lua.createInitialPotionInventory({ slotCount: 4, gold: 0 })).toEqual(
      createInitialPotionInventory({ slotCount: 4, gold: 0 })
    )

    // ── getPotionShopItemDefinitionById / buy price / sell price ──
    for (const itemId of [
      'health-potion',
      'mana-potion',
      'iron-sword',
      'basic-charm',
      'magic-staff',
      'mystery-rock',
      ''
    ]) {
      expect(lua.getPotionShopItemDefinitionById(itemId)).toEqual(
        getPotionShopItemDefinitionById(itemId)
      )
      expect(lua.getPotionShopBuyPriceById(itemId)).toBe(
        getPotionShopBuyPriceById(itemId)
      )
      expect(lua.getPotionShopSellPriceById(itemId)).toBe(
        getPotionShopSellPriceById(itemId)
      )
    }

    const merchant = createInitialPotionInventory({ slotCount: 4 })

    // ── buyPotionShopItem: 성공(빈 인벤토리), 다양한 수량/슬롯 ──
    const emptyPlayer = createInitialPlayerInventory({ slotCount: 3, gold: 500 })
    for (const merchantSlotIndex of [0, 1]) {
      for (const quantity of [undefined, 1, 3, 10]) {
        expect(
          lua.buyPotionShopItem({
            playerInventory: emptyPlayer,
            merchantInventory: merchant,
            merchantSlotIndex,
            quantity
          })
        ).toEqual(
          buyPotionShopItem({
            playerInventory: emptyPlayer,
            merchantInventory: merchant,
            merchantSlotIndex,
            quantity
          })
        )
      }
    }

    // 기존 같은 아이템 스택에 합쳐지는 경우
    const playerWithHealthStack = inventoryFromSlots(500, [
      HEALTH_POTION,
      undefined,
      MANA_POTION
    ])
    expect(
      lua.buyPotionShopItem({
        playerInventory: playerWithHealthStack,
        merchantInventory: merchant,
        merchantSlotIndex: 0,
        quantity: 7
      })
    ).toEqual(
      buyPotionShopItem({
        playerInventory: playerWithHealthStack,
        merchantInventory: merchant,
        merchantSlotIndex: 0,
        quantity: 7
      })
    )

    // 돈 부족
    const brokePlayer = createInitialPlayerInventory({ slotCount: 3, gold: 5 })
    expect(
      lua.buyPotionShopItem({
        playerInventory: brokePlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 0,
        quantity: 1
      })
    ).toEqual(
      buyPotionShopItem({
        playerInventory: brokePlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 0,
        quantity: 1
      })
    )

    // 인벤토리 가득 (서로 다른 아이템으로 채워 스택 불가)
    const fullPlayer = inventoryFromSlots(500, [SWORD, MANA_POTION])
    expect(
      lua.buyPotionShopItem({
        playerInventory: fullPlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 0,
        quantity: 1
      })
    ).toEqual(
      buyPotionShopItem({
        playerInventory: fullPlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 0,
        quantity: 1
      })
    )

    // 재고 없음(빈 상점 슬롯)
    expect(
      lua.buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 3,
        quantity: 1
      })
    ).toEqual(
      buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 3,
        quantity: 1
      })
    )

    // 상점이 취급 않는 아이템(상점 슬롯에 장비 아이템) → 구매가 없음
    const merchantWithSword = inventoryFromSlots(1000, [SWORD])
    expect(
      lua.buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: merchantWithSword,
        merchantSlotIndex: 0,
        quantity: 1
      })
    ).toEqual(
      buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: merchantWithSword,
        merchantSlotIndex: 0,
        quantity: 1
      })
    )

    // 재고 전량 구매 → 상점 슬롯 비워짐
    const smallStockMerchant = inventoryFromSlots(1000, [
      { id: 'health-potion', label: '체력 회복 포션', quantity: 4 }
    ])
    expect(
      lua.buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: smallStockMerchant,
        merchantSlotIndex: 0,
        quantity: 4
      })
    ).toEqual(
      buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: smallStockMerchant,
        merchantSlotIndex: 0,
        quantity: 4
      })
    )

    // 재고 초과 수량
    expect(
      lua.buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: smallStockMerchant,
        merchantSlotIndex: 0,
        quantity: 5
      })
    ).toEqual(
      buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: smallStockMerchant,
        merchantSlotIndex: 0,
        quantity: 5
      })
    )

    // 잘못된 수량(0, 음수, 소수)
    for (const quantity of [0, -1, 2.5]) {
      expect(
        lua.buyPotionShopItem({
          playerInventory: emptyPlayer,
          merchantInventory: merchant,
          merchantSlotIndex: 0,
          quantity
        })
      ).toEqual(
        buyPotionShopItem({
          playerInventory: emptyPlayer,
          merchantInventory: merchant,
          merchantSlotIndex: 0,
          quantity
        })
      )
    }

    // ── sellPotionShopItem ──
    const sellPlayer = inventoryFromSlots(100, [
      HEALTH_POTION,
      SWORD,
      UNKNOWN_ITEM,
      undefined
    ])

    // 부분 판매 / 전량(기본 수량) / 명시 전량
    for (const [playerSlotIndex, quantity] of [
      [0, 2],
      [0, undefined],
      [0, 5],
      [1, 1]
    ] as const) {
      expect(
        lua.sellPotionShopItem({
          playerInventory: sellPlayer,
          merchantInventory: merchant,
          playerSlotIndex,
          quantity
        })
      ).toEqual(
        sellPotionShopItem({
          playerInventory: sellPlayer,
          merchantInventory: merchant,
          playerSlotIndex,
          quantity
        })
      )
    }

    // 취급 않는 아이템(정의 없음)
    expect(
      lua.sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: merchant,
        playerSlotIndex: 2
      })
    ).toEqual(
      sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: merchant,
        playerSlotIndex: 2
      })
    )

    // 빈 슬롯 판매
    expect(
      lua.sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: merchant,
        playerSlotIndex: 3
      })
    ).toEqual(
      sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: merchant,
        playerSlotIndex: 3
      })
    )

    // 보유 수량 초과
    expect(
      lua.sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: merchant,
        playerSlotIndex: 0,
        quantity: 99
      })
    ).toEqual(
      sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: merchant,
        playerSlotIndex: 0,
        quantity: 99
      })
    )

    // 잘못된 수량
    for (const quantity of [0, -3, 1.5]) {
      expect(
        lua.sellPotionShopItem({
          playerInventory: sellPlayer,
          merchantInventory: merchant,
          playerSlotIndex: 0,
          quantity
        })
      ).toEqual(
        sellPotionShopItem({
          playerInventory: sellPlayer,
          merchantInventory: merchant,
          playerSlotIndex: 0,
          quantity
        })
      )
    }

    // 상점 보유금 부족
    const poorMerchant = createInitialPotionInventory({ slotCount: 4, gold: 1 })
    expect(
      lua.sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: poorMerchant,
        playerSlotIndex: 0,
        quantity: 5
      })
    ).toEqual(
      sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: poorMerchant,
        playerSlotIndex: 0,
        quantity: 5
      })
    )

    // 잘못된 슬롯 인덱스는 양쪽 모두 throw
    expect(() =>
      lua?.buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 99
      })
    ).toThrow()
    expect(() =>
      buyPotionShopItem({
        playerInventory: emptyPlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 99
      })
    ).toThrow()
    expect(() =>
      lua?.sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: merchant,
        playerSlotIndex: -1
      })
    ).toThrow()
    expect(() =>
      sellPotionShopItem({
        playerInventory: sellPlayer,
        merchantInventory: merchant,
        playerSlotIndex: -1
      })
    ).toThrow()
  })
})
