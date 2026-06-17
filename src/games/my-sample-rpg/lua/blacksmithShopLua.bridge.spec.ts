import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialBlacksmithInventory,
  getBlacksmithShopBuyPriceById,
  getBlacksmithShopSellPriceById,
  buyBlacksmithShopItem,
  sellBlacksmithShopItem
} from '../blacksmithShop'
import {
  createInitialPlayerInventory,
  setPlayerInventorySlot,
  type PlayerInventory,
  type PlayerInventoryItem
} from '../playerInventory'
import {
  createBlacksmithShopLua,
  type BlacksmithShopLua
} from './blacksmithShopLua'

// blacksmithShop(빈 슬롯 = 배열 null 구멍 + 인벤토리 골드/슬롯 조작)을 callJson + json_null 센티넬로
// 변환한 결과가 원본 TS와 동등한지 실제 WASM으로 검증한다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

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

const IRON_SWORD_ITEM: PlayerInventoryItem = {
  id: 'iron-sword',
  label: '강철 검',
  quantity: 1
}

const HEALTH_POTION_ITEM: PlayerInventoryItem = {
  id: 'health-potion',
  label: '체력 회복 포션',
  quantity: 1
}

const UNKNOWN_ITEM: PlayerInventoryItem = {
  id: 'mystery-rock',
  label: '수수께끼 돌',
  quantity: 1
}

describe('blacksmithShopLua (real wasm)', () => {
  let lua: BlacksmithShopLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS blacksmithShop across create/price/buy/sell incl edge cases', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createBlacksmithShopLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // ── createInitialBlacksmithInventory ──
    const tsDefault = createInitialBlacksmithInventory()
    const luaDefault = lua.createInitialBlacksmithInventory()

    // slot 0 아이템 일치 확인
    expect(luaDefault.slots[0]).toEqual(tsDefault.slots[0])
    // slot 0 id
    expect(luaDefault.slots[0]?.id).toBe('basic-sword')
    // slot 0 label
    expect(luaDefault.slots[0]?.label).toBe(tsDefault.slots[0]?.label)
    // 전체 동등성
    expect(luaDefault).toEqual(tsDefault)

    // 커스텀 slotCount
    for (const slotCount of [1, 5, 10, 20]) {
      expect(lua.createInitialBlacksmithInventory({ slotCount })).toEqual(
        createInitialBlacksmithInventory({ slotCount })
      )
    }
    // 커스텀 gold
    expect(lua.createInitialBlacksmithInventory({ gold: 999 })).toEqual(
      createInitialBlacksmithInventory({ gold: 999 })
    )

    // ── buy price ──
    for (const itemId of ['iron-sword', 'basic-sword', 'health-potion', 'unknown-item', '']) {
      expect(lua.getBlacksmithShopBuyPriceById(itemId)).toBe(
        getBlacksmithShopBuyPriceById(itemId)
      )
    }

    // ── sell price ──
    for (const itemId of ['iron-sword', 'basic-sword', 'health-potion', 'unknown-item', '']) {
      expect(lua.getBlacksmithShopSellPriceById(itemId)).toBe(
        getBlacksmithShopSellPriceById(itemId)
      )
    }

    // ── buyBlacksmithShopItem: 성공 케이스 ──
    const merchant = createInitialBlacksmithInventory()
    const playerWithGold = createInitialPlayerInventory({ slotCount: 5, gold: 100_000 })

    // 슬롯 0 구매 성공
    const buyResult = lua.buyBlacksmithShopItem({
      playerInventory: playerWithGold,
      merchantInventory: merchant,
      merchantSlotIndex: 0
    })
    expect(buyResult).toEqual(
      buyBlacksmithShopItem({
        playerInventory: playerWithGold,
        merchantInventory: merchant,
        merchantSlotIndex: 0
      })
    )
    expect(buyResult.ok).toBe(true)
    expect(buyResult.message).toBe('구매했습니다.')

    // 여러 슬롯 성공
    for (const merchantSlotIndex of [0, 1, 5, 10, 19]) {
      expect(
        lua.buyBlacksmithShopItem({
          playerInventory: playerWithGold,
          merchantInventory: merchant,
          merchantSlotIndex
        })
      ).toEqual(
        buyBlacksmithShopItem({
          playerInventory: playerWithGold,
          merchantInventory: merchant,
          merchantSlotIndex
        })
      )
    }

    // ── buyBlacksmithShopItem: 실패 — 돈 부족 ──
    const brokePlayer = createInitialPlayerInventory({ slotCount: 5, gold: 1 })
    const brokeResult = lua.buyBlacksmithShopItem({
      playerInventory: brokePlayer,
      merchantInventory: merchant,
      merchantSlotIndex: 0
    })
    expect(brokeResult).toEqual(
      buyBlacksmithShopItem({
        playerInventory: brokePlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 0
      })
    )
    expect(brokeResult.ok).toBe(false)
    expect(brokeResult.message).toBe('돈이 부족합니다.')

    // ── buyBlacksmithShopItem: 실패 — 인벤토리 가득 ──
    const fullPlayer = inventoryFromSlots(100_000, [
      IRON_SWORD_ITEM,
      HEALTH_POTION_ITEM,
      UNKNOWN_ITEM
    ])
    const fullResult = lua.buyBlacksmithShopItem({
      playerInventory: fullPlayer,
      merchantInventory: merchant,
      merchantSlotIndex: 0
    })
    expect(fullResult).toEqual(
      buyBlacksmithShopItem({
        playerInventory: fullPlayer,
        merchantInventory: merchant,
        merchantSlotIndex: 0
      })
    )
    expect(fullResult.ok).toBe(false)
    expect(fullResult.message).toBe('인벤토리가 가득 찼습니다.')

    // ── buyBlacksmithShopItem: 실패 — 재고 없음(빈 상점 슬롯) ──
    // 빈 상인 인벤토리를 만들어 빈 슬롯을 구매 시도
    const emptyMerchant = createInitialPlayerInventory({ slotCount: 5, gold: 1_000_000 })
    const noStockResult = lua.buyBlacksmithShopItem({
      playerInventory: playerWithGold,
      merchantInventory: emptyMerchant,
      merchantSlotIndex: 0
    })
    expect(noStockResult).toEqual(
      buyBlacksmithShopItem({
        playerInventory: playerWithGold,
        merchantInventory: emptyMerchant,
        merchantSlotIndex: 0
      })
    )
    expect(noStockResult.ok).toBe(false)
    expect(noStockResult.message).toBe('재고가 없습니다.')

    // ── buyBlacksmithShopItem: 실패 — 가격 없는 아이템(알 수 없는 id) ──
    const merchantWithUnknown = inventoryFromSlots(1_000_000, [UNKNOWN_ITEM])
    const noPriceResult = lua.buyBlacksmithShopItem({
      playerInventory: playerWithGold,
      merchantInventory: merchantWithUnknown,
      merchantSlotIndex: 0
    })
    expect(noPriceResult).toEqual(
      buyBlacksmithShopItem({
        playerInventory: playerWithGold,
        merchantInventory: merchantWithUnknown,
        merchantSlotIndex: 0
      })
    )
    expect(noPriceResult.ok).toBe(false)
    expect(noPriceResult.message).toBe('이 상점에서는 판매하지 않는 아이템입니다.')

    // ── buyBlacksmithShopItem: throw — 범위 밖 슬롯 ──
    expect(() =>
      lua?.buyBlacksmithShopItem({
        playerInventory: playerWithGold,
        merchantInventory: merchant,
        merchantSlotIndex: 999
      })
    ).toThrow()
    expect(() =>
      buyBlacksmithShopItem({
        playerInventory: playerWithGold,
        merchantInventory: merchant,
        merchantSlotIndex: 999
      })
    ).toThrow()

    // ── sellBlacksmithShopItem: 성공 ──
    const playerWithSword = inventoryFromSlots(0, [IRON_SWORD_ITEM, undefined, undefined])
    const sellResult = lua.sellBlacksmithShopItem({
      playerInventory: playerWithSword,
      merchantInventory: merchant,
      playerSlotIndex: 0
    })
    expect(sellResult).toEqual(
      sellBlacksmithShopItem({
        playerInventory: playerWithSword,
        merchantInventory: merchant,
        playerSlotIndex: 0
      })
    )
    expect(sellResult.ok).toBe(true)
    expect(sellResult.message).toBe('판매했습니다.')
    // 판매 후 플레이어 슬롯 비워짐
    expect(sellResult.playerInventory.slots[0]).toBeUndefined()

    // 포션도 판매 가능(판매가 = 포션 price * 0.5)
    const playerWithPotion = inventoryFromSlots(0, [HEALTH_POTION_ITEM])
    expect(
      lua.sellBlacksmithShopItem({
        playerInventory: playerWithPotion,
        merchantInventory: merchant,
        playerSlotIndex: 0
      })
    ).toEqual(
      sellBlacksmithShopItem({
        playerInventory: playerWithPotion,
        merchantInventory: merchant,
        playerSlotIndex: 0
      })
    )

    // ── sellBlacksmithShopItem: 실패 — 빈 슬롯 판매 ──
    const playerEmpty = createInitialPlayerInventory({ slotCount: 3, gold: 0 })
    const emptySlotResult = lua.sellBlacksmithShopItem({
      playerInventory: playerEmpty,
      merchantInventory: merchant,
      playerSlotIndex: 0
    })
    expect(emptySlotResult).toEqual(
      sellBlacksmithShopItem({
        playerInventory: playerEmpty,
        merchantInventory: merchant,
        playerSlotIndex: 0
      })
    )
    expect(emptySlotResult.ok).toBe(false)
    expect(emptySlotResult.message).toBe('판매할 아이템이 없습니다.')

    // ── sellBlacksmithShopItem: 실패 — 알 수 없는 아이템 ──
    const playerWithUnknown = inventoryFromSlots(0, [UNKNOWN_ITEM])
    const unknownSellResult = lua.sellBlacksmithShopItem({
      playerInventory: playerWithUnknown,
      merchantInventory: merchant,
      playerSlotIndex: 0
    })
    expect(unknownSellResult).toEqual(
      sellBlacksmithShopItem({
        playerInventory: playerWithUnknown,
        merchantInventory: merchant,
        playerSlotIndex: 0
      })
    )
    expect(unknownSellResult.ok).toBe(false)
    expect(unknownSellResult.message).toBe('이 상점에서는 매입하지 않는 아이템입니다.')

    // ── sellBlacksmithShopItem: 실패 — 상점 보유금 부족 ──
    const poorMerchantWithSlots = createInitialPlayerInventory({ slotCount: 5, gold: 1 })
    const poorResult = lua.sellBlacksmithShopItem({
      playerInventory: playerWithSword,
      merchantInventory: poorMerchantWithSlots,
      playerSlotIndex: 0
    })
    expect(poorResult).toEqual(
      sellBlacksmithShopItem({
        playerInventory: playerWithSword,
        merchantInventory: poorMerchantWithSlots,
        playerSlotIndex: 0
      })
    )
    expect(poorResult.ok).toBe(false)
    expect(poorResult.message).toBe('상점 보유금이 부족합니다.')

    // ── sellBlacksmithShopItem: throw — 범위 밖 슬롯 ──
    expect(() =>
      lua?.sellBlacksmithShopItem({
        playerInventory: playerWithSword,
        merchantInventory: merchant,
        playerSlotIndex: -1
      })
    ).toThrow()
    expect(() =>
      sellBlacksmithShopItem({
        playerInventory: playerWithSword,
        merchantInventory: merchant,
        playerSlotIndex: -1
      })
    ).toThrow()
  })
})
