// blacksmithShop 연산을 Lua로 실행하는 래퍼. 대장장이 상점 로직은 인벤토리 슬롯/골드를 다루므로
// blacksmith-shop.lua 는 같은 호스트에 로드된 player-inventory.lua, player-equipment.lua,
// player-potion-shop.lua 의 전역(inventory_*, equipment_*, potion_shop_*)을 호출한다.
// 따라서 자체 호스트를 만들 때(input.host 미지정)는 의존 소스를 먼저 로드한다:
//   player-inventory.lua → player-equipment.lua → player-potion-shop.lua → blacksmith-shop.lua
// 빈 슬롯은 JSON에서 null로 오가고 Lua는 json_null 센티넬로 보존한다 → 결과의 null 슬롯은 TS undefined로 정규화한다.
// 데이터 소유는 TS(단일 출처): 장비 정의·포션 정의·상수는 모두 원본 TS에서 import 해 인자로 전달한다.

import playerInventorySource from '../assets/lua/player-inventory.lua?raw'
import playerEquipmentSource from '../assets/lua/player-equipment.lua?raw'
import playerPotionShopSource from '../assets/lua/player-potion-shop.lua?raw'
import blacksmithShopSource from '../assets/lua/blacksmith-shop.lua?raw'

import {
  DEFAULT_BLACKSMITH_INVENTORY_GOLD,
  DEFAULT_BLACKSMITH_INVENTORY_SLOT_COUNT,
  BLACKSMITH_SALE_RATIO,
  BLACKSMITH_INITIAL_STOCK_ITEM_IDS,
  type BlacksmithInventory,
  type BlacksmithShopTransactionResult
} from '../blacksmithShop'
import { PLAYER_EQUIPMENT_ITEM_DEFINITIONS } from '../playerEquipment'
import { POTION_ITEM_DEFINITIONS } from '../potionShop'
import {
  type PlayerInventory,
  type PlayerInventoryItem
} from '../playerInventory'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

type RawInventory = {
  gold: number
  slots: (PlayerInventoryItem | null)[]
}

type RawTransactionResult = {
  ok: boolean
  playerInventory: RawInventory
  merchantInventory: RawInventory
  message: string
}

// Lua는 빈 슬롯을 null로 반환 → TS PlayerInventory의 undefined로 정규화.
const normalizeInventory = (raw: RawInventory): PlayerInventory => ({
  gold: raw.gold,
  slots: raw.slots.map((slot) => (slot === null ? undefined : slot))
})

// 실패(ok=false) 시 TS는 입력 인벤토리를 같은 참조로 돌려준다 → 입력값을 그대로 반환해 의미를 맞춘다.
const normalizeTransactionResult = (
  raw: RawTransactionResult,
  inputPlayerInventory: PlayerInventory,
  inputMerchantInventory: PlayerInventory
): BlacksmithShopTransactionResult =>
  raw.ok
    ? {
        ok: true,
        playerInventory: normalizeInventory(raw.playerInventory),
        merchantInventory: normalizeInventory(raw.merchantInventory),
        message: raw.message
      }
    : {
        ok: false,
        playerInventory: inputPlayerInventory,
        merchantInventory: inputMerchantInventory,
        message: raw.message
      }

export type BlacksmithShopLua = {
  createInitialBlacksmithInventory: (input?: {
    slotCount?: number
    gold?: number
  }) => BlacksmithInventory
  getBlacksmithShopBuyPriceById: (itemId: string) => number | undefined
  getBlacksmithShopSellPriceById: (itemId: string) => number | undefined
  buyBlacksmithShopItem: (input: {
    playerInventory: PlayerInventory
    merchantInventory: PlayerInventory
    merchantSlotIndex: number
  }) => BlacksmithShopTransactionResult
  sellBlacksmithShopItem: (input: {
    playerInventory: PlayerInventory
    merchantInventory: PlayerInventory
    playerSlotIndex: number
  }) => BlacksmithShopTransactionResult
  close: () => void
}

export const createBlacksmithShopLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<BlacksmithShopLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))

  // 자체 호스트를 만든 경우에만 의존 모듈을 로드한다(공유 호스트엔 이미 로드돼 있다).
  if (!input.host) {
    host.runModule(playerInventorySource, '@player-inventory.lua')
    host.runModule(playerEquipmentSource, '@player-equipment.lua')
    host.runModule(playerPotionShopSource, '@player-potion-shop.lua')
  }
  host.runModule(blacksmithShopSource, '@blacksmith-shop.lua')

  return {
    createInitialBlacksmithInventory: ({
      slotCount = DEFAULT_BLACKSMITH_INVENTORY_SLOT_COUNT,
      gold = DEFAULT_BLACKSMITH_INVENTORY_GOLD
    }: { slotCount?: number; gold?: number } = {}): BlacksmithInventory =>
      normalizeInventory(
        host.callJson<RawInventory>(
          'blacksmith_create_initial',
          slotCount,
          gold,
          PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
          [...BLACKSMITH_INITIAL_STOCK_ITEM_IDS]
        )
      ),
    getBlacksmithShopBuyPriceById: (itemId: string): number | undefined => {
      const result = host.callJson<number | null>(
        'blacksmith_buy_price',
        PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
        itemId
      )

      return result === null ? undefined : result
    },
    getBlacksmithShopSellPriceById: (itemId: string): number | undefined => {
      const result = host.callJson<number | null>(
        'blacksmith_sell_price',
        PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
        POTION_ITEM_DEFINITIONS,
        itemId,
        BLACKSMITH_SALE_RATIO
      )

      return result === null ? undefined : result
    },
    buyBlacksmithShopItem: ({
      playerInventory,
      merchantInventory,
      merchantSlotIndex
    }): BlacksmithShopTransactionResult =>
      normalizeTransactionResult(
        host.callJson<RawTransactionResult>(
          'blacksmith_buy',
          playerInventory,
          merchantInventory,
          merchantSlotIndex,
          PLAYER_EQUIPMENT_ITEM_DEFINITIONS
        ),
        playerInventory,
        merchantInventory
      ),
    sellBlacksmithShopItem: ({
      playerInventory,
      merchantInventory,
      playerSlotIndex
    }): BlacksmithShopTransactionResult =>
      normalizeTransactionResult(
        host.callJson<RawTransactionResult>(
          'blacksmith_sell',
          playerInventory,
          merchantInventory,
          playerSlotIndex,
          PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
          POTION_ITEM_DEFINITIONS,
          BLACKSMITH_SALE_RATIO
        ),
        playerInventory,
        merchantInventory
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
