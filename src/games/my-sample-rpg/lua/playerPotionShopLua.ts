// potionShop 연산을 Lua로 실행하는 래퍼. 포션 상점 로직은 인벤토리 슬롯/골드를 다루므로
// player-potion-shop.lua 는 같은 호스트에 로드된 player-inventory.lua 의 전역(inventory_*)을 호출한다.
// 따라서 자체 호스트를 만들 때(input.host 미지정)는 inventory 소스를 먼저, 그다음 potion-shop 소스를 로드한다.
// 빈 슬롯은 JSON에서 null로 오가고 Lua는 json_null 센티넬로 보존한다 → 결과의 null 슬롯은 TS undefined로 정규화한다.
// 데이터 소유는 TS(단일 출처): 포션 정의·장비 정의·상수는 모두 원본 TS에서 import 해 인자로 전달한다.

import playerInventorySource from '../assets/lua/player-inventory.lua?raw'
import playerPotionShopSource from '../assets/lua/player-potion-shop.lua?raw'

import {
  DEFAULT_POTION_INVENTORY_GOLD,
  DEFAULT_POTION_INVENTORY_SLOT_COUNT,
  DEFAULT_POTION_STOCK_QUANTITY,
  POTION_ITEM_DEFINITIONS,
  POTION_SALE_RATIO,
  type PotionShopInventory,
  type PotionShopItemDefinition,
  type PotionShopTransactionResult
} from '../potionShop'
import { PLAYER_EQUIPMENT_ITEM_DEFINITIONS } from '../playerEquipment'
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

export type PlayerPotionShopLua = {
  createInitialPotionInventory: (input?: {
    slotCount?: number
    gold?: number
  }) => PotionShopInventory
  getPotionShopItemDefinitionById: (
    itemId: string
  ) => PotionShopItemDefinition | undefined
  getPotionShopBuyPriceById: (itemId: string) => number | undefined
  getPotionShopSellPriceById: (itemId: string) => number | undefined
  buyPotionShopItem: (input: {
    playerInventory: PlayerInventory
    merchantInventory: PlayerInventory
    merchantSlotIndex: number
    quantity?: number
  }) => PotionShopTransactionResult
  sellPotionShopItem: (input: {
    playerInventory: PlayerInventory
    merchantInventory: PlayerInventory
    playerSlotIndex: number
    quantity?: number
  }) => PotionShopTransactionResult
  close: () => void
}

export const createPlayerPotionShopLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerPotionShopLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))

  // 자체 호스트를 만든 경우에만 inventory 의존을 먼저 로드한다(공유 호스트엔 이미 로드돼 있다).
  if (!input.host) {
    host.runModule(playerInventorySource, '@player-inventory.lua')
  }
  host.runModule(playerPotionShopSource, '@player-potion-shop.lua')

  // 실패(ok=false) 시 TS는 입력 인벤토리를 같은 참조로 돌려준다 → 입력값을 그대로 반환해 의미를 맞춘다.
  const normalizeTransactionResult = (
    raw: RawTransactionResult,
    inputPlayerInventory: PlayerInventory,
    inputMerchantInventory: PlayerInventory
  ): PotionShopTransactionResult =>
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

  return {
    createInitialPotionInventory: ({
      slotCount = DEFAULT_POTION_INVENTORY_SLOT_COUNT,
      gold = DEFAULT_POTION_INVENTORY_GOLD
    }: { slotCount?: number; gold?: number } = {}): PotionShopInventory =>
      normalizeInventory(
        host.callJson<RawInventory>(
          'potion_shop_create_initial',
          slotCount,
          gold,
          POTION_ITEM_DEFINITIONS,
          DEFAULT_POTION_STOCK_QUANTITY
        )
      ),
    getPotionShopItemDefinitionById: (
      itemId: string
    ): PotionShopItemDefinition | undefined => {
      const result = host.callJson<PotionShopItemDefinition | null>(
        'potion_shop_item_definition_by_id',
        POTION_ITEM_DEFINITIONS,
        itemId
      )

      return result === null ? undefined : result
    },
    getPotionShopBuyPriceById: (itemId: string): number | undefined => {
      const result = host.callJson<number | null>(
        'potion_shop_buy_price_by_id',
        POTION_ITEM_DEFINITIONS,
        itemId
      )

      return result === null ? undefined : result
    },
    getPotionShopSellPriceById: (itemId: string): number | undefined => {
      const result = host.callJson<number | null>(
        'potion_shop_sell_price_by_id',
        POTION_ITEM_DEFINITIONS,
        PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
        itemId,
        POTION_SALE_RATIO
      )

      return result === null ? undefined : result
    },
    buyPotionShopItem: ({
      playerInventory,
      merchantInventory,
      merchantSlotIndex,
      quantity
    }): PotionShopTransactionResult =>
      normalizeTransactionResult(
        host.callJson<RawTransactionResult>(
          'potion_shop_buy',
          playerInventory,
          merchantInventory,
          merchantSlotIndex,
          quantity ?? null,
          POTION_ITEM_DEFINITIONS
        ),
        playerInventory,
        merchantInventory
      ),
    sellPotionShopItem: ({
      playerInventory,
      merchantInventory,
      playerSlotIndex,
      quantity
    }): PotionShopTransactionResult =>
      normalizeTransactionResult(
        host.callJson<RawTransactionResult>(
          'potion_shop_sell',
          playerInventory,
          merchantInventory,
          playerSlotIndex,
          quantity ?? null,
          POTION_ITEM_DEFINITIONS,
          PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
          POTION_SALE_RATIO
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
