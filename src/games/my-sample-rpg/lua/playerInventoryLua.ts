// playerInventory 연산을 Lua로 실행하는 래퍼. slots의 빈 슬롯은 JSON에서 null로 오가고,
// Lua는 json_null 센티넬로 보존한다(callJson). 결과의 null 슬롯은 TS의 undefined로 정규화한다.

import playerInventorySource from '../assets/lua/player-inventory.lua?raw'

import {
  DEFAULT_PLAYER_INVENTORY_GOLD,
  DEFAULT_PLAYER_INVENTORY_SLOT_COUNT,
  DEFAULT_PLAYER_INVENTORY_STARTER_ITEMS,
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

// Lua는 빈 슬롯을 null로 반환 → TS PlayerInventory의 undefined로 정규화.
const normalizeInventory = (raw: RawInventory): PlayerInventory => ({
  gold: raw.gold,
  slots: raw.slots.map((slot) => (slot === null ? undefined : slot))
})

export type PlayerInventoryLua = {
  createInitialPlayerInventory: (input?: {
    slotCount?: number
    gold?: number
  }) => PlayerInventory
  setPlayerInventorySlot: (input: {
    inventory: PlayerInventory
    slotIndex: number
    item: PlayerInventoryItem
  }) => PlayerInventory
  clearPlayerInventorySlot: (input: {
    inventory: PlayerInventory
    slotIndex: number
  }) => PlayerInventory
  getPlayerInventoryFilledSlotCount: (inventory: PlayerInventory) => number
  findFirstEmptyPlayerInventorySlotIndex: (
    inventory: PlayerInventory
  ) => number | undefined
  close: () => void
}

export const createPlayerInventoryLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerInventoryLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerInventorySource, '@player-inventory.lua')

  return {
    createInitialPlayerInventory: ({
      slotCount,
      gold = DEFAULT_PLAYER_INVENTORY_GOLD
    }: { slotCount?: number; gold?: number } = {}): PlayerInventory =>
      normalizeInventory(
        host.callJson<RawInventory>(
          'inventory_create',
          slotCount ?? null,
          gold,
          DEFAULT_PLAYER_INVENTORY_STARTER_ITEMS,
          DEFAULT_PLAYER_INVENTORY_SLOT_COUNT
        )
      ),
    setPlayerInventorySlot: ({ inventory, slotIndex, item }): PlayerInventory =>
      normalizeInventory(
        host.callJson<RawInventory>('inventory_set_slot', inventory, slotIndex, item)
      ),
    clearPlayerInventorySlot: ({ inventory, slotIndex }): PlayerInventory =>
      normalizeInventory(
        host.callJson<RawInventory>('inventory_clear_slot', inventory, slotIndex)
      ),
    getPlayerInventoryFilledSlotCount: (inventory: PlayerInventory): number =>
      host.callJson<number>('inventory_filled_slot_count', inventory),
    findFirstEmptyPlayerInventorySlotIndex: (
      inventory: PlayerInventory
    ): number | undefined => {
      const result = host.callJson<number | null>(
        'inventory_first_empty_index',
        inventory
      )

      return result === null ? undefined : result
    },
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
