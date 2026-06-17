// playerConsumables 연산을 Lua로 실행하는 래퍼.
// player-consumables.lua 는 같은 호스트에 로드된 inventory_* / quickslots_* 전역을 호출한다.
// 따라서 자체 호스트를 만들 때(input.host 미지정)는
//   player-inventory.lua → player-quickslots.lua → player-consumables.lua 순서로 로드한다.
// 결과가 json_null(= null) 이면 TS 에서 undefined 로 정규화한다.
// inventory.slots 의 null 구멍도 undefined 로 정규화한다.

import playerInventorySource from '../assets/lua/player-inventory.lua?raw'
import playerQuickslotsSource from '../assets/lua/player-quickslots.lua?raw'
import playerConsumablesSource from '../assets/lua/player-consumables.lua?raw'

import type { PlayerInventory, PlayerInventoryItem } from '../playerInventory'
import type { PlayerProfile } from '../playerProfile'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type UsePlayerInventoryConsumableResult = {
  profile: PlayerProfile
  inventory: PlayerInventory
}

export type UsePlayerQuickslotConsumableResult = UsePlayerInventoryConsumableResult

type RawInventory = {
  gold: number
  slots: (PlayerInventoryItem | null)[]
}

type RawResult = {
  profile: PlayerProfile
  inventory: RawInventory
} | null

const normalizeInventory = (raw: RawInventory): PlayerInventory => ({
  gold: raw.gold,
  slots: raw.slots.map((slot) => (slot === null ? undefined : slot))
})

const normalizeResult = (
  raw: RawResult
): UsePlayerInventoryConsumableResult | undefined => {
  if (raw === null) return undefined
  return {
    profile: raw.profile,
    inventory: normalizeInventory(raw.inventory)
  }
}

export type PlayerConsumablesLua = {
  usePlayerInventoryConsumable: (input: {
    profile: PlayerProfile
    inventory: PlayerInventory
    slotIndex: number
  }) => UsePlayerInventoryConsumableResult | undefined
  usePlayerQuickslotConsumable: (input: {
    profile: PlayerProfile
    inventory: PlayerInventory
    quickslots: { slots: ({ inventorySlotIndex: number } | undefined)[] }
    quickslotIndex: number
  }) => UsePlayerQuickslotConsumableResult | undefined
  close: () => void
}

export const createPlayerConsumablesLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerConsumablesLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))

  if (!input.host) {
    host.runModule(playerInventorySource, '@player-inventory.lua')
    host.runModule(playerQuickslotsSource, '@player-quickslots.lua')
  }
  host.runModule(playerConsumablesSource, '@player-consumables.lua')

  return {
    usePlayerInventoryConsumable: ({ profile, inventory, slotIndex }) =>
      normalizeResult(
        host.callJson<RawResult>(
          'consumables_use_inventory',
          profile,
          inventory,
          slotIndex
        )
      ),
    usePlayerQuickslotConsumable: ({ profile, inventory, quickslots, quickslotIndex }) =>
      normalizeResult(
        host.callJson<RawResult>(
          'consumables_use_quickslot',
          profile,
          inventory,
          quickslots,
          quickslotIndex
        )
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
