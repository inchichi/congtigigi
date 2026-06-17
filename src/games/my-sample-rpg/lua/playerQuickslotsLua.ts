// playerQuickslots 연산을 Lua로 실행하는 래퍼. slots의 빈 슬롯은 JSON에서 null로 오가고,
// Lua는 json_null 센티넬로 보존한다(callJson). 결과의 null 슬롯은 TS의 undefined로 정규화한다.

import playerQuickslotsSource from '../assets/lua/player-quickslots.lua?raw'

import {
  DEFAULT_PLAYER_QUICKSLOT_COUNT,
  type PlayerQuickslotAssignment,
  type PlayerQuickslots
} from '../playerQuickslots'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

type RawQuickslots = {
  slots: (PlayerQuickslotAssignment | null)[]
}

// Lua는 빈 슬롯을 null로 반환 → TS PlayerQuickslots의 undefined로 정규화.
const normalizeQuickslots = (raw: RawQuickslots): PlayerQuickslots => ({
  slots: raw.slots.map((slot) => (slot === null ? undefined : slot))
})

export type PlayerQuickslotsLua = {
  createInitialPlayerQuickslots: (input?: { slotCount?: number }) => PlayerQuickslots
  setPlayerQuickslotAssignment: (input: {
    quickslots: PlayerQuickslots
    quickslotIndex: number
    inventorySlotIndex: number
  }) => PlayerQuickslots
  clearPlayerQuickslotAssignment: (input: {
    quickslots: PlayerQuickslots
    quickslotIndex: number
  }) => PlayerQuickslots
  findPlayerQuickslotIndexByInventorySlotIndex: (
    quickslots: PlayerQuickslots,
    inventorySlotIndex: number
  ) => number | undefined
  getPlayerQuickslotAssignment: (
    quickslots: PlayerQuickslots,
    quickslotIndex: number
  ) => PlayerQuickslotAssignment | undefined
  close: () => void
}

export const createPlayerQuickslotsLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerQuickslotsLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerQuickslotsSource, '@player-quickslots.lua')

  return {
    createInitialPlayerQuickslots: ({
      slotCount = DEFAULT_PLAYER_QUICKSLOT_COUNT
    }: { slotCount?: number } = {}): PlayerQuickslots =>
      normalizeQuickslots(
        host.callJson<RawQuickslots>('quickslots_create_initial', slotCount)
      ),

    setPlayerQuickslotAssignment: ({
      quickslots,
      quickslotIndex,
      inventorySlotIndex
    }): PlayerQuickslots =>
      normalizeQuickslots(
        host.callJson<RawQuickslots>(
          'quickslots_set_assignment',
          quickslots,
          quickslotIndex,
          inventorySlotIndex
        )
      ),

    clearPlayerQuickslotAssignment: ({
      quickslots,
      quickslotIndex
    }): PlayerQuickslots =>
      normalizeQuickslots(
        host.callJson<RawQuickslots>(
          'quickslots_clear_assignment',
          quickslots,
          quickslotIndex
        )
      ),

    findPlayerQuickslotIndexByInventorySlotIndex: (
      quickslots: PlayerQuickslots,
      inventorySlotIndex: number
    ): number | undefined => {
      const result = host.callJson<number | null>(
        'quickslots_find_by_inventory_slot',
        quickslots,
        inventorySlotIndex
      )
      return result === null ? undefined : result
    },

    getPlayerQuickslotAssignment: (
      quickslots: PlayerQuickslots,
      quickslotIndex: number
    ): PlayerQuickslotAssignment | undefined => {
      const result = host.callJson<PlayerQuickslotAssignment | null>(
        'quickslots_get_assignment',
        quickslots,
        quickslotIndex
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
