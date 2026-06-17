// playerSaveState 순수 데이터 reshaping/검증을 Lua로 실행하는 래퍼.
// serialize: {version, ...input} → JSON. normalize/parse: 깊은 검증, 손상 시 undefined.
// 슬롯 배열의 빈 슬롯(TS undefined)은 callJson을 통해 JSON null ↔ Lua json_null로 왕복하고,
// 결과의 null 구멍/없는 값은 다시 TS undefined로 정규화한다(player-inventory 패턴).

import playerSaveStateSource from '../assets/lua/player-save-state.lua?raw'

import type {
  PlayerSaveState,
  PlayerSaveStateInput
} from '../playerSaveState'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

// Lua에서 돌아오는 원시 형태: 빈 슬롯/없는 아이템은 null로 표현된다(undefined가 아님).
type RawSaveState = {
  version: number
  profile: RawProfile
  equipment: RawEquipment
  inventory: RawInventory
  quickslots: RawQuickslots
  skillSlots: RawSkillSlots
}

type RawProfile = Omit<PlayerSaveState['profile'], never>

type RawInventory = {
  gold: number
  slots: (PlayerSaveState['inventory']['slots'][number] | null)[]
}

type RawEquipmentSlot = {
  id: string
  label: string
  item: NonNullable<
    PlayerSaveState['equipment']['slots'][number]['item']
  > | null
}

type RawEquipment = {
  setName: string
  level: number
  slots: RawEquipmentSlot[]
}

type RawQuickslots = {
  slots: (PlayerSaveState['quickslots']['slots'][number] | null)[]
}

type RawSkillSlots = {
  slots: (PlayerSaveState['skillSlots']['slots'][number] | null)[]
}

// Lua null 구멍/없는 값을 TS undefined로 정규화한 완성된 저장 상태로 변환한다.
const normalizeRawSaveState = (raw: RawSaveState): PlayerSaveState => ({
  version: raw.version,
  profile: raw.profile,
  equipment: {
    setName: raw.equipment.setName,
    level: raw.equipment.level,
    slots: raw.equipment.slots.map((slot) => ({
      id: slot.id as PlayerSaveState['equipment']['slots'][number]['id'],
      label: slot.label,
      item: slot.item === null ? undefined : slot.item
    }))
  },
  inventory: {
    gold: raw.inventory.gold,
    slots: raw.inventory.slots.map((slot) => (slot === null ? undefined : slot))
  },
  quickslots: {
    slots: raw.quickslots.slots.map((slot) =>
      slot === null ? undefined : slot
    )
  },
  skillSlots: {
    slots: raw.skillSlots.slots.map((slot) =>
      slot === null ? undefined : slot
    )
  }
})

export type PlayerSaveStateLua = {
  serializePlayerSaveState: (input: PlayerSaveStateInput) => string
  normalizeStoredPlayerSaveState: (
    value: unknown
  ) => PlayerSaveState | undefined
  parseStoredPlayerSaveState: (
    raw: string | null | undefined
  ) => PlayerSaveState | undefined
  close: () => void
}

export const createPlayerSaveStateLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerSaveStateLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerSaveStateSource, '@player-save-state.lua')

  return {
    serializePlayerSaveState: (input: PlayerSaveStateInput): string =>
      host.callJson<string>('save_state_serialize', input),

    normalizeStoredPlayerSaveState: (
      value: unknown
    ): PlayerSaveState | undefined => {
      const result = host.callJson<RawSaveState | null>(
        'save_state_normalize',
        value
      )

      return result === null ? undefined : normalizeRawSaveState(result)
    },

    parseStoredPlayerSaveState: (
      raw: string | null | undefined
    ): PlayerSaveState | undefined => {
      const result = host.callJson<RawSaveState | null>(
        'save_state_parse',
        raw ?? null
      )

      return result === null ? undefined : normalizeRawSaveState(result)
    },

    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
