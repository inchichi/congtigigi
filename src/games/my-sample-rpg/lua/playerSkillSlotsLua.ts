// playerSkillSlots 연산을 Lua로 실행하는 래퍼. slots의 빈 슬롯은 JSON에서 null로 오가고,
// Lua는 json_null 센티넬로 보존한다(callJson). 결과의 null 슬롯은 TS의 undefined로 정규화한다.

import playerSkillSlotsSource from '../assets/lua/player-skill-slots.lua?raw'

import {
  PLAYER_SKILL_SLOT_COUNT,
  PLAYER_SKILL_SLOT_HOTKEY_CODES,
  type PlayerSkillSlotAssignment,
  type PlayerSkillSlots
} from '../playerSkillSlots'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

type RawSkillSlots = {
  slots: (PlayerSkillSlotAssignment | null)[]
}

// Lua는 빈 슬롯을 null로 반환 → TS PlayerSkillSlots의 undefined로 정규화.
const normalizeSkillSlots = (raw: RawSkillSlots): PlayerSkillSlots => ({
  slots: raw.slots.map((slot) => (slot === null ? undefined : slot))
})

export type PlayerSkillSlotsLua = {
  createInitialPlayerSkillSlots: (input?: {
    slotCount?: number
    initialSkillIds?: readonly (string | undefined)[]
  }) => PlayerSkillSlots
  setPlayerSkillSlotAssignment: (input: {
    skillSlots: PlayerSkillSlots
    skillSlotIndex: number
    skillId: string
  }) => PlayerSkillSlots
  clearPlayerSkillSlotAssignment: (input: {
    skillSlots: PlayerSkillSlots
    skillSlotIndex: number
  }) => PlayerSkillSlots
  findPlayerSkillSlotIndexBySkillId: (
    skillSlots: PlayerSkillSlots,
    skillId: string
  ) => number | undefined
  getPlayerSkillSlotAssignment: (
    skillSlots: PlayerSkillSlots,
    skillSlotIndex: number
  ) => PlayerSkillSlotAssignment | undefined
  getPlayerSkillSlotIndexFromCode: (code: string) => number | undefined
  close: () => void
}

export const createPlayerSkillSlotsLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerSkillSlotsLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerSkillSlotsSource, '@player-skill-slots.lua')

  return {
    createInitialPlayerSkillSlots: ({
      slotCount = PLAYER_SKILL_SLOT_COUNT,
      initialSkillIds = []
    }: {
      slotCount?: number
      initialSkillIds?: readonly (string | undefined)[]
    } = {}): PlayerSkillSlots =>
      normalizeSkillSlots(
        host.callJson<RawSkillSlots>(
          'skill_slots_create_initial',
          slotCount,
          // JSON에서 undefined는 직렬화되지 않으므로 null로 변환
          Array.from(initialSkillIds, (id) => id ?? null)
        )
      ),

    setPlayerSkillSlotAssignment: ({
      skillSlots,
      skillSlotIndex,
      skillId
    }): PlayerSkillSlots =>
      normalizeSkillSlots(
        host.callJson<RawSkillSlots>(
          'skill_slots_set_assignment',
          skillSlots,
          skillSlotIndex,
          skillId
        )
      ),

    clearPlayerSkillSlotAssignment: ({
      skillSlots,
      skillSlotIndex
    }): PlayerSkillSlots =>
      normalizeSkillSlots(
        host.callJson<RawSkillSlots>(
          'skill_slots_clear_assignment',
          skillSlots,
          skillSlotIndex
        )
      ),

    findPlayerSkillSlotIndexBySkillId: (
      skillSlots: PlayerSkillSlots,
      skillId: string
    ): number | undefined => {
      const result = host.callJson<number | null>(
        'skill_slots_find_by_skill_id',
        skillSlots,
        skillId
      )
      return result === null ? undefined : result
    },

    getPlayerSkillSlotAssignment: (
      skillSlots: PlayerSkillSlots,
      skillSlotIndex: number
    ): PlayerSkillSlotAssignment | undefined => {
      const result = host.callJson<PlayerSkillSlotAssignment | null>(
        'skill_slots_get_assignment',
        skillSlots,
        skillSlotIndex
      )
      return result === null ? undefined : result
    },

    getPlayerSkillSlotIndexFromCode: (code: string): number | undefined => {
      const result = host.callJson<number | null>(
        'skill_slots_index_from_code',
        // PLAYER_SKILL_SLOT_HOTKEY_CODES는 readonly tuple → 배열로 변환
        Array.from(PLAYER_SKILL_SLOT_HOTKEY_CODES),
        code
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
