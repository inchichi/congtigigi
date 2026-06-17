// playerEquipment 연산을 Lua로 실행하는 래퍼. 빈 슬롯의 item 은 JSON에서 null로 오가고,
// Lua는 json_null 센티넬로 보존한다(callJson). 결과의 null item 은 TS의 undefined로 정규화한다.
// 상수/데이터 소유는 TS(단일 출처): 슬롯 라벨 맵·슬롯 id 순서·스타터 무기 정의·아이템 정의 배열은
// 모두 원본 playerEquipment.ts에서 import 해 인자로 Lua에 전달한다(여기엔 데이터를 복제하지 않는다).

import playerEquipmentSource from '../assets/lua/player-equipment.lua?raw'

import {
  EQUIPMENT_SLOT_IDS,
  EQUIPMENT_SLOT_LABEL_BY_ID,
  PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
  PLAYER_EQUIPMENT_STARTER_WEAPON_ITEM_DEFINITION,
  type PlayerEquipment,
  type PlayerEquipmentItem,
  type PlayerEquipmentItemDefinition,
  type PlayerEquipmentSlot,
  type PlayerEquipmentSlotId
} from '../playerEquipment'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput
} from './luaLogicHost'

// Lua는 빈 슬롯의 item 을 null 로 반환 → TS PlayerEquipmentSlot 의 undefined 로 정규화.
type RawSlot = Omit<PlayerEquipmentSlot, 'item'> & {
  item: PlayerEquipmentItem | null
}
type RawEquipment = Omit<PlayerEquipment, 'slots'> & { slots: RawSlot[] }

const normalizeSlot = (slot: RawSlot): PlayerEquipmentSlot => ({
  ...slot,
  item: slot.item === null ? undefined : slot.item
})

const normalizeEquipment = (raw: RawEquipment): PlayerEquipment => ({
  ...raw,
  slots: raw.slots.map(normalizeSlot)
})

export type PlayerEquipmentLua = {
  createInitialPlayerEquipment: () => PlayerEquipment
  getPlayerEquipmentSlotLabelById: (slotId: PlayerEquipmentSlotId) => string
  createPlayerEquipmentItemFromDefinition: (
    definition: PlayerEquipmentItemDefinition
  ) => PlayerEquipmentItem
  getPlayerEquipmentItemDefinitionById: (
    itemId: string
  ) => PlayerEquipmentItemDefinition | undefined
  getPlayerEquipmentItemDefinitionBySlotId: (
    slotId: PlayerEquipmentSlotId
  ) => PlayerEquipmentItemDefinition | undefined
  getPlayerEquipmentSlotById: (
    equipment: PlayerEquipment,
    slotId: PlayerEquipmentSlotId
  ) => PlayerEquipmentSlot | undefined
  findPlayerEquipmentSlotIndexById: (
    equipment: PlayerEquipment,
    slotId: PlayerEquipmentSlotId
  ) => number
  setPlayerEquipmentSlot: (input: {
    equipment: PlayerEquipment
    slotId: PlayerEquipmentSlotId
    item: PlayerEquipmentItem
  }) => PlayerEquipment
  clearPlayerEquipmentSlot: (input: {
    equipment: PlayerEquipment
    slotId: PlayerEquipmentSlotId
  }) => PlayerEquipment
  close: () => void
}

export const createPlayerEquipmentLua = async (
  input: CreateLuaLogicHostInput = {}
): Promise<PlayerEquipmentLua> => {
  const host = await createLuaLogicHost(input)
  host.runModule(playerEquipmentSource, '@player-equipment.lua')

  return {
    createInitialPlayerEquipment: (): PlayerEquipment =>
      normalizeEquipment(
        host.callJson<RawEquipment>(
          'equipment_create_initial',
          EQUIPMENT_SLOT_IDS,
          EQUIPMENT_SLOT_LABEL_BY_ID,
          PLAYER_EQUIPMENT_STARTER_WEAPON_ITEM_DEFINITION
        )
      ),
    getPlayerEquipmentSlotLabelById: (slotId: PlayerEquipmentSlotId): string =>
      host.callJson<string>(
        'equipment_slot_label_by_id',
        EQUIPMENT_SLOT_LABEL_BY_ID,
        slotId
      ),
    createPlayerEquipmentItemFromDefinition: (
      definition: PlayerEquipmentItemDefinition
    ): PlayerEquipmentItem =>
      host.callJson<PlayerEquipmentItem>(
        'equipment_create_item_from_definition',
        definition
      ),
    getPlayerEquipmentItemDefinitionById: (
      itemId: string
    ): PlayerEquipmentItemDefinition | undefined => {
      const result = host.callJson<PlayerEquipmentItemDefinition | null>(
        'equipment_item_definition_by_id',
        PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
        itemId
      )

      return result === null ? undefined : result
    },
    getPlayerEquipmentItemDefinitionBySlotId: (
      slotId: PlayerEquipmentSlotId
    ): PlayerEquipmentItemDefinition | undefined => {
      const result = host.callJson<PlayerEquipmentItemDefinition | null>(
        'equipment_item_definition_by_slot_id',
        PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
        slotId
      )

      return result === null ? undefined : result
    },
    getPlayerEquipmentSlotById: (
      equipment: PlayerEquipment,
      slotId: PlayerEquipmentSlotId
    ): PlayerEquipmentSlot | undefined => {
      const result = host.callJson<RawSlot | null>(
        'equipment_slot_by_id',
        equipment,
        slotId
      )

      return result === null ? undefined : normalizeSlot(result)
    },
    findPlayerEquipmentSlotIndexById: (
      equipment: PlayerEquipment,
      slotId: PlayerEquipmentSlotId
    ): number =>
      host.callJson<number>(
        'equipment_find_slot_index_by_id',
        equipment,
        slotId
      ),
    setPlayerEquipmentSlot: ({ equipment, slotId, item }): PlayerEquipment =>
      normalizeEquipment(
        host.callJson<RawEquipment>(
          'equipment_set_slot',
          equipment,
          slotId,
          item
        )
      ),
    clearPlayerEquipmentSlot: ({ equipment, slotId }): PlayerEquipment =>
      normalizeEquipment(
        host.callJson<RawEquipment>('equipment_clear_slot', equipment, slotId)
      ),
    close: (): void => {
      host.close()
    }
  }
}
