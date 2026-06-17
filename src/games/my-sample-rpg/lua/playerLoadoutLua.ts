// playerLoadout 연산을 Lua로 실행하는 래퍼.
// player-loadout.lua 는 같은 호스트에 로드된 equipment_* / inventory_* 전역을 호출한다.
// 따라서 자체 호스트를 만들 때(input.host 미지정)는
//   player-inventory.lua → player-equipment.lua → player-loadout.lua 순서로 로드한다.
// 결과가 json_null(= null) 이면 TS 에서 undefined 로 정규화한다.
// inventory.slots 의 null 구멍과 equipment.slots[i].item 의 null 도 undefined 로 정규화한다.

import playerInventorySource from '../assets/lua/player-inventory.lua?raw'
import playerEquipmentSource from '../assets/lua/player-equipment.lua?raw'
import playerLoadoutSource from '../assets/lua/player-loadout.lua?raw'

import {
  EQUIPMENT_SLOT_IDS,
  EQUIPMENT_SLOT_LABEL_BY_ID,
  PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
  PLAYER_EQUIPMENT_STARTER_WEAPON_ITEM_DEFINITION,
  type PlayerEquipment,
  type PlayerEquipmentItem,
  type PlayerEquipmentSlot,
  type PlayerEquipmentSlotId
} from '../playerEquipment'
import type { PlayerInventory, PlayerInventoryItem } from '../playerInventory'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type PlayerLoadoutState = {
  equipment: PlayerEquipment
  inventory: PlayerInventory
}

type RawInventory = {
  gold: number
  slots: (PlayerInventoryItem | null)[]
}

type RawSlot = Omit<PlayerEquipmentSlot, 'item'> & {
  item: PlayerEquipmentItem | null
}

type RawEquipment = Omit<PlayerEquipment, 'slots'> & { slots: RawSlot[] }

type RawLoadoutState = {
  equipment: RawEquipment
  inventory: RawInventory
} | null

const normalizeInventory = (raw: RawInventory): PlayerInventory => ({
  gold: raw.gold,
  slots: raw.slots.map((slot) => (slot === null ? undefined : slot))
})

const normalizeSlot = (slot: RawSlot): PlayerEquipmentSlot => ({
  ...slot,
  item: slot.item === null ? undefined : slot.item
})

const normalizeEquipment = (raw: RawEquipment): PlayerEquipment => ({
  ...raw,
  slots: raw.slots.map(normalizeSlot)
})

const normalizeLoadoutState = (
  raw: RawLoadoutState
): PlayerLoadoutState | undefined => {
  if (raw === null) return undefined
  return {
    equipment: normalizeEquipment(raw.equipment),
    inventory: normalizeInventory(raw.inventory)
  }
}

export type PlayerLoadoutLua = {
  unequipPlayerEquipmentSlot: (input: {
    equipment: PlayerEquipment
    inventory: PlayerInventory
    slotId: PlayerEquipmentSlotId
  }) => PlayerLoadoutState | undefined
  equipPlayerInventorySlot: (input: {
    equipment: PlayerEquipment
    inventory: PlayerInventory
    slotIndex: number
  }) => PlayerLoadoutState | undefined
  close: () => void
}

export const createPlayerLoadoutLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerLoadoutLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))

  if (!input.host) {
    host.runModule(playerInventorySource, '@player-inventory.lua')
    host.runModule(playerEquipmentSource, '@player-equipment.lua')
  }
  host.runModule(playerLoadoutSource, '@player-loadout.lua')

  return {
    unequipPlayerEquipmentSlot: ({ equipment, inventory, slotId }) =>
      normalizeLoadoutState(
        host.callJson<RawLoadoutState>(
          'loadout_unequip_slot',
          equipment,
          inventory,
          slotId,
          PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
          EQUIPMENT_SLOT_IDS,
          EQUIPMENT_SLOT_LABEL_BY_ID,
          PLAYER_EQUIPMENT_STARTER_WEAPON_ITEM_DEFINITION
        )
      ),
    equipPlayerInventorySlot: ({ equipment, inventory, slotIndex }) =>
      normalizeLoadoutState(
        host.callJson<RawLoadoutState>(
          'loadout_equip_inventory_slot',
          equipment,
          inventory,
          slotIndex,
          PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
          EQUIPMENT_SLOT_IDS,
          EQUIPMENT_SLOT_LABEL_BY_ID,
          PLAYER_EQUIPMENT_STARTER_WEAPON_ITEM_DEFINITION
        )
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
