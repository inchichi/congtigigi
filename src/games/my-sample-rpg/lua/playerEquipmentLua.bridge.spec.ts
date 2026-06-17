import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  clearPlayerEquipmentSlot,
  createInitialPlayerEquipment,
  createPlayerEquipmentItemFromDefinition,
  findPlayerEquipmentSlotIndexById,
  getPlayerEquipmentItemDefinitionById,
  getPlayerEquipmentItemDefinitionBySlotId,
  getPlayerEquipmentSlotById,
  getPlayerEquipmentSlotLabelById,
  setPlayerEquipmentSlot,
  PLAYER_EQUIPMENT_ITEM_DEFINITIONS,
  type PlayerEquipmentItem,
  type PlayerEquipmentSlotId
} from '../playerEquipment'
import {
  createPlayerEquipmentLua,
  type PlayerEquipmentLua
} from './playerEquipmentLua'

// playerEquipment(빈 슬롯 item = null 구멍)를 callJson + json_null 센티넬로 변환한 결과가
// TS와 동등한지 실제 WASM으로 검증한다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const ALL_SLOT_IDS: PlayerEquipmentSlotId[] = [
  'weapon',
  'armor',
  'hat',
  'boots',
  'accessory'
]

// 알려지지 않은(존재하지 않는) 슬롯 id — set/clear/find/get 의 미발견 경로를 점검.
const UNKNOWN_SLOT_ID = 'ring' as unknown as PlayerEquipmentSlotId

const IRON_SWORD_ITEM: PlayerEquipmentItem = {
  id: 'iron-sword',
  label: '강철 검',
  level: 2,
  description: '균형이 좋은 근접 무기'
}
const LEATHER_BOOTS_ITEM: PlayerEquipmentItem = {
  id: 'leather-boots',
  label: '가죽 신발',
  level: 2,
  description: '가볍고 단단한 가죽 신발'
}

describe('playerEquipmentLua (real wasm)', () => {
  let lua: PlayerEquipmentLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerEquipment across all deterministic exports', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerEquipmentLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // createInitialPlayerEquipment: 스타터 무기 + 빈 슬롯 구조 동일
    expect(lua.createInitialPlayerEquipment()).toEqual(
      createInitialPlayerEquipment()
    )

    // getPlayerEquipmentSlotLabelById: 모든 슬롯 id
    for (const slotId of ALL_SLOT_IDS) {
      expect(lua.getPlayerEquipmentSlotLabelById(slotId)).toBe(
        getPlayerEquipmentSlotLabelById(slotId)
      )
    }

    // createPlayerEquipmentItemFromDefinition: 정의 배열 전체
    for (const definition of PLAYER_EQUIPMENT_ITEM_DEFINITIONS) {
      expect(lua.createPlayerEquipmentItemFromDefinition(definition)).toEqual(
        createPlayerEquipmentItemFromDefinition(definition)
      )
    }

    // getPlayerEquipmentItemDefinitionById: 존재하는 모든 id + 미존재 id
    for (const definition of PLAYER_EQUIPMENT_ITEM_DEFINITIONS) {
      expect(lua.getPlayerEquipmentItemDefinitionById(definition.id)).toEqual(
        getPlayerEquipmentItemDefinitionById(definition.id)
      )
    }
    for (const missingId of ['', 'does-not-exist', 'basic-swordx', 'BASIC-SWORD']) {
      expect(lua.getPlayerEquipmentItemDefinitionById(missingId)).toBe(
        getPlayerEquipmentItemDefinitionById(missingId)
      )
    }

    // getPlayerEquipmentItemDefinitionBySlotId: 모든 슬롯 id(첫 등장 정의) + 미존재 슬롯
    for (const slotId of ALL_SLOT_IDS) {
      expect(lua.getPlayerEquipmentItemDefinitionBySlotId(slotId)).toEqual(
        getPlayerEquipmentItemDefinitionBySlotId(slotId)
      )
    }
    expect(lua.getPlayerEquipmentItemDefinitionBySlotId(UNKNOWN_SLOT_ID)).toBe(
      getPlayerEquipmentItemDefinitionBySlotId(UNKNOWN_SLOT_ID)
    )

    const base = createInitialPlayerEquipment()

    // findPlayerEquipmentSlotIndexById: 모든 슬롯 id + 미존재(-1)
    for (const slotId of [...ALL_SLOT_IDS, UNKNOWN_SLOT_ID]) {
      expect(lua.findPlayerEquipmentSlotIndexById(base, slotId)).toBe(
        findPlayerEquipmentSlotIndexById(base, slotId)
      )
    }

    // getPlayerEquipmentSlotById: 채워진 슬롯(weapon) + 빈 슬롯 + 미존재
    for (const slotId of [...ALL_SLOT_IDS, UNKNOWN_SLOT_ID]) {
      expect(lua.getPlayerEquipmentSlotById(base, slotId)).toEqual(
        getPlayerEquipmentSlotById(base, slotId)
      )
    }

    // setPlayerEquipmentSlot: 빈 슬롯 채우기 + 기존 무기 슬롯 교체
    for (const slotId of ALL_SLOT_IDS) {
      const item = slotId === 'boots' ? LEATHER_BOOTS_ITEM : IRON_SWORD_ITEM
      expect(lua.setPlayerEquipmentSlot({ equipment: base, slotId, item })).toEqual(
        setPlayerEquipmentSlot({ equipment: base, slotId, item })
      )
    }

    // 누적 변경: set → clear → set 한 결과가 양쪽 동일
    const equipped = setPlayerEquipmentSlot({
      equipment: setPlayerEquipmentSlot({
        equipment: base,
        slotId: 'boots',
        item: LEATHER_BOOTS_ITEM
      }),
      slotId: 'armor',
      item: IRON_SWORD_ITEM
    })
    const luaEquipped = lua.setPlayerEquipmentSlot({
      equipment: lua.setPlayerEquipmentSlot({
        equipment: base,
        slotId: 'boots',
        item: LEATHER_BOOTS_ITEM
      }),
      slotId: 'armor',
      item: IRON_SWORD_ITEM
    })
    expect(luaEquipped).toEqual(equipped)

    // clearPlayerEquipmentSlot: 채워진 슬롯 비우기 + 이미 빈 슬롯 비우기(멱등)
    for (const slotId of ALL_SLOT_IDS) {
      expect(lua.clearPlayerEquipmentSlot({ equipment: equipped, slotId })).toEqual(
        clearPlayerEquipmentSlot({ equipment: equipped, slotId })
      )
    }

    // weapon 슬롯 비운 뒤 다시 조회 — 빈 슬롯 item 정규화(undefined) 동등
    const clearedWeapon = clearPlayerEquipmentSlot({
      equipment: base,
      slotId: 'weapon'
    })
    const luaClearedWeapon = lua.clearPlayerEquipmentSlot({
      equipment: base,
      slotId: 'weapon'
    })
    expect(luaClearedWeapon).toEqual(clearedWeapon)
    expect(lua.getPlayerEquipmentSlotById(luaClearedWeapon, 'weapon')).toEqual(
      getPlayerEquipmentSlotById(clearedWeapon, 'weapon')
    )

    // 잘못된 슬롯 id 는 양쪽 모두 throw
    expect(() =>
      lua?.setPlayerEquipmentSlot({
        equipment: base,
        slotId: UNKNOWN_SLOT_ID,
        item: IRON_SWORD_ITEM
      })
    ).toThrow()
    expect(() =>
      setPlayerEquipmentSlot({
        equipment: base,
        slotId: UNKNOWN_SLOT_ID,
        item: IRON_SWORD_ITEM
      })
    ).toThrow()
    expect(() =>
      lua?.clearPlayerEquipmentSlot({ equipment: base, slotId: UNKNOWN_SLOT_ID })
    ).toThrow()
    expect(() =>
      clearPlayerEquipmentSlot({ equipment: base, slotId: UNKNOWN_SLOT_ID })
    ).toThrow()
  })
})
