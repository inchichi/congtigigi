import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialPlayerEquipment,
  setPlayerEquipmentSlot,
  PLAYER_EQUIPMENT_ITEM_DEFINITIONS
} from '../playerEquipment'
import {
  createInitialPlayerInventory,
  setPlayerInventorySlot,
  type PlayerInventoryItem
} from '../playerInventory'
import {
  equipPlayerInventorySlot,
  unequipPlayerEquipmentSlot
} from '../playerLoadout'
import {
  createPlayerLoadoutLua,
  type PlayerLoadoutLua
} from './playerLoadoutLua'

// playerLoadout(장착/해제) 를 callJson + json_null 센티넬로 변환한 결과가
// 원본 TS 와 동등한지 실제 WASM 으로 검증한다
// (player-inventory.lua + player-equipment.lua 의존 포함).

const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const IRON_SWORD_INV: PlayerInventoryItem = {
  id: 'iron-sword',
  label: '강철 검',
  quantity: 1
}
const BASIC_ARMOR_INV: PlayerInventoryItem = {
  id: 'basic-armor',
  label: '기본 옷',
  quantity: 1
}
const HEALTH_POTION_INV: PlayerInventoryItem = {
  id: 'health-potion',
  label: '체력 회복 포션',
  quantity: 5
}

describe('playerLoadoutLua (real wasm)', () => {
  let lua: PlayerLoadoutLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerLoadout across equip/unequip and edge cases', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerLoadoutLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    const baseEquipment = createInitialPlayerEquipment()
    const baseInventory = createInitialPlayerInventory({ slotCount: 4 })

    // ── equipPlayerInventorySlot: iron-sword → weapon slot (no swap, slot was occupied by basic-sword) ──
    const invWithIronSword = setPlayerInventorySlot({
      inventory: baseInventory,
      slotIndex: 0,
      item: IRON_SWORD_INV
    })
    expect(
      lua.equipPlayerInventorySlot({
        equipment: baseEquipment,
        inventory: invWithIronSword,
        slotIndex: 0
      })
    ).toEqual(
      equipPlayerInventorySlot({
        equipment: baseEquipment,
        inventory: invWithIronSword,
        slotIndex: 0
      })
    )

    // ── equipPlayerInventorySlot: basic-armor → armor slot (empty slot, no swap) ──
    const invWithArmor = setPlayerInventorySlot({
      inventory: baseInventory,
      slotIndex: 0,
      item: BASIC_ARMOR_INV
    })
    expect(
      lua.equipPlayerInventorySlot({
        equipment: baseEquipment,
        inventory: invWithArmor,
        slotIndex: 0
      })
    ).toEqual(
      equipPlayerInventorySlot({
        equipment: baseEquipment,
        inventory: invWithArmor,
        slotIndex: 0
      })
    )

    // ── equipPlayerInventorySlot: swap — equip iron-sword when armor is already equipped ──
    const equipmentWithArmor = setPlayerEquipmentSlot({
      equipment: baseEquipment,
      slotId: 'armor',
      item: {
        id: 'basic-armor',
        label: '기본 옷',
        level: 1,
        description: '초보용 옷'
      }
    })
    const invForSwap = setPlayerInventorySlot({
      inventory: baseInventory,
      slotIndex: 2,
      item: IRON_SWORD_INV
    })
    expect(
      lua.equipPlayerInventorySlot({
        equipment: equipmentWithArmor,
        inventory: invForSwap,
        slotIndex: 2
      })
    ).toEqual(
      equipPlayerInventorySlot({
        equipment: equipmentWithArmor,
        inventory: invForSwap,
        slotIndex: 2
      })
    )

    // ── equipPlayerInventorySlot: empty slot → undefined ──
    expect(
      lua.equipPlayerInventorySlot({
        equipment: baseEquipment,
        inventory: baseInventory,
        slotIndex: 0
      })
    ).toBeUndefined()
    expect(
      equipPlayerInventorySlot({
        equipment: baseEquipment,
        inventory: baseInventory,
        slotIndex: 0
      })
    ).toBeUndefined()

    // ── equipPlayerInventorySlot: no equipment definition (non-equippable item) → undefined ──
    const invWithPotion = setPlayerInventorySlot({
      inventory: baseInventory,
      slotIndex: 0,
      item: HEALTH_POTION_INV
    })
    expect(
      lua.equipPlayerInventorySlot({
        equipment: baseEquipment,
        inventory: invWithPotion,
        slotIndex: 0
      })
    ).toBeUndefined()
    expect(
      equipPlayerInventorySlot({
        equipment: baseEquipment,
        inventory: invWithPotion,
        slotIndex: 0
      })
    ).toBeUndefined()

    // ── unequipPlayerEquipmentSlot: weapon slot (basic-sword → inventory) ──
    expect(
      lua.unequipPlayerEquipmentSlot({
        equipment: baseEquipment,
        inventory: baseInventory,
        slotId: 'weapon'
      })
    ).toEqual(
      unequipPlayerEquipmentSlot({
        equipment: baseEquipment,
        inventory: baseInventory,
        slotId: 'weapon'
      })
    )

    // ── unequipPlayerEquipmentSlot: equip iron-sword first, then unequip it ──
    const equipResult = equipPlayerInventorySlot({
      equipment: baseEquipment,
      inventory: invWithIronSword,
      slotIndex: 0
    })!
    expect(
      lua.unequipPlayerEquipmentSlot({
        equipment: equipResult.equipment,
        inventory: equipResult.inventory,
        slotId: 'weapon'
      })
    ).toEqual(
      unequipPlayerEquipmentSlot({
        equipment: equipResult.equipment,
        inventory: equipResult.inventory,
        slotId: 'weapon'
      })
    )

    // ── unequipPlayerEquipmentSlot: empty equipment slot → undefined ──
    expect(
      lua.unequipPlayerEquipmentSlot({
        equipment: baseEquipment,
        inventory: baseInventory,
        slotId: 'armor'
      })
    ).toBeUndefined()
    expect(
      unequipPlayerEquipmentSlot({
        equipment: baseEquipment,
        inventory: baseInventory,
        slotId: 'armor'
      })
    ).toBeUndefined()

    // ── unequipPlayerEquipmentSlot: no inventory space → undefined ──
    const fullInventory = setPlayerInventorySlot({
      inventory: setPlayerInventorySlot({
        inventory: createInitialPlayerInventory({ slotCount: 1 }),
        slotIndex: 0,
        item: HEALTH_POTION_INV
      }),
      slotIndex: 0,
      item: HEALTH_POTION_INV
    })
    expect(
      lua.unequipPlayerEquipmentSlot({
        equipment: baseEquipment,
        inventory: fullInventory,
        slotId: 'weapon'
      })
    ).toBeUndefined()
    expect(
      unequipPlayerEquipmentSlot({
        equipment: baseEquipment,
        inventory: fullInventory,
        slotId: 'weapon'
      })
    ).toBeUndefined()

    // ── all equipment item definitions can be equipped ──
    for (const definition of PLAYER_EQUIPMENT_ITEM_DEFINITIONS) {
      const invItem: PlayerInventoryItem = {
        id: definition.id,
        label: definition.label,
        quantity: 1
      }
      const invForDef = setPlayerInventorySlot({
        inventory: baseInventory,
        slotIndex: 0,
        item: invItem
      })
      expect(
        lua.equipPlayerInventorySlot({
          equipment: baseEquipment,
          inventory: invForDef,
          slotIndex: 0
        })
      ).toEqual(
        equipPlayerInventorySlot({
          equipment: baseEquipment,
          inventory: invForDef,
          slotIndex: 0
        })
      )
    }
  })
})
