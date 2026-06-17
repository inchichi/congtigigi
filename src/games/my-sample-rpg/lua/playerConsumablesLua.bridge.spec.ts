import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialPlayerInventory,
  setPlayerInventorySlot,
  type PlayerInventoryItem
} from '../playerInventory'
import {
  createInitialPlayerQuickslots,
  setPlayerQuickslotAssignment
} from '../playerQuickslots'
import { createInitialPlayerProfile } from '../playerProfile'
import {
  usePlayerInventoryConsumable,
  usePlayerQuickslotConsumable
} from '../playerConsumables'
import {
  createPlayerConsumablesLua,
  type PlayerConsumablesLua
} from './playerConsumablesLua'

// playerConsumables(포션 소비 + 빠른 슬롯) 를 callJson + json_null 센티넬로 변환한 결과가
// 원본 TS 와 동등한지 실제 WASM 으로 검증한다
// (player-inventory.lua + player-quickslots.lua 의존 포함).

const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const HEALTH_POTION: PlayerInventoryItem = {
  id: 'health-potion',
  label: '체력 회복 포션',
  quantity: 2
}
const MANA_POTION: PlayerInventoryItem = {
  id: 'mana-potion',
  label: '마나 회복 포션',
  quantity: 1
}
const UNKNOWN_ITEM: PlayerInventoryItem = {
  id: 'mystery-rock',
  label: '수수께끼 돌',
  quantity: 3
}

describe('playerConsumablesLua (real wasm)', () => {
  let lua: PlayerConsumablesLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerConsumables across all cases', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerConsumablesLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    const profile = createInitialPlayerProfile()

    // ── usePlayerInventoryConsumable: health-potion (quantity > 1 → set with quantity-1) ──
    const invWithHealthPotion = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 3 }),
      slotIndex: 0,
      item: HEALTH_POTION
    })
    expect(
      lua.usePlayerInventoryConsumable({ profile, inventory: invWithHealthPotion, slotIndex: 0 })
    ).toEqual(
      usePlayerInventoryConsumable({ profile, inventory: invWithHealthPotion, slotIndex: 0 })
    )

    // health-potion fills hp to max (already at max)
    const profileFullHp = { ...profile, hp: { current: 24, max: 24 } }
    expect(
      lua.usePlayerInventoryConsumable({
        profile: profileFullHp,
        inventory: invWithHealthPotion,
        slotIndex: 0
      })
    ).toEqual(
      usePlayerInventoryConsumable({
        profile: profileFullHp,
        inventory: invWithHealthPotion,
        slotIndex: 0
      })
    )

    // ── usePlayerInventoryConsumable: health-potion (quantity = 1 → clear slot) ──
    const invWithOneHealthPotion = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2 }),
      slotIndex: 1,
      item: { ...HEALTH_POTION, quantity: 1 }
    })
    const profileLowHp = { ...profile, hp: { current: 5, max: 24 } }
    expect(
      lua.usePlayerInventoryConsumable({
        profile: profileLowHp,
        inventory: invWithOneHealthPotion,
        slotIndex: 1
      })
    ).toEqual(
      usePlayerInventoryConsumable({
        profile: profileLowHp,
        inventory: invWithOneHealthPotion,
        slotIndex: 1
      })
    )

    // ── usePlayerInventoryConsumable: mana-potion (quantity = 1 → clear slot) ──
    const invWithManaPotion = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 2 }),
      slotIndex: 0,
      item: MANA_POTION
    })
    const profileLowMp = { ...profile, mp: { current: 3, max: 12 } }
    expect(
      lua.usePlayerInventoryConsumable({
        profile: profileLowMp,
        inventory: invWithManaPotion,
        slotIndex: 0
      })
    ).toEqual(
      usePlayerInventoryConsumable({
        profile: profileLowMp,
        inventory: invWithManaPotion,
        slotIndex: 0
      })
    )

    // mana already at max
    const profileFullMp = { ...profile, mp: { current: 12, max: 12 } }
    expect(
      lua.usePlayerInventoryConsumable({
        profile: profileFullMp,
        inventory: invWithManaPotion,
        slotIndex: 0
      })
    ).toEqual(
      usePlayerInventoryConsumable({
        profile: profileFullMp,
        inventory: invWithManaPotion,
        slotIndex: 0
      })
    )

    // ── usePlayerInventoryConsumable: unknown item → undefined ──
    const invWithUnknown = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 1 }),
      slotIndex: 0,
      item: UNKNOWN_ITEM
    })
    expect(
      lua.usePlayerInventoryConsumable({ profile, inventory: invWithUnknown, slotIndex: 0 })
    ).toBeUndefined()
    expect(
      usePlayerInventoryConsumable({ profile, inventory: invWithUnknown, slotIndex: 0 })
    ).toBeUndefined()

    // ── usePlayerInventoryConsumable: empty slot → undefined ──
    const emptyInventory = createInitialPlayerInventory({ slotCount: 2 })
    expect(
      lua.usePlayerInventoryConsumable({ profile, inventory: emptyInventory, slotIndex: 0 })
    ).toBeUndefined()
    expect(
      usePlayerInventoryConsumable({ profile, inventory: emptyInventory, slotIndex: 0 })
    ).toBeUndefined()

    // ── usePlayerQuickslotConsumable: assigned quickslot → delegates to inventory ──
    const invForQuick = setPlayerInventorySlot({
      inventory: createInitialPlayerInventory({ slotCount: 3 }),
      slotIndex: 1,
      item: HEALTH_POTION
    })
    const quickslots = setPlayerQuickslotAssignment({
      quickslots: createInitialPlayerQuickslots(),
      quickslotIndex: 0,
      inventorySlotIndex: 1
    })
    const profileForQuick = { ...profile, hp: { current: 5, max: 24 } }
    expect(
      lua.usePlayerQuickslotConsumable({
        profile: profileForQuick,
        inventory: invForQuick,
        quickslots,
        quickslotIndex: 0
      })
    ).toEqual(
      usePlayerQuickslotConsumable({
        profile: profileForQuick,
        inventory: invForQuick,
        quickslots,
        quickslotIndex: 0
      })
    )

    // ── usePlayerQuickslotConsumable: no assignment → undefined ──
    const emptyQuickslots = createInitialPlayerQuickslots()
    expect(
      lua.usePlayerQuickslotConsumable({
        profile,
        inventory: emptyInventory,
        quickslots: emptyQuickslots,
        quickslotIndex: 0
      })
    ).toBeUndefined()
    expect(
      usePlayerQuickslotConsumable({
        profile,
        inventory: emptyInventory,
        quickslots: emptyQuickslots,
        quickslotIndex: 0
      })
    ).toBeUndefined()

    // ── usePlayerQuickslotConsumable: quickslot points to empty inv slot → undefined ──
    const quickslotsPointingEmpty = setPlayerQuickslotAssignment({
      quickslots: createInitialPlayerQuickslots(),
      quickslotIndex: 2,
      inventorySlotIndex: 0
    })
    expect(
      lua.usePlayerQuickslotConsumable({
        profile,
        inventory: emptyInventory,
        quickslots: quickslotsPointingEmpty,
        quickslotIndex: 2
      })
    ).toBeUndefined()
    expect(
      usePlayerQuickslotConsumable({
        profile,
        inventory: emptyInventory,
        quickslots: quickslotsPointingEmpty,
        quickslotIndex: 2
      })
    ).toBeUndefined()
  })
})
