import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialPlayerInventory,
  setPlayerInventorySlot,
  clearPlayerInventorySlot,
  getPlayerInventoryFilledSlotCount,
  findFirstEmptyPlayerInventorySlotIndex,
  type PlayerInventoryItem
} from '../playerInventory'
import {
  createPlayerInventoryLua,
  type PlayerInventoryLua
} from './playerInventoryLua'

// playerInventory(빈 슬롯 = 배열 null 구멍)를 callJson + json_null 센티넬로 변환한 결과가
// TS와 동등한지 실제 WASM으로 검증한다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const SWORD: PlayerInventoryItem = { id: 'iron-sword', label: '강철 검', quantity: 1 }
const POTION: PlayerInventoryItem = { id: 'health-potion', label: '체력 회복 포션', quantity: 9 }

describe('playerInventoryLua (real wasm)', () => {
  let lua: PlayerInventoryLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerInventory for create/set/clear/count/find', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerInventoryLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // create: 기본(스타터 포함), 슬롯 지정(스타터 없음), 골드 지정
    expect(lua.createInitialPlayerInventory()).toEqual(createInitialPlayerInventory())
    expect(lua.createInitialPlayerInventory({ slotCount: 5 })).toEqual(
      createInitialPlayerInventory({ slotCount: 5 })
    )
    expect(lua.createInitialPlayerInventory({ slotCount: 0 })).toEqual(
      createInitialPlayerInventory({ slotCount: 0 })
    )
    expect(lua.createInitialPlayerInventory({ gold: 50 })).toEqual(
      createInitialPlayerInventory({ gold: 50 })
    )

    const base = createInitialPlayerInventory({ slotCount: 4 })

    // set: 여러 인덱스
    for (const slotIndex of [0, 1, 3]) {
      expect(lua.setPlayerInventorySlot({ inventory: base, slotIndex, item: SWORD })).toEqual(
        setPlayerInventorySlot({ inventory: base, slotIndex, item: SWORD })
      )
    }

    const withItems = setPlayerInventorySlot({
      inventory: setPlayerInventorySlot({ inventory: base, slotIndex: 0, item: SWORD }),
      slotIndex: 2,
      item: POTION
    })

    // clear / count / find
    expect(lua.clearPlayerInventorySlot({ inventory: withItems, slotIndex: 0 })).toEqual(
      clearPlayerInventorySlot({ inventory: withItems, slotIndex: 0 })
    )
    expect(lua.getPlayerInventoryFilledSlotCount(withItems)).toBe(
      getPlayerInventoryFilledSlotCount(withItems)
    )
    expect(lua.findFirstEmptyPlayerInventorySlotIndex(withItems)).toBe(
      findFirstEmptyPlayerInventorySlotIndex(withItems)
    )

    // 가득 찬 경우 find는 undefined
    const full = createInitialPlayerInventory({ slotCount: 2 })
    const fullWith = setPlayerInventorySlot({
      inventory: setPlayerInventorySlot({ inventory: full, slotIndex: 0, item: SWORD }),
      slotIndex: 1,
      item: POTION
    })
    expect(lua.findFirstEmptyPlayerInventorySlotIndex(fullWith)).toBe(
      findFirstEmptyPlayerInventorySlotIndex(fullWith)
    )

    // 잘못된 인덱스는 양쪽 모두 throw
    expect(() => lua?.setPlayerInventorySlot({ inventory: base, slotIndex: 99, item: SWORD })).toThrow()
    expect(() => setPlayerInventorySlot({ inventory: base, slotIndex: 99, item: SWORD })).toThrow()
    expect(() => lua?.clearPlayerInventorySlot({ inventory: base, slotIndex: -1 })).toThrow()
  })
})
