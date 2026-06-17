import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialPlayerQuickslots,
  setPlayerQuickslotAssignment,
  clearPlayerQuickslotAssignment,
  findPlayerQuickslotIndexByInventorySlotIndex,
  getPlayerQuickslotAssignment
} from '../playerQuickslots'
import {
  createPlayerQuickslotsLua,
  type PlayerQuickslotsLua
} from './playerQuickslotsLua'

// playerQuickslots(빈 슬롯 = 배열 null 구멍)를 callJson + json_null 센티넬로 변환한 결과가
// TS와 동등한지 실제 WASM으로 검증한다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

describe('playerQuickslotsLua (real wasm)', () => {
  let lua: PlayerQuickslotsLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerQuickslots for all 5 functions including edge cases', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerQuickslotsLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // ── createInitialPlayerQuickslots ──
    // 기본(6슬롯), 지정 슬롯 수, 0슬롯
    expect(lua.createInitialPlayerQuickslots()).toEqual(
      createInitialPlayerQuickslots()
    )
    expect(lua.createInitialPlayerQuickslots({ slotCount: 4 })).toEqual(
      createInitialPlayerQuickslots({ slotCount: 4 })
    )
    expect(lua.createInitialPlayerQuickslots({ slotCount: 0 })).toEqual(
      createInitialPlayerQuickslots({ slotCount: 0 })
    )
    expect(lua.createInitialPlayerQuickslots({ slotCount: 10 })).toEqual(
      createInitialPlayerQuickslots({ slotCount: 10 })
    )

    const base = createInitialPlayerQuickslots({ slotCount: 4 })

    // ── setPlayerQuickslotAssignment ──
    // 여러 슬롯에 할당
    for (const quickslotIndex of [0, 1, 3]) {
      expect(
        lua.setPlayerQuickslotAssignment({
          quickslots: base,
          quickslotIndex,
          inventorySlotIndex: quickslotIndex * 2
        })
      ).toEqual(
        setPlayerQuickslotAssignment({
          quickslots: base,
          quickslotIndex,
          inventorySlotIndex: quickslotIndex * 2
        })
      )
    }

    // dedup: 같은 inventorySlotIndex를 다른 quickslot으로 옮기면 이전 슬롯이 비워진다
    const withSlot0 = setPlayerQuickslotAssignment({
      quickslots: base,
      quickslotIndex: 0,
      inventorySlotIndex: 5
    })
    const dedupTs = setPlayerQuickslotAssignment({
      quickslots: withSlot0,
      quickslotIndex: 2,
      inventorySlotIndex: 5
    })
    const dedupLua = lua.setPlayerQuickslotAssignment({
      quickslots: withSlot0,
      quickslotIndex: 2,
      inventorySlotIndex: 5
    })
    expect(dedupLua).toEqual(dedupTs)
    // quickslot 0 은 비워져야 한다
    expect(dedupTs.slots[0]).toBeUndefined()
    expect(dedupLua.slots[0]).toBeUndefined()
    // quickslot 2 에 할당돼야 한다
    expect(dedupTs.slots[2]).toEqual({ inventorySlotIndex: 5 })
    expect(dedupLua.slots[2]).toEqual({ inventorySlotIndex: 5 })

    // 범위 초과/미만은 양쪽 모두 throw
    expect(() =>
      lua?.setPlayerQuickslotAssignment({
        quickslots: base,
        quickslotIndex: 99,
        inventorySlotIndex: 0
      })
    ).toThrow()
    expect(() =>
      setPlayerQuickslotAssignment({
        quickslots: base,
        quickslotIndex: 99,
        inventorySlotIndex: 0
      })
    ).toThrow()
    expect(() =>
      lua?.setPlayerQuickslotAssignment({
        quickslots: base,
        quickslotIndex: -1,
        inventorySlotIndex: 0
      })
    ).toThrow()
    expect(() =>
      setPlayerQuickslotAssignment({
        quickslots: base,
        quickslotIndex: -1,
        inventorySlotIndex: 0
      })
    ).toThrow()
    // inventorySlotIndex < 0 은 양쪽 모두 throw
    expect(() =>
      lua?.setPlayerQuickslotAssignment({
        quickslots: base,
        quickslotIndex: 0,
        inventorySlotIndex: -1
      })
    ).toThrow()
    expect(() =>
      setPlayerQuickslotAssignment({
        quickslots: base,
        quickslotIndex: 0,
        inventorySlotIndex: -1
      })
    ).toThrow()

    // ── clearPlayerQuickslotAssignment ──
    const withTwo = setPlayerQuickslotAssignment({
      quickslots: setPlayerQuickslotAssignment({
        quickslots: base,
        quickslotIndex: 0,
        inventorySlotIndex: 3
      }),
      quickslotIndex: 2,
      inventorySlotIndex: 7
    })

    for (const quickslotIndex of [0, 1, 2, 3]) {
      expect(
        lua.clearPlayerQuickslotAssignment({ quickslots: withTwo, quickslotIndex })
      ).toEqual(
        clearPlayerQuickslotAssignment({ quickslots: withTwo, quickslotIndex })
      )
    }

    // 범위 초과는 양쪽 모두 throw
    expect(() =>
      lua?.clearPlayerQuickslotAssignment({ quickslots: base, quickslotIndex: 4 })
    ).toThrow()
    expect(() =>
      clearPlayerQuickslotAssignment({ quickslots: base, quickslotIndex: 4 })
    ).toThrow()
    expect(() =>
      lua?.clearPlayerQuickslotAssignment({ quickslots: base, quickslotIndex: -1 })
    ).toThrow()
    expect(() =>
      clearPlayerQuickslotAssignment({ quickslots: base, quickslotIndex: -1 })
    ).toThrow()

    // ── findPlayerQuickslotIndexByInventorySlotIndex ──
    expect(
      lua.findPlayerQuickslotIndexByInventorySlotIndex(withTwo, 3)
    ).toBe(findPlayerQuickslotIndexByInventorySlotIndex(withTwo, 3))
    expect(
      lua.findPlayerQuickslotIndexByInventorySlotIndex(withTwo, 7)
    ).toBe(findPlayerQuickslotIndexByInventorySlotIndex(withTwo, 7))
    // 미발견 → undefined
    expect(
      lua.findPlayerQuickslotIndexByInventorySlotIndex(withTwo, 99)
    ).toBe(findPlayerQuickslotIndexByInventorySlotIndex(withTwo, 99))
    expect(
      lua.findPlayerQuickslotIndexByInventorySlotIndex(withTwo, 99)
    ).toBeUndefined()
    // 빈 quickslots에서도 미발견
    expect(
      lua.findPlayerQuickslotIndexByInventorySlotIndex(base, 0)
    ).toBe(findPlayerQuickslotIndexByInventorySlotIndex(base, 0))

    // ── getPlayerQuickslotAssignment ──
    expect(
      lua.getPlayerQuickslotAssignment(withTwo, 0)
    ).toEqual(getPlayerQuickslotAssignment(withTwo, 0))
    expect(
      lua.getPlayerQuickslotAssignment(withTwo, 2)
    ).toEqual(getPlayerQuickslotAssignment(withTwo, 2))
    // 빈 슬롯 → undefined
    expect(
      lua.getPlayerQuickslotAssignment(withTwo, 1)
    ).toBe(getPlayerQuickslotAssignment(withTwo, 1))
    expect(
      lua.getPlayerQuickslotAssignment(withTwo, 1)
    ).toBeUndefined()
  })
})
