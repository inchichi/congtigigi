import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialPlayerSkillSlots,
  setPlayerSkillSlotAssignment,
  clearPlayerSkillSlotAssignment,
  findPlayerSkillSlotIndexBySkillId,
  getPlayerSkillSlotAssignment,
  getPlayerSkillSlotIndexFromCode,
  PLAYER_SKILL_SLOT_HOTKEY_CODES
} from '../playerSkillSlots'
import {
  createPlayerSkillSlotsLua,
  type PlayerSkillSlotsLua
} from './playerSkillSlotsLua'

// playerSkillSlots(빈 슬롯 = 배열 null 구멍)를 callJson + json_null 센티넬로 변환한 결과가
// TS와 동등한지 실제 WASM으로 검증한다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

describe('playerSkillSlotsLua (real wasm)', () => {
  let lua: PlayerSkillSlotsLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerSkillSlots for all 6 functions including edge cases', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerSkillSlotsLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // ── createInitialPlayerSkillSlots ──
    // 기본(4슬롯, 빈 initialSkillIds)
    expect(lua.createInitialPlayerSkillSlots()).toEqual(
      createInitialPlayerSkillSlots()
    )
    expect(lua.createInitialPlayerSkillSlots({ slotCount: 3 })).toEqual(
      createInitialPlayerSkillSlots({ slotCount: 3 })
    )
    expect(lua.createInitialPlayerSkillSlots({ slotCount: 0 })).toEqual(
      createInitialPlayerSkillSlots({ slotCount: 0 })
    )
    // initialSkillIds 로 일부 슬롯 채우기
    expect(
      lua.createInitialPlayerSkillSlots({
        slotCount: 4,
        initialSkillIds: ['fireball', undefined, 'heal', undefined]
      })
    ).toEqual(
      createInitialPlayerSkillSlots({
        slotCount: 4,
        initialSkillIds: ['fireball', undefined, 'heal', undefined]
      })
    )
    // 전부 채운 경우
    expect(
      lua.createInitialPlayerSkillSlots({
        slotCount: 4,
        initialSkillIds: ['q-skill', 'w-skill', 'e-skill', 'r-skill']
      })
    ).toEqual(
      createInitialPlayerSkillSlots({
        slotCount: 4,
        initialSkillIds: ['q-skill', 'w-skill', 'e-skill', 'r-skill']
      })
    )

    const base = createInitialPlayerSkillSlots({ slotCount: 4 })

    // ── setPlayerSkillSlotAssignment ──
    // 여러 슬롯에 할당
    for (const skillSlotIndex of [0, 1, 2, 3]) {
      expect(
        lua.setPlayerSkillSlotAssignment({
          skillSlots: base,
          skillSlotIndex,
          skillId: `skill-${skillSlotIndex}`
        })
      ).toEqual(
        setPlayerSkillSlotAssignment({
          skillSlots: base,
          skillSlotIndex,
          skillId: `skill-${skillSlotIndex}`
        })
      )
    }

    // dedup: 같은 skillId를 다른 슬롯으로 옮기면 이전 슬롯이 비워진다
    const withSlot0 = setPlayerSkillSlotAssignment({
      skillSlots: base,
      skillSlotIndex: 0,
      skillId: 'fireball'
    })
    const dedupTs = setPlayerSkillSlotAssignment({
      skillSlots: withSlot0,
      skillSlotIndex: 3,
      skillId: 'fireball'
    })
    const dedupLua = lua.setPlayerSkillSlotAssignment({
      skillSlots: withSlot0,
      skillSlotIndex: 3,
      skillId: 'fireball'
    })
    expect(dedupLua).toEqual(dedupTs)
    expect(dedupTs.slots[0]).toBeUndefined()
    expect(dedupLua.slots[0]).toBeUndefined()
    expect(dedupTs.slots[3]).toEqual({ skillId: 'fireball' })
    expect(dedupLua.slots[3]).toEqual({ skillId: 'fireball' })

    // 범위 초과/미만은 양쪽 모두 throw
    expect(() =>
      lua?.setPlayerSkillSlotAssignment({
        skillSlots: base,
        skillSlotIndex: 99,
        skillId: 'fireball'
      })
    ).toThrow()
    expect(() =>
      setPlayerSkillSlotAssignment({
        skillSlots: base,
        skillSlotIndex: 99,
        skillId: 'fireball'
      })
    ).toThrow()
    expect(() =>
      lua?.setPlayerSkillSlotAssignment({
        skillSlots: base,
        skillSlotIndex: -1,
        skillId: 'fireball'
      })
    ).toThrow()
    expect(() =>
      setPlayerSkillSlotAssignment({
        skillSlots: base,
        skillSlotIndex: -1,
        skillId: 'fireball'
      })
    ).toThrow()
    // 빈 skillId는 양쪽 모두 throw
    expect(() =>
      lua?.setPlayerSkillSlotAssignment({
        skillSlots: base,
        skillSlotIndex: 0,
        skillId: ''
      })
    ).toThrow()
    expect(() =>
      setPlayerSkillSlotAssignment({
        skillSlots: base,
        skillSlotIndex: 0,
        skillId: ''
      })
    ).toThrow()

    // ── clearPlayerSkillSlotAssignment ──
    const withTwo = setPlayerSkillSlotAssignment({
      skillSlots: setPlayerSkillSlotAssignment({
        skillSlots: base,
        skillSlotIndex: 1,
        skillId: 'ice-arrow'
      }),
      skillSlotIndex: 3,
      skillId: 'thunder'
    })

    for (const skillSlotIndex of [0, 1, 2, 3]) {
      expect(
        lua.clearPlayerSkillSlotAssignment({ skillSlots: withTwo, skillSlotIndex })
      ).toEqual(
        clearPlayerSkillSlotAssignment({ skillSlots: withTwo, skillSlotIndex })
      )
    }

    // 범위 초과는 양쪽 모두 throw
    expect(() =>
      lua?.clearPlayerSkillSlotAssignment({ skillSlots: base, skillSlotIndex: 4 })
    ).toThrow()
    expect(() =>
      clearPlayerSkillSlotAssignment({ skillSlots: base, skillSlotIndex: 4 })
    ).toThrow()
    expect(() =>
      lua?.clearPlayerSkillSlotAssignment({ skillSlots: base, skillSlotIndex: -1 })
    ).toThrow()
    expect(() =>
      clearPlayerSkillSlotAssignment({ skillSlots: base, skillSlotIndex: -1 })
    ).toThrow()

    // ── findPlayerSkillSlotIndexBySkillId ──
    expect(
      lua.findPlayerSkillSlotIndexBySkillId(withTwo, 'ice-arrow')
    ).toBe(findPlayerSkillSlotIndexBySkillId(withTwo, 'ice-arrow'))
    expect(
      lua.findPlayerSkillSlotIndexBySkillId(withTwo, 'thunder')
    ).toBe(findPlayerSkillSlotIndexBySkillId(withTwo, 'thunder'))
    // 미발견 → undefined
    expect(
      lua.findPlayerSkillSlotIndexBySkillId(withTwo, 'does-not-exist')
    ).toBe(findPlayerSkillSlotIndexBySkillId(withTwo, 'does-not-exist'))
    expect(
      lua.findPlayerSkillSlotIndexBySkillId(withTwo, 'does-not-exist')
    ).toBeUndefined()
    // 빈 슬롯들에서도 미발견
    expect(
      lua.findPlayerSkillSlotIndexBySkillId(base, 'fireball')
    ).toBe(findPlayerSkillSlotIndexBySkillId(base, 'fireball'))

    // ── getPlayerSkillSlotAssignment ──
    expect(
      lua.getPlayerSkillSlotAssignment(withTwo, 1)
    ).toEqual(getPlayerSkillSlotAssignment(withTwo, 1))
    expect(
      lua.getPlayerSkillSlotAssignment(withTwo, 3)
    ).toEqual(getPlayerSkillSlotAssignment(withTwo, 3))
    // 빈 슬롯 → undefined
    expect(
      lua.getPlayerSkillSlotAssignment(withTwo, 0)
    ).toBe(getPlayerSkillSlotAssignment(withTwo, 0))
    expect(
      lua.getPlayerSkillSlotAssignment(withTwo, 0)
    ).toBeUndefined()

    // ── getPlayerSkillSlotIndexFromCode ──
    for (const code of PLAYER_SKILL_SLOT_HOTKEY_CODES) {
      expect(lua.getPlayerSkillSlotIndexFromCode(code)).toBe(
        getPlayerSkillSlotIndexFromCode(code)
      )
    }
    // 미발견 코드 → undefined
    for (const unknownCode of ['KeyA', 'KeyZ', '', 'keyq', 'Space']) {
      expect(lua.getPlayerSkillSlotIndexFromCode(unknownCode)).toBe(
        getPlayerSkillSlotIndexFromCode(unknownCode)
      )
      expect(lua.getPlayerSkillSlotIndexFromCode(unknownCode)).toBeUndefined()
    }
  })
})
