import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  serializePlayerSaveState,
  normalizeStoredPlayerSaveState,
  parseStoredPlayerSaveState,
  PLAYER_SAVE_STATE_VERSION,
  type PlayerSaveState,
  type PlayerSaveStateInput
} from '../playerSaveState'
import {
  createPlayerSaveStateLua,
  type PlayerSaveStateLua
} from './playerSaveStateLua'

// playerSaveState(깊은 중첩 정규화 + 직렬화)를 callJson + json_null 센티넬로 변환한 결과가
// TS와 동등한지 실제 WASM으로 검증한다. 직렬화 문자열은 키 순서가 비결정적이므로 파싱해 비교한다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createLua = async (): Promise<PlayerSaveStateLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createPlayerSaveStateLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

// 완전하고 유효한 저장 입력(undefined 슬롯 구멍 포함)을 만든다.
const makeValidInput = (): PlayerSaveStateInput => ({
  profile: {
    name: 'zl존 준수',
    job: '전사',
    level: 12,
    experience: { current: 345.5 },
    statPoints: 3,
    availableSkillPoints: 2,
    totalSkillPointsEarned: 7,
    hp: { current: 24, max: 30 },
    mp: { current: 5.25, max: 12 },
    stats: { strength: 5, agility: 4, intelligence: 3, luck: 2 },
    skills: [
      {
        hotkey: '1',
        label: '베기',
        description: '재빠른 근거리 공격',
        level: 2,
        maxLevel: 5
      },
      {
        hotkey: '2',
        label: '방어 자세',
        description: '들어오는 피해를 막아냅니다',
        level: 0,
        maxLevel: 5
      }
    ]
  },
  equipment: {
    setName: '기본 장비',
    level: 1,
    slots: [
      {
        id: 'weapon',
        label: '무기',
        item: {
          id: 'iron-sword',
          label: '강철 검',
          level: 2,
          description: '균형이 좋은 근접 무기'
        }
      },
      { id: 'armor', label: '옷', item: undefined },
      { id: 'hat', label: '모자', item: undefined }
    ]
  },
  inventory: {
    gold: 1000,
    slots: [
      { id: 'health-potion', label: '체력 회복 포션', quantity: 30 },
      undefined,
      { id: 'mana-potion', label: '마나 회복 포션', quantity: 12 },
      undefined
    ]
  },
  quickslots: {
    slots: [{ inventorySlotIndex: 0 }, undefined, { inventorySlotIndex: 2 }]
  },
  skillSlots: {
    slots: [{ skillId: 'cleave' }, undefined, { skillId: 'guard' }]
  }
})

const makeValidSaveState = (): PlayerSaveState => ({
  version: PLAYER_SAVE_STATE_VERSION,
  ...makeValidInput()
})

describe('playerSaveStateLua (real wasm)', () => {
  let lua: PlayerSaveStateLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('serializePlayerSaveState matches TS (parsed, since key order differs)', async () => {
    lua = await createLua()

    const inputs: PlayerSaveStateInput[] = [
      makeValidInput(),
      // 모든 슬롯이 빈 경우(undefined 구멍만)
      {
        ...makeValidInput(),
        inventory: { gold: 0, slots: [undefined, undefined] },
        quickslots: { slots: [undefined, undefined, undefined] },
        skillSlots: { slots: [undefined] }
      },
      // 빈 배열 슬롯
      {
        ...makeValidInput(),
        inventory: { gold: 5, slots: [] },
        quickslots: { slots: [] },
        skillSlots: { slots: [] },
        equipment: { setName: 's', level: 1, slots: [] }
      },
      // 음수/소수/0 숫자
      {
        ...makeValidInput(),
        inventory: { gold: -3.75, slots: [] }
      }
    ]

    for (const input of inputs) {
      const luaJson = lua.serializePlayerSaveState(input)
      const tsJson = serializePlayerSaveState(input)

      // 직렬화 결과는 같은 데이터를 인코딩해야 한다(키 순서는 무관).
      expect(JSON.parse(luaJson)).toEqual(JSON.parse(tsJson))
      // 그리고 그 데이터는 version + 입력 그대로여야 한다.
      expect(JSON.parse(luaJson)).toEqual(
        JSON.parse(JSON.stringify({ version: PLAYER_SAVE_STATE_VERSION, ...input }))
      )
    }
  })

  it('normalizeStoredPlayerSaveState matches TS for valid + many malformed inputs', async () => {
    lua = await createLua()

    const valid = makeValidSaveState()

    const cases: unknown[] = [
      // 정상
      valid,
      // 빈 배열 슬롯들
      {
        ...valid,
        inventory: { gold: 0, slots: [] },
        quickslots: { slots: [] },
        skillSlots: { slots: [] },
        equipment: { setName: 's', level: 1, slots: [] }
      },
      // 슬롯에 null 구멍 + 손상 아이템(관대 처리 → 빈 슬롯)
      {
        ...valid,
        inventory: {
          gold: 1,
          slots: [
            null,
            { id: 'x', label: 'y', quantity: 1 },
            { id: 'broken' }, // quantity 없음 → 빈 슬롯
            { id: 1, label: 'z', quantity: 2 } // id 비문자열 → 빈 슬롯
          ]
        }
      },
      // quickslots/skillSlots 손상 항목 → 빈 슬롯
      {
        ...valid,
        quickslots: {
          slots: [null, { inventorySlotIndex: 'no' }, { inventorySlotIndex: 4 }]
        },
        skillSlots: {
          slots: [null, { skillId: '' }, { skillId: 'ok' }, { notSkill: 1 }]
        }
      },

      // ── 전체 폐기되어야 하는 손상들(undefined 기대) ──
      undefined,
      null,
      42,
      'string',
      true,
      [],
      [1, 2, 3],
      {},
      // 버전 불일치
      { ...valid, version: 0 },
      { ...valid, version: 2 },
      { ...valid, version: '1' },
      // version 없음
      (() => {
        const { version: _omit, ...rest } = valid
        return rest
      })(),
      // profile 손상: 필드 누락/타입 오류
      { ...valid, profile: { ...valid.profile, name: 123 } },
      { ...valid, profile: { ...valid.profile, level: 'x' } },
      { ...valid, profile: { ...valid.profile, job: undefined } },
      // 비유한 숫자(NaN/Inf) → JSON.stringify가 null로 → 정규화 실패
      { ...valid, profile: { ...valid.profile, level: Number.NaN } },
      { ...valid, profile: { ...valid.profile, level: Number.POSITIVE_INFINITY } },
      { ...valid, profile: { ...valid.profile, statPoints: -Number.POSITIVE_INFINITY } },
      // experience/hp/mp/stats 손상
      { ...valid, profile: { ...valid.profile, experience: {} } },
      { ...valid, profile: { ...valid.profile, experience: { current: 'x' } } },
      { ...valid, profile: { ...valid.profile, hp: { current: 1 } } }, // max 없음
      { ...valid, profile: { ...valid.profile, mp: null } },
      { ...valid, profile: { ...valid.profile, stats: { strength: 1, agility: 2, intelligence: 3 } } },
      // skills 배열 아님
      { ...valid, profile: { ...valid.profile, skills: 'nope' } },
      { ...valid, profile: { ...valid.profile, skills: {} } },
      // skills 항목 하나 손상 → 전체 폐기
      {
        ...valid,
        profile: {
          ...valid.profile,
          skills: [{ hotkey: '1', label: 'a', description: 'b', level: 0 }] // maxLevel 없음
        }
      },
      {
        ...valid,
        profile: {
          ...valid.profile,
          skills: [{ hotkey: '1', label: 'a', description: 'b', level: 0, maxLevel: 5 }, null]
        }
      },
      // equipment 손상
      { ...valid, equipment: { ...valid.equipment, setName: 5 } },
      { ...valid, equipment: { ...valid.equipment, level: 'x' } },
      { ...valid, equipment: { ...valid.equipment, slots: 'nope' } },
      // equipment 슬롯 하나 손상(id 누락) → 전체 폐기
      {
        ...valid,
        equipment: {
          ...valid.equipment,
          slots: [{ label: '무기', item: undefined }]
        }
      },
      // equipment 슬롯 item 손상 → 관대 처리(빈 item, 슬롯은 유지)
      {
        ...valid,
        equipment: {
          setName: '기본 장비',
          level: 1,
          slots: [
            { id: 'weapon', label: '무기', item: { id: 'x' } }, // level/desc 없음 → item undefined
            { id: 'armor', label: '옷', item: null }
          ]
        }
      },
      // inventory 손상
      { ...valid, inventory: { gold: 'x', slots: [] } },
      { ...valid, inventory: { gold: 1, slots: {} } },
      { ...valid, inventory: 'nope' },
      // quickslots/skillSlots 손상
      { ...valid, quickslots: { slots: 'nope' } },
      { ...valid, quickslots: {} },
      { ...valid, skillSlots: { slots: 5 } },
      { ...valid, skillSlots: null },
      // 추가 필드는 무시되어야 함(정상으로 통과)
      { ...valid, extra: 'ignored', another: { nested: true } }
    ]

    for (const value of cases) {
      const tsResult = normalizeStoredPlayerSaveState(value)
      const luaResult = lua.normalizeStoredPlayerSaveState(value)

      expect(luaResult).toEqual(tsResult)
    }
  })

  it('parseStoredPlayerSaveState matches TS for valid/malformed/falsy strings', async () => {
    lua = await createLua()

    const valid = makeValidSaveState()
    const validJson = JSON.stringify(valid)

    const cases: (string | null | undefined)[] = [
      validJson,
      // 빈/falsy 입력 → undefined
      '',
      null,
      undefined,
      // 파싱 실패 → undefined
      '{not json',
      'undefined',
      '][',
      '{"version":1,', // 잘린 JSON
      // 파싱은 되지만 손상 → undefined
      JSON.stringify({ version: 2, ...makeValidInput() }),
      JSON.stringify({ version: 1 }),
      JSON.stringify(42),
      JSON.stringify('hi'),
      JSON.stringify([1, 2]),
      JSON.stringify(null),
      // 슬롯 null 구멍이 든 정상 저장본
      JSON.stringify({
        ...valid,
        inventory: {
          gold: 7,
          slots: [null, { id: 'a', label: 'b', quantity: 1 }, null]
        }
      })
    ]

    for (const raw of cases) {
      const tsResult = parseStoredPlayerSaveState(raw)
      const luaResult = lua.parseStoredPlayerSaveState(raw)

      expect(luaResult).toEqual(tsResult)
    }
  })
})
