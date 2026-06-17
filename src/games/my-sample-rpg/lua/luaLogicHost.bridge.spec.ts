import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import { createLuaLogicHost, type LuaLogicHost } from './luaLogicHost'

// callJson(JSON 마샬링)이 구조적 값(객체/배열/중첩/빈 배열/유니코드)을 정확히 왕복하는지
// 실제 Lua 5.3.6 WASM으로 검증한다. (모듈 변환의 전제 — 코덱이 틀리면 여기서 잡힌다.)
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

// 테스트용 전역 함수들 — json_array는 json-codec.lua(프리앰블)가 정의.
const TEST_MODULE = `
function __json_test_identity(value) return value end
function __json_test_empty_array() return json_array({}) end
function __json_test_make(a, b)
  return { sum = a + b, items = json_array({ a, b }), label = "ok", flag = true }
end
function __json_test_sum_counts(items)
  local total = 0
  for _, item in ipairs(items) do
    total = total + (item.count or 1)
  end
  return total
end
`

const createHost = async (): Promise<LuaLogicHost> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  const host = await createLuaLogicHost({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
  host.runModule(TEST_MODULE, '@json-test.lua')

  return host
}

describe('luaLogicHost callJson (real wasm)', () => {
  let host: LuaLogicHost | undefined

  afterEach(() => {
    host?.close()
    host = undefined
  })

  it('round-trips primitives, objects, arrays, nesting, empty arrays, unicode', async () => {
    host = await createHost()

    const samples: unknown[] = [
      0,
      1,
      -7,
      3.5,
      0.085,
      8.35,
      0.04 + 7 * 0.015,
      1 / 3,
      'hi',
      '한글 라벨',
      true,
      false,
      { a: 1, b: 'x', c: true },
      [1, 2, 3],
      [],
      [{ id: 'iron-sword', count: 2 }, { id: 'potion' }],
      { nested: { arr: [1, { k: 2 }], name: '검' }, empty: [] },
      // 배열의 null 구멍(인벤토리 slots 모양) + 객체의 null 값
      [1, null, 2],
      { a: null, b: 'x', c: 3 },
      [{ id: 'a', quantity: 1 }, null, null, { id: 'b', quantity: 5 }]
    ]

    for (const value of samples) {
      expect(host.callJson('__json_test_identity', value)).toEqual(value)
    }
  })

  it('builds and returns structured values', async () => {
    host = await createHost()

    expect(host.callJson('__json_test_empty_array')).toEqual([])
    expect(host.callJson('__json_test_make', 3, 4)).toEqual({
      sum: 7,
      items: [3, 4],
      label: 'ok',
      flag: true
    })
    expect(
      host.callJson('__json_test_sum_counts', [
        { id: 'a', count: 2 },
        { id: 'b' },
        { id: 'c', count: 5 }
      ])
    ).toBe(8)
  })
})
