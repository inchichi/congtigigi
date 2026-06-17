import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import { rollMonsterEquipmentDrop } from '../monsterEquipmentDrops'
import {
  createMonsterEquipmentDropsLua,
  type MonsterEquipmentDropsLua
} from './monsterEquipmentDropsLua'

// 실제 Lua 5.3.6 WASM VM을 node에서 로드해(디스크의 mjs/wasm 주입), Lua 구현이 TS 기준
// 구현과 모든 시퀀스에서 동일한 드롭(또는 undefined)을 내는지 검증한다. RNG는 고정 배열에서
// 순차적으로 값을 내는 클로저로 결정론적으로 만든다(확률 게이트 + 인덱스, 같은 순서로 소비).
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createDrops = async (): Promise<MonsterEquipmentDropsLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createMonsterEquipmentDropsLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

// 각 시퀀스는 [확률 게이트 값, 인덱스 값] 순으로 소비된다.
// 확률 실패(>=0.9), 확률 통과 + 낮은/중간/높은 인덱스, 경계(0.999..)를 두루 다룬다.
const SEQUENCES: number[][] = [
  [0.9, 0.0], // 확률 실패(정확히 0.9)
  [0.95, 0.5], // 확률 실패(높음)
  [1.0, 0.999], // 확률 실패(최대)
  [0.0, 0.0], // 통과 + 인덱스 0
  [0.5, 0.0], // 통과 + 인덱스 0
  [0.89, 0.0833], // 통과 + 낮은 인덱스
  [0.1, 0.25], // 통과 + 중간 인덱스
  [0.4, 0.5], // 통과 + 중간 인덱스
  [0.0, 0.75], // 통과 + 높은 인덱스
  [0.5, 0.999], // 통과 + 높은 인덱스(경계)
  [0.0, 1.0], // 통과 + 인덱스 클램프(=count-1)
  [0.899999, 0.9999999], // 통과 + 마지막 인덱스 경계
  [0.123, 0.456], // 통과 + 임의 중간
  [0.7, 0.333333] // 통과 + 임의 중간
]

const createSequenceRng = (values: number[]): (() => number) => {
  let cursor = 0

  return (): number => {
    const value = values[cursor] ?? 0
    cursor += 1

    return value
  }
}

describe('monsterEquipmentDropsLua (real wasm bridge)', () => {
  let drops: MonsterEquipmentDropsLua | undefined

  afterEach(() => {
    drops?.close()
    drops = undefined
  })

  it('matches the TS monsterEquipmentDrops across every sequence', async () => {
    drops = await createDrops()

    for (const sequence of SEQUENCES) {
      const luaResult = drops.rollMonsterEquipmentDrop(
        createSequenceRng(sequence)
      )
      const referenceResult = rollMonsterEquipmentDrop(
        createSequenceRng(sequence)
      )

      expect(luaResult).toEqual(referenceResult)
    }
  })
})
