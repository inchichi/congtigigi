import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import { getMonsterDisplayName } from '../monsterDisplayName'
import {
  createMonsterDisplayNameLua,
  type MonsterDisplayNameLua
} from './monsterDisplayNameLua'

// 실제 Lua 5.3.6 WASM VM을 node에서 로드해(디스크의 mjs/wasm 주입), Lua 구현이 TS 기준
// 구현과 모든 입력에서 동일한지 검증한다. (컨트롤러 브리지 스펙과 같은 로더 패턴.)
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createDisplayName = async (): Promise<MonsterDisplayNameLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createMonsterDisplayNameLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

const CASES: ReadonlyArray<{ id: string; displayText?: string }> = [
  { id: 'slime-1' },
  { id: 'pig-2', displayText: 'kkulkkul' },
  { id: 'boss' },
  { id: '-leading' },
  { id: 'a-b-c' },
  { id: 'x', displayText: '' },
  { id: 'gold-dragon-3' },
  { id: '' },
  { id: '-' },
  { id: '--' },
  { id: 'a-' },
  { id: '-a' },
  { id: 'name', displayText: 'Custom Name' },
  { id: 'slime-1', displayText: 'Slime King' },
  { id: 'a-b-c-d-e' },
  { id: 'trailing-dash-' },
  { id: 'no-dash-but-text', displayText: 'Override' },
  { id: 'single' }
]

describe('monsterDisplayNameLua (real wasm bridge)', () => {
  let displayName: MonsterDisplayNameLua | undefined

  afterEach(() => {
    displayName?.close()
    displayName = undefined
  })

  it('matches the TS monsterDisplayName across every case', async () => {
    displayName = await createDisplayName()

    for (const args of CASES) {
      expect(displayName.getMonsterDisplayName(args)).toBe(
        getMonsterDisplayName(args)
      )
    }
  })
})
