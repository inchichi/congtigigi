import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import { getPlayerExperienceToNextLevel } from '../playerExperience'
import { PLAYER_MAX_LEVEL } from '../playerProfile'
import {
  createPlayerExperienceLua,
  type PlayerExperienceLua
} from './playerExperienceLua'

// 실제 Lua 5.3.6 WASM VM을 node에서 로드해(디스크의 mjs/wasm 주입), Lua 구현이 TS 기준
// 구현과 모든 레벨에서 동일한지 검증한다. (컨트롤러 브리지 스펙과 같은 로더 패턴.)
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createExperience = async (): Promise<PlayerExperienceLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createPlayerExperienceLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

const LEVELS = [1, 2, 3, 10, 49, 50, 51, 100, 0, -2, 3.6, PLAYER_MAX_LEVEL]

describe('playerExperienceLua (real wasm bridge)', () => {
  let experience: PlayerExperienceLua | undefined

  afterEach(() => {
    experience?.close()
    experience = undefined
  })

  it('matches the TS playerExperience across every level', async () => {
    experience = await createExperience()

    for (const level of LEVELS) {
      expect(experience.getPlayerExperienceToNextLevel(level)).toBe(
        getPlayerExperienceToNextLevel(level)
      )
    }
  })
})
