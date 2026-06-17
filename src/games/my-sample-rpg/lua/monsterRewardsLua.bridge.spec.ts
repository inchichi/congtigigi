import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  getMonsterExperienceDropAmount,
  getMonsterGoldDropAmount,
  getMonsterSkillPointDropAmount
} from '../monsterRewards'
import { createMonsterRewardsLua, type MonsterRewardsLua } from './monsterRewardsLua'

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

const createRewards = async (): Promise<MonsterRewardsLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createMonsterRewardsLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

const LEVELS = [-5, -1, 0, 1, 2, 3, 5, 9, 10, 17, 50, 99, 250, 1.9, 3.2]

describe('monsterRewardsLua (real wasm bridge)', () => {
  let rewards: MonsterRewardsLua | undefined

  afterEach(() => {
    rewards?.close()
    rewards = undefined
  })

  it('matches the TS monsterRewards across every level', async () => {
    rewards = await createRewards()

    for (const level of LEVELS) {
      expect(rewards.getMonsterGoldDropAmount(level)).toBe(
        getMonsterGoldDropAmount(level)
      )
      expect(rewards.getMonsterExperienceDropAmount(level)).toBe(
        getMonsterExperienceDropAmount(level)
      )
      expect(rewards.getMonsterSkillPointDropAmount(level)).toBe(
        getMonsterSkillPointDropAmount(level)
      )
    }
  })
})
