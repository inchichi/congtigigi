import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  getPlayerExperienceToNextLevel,
  grantPlayerExperience
} from '../playerExperience'
import {
  createInitialPlayerProfile,
  PLAYER_MAX_LEVEL,
  type PlayerProfile
} from '../playerProfile'
import {
  createPlayerExperienceLua,
  type PlayerExperienceLua
} from './playerExperienceLua'

// 실제 Lua 5.3.6 WASM VM을 node에서 로드해(디스크의 mjs/wasm 주입), Lua 구현이 TS 기준
// 구현과 동일한지 검증한다. (컨트롤러 브리지 스펙과 같은 로더 패턴.)
// grantPlayerExperience 는 progression_grant_level_up_rewards 위임을 포함하므로 standalone 래퍼가
// player-progression 의존성까지 로드하는 경로도 함께 검증된다.
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

const withOverrides = (overrides: Partial<PlayerProfile>): PlayerProfile => ({
  ...createInitialPlayerProfile(),
  ...overrides
})

const LEVELS = [1, 2, 3, 10, 49, 50, 51, 100, 0, -2, 3.6, PLAYER_MAX_LEVEL]

const PROFILES: PlayerProfile[] = [
  createInitialPlayerProfile(),
  withOverrides({ level: 1, experience: { current: 0 } }),
  withOverrides({ level: 1, experience: { current: 139 } }),
  withOverrides({ level: 1, experience: { current: 100 }, statPoints: 2 }),
  withOverrides({
    level: 5,
    experience: { current: 50 },
    hp: { current: 30, max: 40 }
  }),
  withOverrides({
    level: 30,
    experience: { current: 200 },
    statPoints: 3,
    hp: { current: 100, max: 120 }
  }),
  withOverrides({ level: 99, experience: { current: 0 } }),
  withOverrides({ level: 99, experience: { current: 5000 } }),
  withOverrides({ level: 100, experience: { current: 0 } }),
  withOverrides({ level: 100, experience: { current: 999 } })
]

// 0(보상 없음), 음수/소수(floor·clamp), 한 레벨 직전/직후, 여러 레벨, 캡 초과 점프
const EXPERIENCE_INPUTS = [0, -10, 1, 0.9, 139, 140, 141, 300, 1000, 100000, 9.7]

describe('playerExperienceLua (real wasm bridge)', () => {
  let experience: PlayerExperienceLua | undefined

  afterEach(() => {
    experience?.close()
    experience = undefined
  })

  it('matches the TS getPlayerExperienceToNextLevel across every level', async () => {
    experience = await createExperience()

    for (const level of LEVELS) {
      expect(experience.getPlayerExperienceToNextLevel(level)).toBe(
        getPlayerExperienceToNextLevel(level)
      )
    }
  })

  it('matches the TS grantPlayerExperience across profiles + experience inputs', async () => {
    experience = await createExperience()

    for (const profile of PROFILES) {
      for (const gained of EXPERIENCE_INPUTS) {
        expect(experience.grantPlayerExperience(profile, gained)).toEqual(
          grantPlayerExperience(profile, gained)
        )
      }
    }
  })
})
