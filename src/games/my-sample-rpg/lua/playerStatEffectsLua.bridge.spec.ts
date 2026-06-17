import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import { createInitialPlayerProfile, type PlayerProfile } from '../playerProfile'
import {
  getPlayerPhysicalAttackPower,
  getPlayerMovementSpeedTilesPerSecond,
  getPlayerEvadeChance,
  shouldPlayerEvadeDamage
} from '../playerStatEffects'
import {
  createPlayerStatEffectsLua,
  type PlayerStatEffectsLua
} from './playerStatEffectsLua'

// playerStatEffects의 파생 스탯(객체 입력 → callJson)이 TS와 동등한지 실제 WASM으로 검증.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const withStats = (
  strength: number,
  agility: number,
  luck: number
): Pick<PlayerProfile, 'stats'> => ({
  stats: {
    ...createInitialPlayerProfile().stats,
    strength,
    agility,
    luck
  }
})

const STAT_CASES = [
  withStats(1, 1, 0),
  withStats(5, 4, 2),
  withStats(10, 8, 7),
  withStats(3, 12, 20),
  withStats(0.5, 4, 3),
  withStats(99, 99, 99)
]

describe('playerStatEffectsLua (real wasm)', () => {
  let lua: PlayerStatEffectsLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerStatEffects across stat blocks', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerStatEffectsLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    for (const profile of STAT_CASES) {
      expect(lua.getPlayerPhysicalAttackPower(profile)).toBe(
        getPlayerPhysicalAttackPower(profile)
      )
      expect(lua.getPlayerMovementSpeedTilesPerSecond(profile)).toBe(
        getPlayerMovementSpeedTilesPerSecond(profile)
      )
      expect(lua.getPlayerEvadeChance(profile)).toBe(getPlayerEvadeChance(profile))

      for (const randomValue of [0, 0.03, 0.05, 0.1, 0.34, 0.5, 0.99]) {
        expect(lua.shouldPlayerEvadeDamage(profile, randomValue)).toBe(
          shouldPlayerEvadeDamage(profile, randomValue)
        )
      }
    }
  })
})
