import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  applyMonsterDamage,
  createMonsterCombatState,
  isMonsterDefeated,
  type MonsterCombatState
} from '../monsterCombat'
import { createMonsterCombatLua, type MonsterCombatLua } from './monsterCombatLua'

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

const createCombat = async (): Promise<MonsterCombatLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createMonsterCombatLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

const LEVELS = [1, 2, 3, 5, 10, 50, 0, -1, 3.5]
const HP_MULTIPLIERS = [1, 2, 3, 0, -1, 1.5]
const DAMAGE_MULTIPLIERS = [1, 2, 4, 0, -1, 2.5]
const DAMAGES = [0, 1, 5, 9999, 3.7, -2]
const DEFEAT_HPS = [10, 1, 0, -3]

const expectStateEqual = (
  actual: MonsterCombatState,
  expected: MonsterCombatState
): void => {
  expect(actual.maxHp).toBe(expected.maxHp)
  expect(actual.currentHp).toBe(expected.currentHp)
  expect(actual.contactDamage).toBe(expected.contactDamage)
}

describe('monsterCombatLua (real wasm bridge)', () => {
  let combat: MonsterCombatLua | undefined

  afterEach(() => {
    combat?.close()
    combat = undefined
  })

  it('matches the TS createMonsterCombatState across levels and multipliers', async () => {
    combat = await createCombat()

    for (const level of LEVELS) {
      for (const hpMultiplier of HP_MULTIPLIERS) {
        for (const damageMultiplier of DAMAGE_MULTIPLIERS) {
          const options = { hpMultiplier, damageMultiplier }

          expectStateEqual(
            combat.createMonsterCombatState(level, options),
            createMonsterCombatState(level, options)
          )
        }
      }
    }
  })

  it('matches the TS createMonsterCombatState with defaults', async () => {
    combat = await createCombat()

    expectStateEqual(
      combat.createMonsterCombatState(),
      createMonsterCombatState()
    )
    expectStateEqual(
      combat.createMonsterCombatState(5),
      createMonsterCombatState(5)
    )
  })

  it('matches the TS applyMonsterDamage across damages and states', async () => {
    combat = await createCombat()

    for (const level of LEVELS) {
      for (const damage of DAMAGES) {
        const luaState = combat.createMonsterCombatState(level)
        const tsState = createMonsterCombatState(level)

        expectStateEqual(
          combat.applyMonsterDamage(luaState, damage),
          applyMonsterDamage(tsState, damage)
        )
      }
    }
  })

  it('matches the TS isMonsterDefeated across currentHp values', async () => {
    combat = await createCombat()

    for (const currentHp of DEFEAT_HPS) {
      const state: MonsterCombatState = {
        maxHp: 10,
        currentHp,
        contactDamage: 1
      }

      expect(combat.isMonsterDefeated(state)).toBe(isMonsterDefeated(state))
    }
  })
})
