import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import type { CharacterState } from '../characterState'
import { createMonsterPatrolState } from '../monsterPatrol'
import {
  createMonsterPatrolLua,
  type MonsterPatrolLua
} from './monsterPatrolLua'

// monsterPatrol.createMonsterPatrolState를 Lua로 변환한 결과가 TS 기준 구현과
// 모든 입력에서 동일한지 실제 Lua 5.3.6 WASM VM(디스크의 mjs/wasm 주입)으로 검증한다.
// (stepMonsterPatrol은 가변 Math.random 의존이라 변환 대상이 아니다 — 검증하지 않는다.)
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createPatrol = async (): Promise<MonsterPatrolLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createMonsterPatrolLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

const makeCharacter = (x: number, y: number): CharacterState => ({
  id: 'monster',
  appearanceType: 'slime',
  position: { x, y },
  facing: 'down',
  collisionSize: { width: 1, height: 1 },
  blocksMovement: true,
  controller: {
    kind: 'npc',
    behavior: 'idle',
    moveSpeedTilesPerSecond: 2
  }
})

// 정상 + 경계: 0, 음수, 정수, 소수, 큰 값.
const POSITIONS: Array<{ x: number; y: number }> = [
  { x: 0, y: 0 },
  { x: 5, y: 7 },
  { x: 12.5, y: 3.25 },
  { x: -4, y: -9 },
  { x: -2.75, y: 6.5 },
  { x: 1000000, y: 0.0001 },
  { x: 0.1, y: 0.2 },
  { x: 123.456789, y: 987.654321 }
]

describe('monsterPatrolLua (real wasm bridge)', () => {
  let patrol: MonsterPatrolLua | undefined

  afterEach(() => {
    patrol?.close()
    patrol = undefined
  })

  it('matches TS createMonsterPatrolState across positions', async () => {
    patrol = await createPatrol()

    for (const position of POSITIONS) {
      const character = makeCharacter(position.x, position.y)

      expect(patrol.createMonsterPatrolState(character)).toEqual(
        createMonsterPatrolState(character)
      )
    }
  })

  it('copies position by value (home and target are independent of input shape)', async () => {
    patrol = await createPatrol()

    const character = makeCharacter(8, 11)
    const luaState = patrol.createMonsterPatrolState(character)
    const tsState = createMonsterPatrolState(character)

    expect(luaState).toEqual(tsState)
    expect(luaState.pauseUntilMilliseconds).toBe(0)
    expect(luaState.homePosition).toEqual({ x: 8, y: 11 })
    expect(luaState.targetPosition).toEqual({ x: 8, y: 11 })
    // homePosition과 targetPosition은 서로 다른 객체여야 한다(별도 복사).
    expect(luaState.homePosition).not.toBe(luaState.targetPosition)
  })

  it('matches TS for a fractional origin character', async () => {
    patrol = await createPatrol()

    const character = makeCharacter(2.4000000000000004, -0.000003)

    expect(patrol.createMonsterPatrolState(character)).toEqual(
      createMonsterPatrolState(character)
    )
  })
})
