import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import type { CharacterMoveDirection } from '../characterState'
import {
  getPlayerRollDirectionVector,
  normalizePlayerRollVector,
  getPlayerRollProgress,
  getPlayerRollDistanceTiles,
  getPlayerRollVisualState,
  type PlayerRollVector,
  type PlayerRollVisualState
} from '../playerRoll'
import { createPlayerRollLua, type PlayerRollLua } from './playerRollLua'

// playerRoll의 순수 구르기 수학(벡터/객체 입출력 → callJson)이 TS와 동등한지 실제 WASM으로 검증.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

// 방향: 4개 + (TS 타입에 없지만 default 분기를 검증하기 위한) 미지의 값.
const DIRECTIONS: CharacterMoveDirection[] = ['up', 'down', 'left', 'right']

// 정규화 입력: 0 벡터, 축 단위, 대각선, 음수, 비단위 등.
const VECTORS: PlayerRollVector[] = [
  { x: 0, y: 0 },
  { x: 0, y: -1 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: -1, y: -1 },
  { x: 3, y: 4 },
  { x: -3, y: 4 },
  { x: -0.5, y: 0 },
  { x: 0, y: 0.25 }
]

// 진행도 경계값: 음수, 0, 중간, 1, 1 초과.
const PROGRESS_VALUES = [-1, -0.001, 0, 0.0001, 0.25, 0.5, 0.7531, 1, 1.0001, 5]

// offsetY는 TS에서 -Math.abs(lift)*4 → lift=0(progress 0/1)일 때 -0을 만든다. JSON은 -0을
// 표현할 수 없어 Lua 왕복 결과는 +0이 된다(-0 === +0, 렌더 동일). 부호 비트만 다르므로 양쪽
// 비교 전에 -0을 +0으로 정규화한다(나머지 값/크기는 그대로 정확 비교).
const normalizeVisualState = (
  state: PlayerRollVisualState
): PlayerRollVisualState => ({
  offsetX: state.offsetX + 0,
  offsetY: state.offsetY + 0,
  rotation: state.rotation + 0,
  scaleX: state.scaleX + 0,
  scaleY: state.scaleY + 0
})

describe('playerRollLua (real wasm)', () => {
  let lua: PlayerRollLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerRoll across all converted functions', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerRollLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // getPlayerRollDirectionVector: 각 방향 + default(미지의 값) 분기.
    for (const direction of DIRECTIONS) {
      expect(lua.getPlayerRollDirectionVector(direction)).toEqual(
        getPlayerRollDirectionVector(direction)
      )
    }
    const unknownDirection = 'diagonal' as CharacterMoveDirection
    expect(lua.getPlayerRollDirectionVector(unknownDirection)).toEqual(
      getPlayerRollDirectionVector(unknownDirection)
    )

    // normalizePlayerRollVector: 0 벡터(undefined) + 단위/비단위/음수.
    for (const vector of VECTORS) {
      expect(lua.normalizePlayerRollVector(vector)).toEqual(
        normalizePlayerRollVector(vector)
      )
    }

    // getPlayerRollProgress: 경계 + 일반 경과시간.
    const startCases = [
      { nowMilliseconds: 0, startedAtMilliseconds: 0 },
      { nowMilliseconds: 1000, startedAtMilliseconds: 1000 },
      { nowMilliseconds: 1130, startedAtMilliseconds: 1000 },
      { nowMilliseconds: 1260, startedAtMilliseconds: 1000 },
      { nowMilliseconds: 1500, startedAtMilliseconds: 1000 },
      { nowMilliseconds: 900, startedAtMilliseconds: 1000 },
      { nowMilliseconds: 1037, startedAtMilliseconds: 1000 }
    ]
    for (const input of startCases) {
      expect(lua.getPlayerRollProgress(input)).toBe(getPlayerRollProgress(input))
    }

    // getPlayerRollDistanceTiles: 진행도 경계 포함.
    for (const progress of PROGRESS_VALUES) {
      expect(lua.getPlayerRollDistanceTiles(progress)).toBe(
        getPlayerRollDistanceTiles(progress)
      )
    }

    // getPlayerRollVisualState: 방향 부호 분기(음수 x / x=0 & 음수 y / 양수) × 진행도 경계.
    for (const vector of VECTORS) {
      for (const progress of PROGRESS_VALUES) {
        expect(
          normalizeVisualState(lua.getPlayerRollVisualState({ vector, progress }))
        ).toEqual(
          normalizeVisualState(getPlayerRollVisualState({ vector, progress }))
        )
      }
    }
  })
})
