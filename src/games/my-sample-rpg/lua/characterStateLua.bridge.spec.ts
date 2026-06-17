import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createIdleNpcCharacterController,
  createInitialPlayerCharacter,
  createKeyboardCharacterController,
  createLuaCharacterController,
  createNpcCharacter,
  getCharacterActionFromKey,
  getCharacterControllerIntent,
  getCharacterMoveDirectionFromKey,
  moveCharacterState,
  type CharacterAction,
  type CharacterControllerIntent,
  type CharacterMoveDirection,
  type CharacterState
} from '../characterState'
import {
  createCharacterStateLua,
  type CharacterStateLua
} from './characterStateLua'

// characterState 순수 로직(컨트롤러 kind 디스패치·이동 의도·이동 적용)을 callJson 으로 변환한 결과가
// TS 와 동등한지 실제 WASM 으로 검증한다. 미일치/없음(undefined)은 json_null 왕복.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

// 누른 방향 부분집합들(빈/단일/대각선/3중/4중) — 정규화 벡터의 다양한 경로 점검.
const PRESSED_DIRECTION_SETS: CharacterMoveDirection[][] = [
  [],
  ['up'],
  ['down'],
  ['left'],
  ['right'],
  ['up', 'left'],
  ['up', 'right'],
  ['down', 'left'],
  ['down', 'right'],
  ['up', 'down'],
  ['left', 'right'],
  ['up', 'down', 'left'],
  ['up', 'down', 'left', 'right']
]

const TRIGGERED_ACTION_SETS: CharacterAction[][] = [
  [],
  ['interact'],
  ['attack'],
  ['smash'],
  ['interact', 'attack'],
  ['attack', 'smash', 'interact']
]

const DELTA_MILLISECONDS_CASES = [0, 16, 16.6667, 33.333, 100, 1000]

// 이동 의도의 movement.x/y 는 TS 에서 (음수 방향 성분) * (distance=0) → -0 을 만들 수 있다.
// JSON 은 -0 을 표현하지 못해 Lua 왕복 결과는 +0 이 된다(-0 === +0, 좌표 동일). 부호 비트만
// 다르므로 비교 전 -0 을 +0 으로 정규화한다(나머지 값/크기는 그대로 정확 비교). playerRoll 선례.
const normalizeIntent = (
  intent: CharacterControllerIntent | undefined
): CharacterControllerIntent | undefined => {
  if (intent === undefined) {
    return undefined
  }
  if (intent.movement === undefined) {
    return intent
  }
  return {
    ...intent,
    movement: { x: intent.movement.x + 0, y: intent.movement.y + 0 }
  }
}

describe('characterStateLua (real wasm)', () => {
  let lua: CharacterStateLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS characterState across all deterministic exports', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createCharacterStateLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // ── createKeyboardCharacterController ── 기본 + 명시 속도
    expect(lua.createKeyboardCharacterController()).toEqual(
      createKeyboardCharacterController()
    )
    for (const moveSpeedTilesPerSecond of [0, 1, 4, 8, 12.5]) {
      expect(
        lua.createKeyboardCharacterController({ moveSpeedTilesPerSecond })
      ).toEqual(createKeyboardCharacterController({ moveSpeedTilesPerSecond }))
    }

    // ── createIdleNpcCharacterController ──
    // 기본(대사 없음) / 대사 있음 / 빈 대사·공백만(필터링되어 미포함) / 메시지 시간 포함
    expect(lua.createIdleNpcCharacterController()).toEqual(
      createIdleNpcCharacterController()
    )
    const npcControllerCases: Array<{
      moveSpeedTilesPerSecond?: number
      dialogueLines?: string[]
      messageDurationMilliseconds?: number
    }> = [
      {},
      { moveSpeedTilesPerSecond: 3 },
      { dialogueLines: [] },
      { dialogueLines: ['   ', ''] },
      { dialogueLines: ['  안녕  ', '', '  두번째'] },
      { dialogueLines: ['한 줄'] },
      { messageDurationMilliseconds: 0 },
      { messageDurationMilliseconds: 2500 },
      {
        moveSpeedTilesPerSecond: 6,
        dialogueLines: ['a', '  ', 'b'],
        messageDurationMilliseconds: 1200
      }
    ]
    for (const npcCase of npcControllerCases) {
      expect(lua.createIdleNpcCharacterController(npcCase)).toEqual(
        createIdleNpcCharacterController(npcCase)
      )
    }

    // ── createLuaCharacterController ── config 기본 + 명시, 기본/명시 속도
    const luaControllerCases: Array<{
      scriptId: string
      radiusInTiles: number
      config?: Record<
        string,
        boolean | number | string | boolean[] | number[] | string[]
      >
      moveSpeedTilesPerSecond?: number
    }> = [
      { scriptId: 'guard', radiusInTiles: 3 },
      { scriptId: 'patrol', radiusInTiles: 5.5, moveSpeedTilesPerSecond: 2 },
      {
        scriptId: 'cfg',
        radiusInTiles: 1,
        config: {
          aggressive: true,
          speed: 4,
          name: 'orc',
          waypoints: [1, 2, 3],
          tags: ['a', 'b'],
          flags: [true, false]
        },
        moveSpeedTilesPerSecond: 7
      }
    ]
    for (const luaCase of luaControllerCases) {
      expect(lua.createLuaCharacterController(luaCase)).toEqual(
        createLuaCharacterController(luaCase)
      )
    }

    // ── createInitialPlayerCharacter ── 짝수/홀수 맵 크기(floor 중앙)
    for (const [mapWidth, mapHeight] of [
      [10, 10],
      [11, 7],
      [1, 1],
      [40, 25],
      [0, 0]
    ]) {
      expect(
        lua.createInitialPlayerCharacter({ mapWidth, mapHeight })
      ).toEqual(createInitialPlayerCharacter({ mapWidth, mapHeight }))
    }

    // ── createNpcCharacter ── 최소 입력(기본값) / level·displayText 포함 / facing·blocks·controller 명시
    const npcCharacterCases: Parameters<typeof createNpcCharacter>[0][] = [
      {
        id: 'npc-1',
        appearanceType: 'character_villager',
        position: { x: 3, y: 4 },
        collisionSize: { width: 1, height: 1 }
      },
      {
        id: 'npc-2',
        appearanceType: 'character_merchant',
        level: 5,
        displayText: '상점 주인',
        position: { x: 7, y: 2 },
        facing: 'left',
        collisionSize: { width: 2, height: 1 },
        blocksMovement: false
      },
      {
        id: 'npc-3',
        appearanceType: 'character_guard',
        level: 0,
        displayText: '',
        position: { x: 0, y: 0 },
        collisionSize: { width: 1, height: 2 },
        controller: createIdleNpcCharacterController({
          dialogueLines: ['안녕하세요'],
          messageDurationMilliseconds: 2000
        })
      },
      {
        id: 'npc-4',
        appearanceType: 'character_lua',
        position: { x: 9, y: 9 },
        collisionSize: { width: 1, height: 1 },
        facing: 'up',
        controller: createLuaCharacterController({
          scriptId: 'wander',
          radiusInTiles: 4
        })
      }
    ]
    for (const npcCharacterCase of npcCharacterCases) {
      expect(lua.createNpcCharacter(npcCharacterCase)).toEqual(
        createNpcCharacter(npcCharacterCase)
      )
    }

    // ── getCharacterMoveDirectionFromKey ── 모든 방향키 + 미일치
    for (const key of [
      'ArrowUp',
      'ArrowDown',
      'ArrowLeft',
      'ArrowRight',
      'Enter',
      ' ',
      'a',
      '',
      'arrowup'
    ]) {
      expect(lua.getCharacterMoveDirectionFromKey(key)).toBe(
        getCharacterMoveDirectionFromKey(key)
      )
    }

    // ── getCharacterActionFromKey ── 모든 액션키 + 미일치
    for (const key of [
      'Enter',
      ' ',
      'Space',
      'Spacebar',
      'ArrowUp',
      'a',
      '',
      'enter'
    ]) {
      expect(lua.getCharacterActionFromKey(key)).toBe(
        getCharacterActionFromKey(key)
      )
    }

    // ── getCharacterControllerIntent ── keyboard: 누른 방향 × 액션 × delta 조합
    const keyboardCharacter: CharacterState = createInitialPlayerCharacter({
      mapWidth: 20,
      mapHeight: 20
    })
    const slowKeyboardCharacter: CharacterState = {
      ...keyboardCharacter,
      controller: createKeyboardCharacterController({
        moveSpeedTilesPerSecond: 3
      })
    }
    for (const character of [keyboardCharacter, slowKeyboardCharacter]) {
      for (const pressed of PRESSED_DIRECTION_SETS) {
        for (const triggered of TRIGGERED_ACTION_SETS) {
          for (const deltaMilliseconds of DELTA_MILLISECONDS_CASES) {
            const args = {
              character,
              deltaMilliseconds,
              pressedDirections: new Set<CharacterMoveDirection>(pressed),
              triggeredActions: new Set<CharacterAction>(triggered)
            }
            expect(
              normalizeIntent(lua.getCharacterControllerIntent(args))
            ).toEqual(normalizeIntent(getCharacterControllerIntent(args)))
          }
        }
      }
    }

    // keyboard: pressedDirections/triggeredActions 생략(기본 빈 Set) → undefined intent
    expect(
      lua.getCharacterControllerIntent({
        character: keyboardCharacter,
        deltaMilliseconds: 16
      })
    ).toEqual(
      getCharacterControllerIntent({
        character: keyboardCharacter,
        deltaMilliseconds: 16
      })
    )

    // npc 컨트롤러: idle → 항상 movement 없음 → undefined
    const npcCharacter = createNpcCharacter({
      id: 'npc-intent',
      appearanceType: 'character_villager',
      position: { x: 2, y: 2 },
      collisionSize: { width: 1, height: 1 }
    })
    for (const deltaMilliseconds of DELTA_MILLISECONDS_CASES) {
      const args = {
        character: npcCharacter,
        deltaMilliseconds,
        pressedDirections: new Set<CharacterMoveDirection>(['up', 'right']),
        triggeredActions: new Set<CharacterAction>(['interact'])
      }
      expect(lua.getCharacterControllerIntent(args)).toEqual(
        getCharacterControllerIntent(args)
      )
    }

    // lua 컨트롤러: 항상 undefined
    const luaCharacter: CharacterState = {
      ...keyboardCharacter,
      controller: createLuaCharacterController({
        scriptId: 'guard',
        radiusInTiles: 3
      })
    }
    expect(
      lua.getCharacterControllerIntent({
        character: luaCharacter,
        deltaMilliseconds: 16,
        pressedDirections: new Set<CharacterMoveDirection>(['up']),
        triggeredActions: new Set<CharacterAction>(['interact'])
      })
    ).toBe(
      getCharacterControllerIntent({
        character: luaCharacter,
        deltaMilliseconds: 16,
        pressedDirections: new Set<CharacterMoveDirection>(['up']),
        triggeredActions: new Set<CharacterAction>(['interact'])
      })
    )

    // ── moveCharacterState ── 이동/방향 전환/클램프(경계)/변화 없음
    const moveBase: CharacterState = createInitialPlayerCharacter({
      mapWidth: 10,
      mapHeight: 10
    })
    const wideCharacter: CharacterState = {
      ...moveBase,
      position: { x: 1, y: 1 },
      collisionSize: { width: 2, height: 3 }
    }
    const moveDeltas = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: -1, y: 0 },
      { x: 0, y: 1 },
      { x: 0, y: -1 },
      { x: 1, y: 1 },
      { x: -1, y: -1 },
      { x: 2, y: -3 },
      { x: -3, y: 2 },
      { x: 0.5, y: 0.25 },
      { x: -100, y: 0 },
      { x: 0, y: 100 },
      { x: 100, y: 100 }
    ]
    for (const character of [moveBase, wideCharacter]) {
      for (const delta of moveDeltas) {
        for (const [mapWidth, mapHeight] of [
          [10, 10],
          [5, 5],
          [1, 1]
        ]) {
          const args = { character, delta, mapWidth, mapHeight }
          expect(lua.moveCharacterState(args)).toEqual(
            moveCharacterState(args)
          )
        }
      }
    }

    // moveCharacterState: 변화 없음(원점·동일 facing) → 동일 구조 반환
    const noopArgs = {
      character: moveBase,
      delta: { x: 0, y: 0 },
      mapWidth: 10,
      mapHeight: 10
    }
    expect(lua.moveCharacterState(noopArgs)).toEqual(
      moveCharacterState(noopArgs)
    )

    // moveCharacterState: npc(idle) 캐릭터 이동 — controller 보존 + 위치/방향 갱신
    const npcMoveArgs = {
      character: createNpcCharacter({
        id: 'npc-move',
        appearanceType: 'character_villager',
        position: { x: 2, y: 2 },
        collisionSize: { width: 1, height: 1 },
        controller: createIdleNpcCharacterController({
          dialogueLines: ['지나가지 마시오']
        })
      }),
      delta: { x: 1, y: 0 },
      mapWidth: 10,
      mapHeight: 10
    }
    expect(lua.moveCharacterState(npcMoveArgs)).toEqual(
      moveCharacterState(npcMoveArgs)
    )
  })
})
