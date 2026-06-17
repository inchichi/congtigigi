import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialPlayerControlBindings,
  getPlayerControlActionFromCode,
  getPlayerControlBindingDefinitionById,
  getPlayerControlBindingDefinitions,
  getPlayerControlBindingDisplayText,
  getPlayerControlMovementDirectionFromCode,
  getPlayerControlQuickslotIndexFromCode,
  isPlayerControlCaptureModifierKey,
  isPlayerControlPauseKey,
  normalizeStoredPlayerControlBindings,
  setPlayerControlBinding,
  type PlayerControlBindingId,
  type PlayerControlBindings
} from '../playerControls'
import {
  createPlayerControlsLua,
  type PlayerControlsLua
} from './playerControlsLua'

// playerControls(키 바인딩 매핑/표시/정규화) Lua 포팅이 TS 원본과 동등한지 실제 WASM으로 검증한다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

// 원본 모듈은 findPlayerControlBindingIdByCode 를 export 하지 않으므로,
// 동일 로직을 테스트 안에서 재현해 양쪽 비교의 기준값으로 쓴다.
const referenceFindBindingIdByCode = (
  bindings: PlayerControlBindings,
  code: string
): PlayerControlBindingId | undefined => {
  for (const [bindingId, bindingCode] of Object.entries(bindings)) {
    if (bindingCode === code) {
      return bindingId as PlayerControlBindingId
    }
  }

  return undefined
}

const ALL_BINDING_IDS: PlayerControlBindingId[] = [
  'move-up',
  'move-down',
  'move-left',
  'move-right',
  'interact-enter',
  'interact-space',
  'attack',
  'inventory',
  'stat',
  'equipment',
  'skill',
  'quest',
  'map',
  'portal',
  'pause',
  'quickslot-1',
  'quickslot-2',
  'quickslot-3',
  'quickslot-4',
  'quickslot-5',
  'quickslot-6'
]

// 표시 텍스트의 모든 분기를 덮는 코드들: 라벨맵 명중, Key/Digit/Numpad 접두, 그 외.
const DISPLAY_CODES = [
  'ArrowUp',
  'ArrowDown',
  'ArrowLeft',
  'ArrowRight',
  'Enter',
  'Space',
  'Escape',
  'KeyA',
  'KeyZ',
  'Key',
  'Digit1',
  'Digit0',
  'Digit',
  'Numpad5',
  'Numpad',
  'NumpadAdd',
  'F1',
  'Tab',
  '',
  'Backquote'
]

// 캡처 수정자 키 분기(8개 true) + 비수정자 키들(false).
const MODIFIER_CODES = [
  'ShiftLeft',
  'ShiftRight',
  'ControlLeft',
  'ControlRight',
  'AltLeft',
  'AltRight',
  'MetaLeft',
  'MetaRight',
  'KeyA',
  'Shift',
  'Control',
  '',
  'CapsLock'
]

describe('playerControlsLua (real wasm)', () => {
  let lua: PlayerControlsLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerControls across all deterministic exports', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerControlsLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // ── createInitialPlayerControlBindings ──
    const initial = createInitialPlayerControlBindings()
    expect(lua.createInitialPlayerControlBindings()).toEqual(initial)

    // ── getPlayerControlBindingDefinitions ──
    expect(lua.getPlayerControlBindingDefinitions()).toEqual(
      getPlayerControlBindingDefinitions()
    )

    // ── getPlayerControlBindingDefinitionById (모든 id) ──
    for (const id of ALL_BINDING_IDS) {
      expect(lua.getPlayerControlBindingDefinitionById(id)).toEqual(
        getPlayerControlBindingDefinitionById(id)
      )
    }

    // ── getPlayerControlBindingDisplayText (모든 분기) ──
    for (const code of DISPLAY_CODES) {
      expect(lua.getPlayerControlBindingDisplayText(code)).toBe(
        getPlayerControlBindingDisplayText(code)
      )
    }

    // ── isPlayerControlCaptureModifierKey (true/false 분기) ──
    for (const code of MODIFIER_CODES) {
      expect(lua.isPlayerControlCaptureModifierKey(code)).toBe(
        isPlayerControlCaptureModifierKey(code)
      )
    }

    // ── getPlayerControlMovementDirectionFromCode ──
    const movementCodes = [
      'ArrowUp',
      'ArrowDown',
      'ArrowLeft',
      'ArrowRight',
      'KeyA',
      'Enter',
      'Space',
      'Digit1',
      ''
    ]
    for (const code of movementCodes) {
      expect(lua.getPlayerControlMovementDirectionFromCode(initial, code)).toBe(
        getPlayerControlMovementDirectionFromCode(initial, code)
      )
    }

    // ── getPlayerControlActionFromCode ──
    const actionCodes = [
      'Enter',
      'Space',
      'KeyA',
      'ArrowUp',
      'KeyI',
      'Escape',
      ''
    ]
    for (const code of actionCodes) {
      expect(lua.getPlayerControlActionFromCode(initial, code)).toBe(
        getPlayerControlActionFromCode(initial, code)
      )
    }

    // ── getPlayerControlQuickslotIndexFromCode (각 슬롯 + 미존재) ──
    const quickslotCodes = [
      'Digit1',
      'Digit2',
      'Digit3',
      'Digit4',
      'Digit5',
      'Digit6',
      'Digit7',
      'KeyA',
      ''
    ]
    for (const code of quickslotCodes) {
      expect(lua.getPlayerControlQuickslotIndexFromCode(initial, code)).toBe(
        getPlayerControlQuickslotIndexFromCode(initial, code)
      )
    }

    // ── isPlayerControlPauseKey ──
    for (const code of ['Escape', 'KeyA', 'Enter', '']) {
      expect(lua.isPlayerControlPauseKey(initial, code)).toBe(
        isPlayerControlPauseKey(initial, code)
      )
    }

    // ── findPlayerControlBindingIdByCode (명중 + 미존재) ──
    for (const code of [
      'ArrowUp',
      'KeyA',
      'Digit6',
      'Escape',
      'KeyZ',
      ''
    ]) {
      expect(lua.findPlayerControlBindingIdByCode(initial, code)).toBe(
        referenceFindBindingIdByCode(initial, code)
      )
    }

    // ── setPlayerControlBinding ──
    // 변화 없음(현재 코드 == next): 원본과 동일(동등).
    expect(
      lua.setPlayerControlBinding({
        bindings: initial,
        bindingId: 'attack',
        nextCode: 'KeyA'
      })
    ).toEqual(
      setPlayerControlBinding({
        bindings: initial,
        bindingId: 'attack',
        nextCode: 'KeyA'
      })
    )

    // 충돌 없는 새 코드: 단순 교체.
    expect(
      lua.setPlayerControlBinding({
        bindings: initial,
        bindingId: 'attack',
        nextCode: 'KeyQ'
      })
    ).toEqual(
      setPlayerControlBinding({
        bindings: initial,
        bindingId: 'attack',
        nextCode: 'KeyQ'
      })
    )

    // 충돌 있는 코드(다른 바인딩이 이미 사용): 스왑.
    expect(
      lua.setPlayerControlBinding({
        bindings: initial,
        bindingId: 'attack',
        nextCode: 'KeyI' // inventory 가 쓰는 코드 → 스왑
      })
    ).toEqual(
      setPlayerControlBinding({
        bindings: initial,
        bindingId: 'attack',
        nextCode: 'KeyI'
      })
    )

    // 자기 자신 코드로 재바인딩(conflict == bindingId 경로): 변화 없는 동등.
    expect(
      lua.setPlayerControlBinding({
        bindings: initial,
        bindingId: 'attack',
        nextCode: 'KeyA'
      })
    ).toEqual(
      setPlayerControlBinding({
        bindings: initial,
        bindingId: 'attack',
        nextCode: 'KeyA'
      })
    )

    // 누적 변경(스왑 후 또 스왑)도 동등 유지.
    const afterSwapLua = lua.setPlayerControlBinding({
      bindings: initial,
      bindingId: 'move-up',
      nextCode: 'KeyW'
    })
    const afterSwapTs = setPlayerControlBinding({
      bindings: initial,
      bindingId: 'move-up',
      nextCode: 'KeyW'
    })
    expect(afterSwapLua).toEqual(afterSwapTs)
    expect(
      lua.setPlayerControlBinding({
        bindings: afterSwapLua,
        bindingId: 'move-down',
        nextCode: 'KeyW' // move-up 이 쓰는 코드 → 스왑
      })
    ).toEqual(
      setPlayerControlBinding({
        bindings: afterSwapTs,
        bindingId: 'move-down',
        nextCode: 'KeyW'
      })
    )

    // ── normalizeStoredPlayerControlBindings ──
    // 비-객체 입력(초기값 반환 경로): null/undefined/문자열/숫자/불리언/배열.
    const nonObjectInputs: unknown[] = [
      null,
      undefined,
      'not-an-object',
      42,
      0,
      true,
      false,
      ['ArrowUp', 'ArrowDown']
    ]
    for (const input of nonObjectInputs) {
      expect(lua.normalizeStoredPlayerControlBindings(input)).toEqual(
        normalizeStoredPlayerControlBindings(input)
      )
    }

    // 빈 객체: 전부 default.
    expect(lua.normalizeStoredPlayerControlBindings({})).toEqual(
      normalizeStoredPlayerControlBindings({})
    )

    // 정상 저장본(초기값 그대로): 동일 유지.
    expect(lua.normalizeStoredPlayerControlBindings(initial)).toEqual(
      normalizeStoredPlayerControlBindings(initial)
    )

    // 일부 커스텀 코드 + 빈 문자열(=default) + 비문자열(=default).
    const partialStored = {
      'move-up': 'KeyW',
      'move-down': 'KeyS', // stat 의 default 와 충돌 → resolved 시 default 처리
      attack: '',
      inventory: 123 as unknown as string,
      'quickslot-1': 'KeyZ'
    }
    expect(lua.normalizeStoredPlayerControlBindings(partialStored)).toEqual(
      normalizeStoredPlayerControlBindings(partialStored)
    )

    // 중복 코드를 여러 바인딩에 지정(usedCodes 충돌 해소 경로).
    const dupStored = {
      'move-up': 'KeyX',
      'move-down': 'KeyX',
      'move-left': 'KeyX',
      attack: 'KeyX'
    }
    expect(lua.normalizeStoredPlayerControlBindings(dupStored)).toEqual(
      normalizeStoredPlayerControlBindings(dupStored)
    )

    // 알 수 없는 키가 섞인 객체(무시되어야 함).
    const extraKeysStored = {
      'move-up': 'KeyW',
      unknownKey: 'KeyP',
      somethingElse: 999
    }
    expect(lua.normalizeStoredPlayerControlBindings(extraKeysStored)).toEqual(
      normalizeStoredPlayerControlBindings(extraKeysStored)
    )

    // normalize 결과를 다시 set/find 에 흘려도 동등 유지(왕복 안정성).
    const normalizedLua =
      lua.normalizeStoredPlayerControlBindings(partialStored)
    const normalizedTs = normalizeStoredPlayerControlBindings(partialStored)
    expect(lua.findPlayerControlBindingIdByCode(normalizedLua, 'KeyW')).toBe(
      referenceFindBindingIdByCode(normalizedTs, 'KeyW')
    )
  })
})
