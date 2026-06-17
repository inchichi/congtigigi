// playerControls 순수 로직을 Lua로 실행하는 래퍼.
// PLAYER_CONTROL_BINDING_DEFINITIONS(순서 중요)와 코드→라벨 표시 맵은 TS 소유 —
// runModule 후 player_controls_init()으로 Lua에 주입한다(init-with-data 패턴).
// characterState 의존은 타입 전용(이동방향/액션 유니온)이라 런타임 의존이 없다.

import playerControlsSource from '../assets/lua/player-controls.lua?raw'

import type {
  CharacterAction,
  CharacterMoveDirection
} from '../characterState'
import {
  getPlayerControlBindingDefinitions,
  type PlayerControlBindingDefinition,
  type PlayerControlBindingId,
  type PlayerControlBindings
} from '../playerControls'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

// TS PLAYER_CONTROL_BINDING_LABEL_BY_CODE 와 동일(데이터 소유는 TS, init으로 Lua에 주입).
// 원본은 모듈 내부 const라 export 되지 않으므로 동일 값을 단일 출처로 여기 보관해 주입한다.
const PLAYER_CONTROL_BINDING_LABEL_BY_CODE: Record<string, string> = {
  ArrowUp: '↑',
  ArrowDown: '↓',
  ArrowLeft: '←',
  ArrowRight: '→',
  Enter: 'Enter',
  Space: 'Space',
  Escape: 'Esc'
}

export type PlayerControlsLua = {
  createInitialPlayerControlBindings: () => PlayerControlBindings
  normalizeStoredPlayerControlBindings: (
    nextBindings: unknown
  ) => PlayerControlBindings
  setPlayerControlBinding: (input: {
    bindings: PlayerControlBindings
    bindingId: PlayerControlBindingId
    nextCode: string
  }) => PlayerControlBindings
  getPlayerControlBindingDisplayText: (code: string) => string
  getPlayerControlMovementDirectionFromCode: (
    bindings: PlayerControlBindings,
    code: string
  ) => CharacterMoveDirection | undefined
  getPlayerControlActionFromCode: (
    bindings: PlayerControlBindings,
    code: string
  ) => CharacterAction | undefined
  getPlayerControlQuickslotIndexFromCode: (
    bindings: PlayerControlBindings,
    code: string
  ) => number | undefined
  isPlayerControlPauseKey: (
    bindings: PlayerControlBindings,
    code: string
  ) => boolean
  isPlayerControlCaptureModifierKey: (code: string) => boolean
  getPlayerControlBindingDefinitionById: (
    bindingId: PlayerControlBindingId
  ) => PlayerControlBindingDefinition
  getPlayerControlBindingDefinitions: () => readonly PlayerControlBindingDefinition[]
  findPlayerControlBindingIdByCode: (
    bindings: PlayerControlBindings,
    code: string
  ) => PlayerControlBindingId | undefined
  close: () => void
}

export const createPlayerControlsLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerControlsLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerControlsSource, '@player-controls.lua')

  // init-with-data: 정의 배열(순서 중요) + 코드→라벨 맵을 Lua에 주입.
  // 정의 배열의 단일 출처는 원본 모듈 — getPlayerControlBindingDefinitions()가 반환하는
  // const 배열을 그대로 넘겨 드리프트를 막는다(순서가 normalize/find 의미에 중요).
  host.callJson(
    'player_controls_init',
    getPlayerControlBindingDefinitions(),
    PLAYER_CONTROL_BINDING_LABEL_BY_CODE
  )

  return {
    createInitialPlayerControlBindings: (): PlayerControlBindings =>
      host.callJson<PlayerControlBindings>('player_controls_create_initial'),

    normalizeStoredPlayerControlBindings: (
      nextBindings: unknown
    ): PlayerControlBindings =>
      host.callJson<PlayerControlBindings>(
        'player_controls_normalize_stored',
        nextBindings
      ),

    setPlayerControlBinding: ({
      bindings,
      bindingId,
      nextCode
    }): PlayerControlBindings =>
      host.callJson<PlayerControlBindings>(
        'player_controls_set_binding',
        bindings,
        bindingId,
        nextCode
      ),

    getPlayerControlBindingDisplayText: (code: string): string =>
      host.callJson<string>('player_controls_get_display_text', code),

    getPlayerControlMovementDirectionFromCode: (
      bindings: PlayerControlBindings,
      code: string
    ): CharacterMoveDirection | undefined => {
      const result = host.callJson<CharacterMoveDirection | null>(
        'player_controls_get_movement_direction',
        bindings,
        code
      )

      return result === null ? undefined : result
    },

    getPlayerControlActionFromCode: (
      bindings: PlayerControlBindings,
      code: string
    ): CharacterAction | undefined => {
      const result = host.callJson<CharacterAction | null>(
        'player_controls_get_action',
        bindings,
        code
      )

      return result === null ? undefined : result
    },

    getPlayerControlQuickslotIndexFromCode: (
      bindings: PlayerControlBindings,
      code: string
    ): number | undefined => {
      const result = host.callJson<number | null>(
        'player_controls_get_quickslot_index',
        bindings,
        code
      )

      return result === null ? undefined : result
    },

    isPlayerControlPauseKey: (
      bindings: PlayerControlBindings,
      code: string
    ): boolean =>
      host.callJson<boolean>('player_controls_is_pause_key', bindings, code),

    isPlayerControlCaptureModifierKey: (code: string): boolean =>
      host.callJson<boolean>(
        'player_controls_is_capture_modifier_key',
        code
      ),

    getPlayerControlBindingDefinitionById: (
      bindingId: PlayerControlBindingId
    ): PlayerControlBindingDefinition =>
      host.callJson<PlayerControlBindingDefinition>(
        'player_controls_get_definition_by_id',
        bindingId
      ),

    getPlayerControlBindingDefinitions:
      (): readonly PlayerControlBindingDefinition[] =>
        host.callJson<PlayerControlBindingDefinition[]>(
          'player_controls_get_definitions'
        ),

    findPlayerControlBindingIdByCode: (
      bindings: PlayerControlBindings,
      code: string
    ): PlayerControlBindingId | undefined => {
      const result = host.callJson<PlayerControlBindingId | null>(
        'player_controls_find_binding_id_by_code',
        bindings,
        code
      )

      return result === null ? undefined : result
    },

    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
