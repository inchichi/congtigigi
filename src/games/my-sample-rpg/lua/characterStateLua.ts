// characterState 순수 로직을 Lua VM 에서 실행하는 래퍼 — characterState.ts(TS)와 동일한 API.
// 컨트롤러 kind(keyboard/npc/lua) 판별 유니온·이동 의도·이동 적용을 callJson 으로 마샬링한다.
// pressedDirections/triggeredActions 는 JS Set → Array.from 으로 배열화해 Lua 에 넘긴다(Lua 는 멤버십 테이블로 본다).
// 미일치/없음(undefined)은 Lua json_null 로 오가며, 여기서 undefined 로 정규화한다.
// ZERO 의존 모듈이라 별도 dep .lua 로드는 없다(json-codec 은 호스트가 자동 로드).

import characterStateSource from '../assets/lua/character-state.lua?raw'

import type {
  CharacterAction,
  CharacterControllerIntent,
  CharacterMoveDirection,
  CharacterState,
  KeyboardCharacterController,
  LuaCharacterController,
  LuaCharacterControllerConfig,
  NpcCharacterController
} from '../characterState'
import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

type CreateInitialPlayerCharacterInput = {
  mapWidth: number
  mapHeight: number
}

type CreateNpcCharacterInput = {
  id: string
  appearanceType: string
  level?: number
  displayText?: string
  position: {
    x: number
    y: number
  }
  facing?: CharacterMoveDirection
  collisionSize: {
    width: number
    height: number
  }
  blocksMovement?: boolean
  controller?: CharacterState['controller']
}

type MoveCharacterStateInput = {
  character: CharacterState
  delta: {
    x: number
    y: number
  }
  mapWidth: number
  mapHeight: number
}

type GetCharacterControllerIntentInput = {
  character: CharacterState
  deltaMilliseconds: number
  pressedDirections?: ReadonlySet<CharacterMoveDirection>
  triggeredActions?: ReadonlySet<CharacterAction>
}

export type CharacterStateLua = {
  createKeyboardCharacterController: (input?: {
    moveSpeedTilesPerSecond?: number
  }) => KeyboardCharacterController
  createIdleNpcCharacterController: (input?: {
    moveSpeedTilesPerSecond?: number
    dialogueLines?: string[]
    messageDurationMilliseconds?: number
  }) => NpcCharacterController
  createLuaCharacterController: (input: {
    scriptId: string
    radiusInTiles: number
    config?: LuaCharacterControllerConfig
    moveSpeedTilesPerSecond?: number
  }) => LuaCharacterController
  createInitialPlayerCharacter: (
    input: CreateInitialPlayerCharacterInput
  ) => CharacterState
  createNpcCharacter: (input: CreateNpcCharacterInput) => CharacterState
  getCharacterMoveDirectionFromKey: (
    key: string
  ) => CharacterMoveDirection | undefined
  getCharacterActionFromKey: (key: string) => CharacterAction | undefined
  getCharacterControllerIntent: (
    input: GetCharacterControllerIntentInput
  ) => CharacterControllerIntent | undefined
  moveCharacterState: (input: MoveCharacterStateInput) => CharacterState
  close: () => void
}

export const CHARACTER_STATE_LUA_SOURCE = characterStateSource

export const createCharacterStateLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<CharacterStateLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  // ZERO 의존 — 자기 모듈 소스만 로드.
  host.runModule(characterStateSource, '@character-state.lua')

  return {
    createKeyboardCharacterController: ({
      moveSpeedTilesPerSecond
    }: {
      moveSpeedTilesPerSecond?: number
    } = {}): KeyboardCharacterController =>
      host.callJson<KeyboardCharacterController>(
        'character_create_keyboard_controller',
        { moveSpeedTilesPerSecond }
      ),
    createIdleNpcCharacterController: ({
      moveSpeedTilesPerSecond,
      dialogueLines,
      messageDurationMilliseconds
    }: {
      moveSpeedTilesPerSecond?: number
      dialogueLines?: string[]
      messageDurationMilliseconds?: number
    } = {}): NpcCharacterController =>
      host.callJson<NpcCharacterController>(
        'character_create_idle_npc_controller',
        { moveSpeedTilesPerSecond, dialogueLines, messageDurationMilliseconds }
      ),
    createLuaCharacterController: ({
      scriptId,
      radiusInTiles,
      config,
      moveSpeedTilesPerSecond
    }: {
      scriptId: string
      radiusInTiles: number
      config?: LuaCharacterControllerConfig
      moveSpeedTilesPerSecond?: number
    }): LuaCharacterController =>
      host.callJson<LuaCharacterController>('character_create_lua_controller', {
        scriptId,
        radiusInTiles,
        config,
        moveSpeedTilesPerSecond
      }),
    createInitialPlayerCharacter: (
      createInput: CreateInitialPlayerCharacterInput
    ): CharacterState =>
      host.callJson<CharacterState>(
        'character_create_initial_player',
        createInput
      ),
    createNpcCharacter: (createInput: CreateNpcCharacterInput): CharacterState =>
      host.callJson<CharacterState>('character_create_npc', createInput),
    getCharacterMoveDirectionFromKey: (
      key: string
    ): CharacterMoveDirection | undefined => {
      const result = host.callJson<CharacterMoveDirection | null>(
        'character_move_direction_from_key',
        key
      )

      return result === null ? undefined : result
    },
    getCharacterActionFromKey: (key: string): CharacterAction | undefined => {
      const result = host.callJson<CharacterAction | null>(
        'character_action_from_key',
        key
      )

      return result === null ? undefined : result
    },
    getCharacterControllerIntent: ({
      character,
      deltaMilliseconds,
      pressedDirections,
      triggeredActions
    }: GetCharacterControllerIntentInput):
      | CharacterControllerIntent
      | undefined => {
      const result = host.callJson<CharacterControllerIntent | null>(
        'character_controller_intent',
        {
          character,
          deltaMilliseconds,
          pressedDirections: pressedDirections
            ? Array.from(pressedDirections)
            : [],
          triggeredActions: triggeredActions
            ? Array.from(triggeredActions)
            : []
        }
      )

      return result === null ? undefined : result
    },
    moveCharacterState: (moveInput: MoveCharacterStateInput): CharacterState =>
      host.callJson<CharacterState>('character_move_state', moveInput),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
