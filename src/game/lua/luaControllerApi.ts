import type {
  CharacterState,
  LuaCharacterController
} from '../characterState'

export const LUA_CONTROLLER_REGISTER_METHOD_NAME = 'register'
export const LUA_CONTROLLER_UNREGISTER_METHOD_NAME = 'unregister'
export const LUA_CONTROLLER_STEP_METHOD_NAME = 'step'
export const LUA_CONTROLLER_INTERACT_METHOD_NAME = 'interact'

export type LuaControllerMethodName =
  | typeof LUA_CONTROLLER_REGISTER_METHOD_NAME
  | typeof LUA_CONTROLLER_UNREGISTER_METHOD_NAME
  | typeof LUA_CONTROLLER_STEP_METHOD_NAME
  | typeof LUA_CONTROLLER_INTERACT_METHOD_NAME

export type LuaControllerPosition = {
  x: number
  y: number
}

export type LuaControllerCharacterSnapshot = {
  id: string
  position: LuaControllerPosition
}

export type LuaControllerRegisterInput = {
  character: LuaControllerCharacterSnapshot
  radiusInTiles: number
}

export type LuaControllerUnregisterInput = {
  characterId: string
}

export type LuaControllerStepInput = {
  character: LuaControllerCharacterSnapshot
  deltaSeconds: number
}

export type LuaControllerInteractInput = {
  character: LuaControllerCharacterSnapshot
  sourceCharacterId: string
}

export type LuaControllerMoveIntent = {
  moveX: number
  moveY: number
}

export type LuaControllerUiApi = {
  showMessage: (message: string, durationSeconds?: number) => void
}

export type LuaControllerPublicApi = {
  ui: LuaControllerUiApi
}

export type LuaControllerShowCharacterMessageEvent = {
  kind: 'show-character-message'
  characterId: string
  message: string
  durationMilliseconds: number
}

export type LuaControllerRuntimeEvent = LuaControllerShowCharacterMessageEvent

export type LuaControllerInteractionResponse = {
  message: string
  durationMilliseconds: number
}

export const LUA_CONTROLLER_PUBLIC_API_NAME = 'engine'
export const DEFAULT_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS = 2000
export const MIN_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS = 250

export const createLuaControllerRegisterInput = (
  character: CharacterState,
  controller: LuaCharacterController
): LuaControllerRegisterInput => ({
  character: createLuaControllerCharacterSnapshot(character),
  radiusInTiles: controller.radiusInTiles
})

export const createLuaControllerUnregisterInput = (
  character: CharacterState
): LuaControllerUnregisterInput => ({
  characterId: character.id
})

export const createLuaControllerStepInput = (
  character: CharacterState,
  deltaMilliseconds: number
): LuaControllerStepInput => ({
  character: createLuaControllerCharacterSnapshot(character),
  deltaSeconds: deltaMilliseconds / 1000
})

export const createLuaControllerInteractInput = (
  character: CharacterState,
  sourceCharacter: CharacterState
): LuaControllerInteractInput => ({
  character: createLuaControllerCharacterSnapshot(character),
  sourceCharacterId: sourceCharacter.id
})

export const getLuaControllerRegisterFunctionArguments = (
  input: LuaControllerRegisterInput
): [string, number, number, number] => [
  input.character.id,
  input.character.position.x,
  input.character.position.y,
  input.radiusInTiles
]

export const getLuaControllerUnregisterFunctionArguments = (
  input: LuaControllerUnregisterInput
): [string] => [input.characterId]

export const getLuaControllerStepFunctionArguments = (
  input: LuaControllerStepInput
): [string, number, number, number] => [
  input.character.id,
  input.deltaSeconds,
  input.character.position.x,
  input.character.position.y
]

export const getLuaControllerInteractFunctionArguments = (
  input: LuaControllerInteractInput
): [string, string] => [
  input.character.id,
  input.sourceCharacterId
]

export const createLuaControllerMoveIntent = (
  moveX: number,
  moveY: number
): LuaControllerMoveIntent | undefined => {
  if (moveX === 0 && moveY === 0) {
    return undefined
  }

  return {
    moveX,
    moveY
  }
}

export const createLuaControllerInteractionResponse = (
  message: string | undefined,
  durationSeconds: number | undefined
): LuaControllerInteractionResponse | undefined => {
  if (!message || message.trim().length === 0) {
    return undefined
  }

  return {
    message,
    durationMilliseconds:
      getLuaControllerMessageDurationMilliseconds(durationSeconds)
  }
}

export const getLuaControllerMessageDurationMilliseconds = (
  durationSeconds: number | undefined
): number =>
  typeof durationSeconds === 'number' && Number.isFinite(durationSeconds)
    ? Math.max(
        MIN_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS,
        Math.round(durationSeconds * 1000)
      )
    : DEFAULT_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS

const createLuaControllerCharacterSnapshot = (
  character: CharacterState
): LuaControllerCharacterSnapshot => ({
  id: character.id,
  position: {
    x: character.position.x,
    y: character.position.y
  }
})
