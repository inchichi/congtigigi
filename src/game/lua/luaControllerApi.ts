import type {
  CharacterState,
  LuaCharacterController
} from '../characterState'

export type LuaControllerFunctionContract = {
  registerFunctionName: string
  unregisterFunctionName?: string
  stepFunctionName: string
}

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

export type LuaControllerMoveIntent = {
  moveX: number
  moveY: number
}

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

const createLuaControllerCharacterSnapshot = (
  character: CharacterState
): LuaControllerCharacterSnapshot => ({
  id: character.id,
  position: {
    x: character.position.x,
    y: character.position.y
  }
})
