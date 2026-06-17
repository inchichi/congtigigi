import type {
  CharacterState,
  LuaCharacterController,
  LuaCharacterControllerConfig
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

export type LuaControllerSelfApi = {
  getControllerConfig: () => LuaCharacterControllerConfig
}

export type LuaControllerPublicApi = {
  ui: LuaControllerUiApi
  self: LuaControllerSelfApi
}

export type LuaControllerShowCharacterMessageEvent = {
  kind: 'show-character-message'
  characterId: string
  message: string
  durationMilliseconds: number
}

// Lua 컨트롤러가 engine.ui.show_dialogue 로 비주얼노벨 대화창을 요청할 때 내보내는 이벤트.
// 대사 줄 목록은 Lua 가 소유하고, 초상화/이름 같은 표현은 렌더러가 캐릭터에서 채운다.
export type LuaControllerShowNpcDialogueEvent = {
  kind: 'show-npc-dialogue'
  characterId: string
  lines: string[]
  durationMilliseconds: number
}

// Phase 3 쓰기 채널: Lua 가 호스트에 상태 변경을 "요청"하는 액션 이벤트.
// 실제 적용(APPLY)은 호스트의 기존 순수 reducer 가 담당한다(Lua=요청, TS=적용).
export type LuaControllerRequestQuestStartEvent = {
  kind: 'request-quest-start'
  questId: string
}
export type LuaControllerRequestQuestProgressEvent = {
  kind: 'request-quest-progress'
  questId: string
  objectiveId: string
  amount: number
}
export type LuaControllerRequestQuestCompleteEvent = {
  kind: 'request-quest-complete'
  questId: string
}
export type LuaControllerRequestInventoryAddEvent = {
  kind: 'request-inventory-add'
  itemId: string
  quantity: number
}
export type LuaControllerRequestInventoryRemoveEvent = {
  kind: 'request-inventory-remove'
  itemId: string
  quantity: number
}
// NPC 별 플래그를 호스트가 보관해 다음 상호작용/리로드에서 get_controller_config 로 읽게 한다.
export type LuaControllerSetConfigEvent = {
  kind: 'set-config'
  characterId: string
  key: string
  value: string
}

// Phase 4: 씬 전환 + 사운드 액션. (전투 encounter 는 이 게임의 실시간 온맵 전투 모델에
// 맞는 spawn 프리미티브가 없어 보류 — 설계상 deferrable.)
export type LuaControllerRequestSceneTransitionEvent = {
  kind: 'request-scene-transition'
  sceneId: string
  x: number
  y: number
}
export type LuaControllerPlaySoundEvent = {
  kind: 'play-sound'
  soundId: string
}

export type LuaControllerRuntimeEvent =
  | LuaControllerShowCharacterMessageEvent
  | LuaControllerShowNpcDialogueEvent
  | LuaControllerRequestQuestStartEvent
  | LuaControllerRequestQuestProgressEvent
  | LuaControllerRequestQuestCompleteEvent
  | LuaControllerRequestInventoryAddEvent
  | LuaControllerRequestInventoryRemoveEvent
  | LuaControllerSetConfigEvent
  | LuaControllerRequestSceneTransitionEvent
  | LuaControllerPlaySoundEvent

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
