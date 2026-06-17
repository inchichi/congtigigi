export type InteractionRequestedGameEvent = {
  kind: 'interaction-requested'
  sourceCharacterId: string
}

export type ShowCharacterMessageGameEvent = {
  kind: 'show-character-message'
  characterId: string
  message: string
  durationMilliseconds: number
}

export type ShowNpcDialogueGameEvent = {
  kind: 'show-npc-dialogue'
  characterId: string
  lines: string[]
  durationMilliseconds: number
}

// Phase 3 액션 이벤트는 Lua 런타임 계약과 동일하므로 거기서 가져와 중복 정의를 피한다.
import type {
  LuaControllerRequestQuestStartEvent,
  LuaControllerRequestQuestProgressEvent,
  LuaControllerRequestQuestCompleteEvent,
  LuaControllerRequestInventoryAddEvent,
  LuaControllerRequestInventoryRemoveEvent,
  LuaControllerSetConfigEvent,
  LuaControllerRequestSceneTransitionEvent,
  LuaControllerPlaySoundEvent
} from '../lua/luaControllerApi'

export type GameEvent =
  | InteractionRequestedGameEvent
  | ShowCharacterMessageGameEvent
  | ShowNpcDialogueGameEvent
  | LuaControllerRequestQuestStartEvent
  | LuaControllerRequestQuestProgressEvent
  | LuaControllerRequestQuestCompleteEvent
  | LuaControllerRequestInventoryAddEvent
  | LuaControllerRequestInventoryRemoveEvent
  | LuaControllerSetConfigEvent
  | LuaControllerRequestSceneTransitionEvent
  | LuaControllerPlaySoundEvent

export type GameEventQueue = {
  enqueue: (event: GameEvent) => void
  drain: () => GameEvent[]
  clear: () => void
}

export const createGameEventQueue = (): GameEventQueue => {
  const queuedEvents: GameEvent[] = []

  return {
    enqueue: (event) => {
      queuedEvents.push(event)
    },
    drain: () => queuedEvents.splice(0, queuedEvents.length),
    clear: () => {
      queuedEvents.length = 0
    }
  }
}
