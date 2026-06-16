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

export type GameEvent =
  | InteractionRequestedGameEvent
  | ShowCharacterMessageGameEvent
  | ShowNpcDialogueGameEvent

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
