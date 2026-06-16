import type { CharacterState } from '../characterState'
import type {
  GameEvent,
  ShowCharacterMessageGameEvent
} from '../events/createGameEventQueue'

import type { LuaControllerRuntimeEvent } from '../lua/luaControllerApi'
import type {
  CharacterControllerRuntime,
  CharacterInteractionResponse
} from '../createCharacterControllerRuntime'

// 상호작용 처리 후 렌더러로 넘기는 이벤트: 표시용(말풍선/대화창) + Phase 3 액션(request-*/set-config).
// 런타임 이벤트 유니온과 동일하다(렌더러가 kind 로 분기해 표시 또는 reducer 적용).
type InteractionEmittedEvent = LuaControllerRuntimeEvent
import { resolveCharacterInteractionTarget } from './resolveCharacterInteractionTarget'

type ProcessInteractionEventsInput = {
  events: GameEvent[]
  characters: CharacterState[]
  controllerRuntime: Pick<
    CharacterControllerRuntime,
    'canReceiveInteraction' | 'drainEvents' | 'handleInteraction'
  >
  now: number
  interactionLockUntilByCharacterPair: Map<string, number>
}

export const processInteractionEvents = ({
  events,
  characters,
  controllerRuntime,
  now,
  interactionLockUntilByCharacterPair
}: ProcessInteractionEventsInput): InteractionEmittedEvent[] => {
  const emittedEvents: InteractionEmittedEvent[] = events.filter(
    (event): event is InteractionEmittedEvent =>
      event.kind !== 'interaction-requested'
  )

  for (const event of events) {
    if (event.kind !== 'interaction-requested') {
      continue
    }

    const sourceCharacter = characters.find(
      (character) => character.id === event.sourceCharacterId
    )

    if (!sourceCharacter) {
      continue
    }

    const targetCharacter = resolveCharacterInteractionTarget({
      sourceCharacter,
      targetCharacters: characters,
      canReceiveInteraction: controllerRuntime.canReceiveInteraction
    })

    if (!targetCharacter) {
      continue
    }

    const interactionLockKey = createInteractionLockKey(
      sourceCharacter.id,
      targetCharacter.id
    )
    const lockedUntil =
      interactionLockUntilByCharacterPair.get(interactionLockKey) ?? 0

    if (lockedUntil > now) {
      continue
    }

    const response = controllerRuntime.handleInteraction({
      targetCharacter,
      sourceCharacter
    })
    const runtimeEvents = controllerRuntime.drainEvents()
    const responseDurationMilliseconds =
      response?.durationMilliseconds ?? 0
    const runtimeEventDurationMilliseconds = getMaxMessageEventDuration(
      runtimeEvents
    )
    const hasRuntimeMessageEvent = runtimeEventDurationMilliseconds > 0

    emittedEvents.push(...runtimeEvents)

    if (!hasRuntimeMessageEvent && response) {
      emittedEvents.push(
        createShowCharacterMessageEvent(targetCharacter.id, response)
      )
    }

    const interactionLockDurationMilliseconds = Math.max(
      responseDurationMilliseconds,
      runtimeEventDurationMilliseconds
    )

    if (interactionLockDurationMilliseconds === 0) {
      continue
    }

    interactionLockUntilByCharacterPair.set(
      interactionLockKey,
      now + interactionLockDurationMilliseconds
    )
  }

  return emittedEvents
}

const createShowCharacterMessageEvent = (
  characterId: string,
  response: CharacterInteractionResponse
): ShowCharacterMessageGameEvent => ({
  kind: 'show-character-message',
  characterId,
  message: response.message,
  durationMilliseconds: response.durationMilliseconds
})

const getMaxMessageEventDuration = (
  events: InteractionEmittedEvent[]
): number =>
  events.reduce(
    (maxDurationMilliseconds, event) =>
      Math.max(
        maxDurationMilliseconds,
        'durationMilliseconds' in event ? event.durationMilliseconds : 0
      ),
    0
  )

const createInteractionLockKey = (
  sourceCharacterId: string,
  targetCharacterId: string
): string => `${sourceCharacterId}:${targetCharacterId}`
