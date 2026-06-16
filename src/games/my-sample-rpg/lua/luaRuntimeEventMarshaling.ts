import type {
  LuaControllerRuntimeEvent,
  LuaControllerShowCharacterMessageEvent,
  LuaControllerShowNpcDialogueEvent
} from './luaControllerApi'

// Lua 런타임이 JSON으로 직렬화해 보낸 이벤트를 타입 안전하게 파싱한다.
// Lua 쪽 인코더(createLuaCharacterControllerRuntime 프렐류드의 encode_runtime_event)와 1:1 대응.
// 새 이벤트 kind(미래의 request-* 등)는 EVENT_PARSERS 에 한 항목만 추가하면 된다 — 그래서
// LuaControllerRuntimeEvent 유니온이 "열려" 있어도 파싱이 한 곳에서 일관되게 검증된다.

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string')

// kind 별 파서. 페이로드가 유효하지 않으면 undefined 를 돌려주고, 호출부가 일괄 에러를 던진다.
type RuntimeEventParser = (
  raw: Record<string, unknown>
) => LuaControllerRuntimeEvent | undefined

const EVENT_PARSERS: Record<string, RuntimeEventParser> = {
  'show-character-message': (raw) => {
    if (
      typeof raw.characterId !== 'string' ||
      typeof raw.message !== 'string' ||
      !isFiniteNumber(raw.durationMilliseconds)
    ) {
      return undefined
    }

    const event: LuaControllerShowCharacterMessageEvent = {
      kind: 'show-character-message',
      characterId: raw.characterId,
      message: raw.message,
      durationMilliseconds: raw.durationMilliseconds
    }

    return event
  },
  'show-npc-dialogue': (raw) => {
    if (
      typeof raw.characterId !== 'string' ||
      !isStringArray(raw.lines) ||
      !isFiniteNumber(raw.durationMilliseconds)
    ) {
      return undefined
    }

    const event: LuaControllerShowNpcDialogueEvent = {
      kind: 'show-npc-dialogue',
      characterId: raw.characterId,
      lines: raw.lines,
      durationMilliseconds: raw.durationMilliseconds
    }

    return event
  },
  'request-quest-start': (raw) =>
    typeof raw.questId === 'string'
      ? { kind: 'request-quest-start', questId: raw.questId }
      : undefined,
  'request-quest-progress': (raw) =>
    typeof raw.questId === 'string' &&
    typeof raw.objectiveId === 'string' &&
    isFiniteNumber(raw.amount)
      ? {
          kind: 'request-quest-progress',
          questId: raw.questId,
          objectiveId: raw.objectiveId,
          amount: raw.amount
        }
      : undefined,
  'request-quest-complete': (raw) =>
    typeof raw.questId === 'string'
      ? { kind: 'request-quest-complete', questId: raw.questId }
      : undefined,
  'request-inventory-add': (raw) =>
    typeof raw.itemId === 'string' && isFiniteNumber(raw.quantity)
      ? { kind: 'request-inventory-add', itemId: raw.itemId, quantity: raw.quantity }
      : undefined,
  'request-inventory-remove': (raw) =>
    typeof raw.itemId === 'string' && isFiniteNumber(raw.quantity)
      ? { kind: 'request-inventory-remove', itemId: raw.itemId, quantity: raw.quantity }
      : undefined,
  'set-config': (raw) =>
    typeof raw.characterId === 'string' &&
    typeof raw.key === 'string' &&
    typeof raw.value === 'string'
      ? { kind: 'set-config', characterId: raw.characterId, key: raw.key, value: raw.value }
      : undefined
}

export const parseLuaControllerRuntimeEvents = (
  serializedEvents: string
): LuaControllerRuntimeEvent[] => {
  const parsedEvents = JSON.parse(serializedEvents) as unknown

  if (!Array.isArray(parsedEvents)) {
    throw new Error('Lua runtime event drain did not return an array.')
  }

  return parsedEvents.map((rawEvent) => {
    if (!isObject(rawEvent) || typeof rawEvent.kind !== 'string') {
      throw new Error('Lua runtime event drain returned an invalid event.')
    }

    const event = EVENT_PARSERS[rawEvent.kind]?.(rawEvent)

    if (!event) {
      throw new Error('Lua runtime event drain returned an invalid event.')
    }

    return event
  })
}
