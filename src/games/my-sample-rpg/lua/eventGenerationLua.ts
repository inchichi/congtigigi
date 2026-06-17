// eventGeneration 연산을 Lua로 실행하는 래퍼. 휴일-대사 이벤트 검증/유효성/Tiled NPC 오브젝트 생성을
// callJson 으로 마샬링한다. 자급식 모듈이라 별도 의존 .lua 는 없다(json-codec 은 호스트가 자동 로드).
// 원본 타입을 그대로 재사용하고, 시그니처/반환 타입을 원본과 동일하게 노출한다.

import eventGenerationSource from '../assets/lua/event-generation.lua?raw'

import {
  type HolidayDialogueEventSpec,
  type TiledNpcEventObject
} from '../eventGeneration'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type EventGenerationLua = {
  createHolidayDialogueEventValidationErrors: (
    event: HolidayDialogueEventSpec
  ) => string[]
  isHolidayDialogueEventValid: (event: HolidayDialogueEventSpec) => boolean
  createTiledNpcEventObject: (
    event: HolidayDialogueEventSpec
  ) => TiledNpcEventObject
  close: () => void
}

export const createEventGenerationLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<EventGenerationLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  // 자급식 모듈 — 선행 의존 .lua 없음. 이 모듈 소스만 로드한다.
  host.runModule(eventGenerationSource, '@event-generation.lua')

  return {
    createHolidayDialogueEventValidationErrors: (
      event: HolidayDialogueEventSpec
    ): string[] =>
      host.callJson<string[]>(
        'event_generation_create_validation_errors',
        event
      ),
    isHolidayDialogueEventValid: (event: HolidayDialogueEventSpec): boolean =>
      host.callJson<boolean>('event_generation_is_event_valid', event),
    createTiledNpcEventObject: (
      event: HolidayDialogueEventSpec
    ): TiledNpcEventObject =>
      host.callJson<TiledNpcEventObject>(
        'event_generation_create_tiled_npc_object',
        event
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
