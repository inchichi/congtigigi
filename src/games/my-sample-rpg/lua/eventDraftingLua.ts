// 휴일 대사 이벤트 초안 생성(eventDrafting.ts)을 Lua로 실행하는 래퍼.
// createHolidayDialogueEventDraftFromText 는 객체(HolidayDialogueEventSpec) 또는
// undefined 를 반환하므로 callJson 으로 호출한다(Lua nil → JSON null → undefined 로 정규화).
//
// 이 모듈의 Lua(event-drafting.lua)는 전역 상수 HOLIDAY_DIALOGUE_EVENT_TYPE,
// TALK_EVENT_TRIGGER_TYPE 에 의존한다. 표준상 이 상수들은 event-generation.lua 가
// 먼저 로드되며 전역으로 노출한다:
//   - 공유 퍼사드 호스트(input.host 지정): event-generation 이 이미 먼저 로드돼 있다고 가정 → 이 래퍼는
//     자신의 .lua 만 로드한다(상수 재정의 금지).
//   - 단독 호스트(input.host 미지정): 의존 상수를 먼저 로드한 뒤 event-drafting.lua 를 로드한다.
// 상수 값의 단일 출처는 TS(eventGeneration.ts)이므로 그 값을 import 해 Lua 전역으로 주입한다(드리프트 방지).

import eventDraftingSource from '../assets/lua/event-drafting.lua?raw'

import {
  HOLIDAY_DIALOGUE_EVENT_TYPE,
  TALK_EVENT_TRIGGER_TYPE,
  type HolidayDialogueEventSpec
} from '../eventGeneration'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

// JSON 안전 문자열로 두 상수를 Lua 전역에 주입하는 부트스트랩 청크(단독 호스트 전용).
// 값의 단일 출처는 eventGeneration.ts. event-generation.lua 가 노출하는 것과 동일한 전역명을 쓴다.
const EVENT_GENERATION_CONSTANTS_BOOTSTRAP = [
  `HOLIDAY_DIALOGUE_EVENT_TYPE = ${JSON.stringify(HOLIDAY_DIALOGUE_EVENT_TYPE)}`,
  `TALK_EVENT_TRIGGER_TYPE = ${JSON.stringify(TALK_EVENT_TRIGGER_TYPE)}`
].join('\n')

export type EventDraftingLua = {
  createHolidayDialogueEventDraftFromText: (
    prompt: string
  ) => HolidayDialogueEventSpec | undefined
  close: () => void
}

export const EVENT_DRAFTING_LUA_SOURCE = eventDraftingSource

export const createEventDraftingLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<EventDraftingLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))

  // 자체 호스트를 만든 경우에만 의존 상수를 먼저 로드한다(공유 호스트엔 이미 로드돼 있다).
  if (!input.host) {
    host.runModule(
      EVENT_GENERATION_CONSTANTS_BOOTSTRAP,
      '@event-generation.constants.lua'
    )
  }
  host.runModule(eventDraftingSource, '@event-drafting.lua')

  return {
    createHolidayDialogueEventDraftFromText: (
      prompt: string
    ): HolidayDialogueEventSpec | undefined => {
      const result = host.callJson<HolidayDialogueEventSpec | null>(
        'create_holiday_dialogue_event_draft_from_text',
        prompt
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
