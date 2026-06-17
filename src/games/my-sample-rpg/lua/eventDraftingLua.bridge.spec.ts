import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import { createHolidayDialogueEventDraftFromText } from '../eventDrafting'
import {
  createEventDraftingLua,
  type EventDraftingLua
} from './eventDraftingLua'

// 실제 Lua 5.3.6 WASM VM을 node에서 로드해(디스크의 mjs/wasm 주입), Lua 구현이 TS 기준
// 구현과 모든 입력에서 동일한지 검증한다. (다른 *.bridge.spec.ts와 같은 로더 패턴.)
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createDrafting = async (): Promise<EventDraftingLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createEventDraftingLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

// 크리스마스/할로윈/무매치/대소문자 혼합/유니코드 키워드/혼합 텍스트/우선순위(크리스마스 먼저) 등.
const PROMPTS: readonly string[] = [
  'christmas',
  'Christmas',
  'CHRISTMAS',
  'xmas',
  'XMAS',
  '크리스마스',
  '메리 크리스마스 이벤트를 만들어줘',
  'I want a Christmas event for the town',
  'halloween',
  'Halloween',
  'HALLOWEEN',
  '할로윈',
  '할로윈 이벤트 부탁해',
  'spooky Halloween night',
  '',
  'random prompt with no holiday',
  'birthday party event',
  'XmasAndHalloween',
  'halloween and christmas',
  'this mentions xmas inside a word: xmastree',
  '추석 이벤트',
  '   ',
  'HaLLoWeEn',
  'cHrIsTmAs'
]

describe('eventDraftingLua (real wasm bridge)', () => {
  let drafting: EventDraftingLua | undefined

  afterEach(() => {
    drafting?.close()
    drafting = undefined
  })

  it('matches the TS createHolidayDialogueEventDraftFromText across every prompt', async () => {
    drafting = await createDrafting()

    for (const prompt of PROMPTS) {
      expect(drafting.createHolidayDialogueEventDraftFromText(prompt)).toEqual(
        createHolidayDialogueEventDraftFromText(prompt)
      )
    }
  })

  it('returns undefined (not null) for non-matching prompts', async () => {
    drafting = await createDrafting()

    expect(
      drafting.createHolidayDialogueEventDraftFromText('no holiday here')
    ).toBeUndefined()
  })
})
