import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createHolidayDialogueEventValidationErrors,
  createTiledNpcEventObject,
  isHolidayDialogueEventValid,
  type EventReward,
  type HolidayDialogueEventSpec
} from '../eventGeneration'
import {
  createEventGenerationLua,
  type EventGenerationLua
} from './eventGenerationLua'

// eventGeneration(휴일-대사 이벤트 검증/유효성/Tiled NPC 오브젝트)을 callJson 으로 변환한 결과가
// TS 원본과 동등한지 실제 WASM 으로 검증한다. 모든 검증 분기 + 각 reward enum + 경계값을 덮는다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

// 모든 검증을 통과하는 기준 이벤트.
const createValidEvent = (): HolidayDialogueEventSpec => ({
  event_name: 'spring_festival_01',
  event_type: 'holiday_dialogue',
  npc: {
    id: 'festival_guide',
    display_name: '축제 안내원',
    appearance_type: 'villager'
  },
  trigger: {
    type: 'talk',
    target_scene: 'town_square'
  },
  dialogue: {
    opening_lines: ['안녕하세요!', '축제에 오신 걸 환영해요.'],
    active_lines: ['즐기고 계신가요?'],
    completion_lines: ['또 만나요!']
  },
  reward: { type: 'gold', count: 100 },
  duration: 5
})

// 기준 이벤트에 부분 패치를 적용한 깊은 복제본을 만든다.
const withPatch = (
  patch: (event: HolidayDialogueEventSpec) => void
): HolidayDialogueEventSpec => {
  const event = structuredClone(createValidEvent())
  patch(event)
  return event
}

describe('eventGenerationLua (real wasm)', () => {
  let lua: EventGenerationLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS eventGeneration across validation, validity, and tiled object', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createEventGenerationLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    const rewards: EventReward[] = [
      { type: 'none' },
      { type: 'item', id: 'gift_box', count: 1 },
      { type: 'item', id: '', count: 0 }, // item: 빈 id + 비양수 count
      { type: 'item', id: '  ', count: -3 }, // item: 공백 id + 음수 count
      { type: 'gold', count: 100 },
      { type: 'gold', count: 0 }, // gold: 비양수 count
      { type: 'gold', count: -1 },
      { type: 'experience', count: 50 },
      { type: 'experience', count: 0 }, // experience: 비양수 count
      { type: 'experience', count: -10 }
    ]

    // 각 검증 분기를 개별적으로 위반하는 케이스 + 정상 케이스.
    const cases: HolidayDialogueEventSpec[] = [
      createValidEvent(),
      withPatch((event) => {
        event.event_name = ''
      }),
      withPatch((event) => {
        event.event_name = '   '
      }),
      withPatch((event) => {
        event.event_name = 'Spring_Festival' // 대문자 → 패턴 위반
      }),
      withPatch((event) => {
        event.event_name = 'spring festival' // 공백 → 패턴 위반(빈 것은 아님)
      }),
      withPatch((event) => {
        event.event_name = 'spring-festival' // 하이픈 → 패턴 위반
      }),
      withPatch((event) => {
        event.event_name = 'café' // 비ASCII → 패턴 위반
      }),
      withPatch((event) => {
        event.event_name = 'a1_b2' // 정상(숫자/밑줄 혼합)
      }),
      withPatch((event) => {
        event.event_type = 'wrong_type' as HolidayDialogueEventSpec['event_type']
      }),
      withPatch((event) => {
        event.npc.id = ''
      }),
      withPatch((event) => {
        event.npc.id = '\t\n'
      }),
      withPatch((event) => {
        event.npc.display_name = ''
      }),
      withPatch((event) => {
        event.npc.display_name = '   '
      }),
      withPatch((event) => {
        event.npc.appearance_type = ''
      }),
      withPatch((event) => {
        event.trigger.type = 'scene_enter'
      }),
      withPatch((event) => {
        event.trigger.type = 'quest_start'
      }),
      withPatch((event) => {
        event.trigger.type = 'quest_complete'
      }),
      withPatch((event) => {
        event.trigger.type =
          'invalid_trigger' as HolidayDialogueEventSpec['trigger']['type']
      }),
      withPatch((event) => {
        event.trigger.target_scene = ''
      }),
      withPatch((event) => {
        event.trigger.target_scene = '  '
      }),
      withPatch((event) => {
        event.dialogue.opening_lines = []
      }),
      withPatch((event) => {
        event.dialogue.opening_lines = ['valid', '']
      }),
      withPatch((event) => {
        event.dialogue.opening_lines = ['valid', '   ']
      }),
      withPatch((event) => {
        event.dialogue.active_lines = ['']
      }),
      withPatch((event) => {
        event.dialogue.active_lines = []
      }),
      withPatch((event) => {
        event.dialogue.completion_lines = ['  ']
      }),
      withPatch((event) => {
        event.dialogue.completion_lines = []
      }),
      withPatch((event) => {
        event.duration = 0
      }),
      withPatch((event) => {
        event.duration = -5
      }),
      withPatch((event) => {
        event.duration = 0.5 // 양수 소수 → 정상
      }),
      withPatch((event) => {
        event.duration = 1e15 // 큰 정수값 float
      }),
      // 여러 위반이 동시에 발생해 push 순서까지 동일해야 하는 케이스.
      withPatch((event) => {
        event.event_name = 'Bad Name'
        event.event_type = 'nope' as HolidayDialogueEventSpec['event_type']
        event.npc.id = ''
        event.trigger.type =
          'x' as HolidayDialogueEventSpec['trigger']['type']
        event.dialogue.opening_lines = []
        event.duration = -1
        event.reward = { type: 'item', id: '', count: 0 }
      }),
      // 완전히 빈/공백 투성이 이벤트.
      {
        event_name: '',
        event_type: '' as HolidayDialogueEventSpec['event_type'],
        npc: { id: '', display_name: '', appearance_type: '' },
        trigger: {
          type: '' as HolidayDialogueEventSpec['trigger']['type'],
          target_scene: ''
        },
        dialogue: { opening_lines: [], active_lines: [], completion_lines: [] },
        reward: { type: 'none' },
        duration: 0
      }
    ]

    // reward 변형마다 기준 이벤트에 끼워 검증 분기를 모두 덮는다.
    for (const reward of rewards) {
      cases.push(
        withPatch((event) => {
          event.reward = reward
        })
      )
    }

    for (const event of cases) {
      expect(lua.createHolidayDialogueEventValidationErrors(event)).toEqual(
        createHolidayDialogueEventValidationErrors(event)
      )
      expect(lua.isHolidayDialogueEventValid(event)).toBe(
        isHolidayDialogueEventValid(event)
      )
      expect(lua.createTiledNpcEventObject(event)).toEqual(
        createTiledNpcEventObject(event)
      )
    }

    // createTiledNpcEventObject: 빈 opening_lines 도 [] 로 왕복(객체 아님).
    const emptyLines = withPatch((event) => {
      event.dialogue.opening_lines = []
    })
    expect(lua.createTiledNpcEventObject(emptyLines)).toEqual(
      createTiledNpcEventObject(emptyLines)
    )
    expect(
      lua.createTiledNpcEventObject(emptyLines).properties[
        'controller.dialogueLines'
      ]
    ).toEqual([])

    // opening_lines 얕은 복사 동일성: 단일/다중 라인.
    const singleLine = withPatch((event) => {
      event.dialogue.opening_lines = ['오직 한 줄']
    })
    expect(lua.createTiledNpcEventObject(singleLine)).toEqual(
      createTiledNpcEventObject(singleLine)
    )
  })
})
