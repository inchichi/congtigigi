import { generateJson } from './llmProvider'
import replyWithMessageExample from '../games/my-sample-rpg/assets/lua/reply-with-message.lua?raw'
import vnDialogueExample from '../games/my-sample-rpg/assets/lua/vn-dialogue.lua?raw'
import wanderNearHomeExample from '../games/my-sample-rpg/assets/lua/wander-near-home.lua?raw'

export type GeneratedLuaController = {
  lua_source: string
}

const LUA_CONTROLLER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    lua_source: { type: 'string' }
  },
  required: ['lua_source']
}

// 게임의 공개 Lua 컨트롤러 계약(engine.* 허용목록 포함)을 LLM 에게 그대로 못 박는 시스템 프롬프트.
// Phase 1 허용 API = show_message / show_dialogue / get_controller_config (이미 wasm 브리지로 검증됨).
const createSystemPrompt = (targetCharacterName: string): string =>
  [
    '너는 2D RPG 게임의 NPC 행동을 Lua "컨트롤러 모듈"로 작성하는 코드 생성기다.',
    '반드시 제공된 도구로 lua_source 하나(컨트롤러 모듈 문자열 전체)를 반환한다.',
    '',
    '[컨트롤러 계약]',
    '- 모듈은 `local controller = {}` 로 시작해 마지막에 반드시 `return controller` 로 끝낸다.',
    '- 예약 메서드: register(id, home_x, home_y, radius)=선택, step(id, dt, x, y)=필수, interact(id, source_id)=선택, unregister(id)=선택.',
    '- step 은 매 프레임 이동 방향 (move_x, move_y) 를 반환한다. 제자리 NPC 면 `return 0, 0`.',
    '- 캐릭터별 상태는 controllers[id] 같은 로컬 테이블에 보관한다(스크립트는 여러 캐릭터에 공유될 수 있다).',
    '',
    '[허용된 engine.* API — 이것만 사용]',
    '- engine.ui.show_message(text, duration_seconds): NPC 머리 위 말풍선.',
    '- engine.ui.show_dialogue(lines_table, duration_seconds): 비주얼노벨 대화창(여러 줄을 한 번에). lines_table 은 문자열 배열.',
    '- engine.self.get_controller_config(): 이 캐릭터의 TMX 설정 테이블(예: config.dialogueLines, config.messageDurationSeconds).',
    '',
    '[금지]',
    '- 위 목록 외의 engine.* 호출, require/io/os/package/debug/load/dofile/loadfile, _G 등 전역 접근, break 없는 while true 무한 루프.',
    '',
    `[대상] 이 컨트롤러가 붙는 NPC 이름: "${targetCharacterName}". 사용자의 요청에 맞는 대사/행동을 구현한다.`,
    '대사 text 는 짧고 자연스러운 한국어 문장으로 쓴다. 코드에 주석은 최소화한다.',
    '',
    '[예시 1 — 말풍선 대사 순환]',
    replyWithMessageExample.trim(),
    '',
    '[예시 2 — 비주얼노벨 대화창]',
    vnDialogueExample.trim(),
    '',
    '[예시 3 — 집 주변 배회 + 상호작용]',
    wanderNearHomeExample.trim()
  ].join('\n')

export const generateLuaControllerDraftWithClaude = ({
  apiKey,
  userPrompt,
  targetCharacterName
}: {
  apiKey: string
  userPrompt: string
  targetCharacterName: string
}): Promise<GeneratedLuaController> =>
  generateJson<GeneratedLuaController>({
    apiKey,
    instructions: createSystemPrompt(targetCharacterName),
    input: userPrompt,
    schemaName: 'generated_lua_controller',
    schema: LUA_CONTROLLER_SCHEMA
  })
