---@class ReplyControllerState
---@field dialogue_lines string[] 상호작용할 때 순서대로 보여줄 대사 목록
---@field next_dialogue_index integer 다음에 보여줄 대사 인덱스
---@field duration_seconds number 말풍선을 유지할 시간(초)

---이 스크립트는 `docs/lua-controller-api.md` 와
---`src/game/lua/luaControllerApi.ts` 계약만 사용한다고 가정한다.
---스크립트는 예약된 메서드 이름을 가진 controller table을 반환해야 한다.

---@type table<string, ReplyControllerState>
local controllers = {}
local controller = {}

---@type table<string, string[]>
local default_dialogue_lines = {
  blacksmith = {
    "I AM 대장장이.",
    "YOU 에게 무기를 팔고 수리해줄 예정이지."
  }
}

local function copy_string_list(values)
  local copy = {}

  for index, value in ipairs(values) do
    copy[index] = value
  end

  return copy
end

local function get_dialogue_lines(id, config)
  local configured_lines = config.dialogueLines

  if type(configured_lines) == "table" and #configured_lines > 0 then
    return copy_string_list(configured_lines)
  end

  if default_dialogue_lines[id] ~= nil then
    return copy_string_list(default_dialogue_lines[id])
  end

  return { "Greetings." }
end

---캐릭터별 말풍선 응답 상태를 등록하거나 초기화한다.
---@param id string 캐릭터 식별자
---@param home_x number 처음 등록된 기준 위치 X
---@param home_y number 처음 등록된 기준 위치 Y
---@param radius number 기준 위치에서 허용된 이동 반경
function controller.register(id, home_x, home_y, radius)
  local config = engine.self.get_controller_config()

  controllers[id] = {
    dialogue_lines = get_dialogue_lines(id, config),
    next_dialogue_index = 1,
    duration_seconds =
      type(config.messageDurationSeconds) == "number"
        and config.messageDurationSeconds
        or 2.5
  }
end

---컨트롤러가 제거될 때 캐릭터별 내부 상태를 정리한다.
---@param id string 캐릭터 식별자
function controller.unregister(id)
  controllers[id] = nil
end

---한 프레임 동안 이동할 방향 벡터를 계산한다.
---이 테스트용 컨트롤러는 상호작용만 처리하고 움직이지 않는다.
---@param id string 캐릭터 식별자
---@param dt number 이번 프레임의 경과 시간(초)
---@param x number 현재 캐릭터 위치 X
---@param y number 현재 캐릭터 위치 Y
---@return number move_x 정규화된 X 이동 방향
---@return number move_y 정규화된 Y 이동 방향
function controller.step(id, dt, x, y)
  return 0, 0
end

---플레이어가 상호작용했을 때 공통 UI API로 말풍선을 요청한다.
---@param id string 캐릭터 식별자
---@param source_id string 상호작용을 건 캐릭터 식별자
function controller.interact(id, source_id)
  local controller_state = controllers[id]

  if controller_state == nil then
    return nil, nil
  end

  local dialogue_line =
    controller_state.dialogue_lines[controller_state.next_dialogue_index]
    or controller_state.dialogue_lines[1]

  engine.ui.show_message(dialogue_line, controller_state.duration_seconds)
  controller_state.next_dialogue_index =
    controller_state.next_dialogue_index % #controller_state.dialogue_lines + 1

  return nil, nil
end

return controller
