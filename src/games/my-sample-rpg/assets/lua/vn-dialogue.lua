---비주얼노벨 대화창을 주도하는 NPC 컨트롤러.
---상호작용하면 공개 API `engine.ui.show_dialogue` 로 대사 묶음을 한 번에 넘긴다.
---초상화/이름 같은 표현은 렌더러가 캐릭터에서 채우고, 이 스크립트는 "어떤 대사를
---대화창으로 보여줄지"를 소유한다. `docs/lua-controller-api.md` 계약만 사용한다.

---@class VnDialogueControllerState
---@field lines string[] 대화창에 순서대로 보여줄 대사 목록
---@field duration_seconds number|nil 상호작용 재트리거를 막는 잠금 시간(초)

---@type table<string, VnDialogueControllerState>
local controllers = {}
local controller = {}

local function copy_string_list(values)
  local copy = {}

  if type(values) == "table" then
    for index = 1, #values do
      local value = values[index]

      if value ~= nil and tostring(value) ~= "" then
        copy[#copy + 1] = tostring(value)
      end
    end
  end

  return copy
end

---캐릭터별 대사 상태를 등록한다. 대사는 TMX 의 controller.dialogueLines 에서 읽는다.
---@param id string 캐릭터 식별자
function controller.register(id, home_x, home_y, radius)
  local config = engine.self.get_controller_config()
  local lines = copy_string_list(config.dialogueLines)

  if #lines == 0 then
    lines = { "..." }
  end

  controllers[id] = {
    lines = lines,
    duration_seconds =
      type(config.messageDurationSeconds) == "number"
        and config.messageDurationSeconds
        or nil
  }
end

---@param id string 캐릭터 식별자
function controller.unregister(id)
  controllers[id] = nil
end

---이 컨트롤러는 상호작용만 처리하고 움직이지 않는다.
---@param id string 캐릭터 식별자
---@param dt number
---@param x number
---@param y number
---@return number move_x
---@return number move_y
function controller.step(id, dt, x, y)
  return 0, 0
end

---플레이어가 상호작용하면 대사 묶음 전체를 비주얼노벨 대화창으로 요청한다.
---@param id string 캐릭터 식별자
---@param source_id string 상호작용을 건 캐릭터 식별자
function controller.interact(id, source_id)
  local controller_state = controllers[id]

  if controller_state == nil then
    return nil, nil
  end

  engine.ui.show_dialogue(
    controller_state.lines,
    controller_state.duration_seconds
  )

  return nil, nil
end

return controller
