---@class ReplyControllerState
---@field message string 상호작용을 받았을 때 보여줄 말풍선 메시지
---@field duration_seconds number 말풍선을 유지할 시간(초)

---이 스크립트는 `docs/lua-controller-api.md` 와
---`src/game/lua/luaControllerApi.ts` 계약만 사용한다고 가정한다.

---@type table<string, ReplyControllerState>
local controllers = {}

---@type table<string, string>
local default_messages = {
  blacksmith = "대장장이: 아직은 시연용 마을이야. 나중에 다시 와 줘."
}

---캐릭터별 말풍선 응답 상태를 등록하거나 초기화한다.
---@param id string 캐릭터 식별자
---@param home_x number 처음 등록된 기준 위치 X
---@param home_y number 처음 등록된 기준 위치 Y
---@param radius number 기준 위치에서 허용된 이동 반경
function register_reply_controller(id, home_x, home_y, radius)
  controllers[id] = {
    message = default_messages[id] or "안녕하세요.",
    duration_seconds = 2.5
  }
end

---컨트롤러가 제거될 때 캐릭터별 내부 상태를 정리한다.
---@param id string 캐릭터 식별자
function unregister_reply_controller(id)
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
function step_reply_controller(id, dt, x, y)
  return 0, 0
end

---플레이어가 상호작용했을 때 보여줄 메시지와 시간을 결정한다.
---@param id string 캐릭터 식별자
---@param source_id string 상호작용을 건 캐릭터 식별자
---@return string|nil message 말풍선으로 보여줄 메시지
---@return number|nil duration_seconds 말풍선을 유지할 시간(초)
function interact_reply_controller(id, source_id)
  local controller = controllers[id]

  if controller == nil then
    return nil, nil
  end

  return controller.message, controller.duration_seconds
end
