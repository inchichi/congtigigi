---@class WanderControllerState
---@field home_x number 처음 등록된 기준 위치 X
---@field home_y number 처음 등록된 기준 위치 Y
---@field radius number 기준 위치에서 얼마나 멀리 배회할 수 있는지 나타내는 반경
---@field pause_remaining number 잠시 멈춰 있어야 하는 남은 시간(초)
---@field target_x number 현재 이동하려는 목표 위치 X
---@field target_y number 현재 이동하려는 목표 위치 Y

---이 스크립트는 `docs/lua-controller-api.md` 와
---`src/game/lua/luaControllerApi.ts` 계약만 사용한다고 가정한다.

---@type table<string, WanderControllerState>
local controllers = {}

---값을 최소/최대 범위 안으로 제한한다.
---@param value number 제한할 값
---@param minimum number 최소값
---@param maximum number 최대값
---@return number
local function clamp(value, minimum, maximum)
  if value < minimum then
    return minimum
  end

  if value > maximum then
    return maximum
  end

  return value
end

---배회 반경 안에서 다음 목표 지점을 무작위로 다시 고른다.
---@param controller WanderControllerState
local function choose_target(controller)
  local angle = math.random() * math.pi * 2
  local distance = math.random() * controller.radius

  controller.target_x = controller.home_x + math.cos(angle) * distance
  controller.target_y = controller.home_y + math.sin(angle) * distance
end

---캐릭터별 배회 컨트롤러 상태를 등록하거나 초기화한다.
---웹 쪽에서 NPC를 처음 Lua 컨트롤러에 연결할 때 호출한다.
---@param id string 캐릭터 식별자
---@param home_x number 배회의 기준 위치 X
---@param home_y number 배회의 기준 위치 Y
---@param radius number 기준 위치에서 허용할 최대 배회 반경
function register_wander_controller(id, home_x, home_y, radius)
  controllers[id] = {
    home_x = home_x,
    home_y = home_y,
    radius = radius,
    pause_remaining = 0,
    target_x = home_x,
    target_y = home_y
  }
end

---컨트롤러가 제거될 때 캐릭터별 내부 상태를 정리한다.
---@param id string 캐릭터 식별자
function unregister_wander_controller(id)
  controllers[id] = nil
end

---한 프레임 동안 이동할 방향 벡터를 계산한다.
---목표 지점에 도달하면 잠시 멈춘 뒤 새 목표를 다시 고른다.
---@param id string 캐릭터 식별자
---@param dt number 이번 프레임의 경과 시간(초)
---@param x number 현재 캐릭터 위치 X
---@param y number 현재 캐릭터 위치 Y
---@return number move_x 정규화된 X 이동 방향
---@return number move_y 정규화된 Y 이동 방향
function step_wander_controller(id, dt, x, y)
  local controller = controllers[id]

  if controller == nil then
    return 0, 0
  end

  if controller.pause_remaining > 0 then
    controller.pause_remaining = math.max(0, controller.pause_remaining - dt)
    return 0, 0
  end

  local dx = controller.target_x - x
  local dy = controller.target_y - y
  local distance = math.sqrt(dx * dx + dy * dy)

  if distance < 0.1 then
    choose_target(controller)
    controller.pause_remaining = 0.5 + math.random() * 1.0
    return 0, 0
  end

  local normalized_x = dx / distance
  local normalized_y = dy / distance

  return clamp(normalized_x, -1, 1), clamp(normalized_y, -1, 1)
end
