-- TS characterState.ts → Lua. 캐릭터 상태/컨트롤러 순수 로직(이동·방향·의도).
-- ZERO 의존 — 완전 자급. RNG 없음. deltaMilliseconds, pressedKeys/triggeredActions 는 모두 인자로 들어온다.
-- pressedDirections/triggeredActions 는 JS Set → 래퍼가 배열(Array.from)로 전달, Lua 는 멤버십 테이블로 본다.
-- 컨트롤러는 kind(keyboard/npc/lua) 판별 유니온 — kind 디스패치를 보존한다.
-- 0-based 인덱스/구멍 없음(이 모듈엔 sparse 배열 없음). 빈 배열은 json_array() 로 만든다.

local PLAYER_CHARACTER_ID = 'player'
local PLAYER_CHARACTER_APPEARANCE_TYPE = 'character_adventurer_brown_hair'
local DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND = 8

-- ── private 헬퍼 (TS 의 module-private 함수들을 Lua 로 재현) ──

local function clamp(value, minimum, maximum)
  return math.min(math.max(value, minimum), maximum)
end

-- 배열을 멤버십 집합 테이블로(set.has 흉내). 빈/없음이면 빈 테이블.
local function to_set(values)
  local set = {}
  if values ~= nil and values ~= json_null then
    for i = 1, #values do
      set[values[i]] = true
    end
  end
  return set
end

-- getNormalizedMovementVector: pressedDirections(집합)에서 정규화된 이동 벡터. 없으면 json_null.
-- Math.hypot(x,y) 는 x,y ∈ {-1,0,1} 범위에서 sqrt(x*x+y*y) 와 비트 동일하다.
local function get_normalized_movement_vector(pressed_set)
  local x = 0
  local y = 0
  if pressed_set['left'] then x = x - 1 end
  if pressed_set['right'] then x = x + 1 end
  if pressed_set['up'] then y = y - 1 end
  if pressed_set['down'] then y = y + 1 end
  if x == 0 and y == 0 then
    return json_null
  end
  local magnitude = math.sqrt(x * x + y * y)
  return { x = x / magnitude, y = y / magnitude }
end

-- getMovementDeltaFromDirection: 방향이 없으면 json_null, 있으면 거리(타일) 곱.
local function get_movement_delta_from_direction(direction, move_speed_tiles_per_second, delta_milliseconds)
  if direction == nil or direction == json_null then
    return json_null
  end
  local distance_in_tiles = (move_speed_tiles_per_second * delta_milliseconds) / 1000
  return {
    x = direction.x * distance_in_tiles,
    y = direction.y * distance_in_tiles
  }
end

-- getNpcControllerDelta: behavior 'idle' → 이동 없음(json_null).
local function get_npc_controller_delta(controller)
  if controller.behavior == 'idle' then
    return json_null
  end
  return json_null
end

-- createCharacterControllerIntent: 이동/액션이 모두 비면 nil(undefined), 아니면 intent 객체.
local function create_character_controller_intent(movement, actions)
  local has_movement = movement ~= nil and movement ~= json_null
  local has_actions = actions ~= nil and actions ~= json_null and #actions > 0
  if not has_movement and not has_actions then
    return json_null
  end
  local intent = {}
  if has_movement then
    intent.movement = movement
  end
  if has_actions then
    intent.actions = actions
  end
  return intent
end

-- getFacingFromDelta: delta 로부터 다음 facing.
local function get_facing_from_delta(current_facing, delta)
  if delta.x == 0 and delta.y == 0 then
    return current_facing
  end
  if math.abs(delta.x) >= math.abs(delta.y) and delta.x ~= 0 then
    if delta.x > 0 then return 'right' else return 'left' end
  end
  if delta.y ~= 0 then
    if delta.y > 0 then return 'down' else return 'up' end
  end
  return current_facing
end

-- 문자열 키 테이블 얕은 복사(TS spread).
local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- 문자열 trim(양끝 공백 제거 — JS String.prototype.trim 의 ASCII 공백 범위와 호환).
local function trim(s)
  return (s:gsub('^%s+', ''):gsub('%s+$', ''))
end

-- ── createKeyboardCharacterController ──
-- options: { moveSpeedTilesPerSecond? } (없으면 빈 객체로 들어옴). 기본 속도 적용.
function character_create_keyboard_controller(options)
  local opts = options
  if opts == nil or opts == json_null then opts = {} end
  local speed = opts.moveSpeedTilesPerSecond
  if speed == nil or speed == json_null then
    speed = DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND
  end
  return {
    kind = 'keyboard',
    moveSpeedTilesPerSecond = speed
  }
end

-- ── createIdleNpcCharacterController ──
-- dialogueLines 는 trim+빈줄 제거 후 비어있지 않을 때만 포함. messageDurationMilliseconds 는 정의됐을 때만 포함.
function character_create_idle_npc_controller(options)
  local opts = options
  if opts == nil or opts == json_null then opts = {} end
  local speed = opts.moveSpeedTilesPerSecond
  if speed == nil or speed == json_null then
    speed = DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND
  end

  local controller = {
    kind = 'npc',
    behavior = 'idle',
    moveSpeedTilesPerSecond = speed
  }

  local raw_lines = opts.dialogueLines
  if raw_lines ~= nil and raw_lines ~= json_null then
    local cleaned = json_array({})
    for i = 1, #raw_lines do
      local line = trim(raw_lines[i])
      if #line > 0 then
        cleaned[#cleaned + 1] = line
      end
    end
    if #cleaned > 0 then
      controller.dialogueLines = cleaned
    end
  end

  local duration = opts.messageDurationMilliseconds
  if duration ~= nil and duration ~= json_null then
    controller.messageDurationMilliseconds = duration
  end

  return controller
end

-- ── createLuaCharacterController ──
-- input: { scriptId, radiusInTiles, config?, moveSpeedTilesPerSecond? }. config 기본 {}.
function character_create_lua_controller(input)
  local speed = input.moveSpeedTilesPerSecond
  if speed == nil or speed == json_null then
    speed = DEFAULT_CHARACTER_MOVE_SPEED_TILES_PER_SECOND
  end
  local config = input.config
  if config == nil or config == json_null then
    config = {}
  end
  return {
    kind = 'lua',
    scriptId = input.scriptId,
    radiusInTiles = input.radiusInTiles,
    moveSpeedTilesPerSecond = speed,
    config = config
  }
end

-- ── createInitialPlayerCharacter ── 맵 중앙(floor) 에 player 캐릭터.
function character_create_initial_player(input)
  return {
    id = PLAYER_CHARACTER_ID,
    appearanceType = PLAYER_CHARACTER_APPEARANCE_TYPE,
    position = {
      x = math.floor(input.mapWidth / 2),
      y = math.floor(input.mapHeight / 2)
    },
    facing = 'down',
    collisionSize = { width = 1, height = 1 },
    blocksMovement = true,
    controller = character_create_keyboard_controller({})
  }
end

-- ── createNpcCharacter ──
-- facing 기본 'down', blocksMovement 기본 true, controller 기본 idle npc.
-- level/displayText 는 정의됐을 때만 포함.
function character_create_npc(input)
  local facing = input.facing
  if facing == nil or facing == json_null then facing = 'down' end

  local blocks = input.blocksMovement
  if blocks == nil or blocks == json_null then blocks = true end

  local controller = input.controller
  if controller == nil or controller == json_null then
    controller = character_create_idle_npc_controller({})
  end

  local character = {
    id = input.id,
    appearanceType = input.appearanceType,
    position = input.position,
    facing = facing,
    collisionSize = input.collisionSize,
    blocksMovement = blocks,
    controller = controller
  }

  if input.level ~= nil and input.level ~= json_null then
    character.level = input.level
  end
  if input.displayText ~= nil and input.displayText ~= json_null then
    character.displayText = input.displayText
  end

  return character
end

-- ── getCharacterMoveDirectionFromKey ── (미일치 시 json_null → 래퍼 undefined)
function character_move_direction_from_key(key)
  if key == 'ArrowUp' then return 'up' end
  if key == 'ArrowDown' then return 'down' end
  if key == 'ArrowLeft' then return 'left' end
  if key == 'ArrowRight' then return 'right' end
  return json_null
end

-- ── getCharacterActionFromKey ── (미일치 시 json_null → 래퍼 undefined)
function character_action_from_key(key)
  if key == 'Enter' or key == ' ' or key == 'Space' or key == 'Spacebar' then
    return 'interact'
  end
  return json_null
end

-- ── getCharacterControllerIntent ──
-- input: { character, deltaMilliseconds, pressedDirections?(array), triggeredActions?(array) }.
-- 컨트롤러 kind 로 디스패치. lua kind 는 항상 nil(undefined).
function character_controller_intent(input)
  local character = input.character
  local controller = character.controller
  local kind = controller.kind

  if kind == 'keyboard' then
    local pressed_set = to_set(input.pressedDirections)
    local direction = get_normalized_movement_vector(pressed_set)
    local movement = get_movement_delta_from_direction(
      direction,
      controller.moveSpeedTilesPerSecond,
      input.deltaMilliseconds
    )
    -- triggeredActions: size>0 일 때만 배열, 아니면 undefined. 순서는 입력 배열 그대로 유지.
    local raw_actions = input.triggeredActions
    local actions = json_null
    if raw_actions ~= nil and raw_actions ~= json_null and #raw_actions > 0 then
      actions = json_array(raw_actions)
    end
    return create_character_controller_intent(movement, actions)
  end

  if kind == 'npc' then
    local movement = get_npc_controller_delta(controller)
    return create_character_controller_intent(movement, json_null)
  end

  -- 'lua'
  return json_null
end

-- ── moveCharacterState ──
-- input: { character, delta, mapWidth, mapHeight }. 변화 없으면 원본 그대로 반환(TS 와 구조 동등).
function character_move_state(input)
  local character = input.character
  local delta = input.delta
  local next_facing = get_facing_from_delta(character.facing, delta)
  local clamped = {
    x = clamp(
      character.position.x + delta.x,
      0,
      input.mapWidth - character.collisionSize.width
    ),
    y = clamp(
      character.position.y + delta.y,
      0,
      input.mapHeight - character.collisionSize.height
    )
  }

  if clamped.x == character.position.x
    and clamped.y == character.position.y
    and next_facing == character.facing then
    return character
  end

  local out = shallow_copy(character)
  out.position = clamped
  out.facing = next_facing
  return out
end
