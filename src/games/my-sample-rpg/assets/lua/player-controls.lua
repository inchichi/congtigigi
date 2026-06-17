-- player-controls.lua
-- TS playerControls.ts → Lua. 키 바인딩(코드) 매핑/표시/정규화 순수 로직.
-- 데이터 소유는 TS(단일 출처): PLAYER_CONTROL_BINDING_DEFINITIONS(배열, 순서 중요),
-- 코드→라벨 표시 맵은 player_controls_init()으로 주입한다(여기엔 복제하지 않는다).
-- characterState 의존은 타입 전용(이동방향/액션 문자열 유니온)이라 런타임 의존이 없다.

local _definitions = nil            -- player_controls_init()이 저장하는 정의 배열(순서 보존)
local _index_by_id = nil            -- id → 0-based 인덱스(TS PLAYER_CONTROL_BINDING_INDEX_BY_ID)
local _label_by_code = nil          -- 코드 → 표시 라벨 맵(TS PLAYER_CONTROL_BINDING_LABEL_BY_CODE)

-- ── 초기화 ──────────────────────────────────────────────────
-- definitions: { { id, label, defaultCode }, ... } (TS 배열 순서 그대로)
-- label_by_code: { ArrowUp = '↑', ... } 등 코드→표시 맵
function player_controls_init(definitions, label_by_code)
  _definitions = definitions
  _label_by_code = label_by_code
  _index_by_id = {}
  for i = 1, #_definitions do
    _index_by_id[_definitions[i].id] = i - 1
  end
  return true
end

-- ── 내부 헬퍼 ────────────────────────────────────────────────

-- TS: code.startsWith(prefix) — Lua 평탄 비교(plain find).
local function starts_with(s, prefix)
  return s:sub(1, #prefix) == prefix
end

-- TS findPlayerControlBindingIdByCode: bindings(객체)에서 코드 일치 첫 id, 미발견 nil.
-- bindings 는 createInitial 이 만든 순서(정의 배열 순서)를 따른다 — 정의 순서로 순회해 동일 보장.
local function find_binding_id_by_code(bindings, code)
  for i = 1, #_definitions do
    local id = _definitions[i].id
    if bindings[id] == code then
      return id
    end
  end
  return nil
end

-- ── createInitialPlayerControlBindings ──
function player_controls_create_initial()
  local out = {}
  for i = 1, #_definitions do
    local def = _definitions[i]
    out[def.id] = def.defaultCode
  end
  return out
end

-- ── normalizeStoredPlayerControlBindings ──
-- next_bindings 가 테이블(객체)이 아니면(=nil/json_null/숫자/문자열/불리언) 초기값 반환.
-- TS: !nextBindings || typeof !== 'object' → createInitial. JS 에선 배열도 object 라
-- 통과하지만, 배열이면 storedBindings[id] 가 없어 전부 defaultCode 가 되므로 동일하게 처리된다.
function player_controls_normalize_stored(next_bindings)
  if type(next_bindings) ~= 'table' or rawequal(next_bindings, json_null) then
    return player_controls_create_initial()
  end

  local normalized = player_controls_create_initial()
  local used_codes = {}

  for i = 1, #_definitions do
    local def = _definitions[i]
    local id = def.id
    local default_code = def.defaultCode
    local candidate = next_bindings[id]
    local binding_code
    if type(candidate) == 'string' and #candidate > 0 then
      binding_code = candidate
    else
      binding_code = default_code
    end
    local resolved_code
    if used_codes[binding_code] then
      resolved_code = default_code
    else
      resolved_code = binding_code
    end
    normalized[id] = resolved_code
    used_codes[resolved_code] = true
  end

  return normalized
end

-- ── setPlayerControlBinding ──
-- bindings 의 binding_id 를 next_code 로 바꾼다. 충돌(다른 id 가 next_code 사용 중)이면
-- 그 id 에 기존 코드를 넘겨 스왑한다. 변화 없으면(현재 코드 == next_code) 원본 반환.
function player_controls_set_binding(bindings, binding_id, next_code)
  local current_code = bindings[binding_id]

  if current_code == next_code then
    return bindings
  end

  local conflict_id = find_binding_id_by_code(bindings, next_code)

  local out = {}
  for k, v in pairs(bindings) do
    out[k] = v
  end
  out[binding_id] = next_code

  if conflict_id ~= nil and conflict_id ~= binding_id then
    out[conflict_id] = current_code
  end

  return out
end

-- ── getPlayerControlBindingDisplayText ──
-- 라벨 맵에 있으면 그 값. 없으면 Key/Digit/Numpad 접두 슬라이스, 그 외엔 코드 그대로.
function player_controls_get_display_text(code)
  local mapped = _label_by_code[code]
  if mapped ~= nil then
    return mapped
  end

  if starts_with(code, 'Key') then
    return code:sub(4)       -- TS slice(3): 0-based 3 → Lua 1-based 4
  end

  if starts_with(code, 'Digit') then
    return code:sub(6)       -- TS slice(5)
  end

  if starts_with(code, 'Numpad') then
    return 'Num ' .. code:sub(7)  -- TS `Num ${code.slice(6)}`
  end

  return code
end

-- ── getPlayerControlMovementDirectionFromCode ── (미발견 json_null)
function player_controls_get_movement_direction(bindings, code)
  if code == bindings['move-up'] then
    return 'up'
  end
  if code == bindings['move-down'] then
    return 'down'
  end
  if code == bindings['move-left'] then
    return 'left'
  end
  if code == bindings['move-right'] then
    return 'right'
  end
  return json_null
end

-- ── getPlayerControlActionFromCode ── (미발견 json_null)
function player_controls_get_action(bindings, code)
  if code == bindings['interact-enter'] or code == bindings['interact-space'] then
    return 'interact'
  end
  if code == bindings.attack then
    return 'attack'
  end
  return json_null
end

-- ── getPlayerControlQuickslotIndexFromCode ── (0-based index, 미발견 json_null)
function player_controls_get_quickslot_index(bindings, code)
  for index = 0, 5 do
    if code == bindings['quickslot-' .. (index + 1)] then
      return index
    end
  end
  return json_null
end

-- ── isPlayerControlPauseKey ──
function player_controls_is_pause_key(bindings, code)
  return code == bindings.pause
end

-- ── isPlayerControlCaptureModifierKey ──
function player_controls_is_capture_modifier_key(code)
  return code == 'ShiftLeft'
    or code == 'ShiftRight'
    or code == 'ControlLeft'
    or code == 'ControlRight'
    or code == 'AltLeft'
    or code == 'AltRight'
    or code == 'MetaLeft'
    or code == 'MetaRight'
end

-- ── getPlayerControlBindingDefinitionById ── (TS는 인덱스 맵으로 정의 반환)
function player_controls_get_definition_by_id(binding_id)
  local index = _index_by_id[binding_id]
  return _definitions[index + 1]
end

-- ── getPlayerControlBindingDefinitions ── (전체 정의 배열)
function player_controls_get_definitions()
  local out = json_array({})
  for i = 1, #_definitions do
    out[i] = _definitions[i]
  end
  return out
end

-- ── findPlayerControlBindingIdByCode ── (미발견 json_null)
function player_controls_find_binding_id_by_code(bindings, code)
  local id = find_binding_id_by_code(bindings, code)
  if id == nil then
    return json_null
  end
  return id
end
