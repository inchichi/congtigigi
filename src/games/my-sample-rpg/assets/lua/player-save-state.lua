-- TS playerSaveState.ts → Lua. 순수 데이터 reshaping/검증.
-- serialize: {version=...} + 입력 필드 → json_encode.
-- normalize: 신뢰할 수 없는 값을 깊게 검증/정규화, 손상이면 json_null(=TS undefined) 반환.
-- parse: json_decode를 pcall로 감싸 실패 시 json_null(=TS undefined).
--
-- TS의 모든 가드(typeof/Number.isFinite/Array.isArray)·기본값·널 코얼레싱을 그대로 반영한다.
-- 슬롯 배열의 빈 슬롯(TS undefined → JSON null)은 json_null 센티넬로 왕복한다.
-- 결과의 undefined 필드(없는 값)는 단일 반환값일 때 json_null로 표현해 래퍼가 undefined로 되돌린다.

-- 버전은 TS 상수와 일치해야 한다(PLAYER_SAVE_STATE_VERSION = 1).
local SAVE_STATE_VERSION = 1

-- TS isRecord: 객체이며 null/배열이 아님.
-- Lua에선: 테이블이며, json_null 센티넬이 아니고, json_array(배열) 표시가 없어야 한다.
local function is_record(value)
  if type(value) ~= 'table' then
    return false
  end
  if value == json_null then
    return false
  end
  local mt = getmetatable(value)
  if mt and mt.__json == 'array' then
    return false
  end
  if mt and mt.__json == 'null' then
    return false
  end
  return true
end

-- TS Array.isArray: 디코더가 모든 배열에 __json='array' 표시를 단다.
local function is_array(value)
  if type(value) ~= 'table' then
    return false
  end
  local mt = getmetatable(value)
  return mt ~= nil and mt.__json == 'array'
end

-- TS isFiniteNumber: typeof number && Number.isFinite. NaN/Inf 제외.
local function is_finite_number(value)
  if type(value) ~= 'number' then
    return false
  end
  if value ~= value then -- NaN
    return false
  end
  if value == math.huge or value == -math.huge then
    return false
  end
  return true
end

-- TS isString.
local function is_string(value)
  return type(value) == 'string'
end

-- JSON에서 빈 슬롯(undefined→null)은 json_null로 디코드된다. TS의 `rawSlot == null` 판정.
local function is_nullish(value)
  return value == nil or value == json_null
end

local function normalize_resource(value)
  if not is_record(value) then
    return nil
  end
  if not is_finite_number(value.current) or not is_finite_number(value.max) then
    return nil
  end
  return { current = value.current, max = value.max }
end

local function normalize_experience(value)
  if not is_record(value) or not is_finite_number(value.current) then
    return nil
  end
  return { current = value.current }
end

local function normalize_stat_block(value)
  if not is_record(value) then
    return nil
  end
  local strength = value.strength
  local agility = value.agility
  local intelligence = value.intelligence
  local luck = value.luck
  if
    not is_finite_number(strength)
    or not is_finite_number(agility)
    or not is_finite_number(intelligence)
    or not is_finite_number(luck)
  then
    return nil
  end
  return {
    strength = strength,
    agility = agility,
    intelligence = intelligence,
    luck = luck
  }
end

local function normalize_skill_slot(value)
  if not is_record(value) then
    return nil
  end
  if
    not is_string(value.hotkey)
    or not is_string(value.label)
    or not is_string(value.description)
    or not is_finite_number(value.level)
    or not is_finite_number(value.maxLevel)
  then
    return nil
  end
  return {
    hotkey = value.hotkey,
    label = value.label,
    description = value.description,
    level = value.level,
    maxLevel = value.maxLevel
  }
end

local function normalize_profile(value)
  if not is_record(value) then
    return nil
  end
  if
    not is_string(value.name)
    or not is_string(value.job)
    or not is_finite_number(value.level)
    or not is_finite_number(value.statPoints)
    or not is_finite_number(value.availableSkillPoints)
    or not is_finite_number(value.totalSkillPointsEarned)
  then
    return nil
  end

  local experience = normalize_experience(value.experience)
  local hp = normalize_resource(value.hp)
  local mp = normalize_resource(value.mp)
  local stats = normalize_stat_block(value.stats)

  if
    experience == nil
    or hp == nil
    or mp == nil
    or stats == nil
    or not is_array(value.skills)
  then
    return nil
  end

  -- 스킬 슬롯은 고정 구성이라 하나라도 손상되면 저장본 전체를 폐기한다.
  local skills = json_array({})
  for i = 1, #value.skills do
    local skill = normalize_skill_slot(value.skills[i])
    if skill == nil then
      return nil
    end
    skills[i] = skill
  end

  return {
    name = value.name,
    job = value.job,
    level = value.level,
    experience = experience,
    statPoints = value.statPoints,
    availableSkillPoints = value.availableSkillPoints,
    totalSkillPointsEarned = value.totalSkillPointsEarned,
    hp = hp,
    mp = mp,
    stats = stats,
    skills = skills
  }
end

local function normalize_inventory_item(value)
  if not is_record(value) then
    return nil
  end
  if
    not is_string(value.id)
    or not is_string(value.label)
    or not is_finite_number(value.quantity)
  then
    return nil
  end
  return { id = value.id, label = value.label, quantity = value.quantity }
end

local function normalize_inventory(value)
  if
    not is_record(value)
    or not is_finite_number(value.gold)
    or not is_array(value.slots)
  then
    return nil
  end

  -- 빈 슬롯(null)·손상 아이템은 관대하게 빈 슬롯(json_null)으로 둔다.
  local slots = json_array({})
  for i = 1, #value.slots do
    local raw_slot = value.slots[i]
    if is_nullish(raw_slot) then
      slots[i] = json_null
    else
      local item = normalize_inventory_item(raw_slot)
      slots[i] = item == nil and json_null or item
    end
  end

  return { gold = value.gold, slots = slots }
end

local function normalize_equipment_item(value)
  if not is_record(value) then
    return nil
  end
  if
    not is_string(value.id)
    or not is_string(value.label)
    or not is_string(value.description)
    or not is_finite_number(value.level)
  then
    return nil
  end
  return {
    id = value.id,
    label = value.label,
    level = value.level,
    description = value.description
  }
end

local function normalize_equipment_slot(value)
  if not is_record(value) or not is_string(value.id) or not is_string(value.label) then
    return nil
  end

  -- item == null(빈 장비) 또는 손상 → undefined. 그 외 정규화.
  local item
  if is_nullish(value.item) then
    item = json_null
  else
    local normalized = normalize_equipment_item(value.item)
    item = normalized == nil and json_null or normalized
  end

  return {
    id = value.id,
    label = value.label,
    item = item
  }
end

local function normalize_equipment(value)
  if
    not is_record(value)
    or not is_string(value.setName)
    or not is_finite_number(value.level)
    or not is_array(value.slots)
  then
    return nil
  end

  -- 장비 슬롯은 고정 구성이라 하나라도 손상되면 폐기한다.
  local slots = json_array({})
  for i = 1, #value.slots do
    local slot = normalize_equipment_slot(value.slots[i])
    if slot == nil then
      return nil
    end
    slots[i] = slot
  end

  return { setName = value.setName, level = value.level, slots = slots }
end

local function normalize_quickslot_assignment(value)
  if not is_record(value) or not is_finite_number(value.inventorySlotIndex) then
    return nil
  end
  return { inventorySlotIndex = value.inventorySlotIndex }
end

local function normalize_quickslots(value)
  if not is_record(value) or not is_array(value.slots) then
    return nil
  end

  local slots = json_array({})
  for i = 1, #value.slots do
    local raw_slot = value.slots[i]
    if is_nullish(raw_slot) then
      slots[i] = json_null
    else
      local assignment = normalize_quickslot_assignment(raw_slot)
      slots[i] = assignment == nil and json_null or assignment
    end
  end

  return { slots = slots }
end

local function normalize_skill_slot_assignment(value)
  if not is_record(value) or not is_string(value.skillId) or #value.skillId == 0 then
    return nil
  end
  return { skillId = value.skillId }
end

local function normalize_skill_slots(value)
  if not is_record(value) or not is_array(value.slots) then
    return nil
  end

  local slots = json_array({})
  for i = 1, #value.slots do
    local raw_slot = value.slots[i]
    if is_nullish(raw_slot) then
      slots[i] = json_null
    else
      local assignment = normalize_skill_slot_assignment(raw_slot)
      slots[i] = assignment == nil and json_null or assignment
    end
  end

  return { slots = slots }
end

-- ── 공개 전역 함수 ──

-- serializePlayerSaveState: {version, ...input} → JSON 문자열.
-- input은 profile/equipment/inventory/quickslots/skillSlots를 가진 레코드(이미 디코드됨).
-- 슬롯의 빈 칸은 json_null로 들어와 그대로 null로 인코딩된다.
function save_state_serialize(input)
  local save_state = {
    version = SAVE_STATE_VERSION,
    profile = input.profile,
    equipment = input.equipment,
    inventory = input.inventory,
    quickslots = input.quickslots,
    skillSlots = input.skillSlots
  }
  return __json_encode(save_state)
end

-- normalizeStoredPlayerSaveState: 신뢰할 수 없는 값 → 검증된 저장 상태 또는 json_null(undefined).
function save_state_normalize(value)
  if not is_record(value) or value.version ~= SAVE_STATE_VERSION then
    return json_null
  end

  local profile = normalize_profile(value.profile)
  local equipment = normalize_equipment(value.equipment)
  local inventory = normalize_inventory(value.inventory)
  local quickslots = normalize_quickslots(value.quickslots)
  local skill_slots = normalize_skill_slots(value.skillSlots)

  if
    profile == nil
    or equipment == nil
    or inventory == nil
    or quickslots == nil
    or skill_slots == nil
  then
    return json_null
  end

  return {
    version = SAVE_STATE_VERSION,
    profile = profile,
    equipment = equipment,
    inventory = inventory,
    quickslots = quickslots,
    skillSlots = skill_slots
  }
end

-- parseStoredPlayerSaveState: 직렬화 문자열 → 검증된 상태. 파싱 실패/빈 입력 시 json_null(undefined).
-- TS의 `if (!raw)`는 ''(빈 문자열)·null·undefined를 모두 falsy로 본다.
function save_state_parse(raw)
  if is_nullish(raw) or raw == '' or raw == false then
    return json_null
  end
  if type(raw) ~= 'string' then
    -- TS는 string|null|undefined만 받지만, 방어적으로 비문자열은 파싱 실패로 처리.
    return json_null
  end

  local ok, decoded = pcall(__json_decode, raw)
  if not ok then
    return json_null
  end

  return save_state_normalize(decoded)
end
