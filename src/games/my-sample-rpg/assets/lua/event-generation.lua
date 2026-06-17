-- TS eventGeneration.ts → Lua. 자급식(런타임 import 없음) 휴일-대사 이벤트 검증 + Tiled NPC 오브젝트 생성.
-- 모두 객체/배열 인자·반환이라 callJson 으로 마샬링한다.
--
-- 검증 규칙은 원본과 1:1 매핑:
--   .trim().length === 0   → trim 후 빈 문자열
--   /^[a-z0-9_]+$/u        → Lua 패턴 ^[%l%d_]+$ (순수 ASCII라 /u 는 무효, %l=a-z, %d=0-9)
--   Number.isFinite + <=0  → 유한수이며 0 초과
--   Array.every(non-empty) → 모든 요소가 trim 후 비지 않은 문자열
-- private 헬퍼 validateReward / isNonEmptyString 는 여기에 인라인으로 재구현한다.

local HOLIDAY_DIALOGUE_EVENT_TYPE = 'holiday_dialogue'
local TALK_EVENT_TRIGGER_TYPE = 'talk'
local SCENE_ENTER_EVENT_TRIGGER_TYPE = 'scene_enter'
local QUEST_START_EVENT_TRIGGER_TYPE = 'quest_start'
local QUEST_COMPLETE_EVENT_TRIGGER_TYPE = 'quest_complete'
local HOLIDAY_DIALOGUE_CONTROLLER_SCRIPT_ID = 'reply-with-message'

-- JS String.prototype.trim 의 ASCII 부분(공백/탭/개행/CR/FF/VT)을 양끝에서 제거한다.
local function trim(value)
  return (value:gsub('^[%s]*(.-)[%s]*$', '%1'))
end

-- JS .trim().length === 0 와 동치: 문자열이며 trim 후 비어 있지 않은지.
-- (값이 문자열이 아닐 가능성은 타입상 없지만 방어적으로 처리.)
local function is_non_empty_string(value)
  return type(value) == 'string' and #trim(value) > 0
end

-- JS Number.isFinite: number 타입이며 NaN/±Inf 가 아님.
-- JSON 은 NaN/Inf 를 표현 못 하므로(직렬화 시 null) JSON_NULL/비숫자는 비유한으로 본다.
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

-- /^[a-z0-9_]+$/ 와 동치: 1글자 이상, 모두 소문자/숫자/밑줄.
local function matches_event_name_pattern(value)
  if type(value) ~= 'string' then
    return false
  end
  return value:match('^[%l%d_]+$') ~= nil
end

-- private validateReward 재구현: reward.type 분기별로 errors 에 메시지 push.
local function validate_reward(reward, errors)
  local reward_type = reward.type
  if reward_type == 'none' then
    return
  elseif reward_type == 'item' then
    if #trim(reward.id) == 0 then
      errors[#errors + 1] = 'reward.id must not be empty when reward.type is "item"'
    end
    if not is_finite_number(reward.count) or reward.count <= 0 then
      errors[#errors + 1] = 'reward.count must be a positive finite number when reward.type is "item"'
    end
    return
  elseif reward_type == 'gold' or reward_type == 'experience' then
    if not is_finite_number(reward.count) or reward.count <= 0 then
      errors[#errors + 1] = 'reward.count must be a positive finite number when reward.type is "' .. reward_type .. '"'
    end
    return
  end
  -- TS switch 에 default 가 없어 알 수 없는 type 은 아무 것도 push 하지 않는다(동일 동작).
end

-- Array.every(isNonEmptyString) 와 동치: 모든 요소가 비지 않은 문자열인지.
local function every_non_empty_string(lines)
  for i = 1, #lines do
    if not is_non_empty_string(lines[i]) then
      return false
    end
  end
  return true
end

-- ── createHolidayDialogueEventValidationErrors ──
-- 원본의 push 순서를 정확히 보존한다. 결과는 항상 배열(빈 경우 [])이라 json_array 로 감싼다.
function event_generation_create_validation_errors(event)
  local errors = json_array({})

  if #trim(event.event_name) == 0 then
    errors[#errors + 1] = 'event_name must not be empty'
  end

  if not matches_event_name_pattern(event.event_name) then
    errors[#errors + 1] = 'event_name must use lowercase letters, numbers, and underscores only'
  end

  if event.event_type ~= HOLIDAY_DIALOGUE_EVENT_TYPE then
    errors[#errors + 1] = 'event_type must be "' .. HOLIDAY_DIALOGUE_EVENT_TYPE .. '"'
  end

  if #trim(event.npc.id) == 0 then
    errors[#errors + 1] = 'npc.id must not be empty'
  end

  if #trim(event.npc.display_name) == 0 then
    errors[#errors + 1] = 'npc.display_name must not be empty'
  end

  if #trim(event.npc.appearance_type) == 0 then
    errors[#errors + 1] = 'npc.appearance_type must not be empty'
  end

  local trigger_type = event.trigger.type
  if trigger_type ~= TALK_EVENT_TRIGGER_TYPE
    and trigger_type ~= SCENE_ENTER_EVENT_TRIGGER_TYPE
    and trigger_type ~= QUEST_START_EVENT_TRIGGER_TYPE
    and trigger_type ~= QUEST_COMPLETE_EVENT_TRIGGER_TYPE then
    errors[#errors + 1] = 'trigger.type must be a supported trigger type'
  end

  if #trim(event.trigger.target_scene) == 0 then
    errors[#errors + 1] = 'trigger.target_scene must not be empty'
  end

  if #event.dialogue.opening_lines == 0 then
    errors[#errors + 1] = 'dialogue.opening_lines must contain at least one line'
  end

  if not every_non_empty_string(event.dialogue.opening_lines) then
    errors[#errors + 1] = 'dialogue.opening_lines must contain non-empty strings'
  end

  if not every_non_empty_string(event.dialogue.active_lines) then
    errors[#errors + 1] = 'dialogue.active_lines must contain non-empty strings'
  end

  if not every_non_empty_string(event.dialogue.completion_lines) then
    errors[#errors + 1] = 'dialogue.completion_lines must contain non-empty strings'
  end

  if not is_finite_number(event.duration) or event.duration <= 0 then
    errors[#errors + 1] = 'duration must be a positive finite number'
  end

  validate_reward(event.reward, errors)

  return errors
end

-- ── isHolidayDialogueEventValid ──
function event_generation_is_event_valid(event)
  return #event_generation_create_validation_errors(event) == 0
end

-- ── createTiledNpcEventObject ──
-- 평면 객체 생성 + opening_lines 얕은 복사([...]). 빈 배열도 [] 로 나오게 json_array 로 복사.
function event_generation_create_tiled_npc_object(event)
  local dialogue_lines = json_array({})
  local opening = event.dialogue.opening_lines
  for i = 1, #opening do
    dialogue_lines[i] = opening[i]
  end

  return {
    name = event.npc.id,
    type = event.npc.appearance_type,
    properties = {
      displayText = event.npc.display_name,
      ['controller.scriptId'] = HOLIDAY_DIALOGUE_CONTROLLER_SCRIPT_ID,
      ['controller.dialogueLines'] = dialogue_lines,
      ['controller.messageDurationSeconds'] = event.duration
    }
  }
end
