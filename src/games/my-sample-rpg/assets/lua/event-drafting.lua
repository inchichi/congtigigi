-- 휴일 대사 이벤트 초안 생성 규칙 (TS eventDrafting.ts에서 Lua로 변환).
-- prompt(string) 하나를 받아 HolidayDialogueEventSpec 객체 또는 nil(=TS undefined)을 반환한다.
-- 객체/nil 반환이라 호스트는 callJson 으로 호출한다(__logic_call_json 디스패처).
--
-- HOLIDAY_DIALOGUE_EVENT_TYPE("holiday_dialogue")와 TALK_EVENT_TRIGGER_TYPE("talk")는
-- event-generation.lua 가 먼저 로드되며 전역으로 노출한다(표준 단독/퍼사드 경로).
--
-- TS: prompt.toLowerCase() 후 두 키워드 배열에 includes 매치.
-- Lua: string.lower + string.find(s, k, 1, true)(plain-find) — 이 키들에 대해 JS와 동일.

local function includes_any(value, keywords)
  for i = 1, #keywords do
    if string.find(value, keywords[i], 1, true) ~= nil then
      return true
    end
  end
  return false
end

local CHRISTMAS_KEYWORDS = { '크리스마스', 'christmas', 'xmas' }
local HALLOWEEN_KEYWORDS = { '할로윈', 'halloween' }

function create_holiday_dialogue_event_draft_from_text(prompt)
  local normalized_prompt = string.lower(prompt)

  if includes_any(normalized_prompt, CHRISTMAS_KEYWORDS) then
    return {
      event_name = 'christmas_event',
      event_type = HOLIDAY_DIALOGUE_EVENT_TYPE,
      npc = {
        id = 'santa',
        display_name = '산타',
        appearance_type = 'character_villager_brown_tunic'
      },
      trigger = {
        type = TALK_EVENT_TRIGGER_TYPE,
        target_scene = 'town'
      },
      dialogue = {
        opening_lines = json_array({ '메리 크리스마스!', '오늘은 특별한 날이란다.' }),
        active_lines = json_array({}),
        completion_lines = json_array({})
      },
      reward = {
        type = 'item',
        id = 'gift',
        count = 1
      },
      duration = 7
    }
  end

  if includes_any(normalized_prompt, HALLOWEEN_KEYWORDS) then
    return {
      event_name = 'halloween_event',
      event_type = HOLIDAY_DIALOGUE_EVENT_TYPE,
      npc = {
        id = 'halloween_npc',
        display_name = '할로윈 NPC',
        appearance_type = 'character_commoner_tan_tunic'
      },
      trigger = {
        type = TALK_EVENT_TRIGGER_TYPE,
        target_scene = 'town'
      },
      dialogue = {
        opening_lines = json_array({ '할로윈이야!', '사탕을 받아가!' }),
        active_lines = json_array({}),
        completion_lines = json_array({})
      },
      reward = {
        type = 'item',
        id = 'candy',
        count = 3
      },
      duration = 7
    }
  end

  return nil
end
