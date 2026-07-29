-- quest-log.lua
-- TS questLog.ts → Lua. 퀘스트 정의(QUEST_DEFINITIONS)는 TS 소유 — quest_log_init()으로 주입.
-- QuestLogState JSON: { progressByQuestId: { "q001": { id, status, objectives:{id:count}, trackerVisible } } }

local _definitions = nil          -- quest_log_init()이 저장하는 배열
local _definitions_by_id = nil    -- id → definition 맵

-- ── 초기화 ──────────────────────────────────────────────────

function quest_log_init(definitions)
  _definitions = definitions
  _definitions_by_id = {}
  for i = 1, #_definitions do
    local def = _definitions[i]
    _definitions_by_id[def.id] = def
  end
  return true
end

-- ── 내부 헬퍼 ────────────────────────────────────────────────

local function get_definition(quest_id)
  local def = _definitions_by_id[quest_id]
  if not def then
    error('Unknown quest id ' .. tostring(quest_id))
  end
  return def
end

local function get_progress(quest_log, quest_id)
  local progress = quest_log.progressByQuestId[quest_id]
  if not progress then
    error('Unknown quest id ' .. tostring(quest_id))
  end
  return progress
end

local function copy_objectives(objectives)
  local out = {}
  for k, v in pairs(objectives) do
    out[k] = v
  end
  return out
end

local function copy_progress(progress)
  return {
    id             = progress.id,
    status         = progress.status,
    objectives     = copy_objectives(progress.objectives),
    trackerVisible = progress.trackerVisible
  }
end

local function copy_quest_log(quest_log)
  local out = {}
  for quest_id, progress in pairs(quest_log.progressByQuestId) do
    out[quest_id] = copy_progress(progress)
  end
  return { progressByQuestId = out }
end

local function update_quest_progress(quest_log, new_progress)
  local by_id = {}
  for quest_id, progress in pairs(quest_log.progressByQuestId) do
    by_id[quest_id] = progress
  end
  by_id[new_progress.id] = new_progress
  return { progressByQuestId = by_id }
end

local function create_initial_progress(def)
  local objectives = {}
  for i = 1, #def.objectives do
    objectives[def.objectives[i].id] = 0
  end
  return {
    id             = def.id,
    status         = 'not-started',
    objectives     = objectives,
    trackerVisible = false
  }
end

local function is_quest_unlocked(quest_log, quest_id)
  local def = get_definition(quest_id)
  for i = 1, #def.prerequisiteQuestIds do
    local prereq_id = def.prerequisiteQuestIds[i]
    if get_progress(quest_log, prereq_id).status ~= 'completed' then
      return false
    end
  end
  return true
end

local function are_objectives_complete(def, progress)
  for i = 1, #def.objectives do
    local obj = def.objectives[i]
    local current = progress.objectives[obj.id] or 0
    if current < obj.required then
      return false
    end
  end
  return true
end

local function record_objective(quest_log, quest_id, objective_id, amount)
  amount = amount or 1
  local quest = get_progress(quest_log, quest_id)
  if quest.status ~= 'active' then
    return quest_log
  end
  local def = get_definition(quest_id)
  -- find objective
  local objective = nil
  for i = 1, #def.objectives do
    if def.objectives[i].id == objective_id then
      objective = def.objectives[i]
      break
    end
  end
  if not objective then
    return quest_log
  end
  local current = quest.objectives[objective_id] or 0
  local floor_amount = math.floor(amount)
  if floor_amount < 0 then floor_amount = 0 end
  local next_current = math.min(objective.required, current + floor_amount)
  local next_objectives = copy_objectives(quest.objectives)
  next_objectives[objective_id] = next_current
  local next_quest = {
    id             = quest.id,
    status         = quest.status,
    objectives     = next_objectives,
    trackerVisible = quest.trackerVisible
  }
  if are_objectives_complete(def, next_quest) then
    next_quest.status = 'ready-to-turn-in'
  end
  return update_quest_progress(quest_log, next_quest)
end

local function record_matching_objective(quest_log, matches_fn)
  local next_log = quest_log
  for i = 1, #_definitions do
    local def = _definitions[i]
    local quest = get_progress(next_log, def.id)
    if quest.status == 'active' then
      for j = 1, #def.objectives do
        local objective = def.objectives[j]
        if matches_fn(objective) then
          next_log = record_objective(next_log, def.id, objective.id, 1)
        end
      end
    end
  end
  return next_log
end

local function get_definitions_for_npc(npc_id)
  local result = {}
  for i = 1, #_definitions do
    if _definitions[i].giverNpcId == npc_id then
      result[#result + 1] = _definitions[i]
    end
  end
  return result
end

-- ── 공개 함수 ─────────────────────────────────────────────────

function quest_log_create_initial()
  local by_id = {}
  for i = 1, #_definitions do
    local def = _definitions[i]
    by_id[def.id] = create_initial_progress(def)
  end
  return { progressByQuestId = by_id }
end

function quest_log_get_progress(quest_log, quest_id)
  return get_progress(quest_log, quest_id)
end

function quest_log_get_definition(quest_id)
  return get_definition(quest_id)
end

function quest_log_start(quest_log, quest_id)
  local quest = get_progress(quest_log, quest_id)
  if quest.status ~= 'not-started' or not is_quest_unlocked(quest_log, quest_id) then
    return quest_log
  end
  local new_progress = {
    id             = quest.id,
    status         = 'active',
    objectives     = copy_objectives(quest.objectives),
    trackerVisible = true
  }
  return update_quest_progress(quest_log, new_progress)
end

function quest_log_record_objective(quest_log, quest_id, objective_id, amount)
  return record_objective(quest_log, quest_id, objective_id, amount or 1)
end

function quest_log_record_monster_defeat(quest_log, scene_id, appearance_type)
  return record_matching_objective(quest_log, function(obj)
    return obj.type == 'monster-defeat'
      and obj.target.sceneId == scene_id
      and obj.target.appearanceType == appearance_type
  end)
end

function quest_log_record_item_use(quest_log, item_id)
  return record_matching_objective(quest_log, function(obj)
    return obj.type == 'item-use' and obj.target.itemId == item_id
  end)
end

function quest_log_record_shop_open(quest_log, shop_id)
  return record_matching_objective(quest_log, function(obj)
    return obj.type == 'shop-open' and obj.target.shopId == shop_id
  end)
end

function quest_log_record_scene_enter(quest_log, scene_id)
  return record_matching_objective(quest_log, function(obj)
    return obj.type == 'scene-enter' and obj.target.sceneId == scene_id
  end)
end

function quest_log_record_talk(quest_log, npc_id)
  return record_matching_objective(quest_log, function(obj)
    return obj.type == 'talk' and obj.target.npcId == npc_id
  end)
end

function quest_log_set_tracker_visible(quest_log, quest_id, tracker_visible)
  local quest = get_progress(quest_log, quest_id)
  if tracker_visible then
    if quest.status ~= 'active' and quest.status ~= 'ready-to-turn-in' then
      return quest_log
    end
  end
  if quest.trackerVisible == tracker_visible then
    return quest_log
  end
  local new_progress = {
    id             = quest.id,
    status         = quest.status,
    objectives     = copy_objectives(quest.objectives),
    trackerVisible = tracker_visible
  }
  return update_quest_progress(quest_log, new_progress)
end

function quest_log_hide_trackers(quest_log)
  local by_id = {}
  for quest_id, quest in pairs(quest_log.progressByQuestId) do
    by_id[quest_id] = {
      id             = quest.id,
      status         = quest.status,
      objectives     = copy_objectives(quest.objectives),
      trackerVisible = false
    }
  end
  return { progressByQuestId = by_id }
end

function quest_log_abandon(quest_log, quest_id)
  local def = get_definition(quest_id)
  return update_quest_progress(quest_log, create_initial_progress(def))
end

function quest_log_complete(quest_log, quest_id)
  local quest = get_progress(quest_log, quest_id)
  local def   = get_definition(quest_id)
  if quest.status ~= 'ready-to-turn-in' then
    return {
      nextQuestLog     = quest_log,
      didComplete      = false,
      goldReward       = 0,
      experienceReward = 0,
      itemRewards      = json_array({})
    }
  end
  local new_progress = {
    id             = quest.id,
    status         = 'completed',
    objectives     = copy_objectives(quest.objectives),
    trackerVisible = false
  }
  local next_log = update_quest_progress(quest_log, new_progress)
  -- clone item rewards
  local item_rewards = json_array({})
  local raw_items = def.rewards.items
  for i = 1, #raw_items do
    item_rewards[i] = {
      id       = raw_items[i].id,
      label    = raw_items[i].label,
      quantity = raw_items[i].quantity
    }
  end
  return {
    nextQuestLog     = next_log,
    didComplete      = true,
    goldReward       = def.rewards.gold,
    experienceReward = def.rewards.experience,
    itemRewards      = item_rewards
  }
end

function quest_log_get_visible_trackers(quest_log)
  local result = json_array({})
  for i = 1, #_definitions do
    local def   = _definitions[i]
    local quest = get_progress(quest_log, def.id)
    if quest.trackerVisible then
      if quest.status == 'active' then
        local objective = def.objectives[1]
        result[#result + 1] = {
          questId = def.id,
          text    = def.trackerLabel .. ' '
                      .. tostring(quest.objectives[objective.id] or 0)
                      .. '/' .. tostring(objective.required),
          -- TS와 동형: 트래커의 대상 이미지 링크용 활성 목표 정의를 함께 내보낸다.
          objective = objective
        }
      elseif quest.status == 'ready-to-turn-in' then
        local text
        if def.turnInTrackerText then
          text = def.turnInTrackerText
        else
          text = def.title .. ': ' .. def.giverName .. '\xEC\x97\x90\xEA\xB2\x8C \xEB\x8F\x8C\xEC\x95\x84\xEA\xB0\x80\xEA\xB8\xB0'
        end
        result[#result + 1] = { questId = def.id, text = text }
      end
    end
  end
  return result
end

function quest_log_get_npc_badge(quest_log, quest_id)
  local quest = get_progress(quest_log, quest_id)
  if quest.status == 'not-started' then
    if is_quest_unlocked(quest_log, quest_id) then
      return 'new'
    else
      return json_null
    end
  elseif quest.status == 'ready-to-turn-in' then
    return 'finish'
  else
    return json_null
  end
end

function quest_log_get_npc_badge_for_npc(quest_log, npc_id)
  local defs = get_definitions_for_npc(npc_id)
  -- finish takes priority
  for i = 1, #defs do
    if get_progress(quest_log, defs[i].id).status == 'ready-to-turn-in' then
      return 'finish'
    end
  end
  -- new: any not-started unlocked
  for i = 1, #defs do
    local quest = get_progress(quest_log, defs[i].id)
    if quest.status == 'not-started' and is_quest_unlocked(quest_log, defs[i].id) then
      return 'new'
    end
  end
  return json_null
end

function quest_log_get_next_interaction_for_npc(quest_log, npc_id)
  local defs = get_definitions_for_npc(npc_id)
  -- ready-to-turn-in → complete
  for i = 1, #defs do
    if get_progress(quest_log, defs[i].id).status == 'ready-to-turn-in' then
      local def = defs[i]
      return {
        questId    = def.id,
        action     = 'complete',
        definition = def,
        progress   = get_progress(quest_log, def.id)
      }
    end
  end
  -- active → active
  for i = 1, #defs do
    if get_progress(quest_log, defs[i].id).status == 'active' then
      local def = defs[i]
      return {
        questId    = def.id,
        action     = 'active',
        definition = def,
        progress   = get_progress(quest_log, def.id)
      }
    end
  end
  -- not-started unlocked → start
  for i = 1, #defs do
    local quest = get_progress(quest_log, defs[i].id)
    if quest.status == 'not-started' and is_quest_unlocked(quest_log, defs[i].id) then
      local def = defs[i]
      return {
        questId    = def.id,
        action     = 'start',
        definition = def,
        progress   = get_progress(quest_log, def.id)
      }
    end
  end
  return json_null
end

function quest_log_format_text(text, player_name)
  -- replace {playerName} with player_name
  -- Lua pattern: %{playerName%}
  return (text:gsub('{playerName}', player_name))
end

function quest_log_format_text_lines(lines, player_name)
  local out = json_array({})
  for i = 1, #lines do
    out[i] = (lines[i]:gsub('{playerName}', player_name))
  end
  return out
end
