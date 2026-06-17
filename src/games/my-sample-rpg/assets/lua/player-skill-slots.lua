-- TS playerSkillSlots.ts → Lua. slots는 (assignment|undefined)[] — JSON에서 빈 슬롯은 null,
-- Lua에선 json_null 센티넬로 보존된다(배열 길이 유지). slot_index는 JS 0-based.

local function copy_slots(slots)
  local out = json_array({})
  for i = 1, #slots do
    out[i] = slots[i]
  end
  return out
end

-- createInitialPlayerSkillSlots({ slotCount, initialSkillIds })
-- initial_skill_ids_json: 배열(빈 배열도 가능). slots[i]= {skillId=...} if defined, else json_null.
function skill_slots_create_initial(slot_count, initial_skill_ids_json)
  local slots = json_array({})
  for i = 1, slot_count do
    local skill_id = initial_skill_ids_json[i]
    if skill_id ~= nil and skill_id ~= json_null and skill_id ~= '' then
      slots[i] = { skillId = skill_id }
    else
      slots[i] = json_null
    end
  end
  return { slots = slots }
end

-- setPlayerSkillSlotAssignment({ skillSlots, skillSlotIndex, skillId })
-- throws if out of range or skillId empty; deduplicates.
function skill_slots_set_assignment(skill_slots, slot_index_0based, skill_id)
  if slot_index_0based < 0 or slot_index_0based >= #skill_slots.slots then
    error('Invalid skill slot index ' .. slot_index_0based)
  end
  if skill_id == nil or skill_id == '' then
    error('Invalid skill id')
  end

  -- 같은 skillId 를 가진 다른 슬롯을 json_null 로 지움(dedup).
  local slots = copy_slots(skill_slots.slots)
  for i = 1, #slots do
    local slot = slots[i]
    if slot ~= json_null and slot ~= nil
       and slot.skillId == skill_id
       and (i - 1) ~= slot_index_0based then
      slots[i] = json_null
    end
  end

  slots[slot_index_0based + 1] = { skillId = skill_id }
  return { slots = slots }
end

-- clearPlayerSkillSlotAssignment({ skillSlots, skillSlotIndex })
function skill_slots_clear_assignment(skill_slots, slot_index_0based)
  if slot_index_0based < 0 or slot_index_0based >= #skill_slots.slots then
    error('Invalid skill slot index ' .. slot_index_0based)
  end

  local slots = copy_slots(skill_slots.slots)
  slots[slot_index_0based + 1] = json_null
  return { slots = slots }
end

-- findPlayerSkillSlotIndexBySkillId(skillSlots, skillId)
-- 0-based 인덱스 반환, 미발견 시 json_null.
function skill_slots_find_by_skill_id(skill_slots, skill_id)
  for i = 1, #skill_slots.slots do
    local slot = skill_slots.slots[i]
    if slot ~= json_null and slot ~= nil and slot.skillId == skill_id then
      return i - 1
    end
  end
  return json_null
end

-- getPlayerSkillSlotAssignment(skillSlots, skillSlotIndex)
-- {skillId=...} 또는 json_null.
function skill_slots_get_assignment(skill_slots, slot_index_0based)
  local slot = skill_slots.slots[slot_index_0based + 1]
  if slot == nil or slot == json_null then
    return json_null
  end
  return slot
end

-- getPlayerSkillSlotIndexFromCode(hotkeyCodesJson, code)
-- hotkey_codes_json: 배열(예: ["KeyQ","KeyW","KeyE","KeyR"]).
-- 0-based 인덱스 반환, 미발견 시 json_null.
function skill_slots_index_from_code(hotkey_codes_json, code)
  for i = 1, #hotkey_codes_json do
    if hotkey_codes_json[i] == code then
      return i - 1
    end
  end
  return json_null
end
