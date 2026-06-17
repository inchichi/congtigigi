-- TS playerEquipment.ts → Lua. 장비 슬롯(weapon/armor/hat/boots/accessory) 객체를 다룬다(JSON 마샬링).
-- 빈 슬롯의 item 은 TS에서 undefined → JSON null, Lua에선 json_null 센티넬로 보존하고 래퍼가 undefined로 정규화한다.
-- slot_index 는 JS 0-based, Lua slots 배열은 1-based. findIndex 미발견은 TS와 동일하게 -1을 반환한다.
--
-- 데이터 소유는 TS(단일 출처): 슬롯 라벨 맵, 슬롯 id 순서, 스타터 무기 정의, 아이템 정의 배열은
-- 모두 호출 시 인자로 전달된다(여기엔 큰 데이터 테이블을 복제하지 않는다).

-- profile/slot 처럼 문자열 키 테이블을 얕게 복사한다(TS의 spread 패턴).
local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- slots 시퀀스를 새 배열로 복사한다(요소는 그대로 — TS의 [...slots]).
local function copy_slots(slots)
  local out = json_array({})
  for i = 1, #slots do
    out[i] = slots[i]
  end
  return out
end

-- ── createPlayerEquipmentItemFromDefinition ──
-- definition 에서 PlayerEquipmentItem 으로 추리는 순수 변환.
function equipment_create_item_from_definition(definition)
  return {
    id = definition.id,
    label = definition.label,
    level = definition.level,
    description = definition.description
  }
end

-- ── createInitialPlayerEquipment ──
-- slot_ids: 슬롯 id 순서(배열), slot_label_by_id: id→라벨 맵(객체), starter_weapon: 스타터 무기 정의.
-- weapon 슬롯만 스타터 무기 아이템, 나머지는 빈 슬롯(json_null → 래퍼에서 undefined).
function equipment_create_initial(slot_ids, slot_label_by_id, starter_weapon)
  local slots = json_array({})
  for i = 1, #slot_ids do
    local slot_id = slot_ids[i]
    local item
    if slot_id == 'weapon' then
      item = equipment_create_item_from_definition(starter_weapon)
    else
      item = json_null
    end
    slots[i] = {
      id = slot_id,
      label = slot_label_by_id[slot_id],
      item = item
    }
  end

  return { setName = '기본 장비', level = 1, slots = slots }
end

-- ── getPlayerEquipmentSlotLabelById ──
function equipment_slot_label_by_id(slot_label_by_id, slot_id)
  return slot_label_by_id[slot_id]
end

-- ── getPlayerEquipmentItemDefinitionById ── (정의 배열에서 id 일치 1건, 미발견 시 json_null)
function equipment_item_definition_by_id(definitions, item_id)
  for i = 1, #definitions do
    if definitions[i].id == item_id then
      return definitions[i]
    end
  end
  return json_null
end

-- ── getPlayerEquipmentItemDefinitionBySlotId ──
-- TS는 slotId 별 첫 등장 정의를 맵에 담는다 → 배열 순서상 slotId 가 처음 나오는 정의를 반환.
function equipment_item_definition_by_slot_id(definitions, slot_id)
  for i = 1, #definitions do
    if definitions[i].slotId == slot_id then
      return definitions[i]
    end
  end
  return json_null
end

-- ── findPlayerEquipmentSlotIndexById ── (TS findIndex: 미발견 -1, 발견 시 0-based)
function equipment_find_slot_index_by_id(equipment, slot_id)
  for i = 1, #equipment.slots do
    if equipment.slots[i].id == slot_id then
      return i - 1
    end
  end
  return -1
end

-- ── getPlayerEquipmentSlotById ── (미발견 시 json_null)
function equipment_slot_by_id(equipment, slot_id)
  for i = 1, #equipment.slots do
    if equipment.slots[i].id == slot_id then
      return equipment.slots[i]
    end
  end
  return json_null
end

-- ── setPlayerEquipmentSlot ── 대상 슬롯의 item 을 새 item 으로 교체(슬롯 다른 키 보존).
function equipment_set_slot(equipment, slot_id, item)
  local slot_index = equipment_find_slot_index_by_id(equipment, slot_id)

  if slot_index < 0 then
    error('Invalid equipment slot id ' .. slot_id)
  end

  local slots = copy_slots(equipment.slots)
  local next_slot = shallow_copy(slots[slot_index + 1])
  next_slot.item = item
  slots[slot_index + 1] = next_slot

  local out = shallow_copy(equipment)
  out.slots = slots
  return out
end

-- ── clearPlayerEquipmentSlot ── 대상 슬롯의 item 을 비운다(json_null → 래퍼에서 undefined).
function equipment_clear_slot(equipment, slot_id)
  local slot_index = equipment_find_slot_index_by_id(equipment, slot_id)

  if slot_index < 0 then
    error('Invalid equipment slot id ' .. slot_id)
  end

  local slots = copy_slots(equipment.slots)
  local next_slot = shallow_copy(slots[slot_index + 1])
  next_slot.item = json_null
  slots[slot_index + 1] = next_slot

  local out = shallow_copy(equipment)
  out.slots = slots
  return out
end
