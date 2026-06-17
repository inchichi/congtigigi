-- TS playerLoadout.ts → Lua. 장착/해제 로직.
-- equipment_* 전역은 같은 호스트에 로드된 player-equipment.lua 에서 옴.
-- inventory_* 전역은 같은 호스트에 로드된 player-inventory.lua 에서 옴.
-- 결과가 없으면 json_null 을 반환(래퍼에서 undefined 로 정규화).
-- 데이터 소유는 TS(단일 출처): 아이템 정의 배열·슬롯 id 배열·슬롯 라벨 맵·스타터 무기 정의는
-- 모두 호출 시 인자로 전달된다.

-- loadout_unequip_slot(equipment, inventory, slot_id, item_defs, slot_ids, slot_label_by_id, starter_weapon)
-- → {equipment=..., inventory=...} 또는 json_null
function loadout_unequip_slot(equipment, inventory, slot_id, item_defs, slot_ids, slot_label_by_id, starter_weapon)
  local equipment_slot = equipment_slot_by_id(equipment, slot_id)
  if equipment_slot == json_null then
    return json_null
  end
  local item = equipment_slot.item
  if item == nil or item == json_null then
    return json_null
  end

  local empty_slot_index = inventory_first_empty_index(inventory)
  if empty_slot_index == json_null then
    return json_null
  end

  local inv_item = { id = item.id, label = item.label, quantity = 1 }
  local next_inventory = inventory_set_slot(inventory, empty_slot_index, inv_item)
  local next_equipment = equipment_clear_slot(equipment, slot_id)

  return { equipment = next_equipment, inventory = next_inventory }
end

-- loadout_equip_inventory_slot(equipment, inventory, slot_index, item_defs, slot_ids, slot_label_by_id, starter_weapon)
-- → {equipment=..., inventory=...} 또는 json_null
-- slot_index 는 0-based.
function loadout_equip_inventory_slot(equipment, inventory, slot_index, item_defs, slot_ids, slot_label_by_id, starter_weapon)
  local item = inventory.slots[slot_index + 1]
  if item == nil or item == json_null then
    return json_null
  end

  local definition = equipment_item_definition_by_id(item_defs, item.id)
  if definition == json_null then
    return json_null
  end

  local eq_slot = equipment_slot_by_id(equipment, definition.slotId)
  if eq_slot == json_null then
    return json_null
  end

  local equipped_item = eq_slot.item  -- may be json_null or a table

  local next_equipment = equipment_set_slot(
    equipment,
    definition.slotId,
    equipment_create_item_from_definition(definition)
  )
  local next_inventory = inventory_clear_slot(inventory, slot_index)

  if equipped_item ~= nil and equipped_item ~= json_null then
    local swap_item = { id = equipped_item.id, label = equipped_item.label, quantity = 1 }
    next_inventory = inventory_set_slot(next_inventory, slot_index, swap_item)
  end

  return { equipment = next_equipment, inventory = next_inventory }
end
