-- TS playerQuickslots.ts → Lua. slots는 (assignment|undefined)[] — JSON에서 빈 슬롯은 null,
-- Lua에선 json_null 센티넬로 보존된다(배열 길이 유지). quickslot_index는 JS 0-based.

local function copy_slots(slots)
  local out = json_array({})
  for i = 1, #slots do
    out[i] = slots[i]
  end
  return out
end

-- createInitialPlayerQuickslots({ slotCount })
-- slot_count 개의 null 슬롯으로 초기화된 quickslots 반환.
function quickslots_create_initial(slot_count)
  local slots = json_array({})
  for i = 1, slot_count do
    slots[i] = json_null
  end
  return { slots = slots }
end

-- setPlayerQuickslotAssignment({ quickslots, quickslotIndex, inventorySlotIndex })
-- quickslot_index_0based 범위 검증, inventorySlotIndex < 0 검증, 중복 제거 후 할당.
function quickslots_set_assignment(quickslots, quickslot_index_0based, inventory_slot_index)
  if quickslot_index_0based < 0 or quickslot_index_0based >= #quickslots.slots then
    error('Invalid quickslot index ' .. quickslot_index_0based)
  end
  if inventory_slot_index < 0 then
    error('Invalid inventory slot index ' .. inventory_slot_index)
  end

  -- 같은 inventorySlotIndex 를 가진 다른 슬롯을 nil 로 지움(dedup).
  local slots = copy_slots(quickslots.slots)
  for i = 1, #slots do
    local slot = slots[i]
    if slot ~= json_null and slot ~= nil
       and slot.inventorySlotIndex == inventory_slot_index
       and (i - 1) ~= quickslot_index_0based then
      slots[i] = json_null
    end
  end

  slots[quickslot_index_0based + 1] = { inventorySlotIndex = inventory_slot_index }
  return { slots = slots }
end

-- clearPlayerQuickslotAssignment({ quickslots, quickslotIndex })
function quickslots_clear_assignment(quickslots, quickslot_index_0based)
  if quickslot_index_0based < 0 or quickslot_index_0based >= #quickslots.slots then
    error('Invalid quickslot index ' .. quickslot_index_0based)
  end

  local slots = copy_slots(quickslots.slots)
  slots[quickslot_index_0based + 1] = json_null
  return { slots = slots }
end

-- findPlayerQuickslotIndexByInventorySlotIndex(quickslots, inventorySlotIndex)
-- 0-based 인덱스 반환, 미발견 시 json_null.
function quickslots_find_by_inventory_slot(quickslots, inventory_slot_index)
  for i = 1, #quickslots.slots do
    local slot = quickslots.slots[i]
    if slot ~= json_null and slot ~= nil
       and slot.inventorySlotIndex == inventory_slot_index then
      return i - 1
    end
  end
  return json_null
end

-- getPlayerQuickslotAssignment(quickslots, quickslotIndex)
-- {inventorySlotIndex=...} 또는 json_null.
function quickslots_get_assignment(quickslots, quickslot_index_0based)
  local slot = quickslots.slots[quickslot_index_0based + 1]
  if slot == nil or slot == json_null then
    return json_null
  end
  return slot
end
