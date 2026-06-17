-- TS playerInventory.ts → Lua. slots는 (item|undefined)[] — JSON에서 빈 슬롯은 null,
-- Lua에선 json_null 센티넬로 보존된다(배열 길이 유지). slot_index는 JS 0-based, Lua는 1-based.
-- 상수/스타터 아이템(데이터)은 TS에서 인자로 전달한다.

local function copy_slots(slots)
  local out = json_array({})
  for i = 1, #slots do
    out[i] = slots[i]
  end
  return out
end

local function copy_item(item)
  return { id = item.id, label = item.label, quantity = item.quantity }
end

function inventory_create(slot_count, gold, starter_items, default_slot_count)
  local resolved_count = slot_count
  local use_starter = false
  if slot_count == json_null then
    resolved_count = default_slot_count
    use_starter = true
  end

  local slots = json_array({})
  for i = 1, resolved_count do
    if use_starter and starter_items[i] ~= nil then
      slots[i] = copy_item(starter_items[i])
    else
      slots[i] = json_null
    end
  end

  return { gold = gold, slots = slots }
end

function inventory_set_slot(inventory, slot_index, item)
  if slot_index < 0 or slot_index >= #inventory.slots then
    error('Invalid inventory slot index ' .. slot_index)
  end
  local slots = copy_slots(inventory.slots)
  slots[slot_index + 1] = item
  return { gold = inventory.gold, slots = slots }
end

function inventory_clear_slot(inventory, slot_index)
  if slot_index < 0 or slot_index >= #inventory.slots then
    error('Invalid inventory slot index ' .. slot_index)
  end
  local slots = copy_slots(inventory.slots)
  slots[slot_index + 1] = json_null
  return { gold = inventory.gold, slots = slots }
end

function inventory_filled_slot_count(inventory)
  local count = 0
  for i = 1, #inventory.slots do
    if inventory.slots[i] ~= json_null then
      count = count + 1
    end
  end
  return count
end

function inventory_first_empty_index(inventory)
  for i = 1, #inventory.slots do
    if inventory.slots[i] == json_null then
      return i - 1
    end
  end
  return json_null
end
