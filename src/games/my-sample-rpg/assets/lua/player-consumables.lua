-- TS playerConsumables.ts → Lua. 소비 아이템(포션) 사용 로직.
-- inventory_* 전역은 같은 호스트에 로드된 player-inventory.lua 에서 옴.
-- quickslots_* 전역은 같은 호스트에 로드된 player-quickslots.lua 에서 옴.
-- 결과가 없으면 json_null 을 반환(래퍼에서 undefined 로 정규화).

local POTION_RESTORE_AMOUNT = 10

-- 문자열 키 테이블을 얕게 복사한다(TS 의 spread 패턴).
local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- consumeInventorySlot: quantity > 1 이면 set(quantity-1), 아니면 clear.
local function consume_inventory_slot(inventory, slot_index, item)
  if item.quantity > 1 then
    local reduced = shallow_copy(item)
    reduced.quantity = item.quantity - 1
    return inventory_set_slot(inventory, slot_index, reduced)
  else
    return inventory_clear_slot(inventory, slot_index)
  end
end

-- consumables_use_inventory(profile_json, inventory_json, slot_index)
-- → {profile=..., inventory=...} 또는 json_null
-- slot_index 는 0-based.
function consumables_use_inventory(profile, inventory, slot_index)
  local item = inventory.slots[slot_index + 1]
  if item == nil or item == json_null then
    return json_null
  end

  if item.id == 'health-potion' then
    local hp = profile.hp
    local next_hp = shallow_copy(hp)
    next_hp.current = math.min(hp.max, hp.current + POTION_RESTORE_AMOUNT)
    local next_profile = shallow_copy(profile)
    next_profile.hp = next_hp
    return {
      profile = next_profile,
      inventory = consume_inventory_slot(inventory, slot_index, item)
    }

  elseif item.id == 'mana-potion' then
    local mp = profile.mp
    local next_mp = shallow_copy(mp)
    next_mp.current = math.min(mp.max, mp.current + POTION_RESTORE_AMOUNT)
    local next_profile = shallow_copy(profile)
    next_profile.mp = next_mp
    return {
      profile = next_profile,
      inventory = consume_inventory_slot(inventory, slot_index, item)
    }

  else
    return json_null
  end
end

-- consumables_use_quickslot(profile_json, inventory_json, quickslots_json, quickslot_index)
-- → {profile=..., inventory=...} 또는 json_null
-- quickslot_index 는 0-based.
function consumables_use_quickslot(profile, inventory, quickslots, quickslot_index)
  local assignment = quickslots_get_assignment(quickslots, quickslot_index)
  if assignment == nil or assignment == json_null then
    return json_null
  end
  return consumables_use_inventory(profile, inventory, assignment.inventorySlotIndex)
end
