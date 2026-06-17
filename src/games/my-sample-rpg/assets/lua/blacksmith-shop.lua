-- TS blacksmithShop.ts → Lua. 대장장이 상점의 결정적 로직(구매/판매/가격/초기 재고)을 다룬다(JSON 마샬링).
-- 인벤토리 슬롯 조작은 같은 호스트에 로드된 player-inventory.lua 의 전역(inventory_*)을 호출한다.
-- 아이템 정의 조회는 player-equipment.lua 의 equipment_item_definition_by_id 를 호출한다.
-- 판매가 계산은 player-potion-shop.lua 의 potion_shop_item_definition_by_id 도 활용한다.
-- slots 의 빈 슬롯은 TS undefined → JSON null, Lua 에선 json_null 센티넬로 보존하고 래퍼가 undefined 로 정규화한다.
-- slot_index 는 JS 0-based, Lua slots 배열은 1-based.
--
-- 데이터 소유는 TS(단일 출처): 장비 정의 배열·포션 정의 배열·상수는 모두 호출 시 인자로 전달된다.

-- 문자열 키 테이블을 얕게 복사한다(TS 의 spread 패턴).
local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- withInventoryGold: { ...inventory, gold } (얕은 복사 후 gold 교체).
local function with_inventory_gold(inventory, gold)
  local out = shallow_copy(inventory)
  out.gold = gold
  return out
end

local function transaction_failure(player_inventory, merchant_inventory, message)
  return {
    ok = false,
    playerInventory = player_inventory,
    merchantInventory = merchant_inventory,
    message = message
  }
end

-- assertInventorySlotIndex: 범위 밖이면 TS 와 동일 메시지로 error.
local function assert_slot_index(inventory, slot_index)
  if slot_index < 0 or slot_index >= #inventory.slots then
    error('Invalid inventory slot index ' .. slot_index)
  end
end

-- ── blacksmith_create_initial ──
-- equipment_item_defs_json: 장비 정의 배열(JSON 마샬링됨), initial_stock_ids_json: 초기 재고 id 배열.
-- slot_count 개의 슬롯을 만들고 앞에서부터 initial_stock_ids 순서대로 장비 아이템을 채운다.
function blacksmith_create_initial(slot_count, gold, equipment_item_defs_json, initial_stock_ids_json)
  local slots = json_array({})
  for i = 1, slot_count do
    slots[i] = json_null
  end

  local fill_count = math.min(slot_count, #initial_stock_ids_json)
  for i = 1, fill_count do
    local item_id = initial_stock_ids_json[i]
    local def = equipment_item_definition_by_id(equipment_item_defs_json, item_id)
    if def == json_null then
      error('Missing blacksmith stock item definition for ' .. item_id)
    end
    slots[i] = { id = def.id, label = def.label, quantity = 1 }
  end

  return { gold = gold, slots = slots }
end

-- ── blacksmith_buy_price ──
-- 장비 정의에서 price 를 반환한다(미발견 시 json_null).
function blacksmith_buy_price(equipment_item_defs_json, item_id)
  local def = equipment_item_definition_by_id(equipment_item_defs_json, item_id)
  if def == json_null then
    return json_null
  end
  return def.price
end

-- blacksmith_trade_item_price: 장비 정의 price ?? 포션 정의 price (둘 다 없으면 json_null)
local function blacksmith_trade_item_price(equipment_item_defs_json, potion_item_defs_json, item_id)
  local eq_def = equipment_item_definition_by_id(equipment_item_defs_json, item_id)
  if eq_def ~= json_null then
    return eq_def.price
  end
  local pot_def = potion_shop_item_definition_by_id(potion_item_defs_json, item_id)
  if pot_def ~= json_null then
    return pot_def.price
  end
  return json_null
end

-- ── blacksmith_sell_price ──
-- max(1, floor(trade_price * sale_ratio)), 미발견 시 json_null.
function blacksmith_sell_price(equipment_item_defs_json, potion_item_defs_json, item_id, sale_ratio)
  local price = blacksmith_trade_item_price(equipment_item_defs_json, potion_item_defs_json, item_id)
  if price == json_null then
    return json_null
  end
  return math.max(1, math.floor(price * sale_ratio))
end

-- ── blacksmith_buy ──
-- 플레이어가 상인의 merchant_slot_index 슬롯 아이템을 구매한다(수량 1).
-- 성공: 플레이어 골드 -= price, 플레이어 인벤토리에 아이템 추가; 상인 슬롯 비움, 골드 += price.
function blacksmith_buy(
  player_inv_json,
  merchant_inv_json,
  merchant_slot_index,
  equipment_item_defs_json
)
  assert_slot_index(merchant_inv_json, merchant_slot_index)

  local merchant_item = merchant_inv_json.slots[merchant_slot_index + 1]
  if merchant_item == nil or merchant_item == json_null then
    return transaction_failure(player_inv_json, merchant_inv_json, '재고가 없습니다.')
  end

  local item_price = blacksmith_buy_price(equipment_item_defs_json, merchant_item.id)
  if item_price == json_null then
    return transaction_failure(
      player_inv_json,
      merchant_inv_json,
      '이 상점에서는 판매하지 않는 아이템입니다.'
    )
  end

  if player_inv_json.gold < item_price then
    return transaction_failure(player_inv_json, merchant_inv_json, '돈이 부족합니다.')
  end

  local empty_player_slot_index = inventory_first_empty_index(player_inv_json)
  if empty_player_slot_index == json_null then
    return transaction_failure(player_inv_json, merchant_inv_json, '인벤토리가 가득 찼습니다.')
  end

  local cloned_item = { id = merchant_item.id, label = merchant_item.label, quantity = merchant_item.quantity }
  local next_player_inventory = with_inventory_gold(
    inventory_set_slot(player_inv_json, empty_player_slot_index, cloned_item),
    player_inv_json.gold - item_price
  )
  local next_merchant_inventory = with_inventory_gold(
    inventory_clear_slot(merchant_inv_json, merchant_slot_index),
    merchant_inv_json.gold + item_price
  )

  return {
    ok = true,
    playerInventory = next_player_inventory,
    merchantInventory = next_merchant_inventory,
    message = '구매했습니다.'
  }
end

-- ── blacksmith_sell ──
-- 플레이어의 player_slot_index 슬롯 아이템을 상인에게 판매한다.
-- 성공: 플레이어 슬롯 비움, 골드 += price; 상인 골드 -= price (상인은 아이템 인벤토리에 추가 안 함).
function blacksmith_sell(
  player_inv_json,
  merchant_inv_json,
  player_slot_index,
  equipment_item_defs_json,
  potion_item_defs_json,
  sale_ratio
)
  assert_slot_index(player_inv_json, player_slot_index)

  local player_item = player_inv_json.slots[player_slot_index + 1]
  if player_item == nil or player_item == json_null then
    return transaction_failure(player_inv_json, merchant_inv_json, '판매할 아이템이 없습니다.')
  end

  local item_price = blacksmith_sell_price(
    equipment_item_defs_json,
    potion_item_defs_json,
    player_item.id,
    sale_ratio
  )
  if item_price == json_null then
    return transaction_failure(
      player_inv_json,
      merchant_inv_json,
      '이 상점에서는 매입하지 않는 아이템입니다.'
    )
  end

  if merchant_inv_json.gold < item_price then
    return transaction_failure(player_inv_json, merchant_inv_json, '상점 보유금이 부족합니다.')
  end

  local next_player_inventory = with_inventory_gold(
    inventory_clear_slot(player_inv_json, player_slot_index),
    player_inv_json.gold + item_price
  )
  local next_merchant_inventory = with_inventory_gold(
    merchant_inv_json,
    merchant_inv_json.gold - item_price
  )

  return {
    ok = true,
    playerInventory = next_player_inventory,
    merchantInventory = next_merchant_inventory,
    message = '판매했습니다.'
  }
end
