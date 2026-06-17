-- TS potionShop.ts → Lua. 포션 상점의 결정적 로직(구매/판매/가격/초기 재고)을 다룬다(JSON 마샬링).
-- 인벤토리 슬롯 조작은 같은 호스트에 로드된 player-inventory.lua 의 전역(inventory_*)을 호출한다.
-- slots 의 빈 슬롯은 TS undefined → JSON null, Lua 에선 json_null 센티넬로 보존하고 래퍼가 undefined 로 정규화한다.
-- slot_index 는 JS 0-based, Lua slots 배열은 1-based.
--
-- 데이터 소유는 TS(단일 출처): 포션 정의 배열·장비 정의 배열·상수는 모두 호출 시 인자로 전달된다
-- (여기엔 큰 데이터 테이블을 복제하지 않는다).

-- 문자열 키 테이블을 얕게 복사한다(TS 의 spread 패턴).
local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- ── getPotionShopItemDefinitionById ── (포션 정의 배열에서 id 일치 1건, 미발견 시 json_null)
function potion_shop_item_definition_by_id(definitions, item_id)
  for i = 1, #definitions do
    if definitions[i].id == item_id then
      return definitions[i]
    end
  end
  return json_null
end

-- ── getPotionShopBuyPriceById ── (포션 정의의 price, 미발견 시 json_null)
function potion_shop_buy_price_by_id(definitions, item_id)
  local definition = potion_shop_item_definition_by_id(definitions, item_id)
  if definition == json_null then
    return json_null
  end
  return definition.price
end

-- getPotionShopTradeItemPriceById: 포션 정의 price ?? 장비 정의 price (둘 다 없으면 json_null)
local function potion_shop_trade_item_price_by_id(potion_definitions, equipment_definitions, item_id)
  local potion_definition = potion_shop_item_definition_by_id(potion_definitions, item_id)
  if potion_definition ~= json_null then
    return potion_definition.price
  end
  for i = 1, #equipment_definitions do
    if equipment_definitions[i].id == item_id then
      return equipment_definitions[i].price
    end
  end
  return json_null
end

-- ── getPotionShopSellPriceById ── max(1, floor(trade_price * sale_ratio)), 미발견 시 json_null
function potion_shop_sell_price_by_id(potion_definitions, equipment_definitions, item_id, sale_ratio)
  local buy_price = potion_shop_trade_item_price_by_id(
    potion_definitions,
    equipment_definitions,
    item_id
  )
  if buy_price == json_null then
    return json_null
  end
  return math.max(1, math.floor(buy_price * sale_ratio))
end

-- ── createInitialPotionInventory ──
-- 슬롯 수만큼 빈 슬롯(json_null)을 만들고, 앞에서부터 포션 정의 순서대로 초기 재고를 채운다
-- (POTION_INITIAL_STOCK_ITEM_IDS == 정의들의 id 순서이므로 정의 i 를 그대로 사용).
function potion_shop_create_initial(slot_count, gold, potion_definitions, default_stock_quantity)
  local slots = json_array({})
  for i = 1, slot_count do
    slots[i] = json_null
  end
  local stock_count = #potion_definitions
  for i = 1, slot_count do
    if i > stock_count then
      break
    end
    local definition = potion_definitions[i]
    slots[i] = {
      id = definition.id,
      label = definition.label,
      quantity = default_stock_quantity
    }
  end
  return { gold = gold, slots = slots }
end

-- resolveTradeQuantity: requested ?? default, 정수 && > 0 이면 그 값, 아니면 nil(사용처에서 실패 처리).
local function resolve_trade_quantity(requested_quantity, default_quantity)
  local quantity = requested_quantity
  if quantity == json_null then
    quantity = default_quantity
  end
  if math.type(quantity) == 'integer' and quantity > 0 then
    return quantity
  end
  -- 정수값 float 도 Number.isInteger 와 동일하게 정수로 취급.
  if type(quantity) == 'number' and quantity == math.floor(quantity)
    and quantity > 0 and quantity ~= math.huge then
    return math.floor(quantity)
  end
  return nil
end

-- findPlayerInventoryStackSlotIndexByItemId: 같은 id 의 첫 슬롯(0-based), 미발견 시 nil.
local function find_stack_slot_index_by_item_id(inventory, item_id)
  for i = 1, #inventory.slots do
    local slot = inventory.slots[i]
    if slot ~= json_null and slot.id == item_id then
      return i - 1
    end
  end
  return nil
end

-- assertInventorySlotIndex: 범위 밖이면 TS 와 동일 메시지로 error.
local function assert_slot_index(inventory, slot_index)
  if slot_index < 0 or slot_index >= #inventory.slots then
    error('Invalid inventory slot index ' .. slot_index)
  end
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

-- ── buyPotionShopItem ── (구매가는 포션 정의만 사용 → 장비 정의 불필요)
function potion_shop_buy(
  player_inventory,
  merchant_inventory,
  merchant_slot_index,
  requested_quantity,
  potion_definitions
)
  assert_slot_index(merchant_inventory, merchant_slot_index)

  local merchant_item = merchant_inventory.slots[merchant_slot_index + 1]
  if merchant_item == nil or merchant_item == json_null then
    return transaction_failure(player_inventory, merchant_inventory, '재고가 없습니다.')
  end

  local item_price = potion_shop_buy_price_by_id(potion_definitions, merchant_item.id)
  if item_price == json_null then
    return transaction_failure(
      player_inventory,
      merchant_inventory,
      '이 상점에서는 판매하지 않는 아이템입니다.'
    )
  end

  local quantity = resolve_trade_quantity(requested_quantity, 1)
  if quantity == nil then
    return transaction_failure(
      player_inventory,
      merchant_inventory,
      '구매 수량은 1개 이상의 정수여야 합니다.'
    )
  end

  if merchant_item.quantity < quantity then
    return transaction_failure(player_inventory, merchant_inventory, '재고가 부족합니다.')
  end

  local total_price = item_price * quantity
  if player_inventory.gold < total_price then
    return transaction_failure(player_inventory, merchant_inventory, '돈이 부족합니다.')
  end

  local player_slot_index = find_stack_slot_index_by_item_id(player_inventory, merchant_item.id)
  if player_slot_index == nil then
    local empty_index = inventory_first_empty_index(player_inventory)
    if empty_index ~= json_null then
      player_slot_index = empty_index
    end
  end

  if player_slot_index == nil then
    return transaction_failure(
      player_inventory,
      merchant_inventory,
      '인벤토리가 가득 찼습니다.'
    )
  end

  local existing_player_item = player_inventory.slots[player_slot_index + 1]
  local next_player_item_quantity = quantity
  if existing_player_item ~= nil and existing_player_item ~= json_null then
    next_player_item_quantity = existing_player_item.quantity + quantity
  end

  local next_player_inventory = with_inventory_gold(
    inventory_set_slot(player_inventory, player_slot_index, {
      id = merchant_item.id,
      label = merchant_item.label,
      quantity = next_player_item_quantity
    }),
    player_inventory.gold - total_price
  )

  local next_merchant_base
  if merchant_item.quantity == quantity then
    next_merchant_base = inventory_clear_slot(merchant_inventory, merchant_slot_index)
  else
    local reduced_item = shallow_copy(merchant_item)
    reduced_item.quantity = merchant_item.quantity - quantity
    next_merchant_base = inventory_set_slot(merchant_inventory, merchant_slot_index, reduced_item)
  end
  local next_merchant_inventory = with_inventory_gold(
    next_merchant_base,
    merchant_inventory.gold + total_price
  )

  return {
    ok = true,
    playerInventory = next_player_inventory,
    merchantInventory = next_merchant_inventory,
    message = quantity .. '개를 구매했습니다.'
  }
end

-- ── sellPotionShopItem ──
function potion_shop_sell(
  player_inventory,
  merchant_inventory,
  player_slot_index,
  requested_quantity,
  potion_definitions,
  equipment_definitions,
  sale_ratio
)
  assert_slot_index(player_inventory, player_slot_index)

  local player_item = player_inventory.slots[player_slot_index + 1]
  if player_item == nil or player_item == json_null then
    return transaction_failure(
      player_inventory,
      merchant_inventory,
      '판매할 아이템이 없습니다.'
    )
  end

  local item_price = potion_shop_sell_price_by_id(
    potion_definitions,
    equipment_definitions,
    player_item.id,
    sale_ratio
  )
  if item_price == json_null then
    return transaction_failure(
      player_inventory,
      merchant_inventory,
      '이 상점에서는 매입하지 않는 아이템입니다.'
    )
  end

  local quantity = resolve_trade_quantity(requested_quantity, player_item.quantity)
  if quantity == nil then
    return transaction_failure(
      player_inventory,
      merchant_inventory,
      '판매 수량은 1개 이상의 정수여야 합니다.'
    )
  end

  if player_item.quantity < quantity then
    return transaction_failure(
      player_inventory,
      merchant_inventory,
      '보유 수량보다 많이 판매할 수 없습니다.'
    )
  end

  local total_price = item_price * quantity
  if merchant_inventory.gold < total_price then
    return transaction_failure(
      player_inventory,
      merchant_inventory,
      '상점 보유금이 부족합니다.'
    )
  end

  local next_player_base
  if quantity == player_item.quantity then
    next_player_base = inventory_clear_slot(player_inventory, player_slot_index)
  else
    local reduced_item = shallow_copy(player_item)
    reduced_item.quantity = player_item.quantity - quantity
    next_player_base = inventory_set_slot(player_inventory, player_slot_index, reduced_item)
  end
  local next_player_inventory = with_inventory_gold(
    next_player_base,
    player_inventory.gold + total_price
  )

  local next_merchant_inventory = with_inventory_gold(
    merchant_inventory,
    merchant_inventory.gold - total_price
  )

  return {
    ok = true,
    playerInventory = next_player_inventory,
    merchantInventory = next_merchant_inventory,
    message = quantity .. '개를 판매했습니다.'
  }
end
