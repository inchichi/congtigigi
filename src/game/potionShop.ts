import {
  clearPlayerInventorySlot,
  findFirstEmptyPlayerInventorySlotIndex,
  setPlayerInventorySlot,
  type PlayerInventory,
  type PlayerInventoryItem
} from './playerInventory'
import { getPlayerEquipmentItemDefinitionById } from './playerEquipment'

export type PotionShopInventory = PlayerInventory

export type PotionShopTransactionResult = {
  ok: boolean
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  message: string
}

type CreateInitialPotionInventoryInput = {
  slotCount?: number
  gold?: number
}

type BuyPotionShopItemInput = {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  merchantSlotIndex: number
}

type SellPotionShopItemInput = {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  playerSlotIndex: number
}

type PotionShopItemDefinition = {
  id: 'health-potion' | 'mana-potion'
  label: string
  description: string
  price: number
}

const DEFAULT_POTION_INVENTORY_SLOT_COUNT = 12
const DEFAULT_POTION_INVENTORY_GOLD = 1_000_000_000_000_000
const POTION_SALE_RATIO = 0.5
const DEFAULT_POTION_STOCK_QUANTITY = 30
const POTION_ITEM_DEFINITIONS: PotionShopItemDefinition[] = [
  {
    id: 'health-potion',
    label: '체력 회복 포션',
    description: '체력을 회복하는 물약',
    price: 10
  },
  {
    id: 'mana-potion',
    label: '마나 회복 포션',
    description: '마나를 회복하는 물약',
    price: 15
  }
]
const POTION_INITIAL_STOCK_ITEM_IDS = POTION_ITEM_DEFINITIONS.map(
  (definition) => definition.id
)
const POTION_ITEM_DEFINITION_BY_ID: Map<string, PotionShopItemDefinition> = new Map(
  POTION_ITEM_DEFINITIONS.map((definition) => [definition.id, definition] as const)
)

export const createInitialPotionInventory = ({
  slotCount = DEFAULT_POTION_INVENTORY_SLOT_COUNT,
  gold = DEFAULT_POTION_INVENTORY_GOLD
}: CreateInitialPotionInventoryInput = {}): PotionShopInventory => {
  const slots: Array<PlayerInventoryItem | undefined> = Array.from(
    { length: slotCount },
    () => undefined
  )

  for (
    let index = 0;
    index < slotCount && index < POTION_INITIAL_STOCK_ITEM_IDS.length;
    index += 1
  ) {
    const itemDefinition = getPotionShopItemDefinitionById(
      POTION_INITIAL_STOCK_ITEM_IDS[index]
    )

    if (!itemDefinition) {
      throw new Error(
        `Missing potion stock item definition for ${POTION_INITIAL_STOCK_ITEM_IDS[index]}`
      )
    }

    slots[index] = createPlayerInventoryItemFromPotionDefinition(
      itemDefinition
    )
  }

  return {
    gold,
    slots
  }
}

export const getPotionShopItemDefinitionById = (
  itemId: string
): PotionShopItemDefinition | undefined => POTION_ITEM_DEFINITION_BY_ID.get(itemId)

export const getPotionShopBuyPriceById = (
  itemId: string
): number | undefined => getPotionShopItemDefinitionById(itemId)?.price

export const getPotionShopSellPriceById = (
  itemId: string
): number | undefined => {
  const buyPrice = getPotionShopTradeItemPriceById(itemId)

  if (buyPrice === undefined) {
    return undefined
  }

  return Math.max(1, Math.floor(buyPrice * POTION_SALE_RATIO))
}

export const buyPotionShopItem = ({
  playerInventory,
  merchantInventory,
  merchantSlotIndex
}: BuyPotionShopItemInput): PotionShopTransactionResult => {
  assertInventorySlotIndex(merchantInventory, merchantSlotIndex)

  const merchantItem = merchantInventory.slots[merchantSlotIndex]

  if (!merchantItem) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '재고가 없습니다.'
    })
  }

  const itemPrice = getPotionShopBuyPriceById(merchantItem.id)

  if (itemPrice === undefined) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '이 상점에서는 판매하지 않는 아이템입니다.'
    })
  }

  if (playerInventory.gold < itemPrice) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '돈이 부족합니다.'
    })
  }

  const emptyPlayerSlotIndex = findFirstEmptyPlayerInventorySlotIndex(
    playerInventory
  )

  if (emptyPlayerSlotIndex === undefined) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '인벤토리가 가득 찼습니다.'
    })
  }

  const nextPlayerInventory = withInventoryGold(
    setPlayerInventorySlot({
      inventory: playerInventory,
      slotIndex: emptyPlayerSlotIndex,
      item: clonePlayerInventoryItem(merchantItem)
    }),
    playerInventory.gold - itemPrice
  )
  const nextMerchantInventory = withInventoryGold(
    clearPlayerInventorySlot({
      inventory: merchantInventory,
      slotIndex: merchantSlotIndex
    }),
    merchantInventory.gold + itemPrice
  )

  return {
    ok: true,
    playerInventory: nextPlayerInventory,
    merchantInventory: nextMerchantInventory,
    message: '구매했습니다.'
  }
}

export const sellPotionShopItem = ({
  playerInventory,
  merchantInventory,
  playerSlotIndex
}: SellPotionShopItemInput): PotionShopTransactionResult => {
  assertInventorySlotIndex(playerInventory, playerSlotIndex)

  const playerItem = playerInventory.slots[playerSlotIndex]

  if (!playerItem) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '판매할 아이템이 없습니다.'
    })
  }

  const itemPrice = getPotionShopSellPriceById(playerItem.id)

  if (itemPrice === undefined) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '이 상점에서는 매입하지 않는 아이템입니다.'
    })
  }

  if (merchantInventory.gold < itemPrice) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '상점 보유금이 부족합니다.'
    })
  }

  const nextPlayerInventory = withInventoryGold(
    clearPlayerInventorySlot({
      inventory: playerInventory,
      slotIndex: playerSlotIndex
    }),
    playerInventory.gold + itemPrice
  )
  const nextMerchantInventory = withInventoryGold(
    merchantInventory,
    merchantInventory.gold - itemPrice
  )

  return {
    ok: true,
    playerInventory: nextPlayerInventory,
    merchantInventory: nextMerchantInventory,
    message: '판매했습니다.'
  }
}

const getPotionShopTradeItemPriceById = (
  itemId: string
): number | undefined =>
  getPotionShopItemDefinitionById(itemId)?.price ??
  getPlayerEquipmentItemDefinitionById(itemId)?.price

const createPlayerInventoryItemFromPotionDefinition = (
  definition: PotionShopItemDefinition
): PlayerInventoryItem => ({
  id: definition.id,
  label: definition.label,
  quantity: DEFAULT_POTION_STOCK_QUANTITY
})

const clonePlayerInventoryItem = (
  item: PlayerInventoryItem
): PlayerInventoryItem => ({
  id: item.id,
  label: item.label,
  quantity: item.quantity
})

const withInventoryGold = (
  inventory: PlayerInventory,
  gold: number
): PlayerInventory => ({
  ...inventory,
  gold
})

const createPotionShopTransactionFailure = ({
  playerInventory,
  merchantInventory,
  message
}: {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  message: string
}): PotionShopTransactionResult => ({
  ok: false,
  playerInventory,
  merchantInventory,
  message
})

const assertInventorySlotIndex = (
  inventory: PlayerInventory,
  slotIndex: number
) => {
  if (slotIndex < 0 || slotIndex >= inventory.slots.length) {
    throw new Error(`Invalid inventory slot index ${slotIndex}`)
  }
}
