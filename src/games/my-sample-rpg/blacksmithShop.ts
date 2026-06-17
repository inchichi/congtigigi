import {
  clearPlayerInventorySlot,
  findFirstEmptyPlayerInventorySlotIndex,
  setPlayerInventorySlot,
  type PlayerInventory,
  type PlayerInventoryItem
} from './playerInventory'
import {
  getPlayerEquipmentItemDefinitionById,
  type PlayerEquipmentItemDefinition
} from './playerEquipment'
import { getPotionShopItemDefinitionById } from './potionShop'

export type BlacksmithInventory = PlayerInventory

export type BlacksmithShopTransactionResult = {
  ok: boolean
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  message: string
}

type CreateInitialBlacksmithInventoryInput = {
  slotCount?: number
  gold?: number
}

type BuyBlacksmithShopItemInput = {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  merchantSlotIndex: number
}

type SellBlacksmithShopItemInput = {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  playerSlotIndex: number
}

const DEFAULT_BLACKSMITH_INVENTORY_GOLD = 1_000_000_000_000_000
const BLACKSMITH_SALE_RATIO = 0.5
const BLACKSMITH_INITIAL_STOCK_ITEM_IDS = [
  'basic-sword',
  'basic-armor',
  'basic-boots',
  'basic-charm',
  'bronze-sword',
  'iron-sword',
  'battle-axe',
  'long-spear',
  'quick-dagger',
  'spiked-mace',
  'magic-staff',
  'iron-armor',
  'Leather_Armor',
  'Leather_Helmet',
  'Chain_Armor',
  'Chain_Helmet',
  'Iron_Armor',
  'Iron_Helmet',
  'leather-boots',
  'smith-charm'
] as const
const DEFAULT_BLACKSMITH_INVENTORY_SLOT_COUNT =
  BLACKSMITH_INITIAL_STOCK_ITEM_IDS.length

export const createInitialBlacksmithInventory = ({
  slotCount = DEFAULT_BLACKSMITH_INVENTORY_SLOT_COUNT,
  gold = DEFAULT_BLACKSMITH_INVENTORY_GOLD
}: CreateInitialBlacksmithInventoryInput = {}): BlacksmithInventory => {
  const slots: Array<PlayerInventoryItem | undefined> = Array.from(
    { length: slotCount },
    () => undefined
  )

  for (
    let index = 0;
    index < slotCount && index < BLACKSMITH_INITIAL_STOCK_ITEM_IDS.length;
    index += 1
  ) {
    const itemDefinition = getPlayerEquipmentItemDefinitionById(
      BLACKSMITH_INITIAL_STOCK_ITEM_IDS[index]
    )

    if (!itemDefinition) {
      throw new Error(
        `Missing blacksmith stock item definition for ${BLACKSMITH_INITIAL_STOCK_ITEM_IDS[index]}`
      )
    }

    slots[index] = createPlayerInventoryItemFromEquipmentDefinition(
      itemDefinition
    )
  }

  return {
    gold,
    slots
  }
}

export const getBlacksmithShopItemPriceById = (
  itemId: string
): number | undefined => getPlayerEquipmentItemDefinitionById(itemId)?.price

export const getBlacksmithShopBuyPriceById = getBlacksmithShopItemPriceById

export const getBlacksmithShopSellPriceById = (
  itemId: string
): number | undefined => {
  const buyPrice = getBlacksmithShopTradeItemPriceById(itemId)

  if (buyPrice === undefined) {
    return undefined
  }

  return Math.max(1, Math.floor(buyPrice * BLACKSMITH_SALE_RATIO))
}

export const buyBlacksmithShopItem = ({
  playerInventory,
  merchantInventory,
  merchantSlotIndex
}: BuyBlacksmithShopItemInput): BlacksmithShopTransactionResult => {
  assertInventorySlotIndex(merchantInventory, merchantSlotIndex)

  const merchantItem = merchantInventory.slots[merchantSlotIndex]

  if (!merchantItem) {
    return createBlacksmithShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '재고가 없습니다.'
    })
  }

  const itemPrice = getBlacksmithShopItemPriceById(merchantItem.id)

  if (itemPrice === undefined) {
    return createBlacksmithShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '이 상점에서는 판매하지 않는 아이템입니다.'
    })
  }

  if (playerInventory.gold < itemPrice) {
    return createBlacksmithShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '돈이 부족합니다.'
    })
  }

  const emptyPlayerSlotIndex = findFirstEmptyPlayerInventorySlotIndex(
    playerInventory
  )

  if (emptyPlayerSlotIndex === undefined) {
    return createBlacksmithShopTransactionFailure({
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

export const sellBlacksmithShopItem = ({
  playerInventory,
  merchantInventory,
  playerSlotIndex
}: SellBlacksmithShopItemInput): BlacksmithShopTransactionResult => {
  assertInventorySlotIndex(playerInventory, playerSlotIndex)

  const playerItem = playerInventory.slots[playerSlotIndex]

  if (!playerItem) {
    return createBlacksmithShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '판매할 아이템이 없습니다.'
    })
  }

  const itemPrice = getBlacksmithShopSellPriceById(playerItem.id)

  if (itemPrice === undefined) {
    return createBlacksmithShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '이 상점에서는 매입하지 않는 아이템입니다.'
    })
  }

  if (merchantInventory.gold < itemPrice) {
    return createBlacksmithShopTransactionFailure({
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

export const getBlacksmithShopTradeItemPriceById = (
  itemId: string
): number | undefined =>
  getPlayerEquipmentItemDefinitionById(itemId)?.price ??
  getPotionShopItemDefinitionById(itemId)?.price

const createPlayerInventoryItemFromEquipmentDefinition = (
  definition: PlayerEquipmentItemDefinition
): PlayerInventoryItem => ({
  id: definition.id,
  label: definition.label,
  quantity: 1
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
  gold,
  slots: [...inventory.slots]
})

const createBlacksmithShopTransactionFailure = ({
  playerInventory,
  merchantInventory,
  message
}: {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  message: string
}): BlacksmithShopTransactionResult => ({
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
