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
  quantity?: number
}

type SellPotionShopItemInput = {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  playerSlotIndex: number
  quantity?: number
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
const DEFAULT_POTION_STOCK_QUANTITY = 999_999
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
  merchantSlotIndex,
  quantity: requestedQuantity
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

  const quantity = resolveTradeQuantity({
    requestedQuantity,
    defaultQuantity: 1
  })

  if (quantity === undefined) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '구매 수량은 1개 이상의 정수여야 합니다.'
    })
  }

  if (merchantItem.quantity < quantity) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '재고가 부족합니다.'
    })
  }

  const totalPrice = itemPrice * quantity

  if (playerInventory.gold < totalPrice) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '돈이 부족합니다.'
    })
  }

  const playerInventorySlotIndex =
    findPlayerInventoryStackSlotIndexByItemId(playerInventory, merchantItem.id) ??
    findFirstEmptyPlayerInventorySlotIndex(playerInventory)

  if (playerInventorySlotIndex === undefined) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '인벤토리가 가득 찼습니다.'
    })
  }

  const existingPlayerItem = playerInventory.slots[playerInventorySlotIndex]
  const nextPlayerItemQuantity = existingPlayerItem
    ? existingPlayerItem.quantity + quantity
    : quantity
  const nextPlayerInventory = withInventoryGold(
    setPlayerInventorySlot({
      inventory: playerInventory,
      slotIndex: playerInventorySlotIndex,
      item: {
        id: merchantItem.id,
        label: merchantItem.label,
        quantity: nextPlayerItemQuantity
      }
    }),
    playerInventory.gold - totalPrice
  )
  const nextMerchantInventory = withInventoryGold(
    merchantItem.quantity === quantity
      ? clearPlayerInventorySlot({
          inventory: merchantInventory,
          slotIndex: merchantSlotIndex
        })
      : setPlayerInventorySlot({
          inventory: merchantInventory,
          slotIndex: merchantSlotIndex,
          item: {
            ...merchantItem,
            quantity: merchantItem.quantity - quantity
          }
        }),
    merchantInventory.gold + totalPrice
  )

  return {
    ok: true,
    playerInventory: nextPlayerInventory,
    merchantInventory: nextMerchantInventory,
    message: `${quantity}개를 구매했습니다.`
  }
}

export const sellPotionShopItem = ({
  playerInventory,
  merchantInventory,
  playerSlotIndex,
  quantity: requestedQuantity
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

  const quantity = resolveTradeQuantity({
    requestedQuantity,
    defaultQuantity: playerItem.quantity
  })

  if (quantity === undefined) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '판매 수량은 1개 이상의 정수여야 합니다.'
    })
  }

  if (playerItem.quantity < quantity) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '보유 수량보다 많이 판매할 수 없습니다.'
    })
  }

  const totalPrice = itemPrice * quantity

  if (merchantInventory.gold < totalPrice) {
    return createPotionShopTransactionFailure({
      playerInventory,
      merchantInventory,
      message: '상점 보유금이 부족합니다.'
    })
  }

  const nextPlayerInventory = withInventoryGold(
    quantity === playerItem.quantity
      ? clearPlayerInventorySlot({
          inventory: playerInventory,
          slotIndex: playerSlotIndex
        })
      : setPlayerInventorySlot({
          inventory: playerInventory,
          slotIndex: playerSlotIndex,
          item: {
            ...playerItem,
            quantity: playerItem.quantity - quantity
          }
        }),
    playerInventory.gold + totalPrice
  )
  const nextMerchantInventory = withInventoryGold(
    merchantInventory,
    merchantInventory.gold - totalPrice
  )

  return {
    ok: true,
    playerInventory: nextPlayerInventory,
    merchantInventory: nextMerchantInventory,
    message: `${quantity}개를 판매했습니다.`
  }
}

const getPotionShopTradeItemPriceById = (
  itemId: string
): number | undefined =>
  getPotionShopItemDefinitionById(itemId)?.price ??
  getPlayerEquipmentItemDefinitionById(itemId)?.price

const createPlayerInventoryItemFromPotionDefinition = (
  definition: PotionShopItemDefinition,
  quantity = DEFAULT_POTION_STOCK_QUANTITY
): PlayerInventoryItem => ({
  id: definition.id,
  label: definition.label,
  quantity
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

const resolveTradeQuantity = ({
  requestedQuantity,
  defaultQuantity
}: {
  requestedQuantity: number | undefined
  defaultQuantity: number
}): number | undefined => {
  const quantity = requestedQuantity ?? defaultQuantity

  return Number.isInteger(quantity) && quantity > 0 ? quantity : undefined
}

const findPlayerInventoryStackSlotIndexByItemId = (
  inventory: PlayerInventory,
  itemId: string
): number | undefined => {
  const slotIndex = inventory.slots.findIndex((slot) => slot?.id === itemId)

  return slotIndex < 0 ? undefined : slotIndex
}

const assertInventorySlotIndex = (
  inventory: PlayerInventory,
  slotIndex: number
) => {
  if (slotIndex < 0 || slotIndex >= inventory.slots.length) {
    throw new Error(`Invalid inventory slot index ${slotIndex}`)
  }
}
