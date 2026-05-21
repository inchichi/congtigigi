import {
  buyPotionShopItem,
  getPotionShopBuyPriceById,
  getPotionShopItemDefinitionById,
  getPotionShopSellPriceById,
  sellPotionShopItem
} from '../game/potionShop'
import {
  getPlayerEquipmentItemDefinitionById,
  getPlayerEquipmentSlotLabelById
} from '../game/playerEquipment'
import {
  findFirstEmptyPlayerInventorySlotIndex,
  getPlayerInventoryFilledSlotCount
} from '../game/playerInventory'
import type { PlayerInventory } from '../game/playerInventory'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreatePotionShopOverlayInput = {
  mountElement: HTMLElement
  getPlayerInventory: () => PlayerInventory
  getMerchantInventory: () => PlayerInventory
  getIsOpen: () => boolean
  onRequestOpenChange: (isOpen: boolean) => void
  onRequestTradeStateChange: (
    nextPlayerInventory: PlayerInventory,
    nextMerchantInventory: PlayerInventory
  ) => void
}

export type PotionShopOverlay = {
  syncFrame: () => void
  destroy: () => void
}

type TradeMode = 'buy' | 'sell'

type TradeRowElements = {
  button: HTMLButtonElement
  icon: HTMLSpanElement
  content: HTMLDivElement
  name: HTMLDivElement
  meta: HTMLDivElement
  price: HTMLDivElement
}

type TradePaneElements = {
  root: HTMLElement
  header: HTMLDivElement
  title: HTMLDivElement
  summary: HTMLDivElement
  list: HTMLDivElement
  emptyState: HTMLDivElement
  rows: TradeRowElements[]
  sync: () => void
}

const TINY_DUNGEON_TILESET_IMAGE_URL = new URL(
  '../assets/tilesets/tiny-dungeon-16.png',
  import.meta.url
).href
const TINY_DUNGEON_TILESET_WIDTH = 192
const TINY_DUNGEON_TILESET_HEIGHT = 176
const POTION_ICON_FRAME_BY_ID = {
  'health-potion': {
    imageUrl: TINY_DUNGEON_TILESET_IMAGE_URL,
    imageWidth: TINY_DUNGEON_TILESET_WIDTH,
    imageHeight: TINY_DUNGEON_TILESET_HEIGHT,
    frame: {
      x: 112,
      y: 144,
      width: 16,
      height: 16
    }
  },
  'mana-potion': {
    imageUrl: TINY_DUNGEON_TILESET_IMAGE_URL,
    imageWidth: TINY_DUNGEON_TILESET_WIDTH,
    imageHeight: TINY_DUNGEON_TILESET_HEIGHT,
    frame: {
      x: 128,
      y: 144,
      width: 16,
      height: 16
    }
  }
} as const
const PLAYER_PORTRAIT_FRAME = {
  x: 32,
  y: 128,
  width: 16,
  height: 16
}
const POTION_MERCHANT_PORTRAIT_FRAME = {
  x: 48,
  y: 128,
  width: 16,
  height: 16
}
const POISON_SHOP_ROW_ICON_SCALE = 0.85
const PORTRAIT_SCALE = 3

export const createPotionShopOverlay = ({
  mountElement,
  getPlayerInventory,
  getMerchantInventory,
  getIsOpen,
  onRequestOpenChange,
  onRequestTradeStateChange
}: CreatePotionShopOverlayInput): PotionShopOverlay => {
  const overlayRoot = document.createElement('div')
  const backdropButton = document.createElement('button')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const header = document.createElement('div')
  const merchantCard = document.createElement('div')
  const merchantPortrait = document.createElement('span')
  const merchantCardText = document.createElement('div')
  const merchantName = document.createElement('div')
  const merchantGold = document.createElement('div')
  const centerCard = document.createElement('div')
  const titleElement = document.createElement('div')
  const subtitleElement = document.createElement('div')
  const closeButton = document.createElement('button')
  const playerCard = document.createElement('div')
  const playerPortrait = document.createElement('span')
  const playerCardText = document.createElement('div')
  const playerName = document.createElement('div')
  const playerGold = document.createElement('div')
  const panes = document.createElement('div')
  const statusElement = document.createElement('div')
  const footerElement = document.createElement('div')
  const merchantPane = createTradePane({
    kind: 'buy',
    title: '구매',
    subtitle: '상점 재고',
    getInventory: getMerchantInventory,
    getCounterInventory: getPlayerInventory,
    onRowClick: handleMerchantRowClick
  })
  const playerPane = createTradePane({
    kind: 'sell',
    title: '판매',
    subtitle: '내 인벤토리',
    getInventory: getPlayerInventory,
    getCounterInventory: getMerchantInventory,
    onRowClick: handlePlayerRowClick
  })
  let statusMessage: string | undefined
  let wasOpen = false

  overlayRoot.className = 'blacksmith-shop-overlay potion-shop-overlay'

  backdropButton.type = 'button'
  backdropButton.className = 'blacksmith-shop-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.tabIndex = -1
  backdropButton.setAttribute('aria-hidden', 'true')

  panel.className = 'blacksmith-shop-overlay__panel blacksmith-shop-overlay__panel--shop'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'false')
  panel.setAttribute('aria-labelledby', 'potion-shop-title')

  panelBody.className = 'blacksmith-shop-overlay__panel-body'
  header.className = 'blacksmith-shop-overlay__header'

  merchantCard.className = 'blacksmith-shop-overlay__portrait-card'
  merchantPortrait.className = 'blacksmith-shop-overlay__portrait'
  merchantCardText.className = 'blacksmith-shop-overlay__portrait-text'
  merchantName.className = 'blacksmith-shop-overlay__portrait-name'
  merchantName.textContent = '물약상인'
  merchantGold.className = 'blacksmith-shop-overlay__portrait-gold'
  merchantGold.hidden = true
  merchantCardText.append(merchantName, merchantGold)
  merchantCard.append(merchantPortrait, merchantCardText)

  centerCard.className = 'blacksmith-shop-overlay__center-card'
  titleElement.id = 'potion-shop-title'
  titleElement.className = 'blacksmith-shop-overlay__title'
  titleElement.textContent = '물약 상점'
  subtitleElement.className = 'blacksmith-shop-overlay__subtitle'
  subtitleElement.textContent = '왼쪽은 구매, 오른쪽은 판매'

  closeButton.type = 'button'
  closeButton.className = 'blacksmith-shop-overlay__close'
  closeButton.hidden = true
  closeButton.setAttribute('aria-label', '물약 상점 닫기')
  closeButton.title = '거래 닫기 (Esc)'
  closeButton.textContent = '×'

  centerCard.append(titleElement, subtitleElement)

  playerCard.className = 'blacksmith-shop-overlay__portrait-card'
  playerPortrait.className = 'blacksmith-shop-overlay__portrait'
  playerCardText.className = 'blacksmith-shop-overlay__portrait-text'
  playerName.className = 'blacksmith-shop-overlay__portrait-name'
  playerName.textContent = '플레이어'
  playerGold.className = 'blacksmith-shop-overlay__portrait-gold'
  playerCardText.append(playerName, playerGold)
  playerCard.append(playerPortrait, playerCardText)

  panes.className = 'blacksmith-shop-overlay__panes'
  statusElement.className = 'blacksmith-shop-overlay__status'
  statusElement.textContent = '물약을 클릭해서 사고팔 수 있습니다'
  footerElement.className = 'blacksmith-shop-overlay__footer'
  footerElement.textContent = 'Esc로 닫기'

  setPortraitFrame(merchantPortrait, POTION_MERCHANT_PORTRAIT_FRAME, PORTRAIT_SCALE)
  setPortraitFrame(playerPortrait, PLAYER_PORTRAIT_FRAME, PORTRAIT_SCALE)

  header.append(merchantCard, centerCard, playerCard)
  panes.append(merchantPane.root, playerPane.root)
  panelBody.append(header, panes, statusElement, footerElement)
  panel.append(panelBody)
  overlayRoot.append(backdropButton, panel)
  mountElement.append(overlayRoot)

  function syncLayout() {
    const isOpen = getIsOpen()
    const playerInventory = getPlayerInventory()

    if (isOpen && !wasOpen) {
      statusMessage = undefined
    }

    overlayRoot.hidden = !isOpen
    overlayRoot.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      backdropButton.hidden = true
      panel.hidden = true
      closeButton.hidden = true
      statusMessage = undefined
      wasOpen = false
      return
    }

    backdropButton.hidden = false
    closeButton.hidden = false
    panel.hidden = false
    panel.style.transformOrigin = 'center center'
    panel.style.transform = `translate(-50%, -50%) scale(${getResponsiveUiScale()})`
    panel.style.backgroundImage = 'none'
    panel.style.backgroundRepeat = 'no-repeat'
    panel.style.backgroundPosition = '0 0'
    panel.style.backgroundSize = '100% 100%'
    panel.style.width = 'max-content'
    panel.style.height = 'max-content'
    panelBody.style.width = 'max-content'
    panelBody.style.height = 'max-content'

    titleElement.textContent = '물약 상점'
    subtitleElement.textContent = '왼쪽은 구매, 오른쪽은 판매'
    statusElement.textContent =
      statusMessage ?? '체력 물약과 마나 물약을 준비해뒀어.'
    footerElement.textContent = 'Esc로 닫기'

    playerGold.textContent = `${formatGoldAmount(playerInventory.gold)}`
    merchantPane.sync()
    playerPane.sync()
    wasOpen = true
  }

  function handleMerchantRowClick(slotIndex: number) {
    const nextState = buyPotionShopItem({
      playerInventory: getPlayerInventory(),
      merchantInventory: getMerchantInventory(),
      merchantSlotIndex: slotIndex
    })

    statusMessage = nextState.message
    onRequestTradeStateChange(
      nextState.playerInventory,
      nextState.merchantInventory
    )
    syncLayout()
  }

  function handlePlayerRowClick(slotIndex: number) {
    const nextState = sellPotionShopItem({
      playerInventory: getPlayerInventory(),
      merchantInventory: getMerchantInventory(),
      playerSlotIndex: slotIndex
    })

    statusMessage = nextState.message
    onRequestTradeStateChange(
      nextState.playerInventory,
      nextState.merchantInventory
    )
    syncLayout()
  }

  function handleCloseButtonClick(event: MouseEvent) {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  function handleCloseButtonPointerDown(event: PointerEvent) {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  function handleBackdropClick(event: MouseEvent) {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  closeButton.addEventListener('click', handleCloseButtonClick)
  closeButton.addEventListener('pointerdown', handleCloseButtonPointerDown)
  backdropButton.addEventListener('click', handleBackdropClick)

  const destroy = () => {
    closeButton.removeEventListener('click', handleCloseButtonClick)
    closeButton.removeEventListener('pointerdown', handleCloseButtonPointerDown)
    backdropButton.removeEventListener('click', handleBackdropClick)
    overlayRoot.remove()
  }

  syncLayout()

  return {
    syncFrame: syncLayout,
    destroy
  }

  function createTradePane({
    kind,
    title,
    subtitle,
    getInventory,
    getCounterInventory,
    onRowClick
  }: {
    kind: TradeMode
    title: string
    subtitle: string
    getInventory: () => PlayerInventory
    getCounterInventory: () => PlayerInventory
    onRowClick: (slotIndex: number) => void
  }): TradePaneElements {
    const root = document.createElement('section')
    const header = document.createElement('div')
    const titleElement = document.createElement('div')
    const summaryElement = document.createElement('div')
    const list = document.createElement('div')
    const emptyState = document.createElement('div')
    const rows: TradeRowElements[] = []

    root.className = `blacksmith-shop-overlay__pane blacksmith-shop-overlay__pane--${kind}`
    header.className = 'blacksmith-shop-overlay__pane-header'
    titleElement.className = 'blacksmith-shop-overlay__pane-title'
    titleElement.textContent = title
    summaryElement.className = 'blacksmith-shop-overlay__pane-summary'
    summaryElement.textContent = subtitle
    list.className = 'blacksmith-shop-overlay__list'
    emptyState.className = 'blacksmith-shop-overlay__empty-state'
    emptyState.textContent =
      kind === 'buy'
        ? '상점 재고가 비어 있습니다'
        : '판매할 아이템이 없습니다'

    const initialSlots = getInventory().slots.length

    for (let index = 0; index < initialSlots; index += 1) {
      const button = document.createElement('button')
      const icon = document.createElement('span')
      const content = document.createElement('div')
      const name = document.createElement('div')
      const meta = document.createElement('div')
      const price = document.createElement('div')

      button.type = 'button'
      button.className = 'blacksmith-shop-overlay__row'
      button.addEventListener('click', (event) => {
        event.preventDefault()
        event.stopPropagation()
        onRowClick(index)
      })

      icon.className = 'blacksmith-shop-overlay__row-icon'
      icon.setAttribute('aria-hidden', 'true')

      content.className = 'blacksmith-shop-overlay__row-content'
      name.className = 'blacksmith-shop-overlay__row-name'
      meta.className = 'blacksmith-shop-overlay__row-meta'
      price.className = 'blacksmith-shop-overlay__row-price'

      content.append(name, meta)
      button.append(icon, content, price)
      list.append(button)

      rows.push({
        button,
        icon,
        content,
        name,
        meta,
        price
      })
    }

    header.append(titleElement, summaryElement)
    root.append(header, list, emptyState)

    const syncPane = () => {
      const inventory = getInventory()
      const counterInventory = getCounterInventory()
      const counterHasSpace =
        findFirstEmptyPlayerInventorySlotIndex(counterInventory) !== undefined
      const visibleRows = rows.map((row, index) => {
        const item = inventory.slots[index]

        return syncPaneRow({
          kind,
          row,
          item,
          inventory,
          counterInventory,
          counterHasSpace
        })
      })
      const visibleCount = visibleRows.filter(Boolean).length

      summaryElement.textContent =
        kind === 'buy'
          ? `상점 재고 · ${getPlayerInventoryFilledSlotCount(inventory)} / ${inventory.slots.length} 칸`
          : `내 인벤토리 · ${getPlayerInventoryFilledSlotCount(inventory)} / ${inventory.slots.length} 칸`

      emptyState.hidden = visibleCount > 0
    }

    return {
      root,
      header,
      title: titleElement,
      summary: summaryElement,
      list,
      emptyState,
      rows,
      sync: syncPane
    }
  }

  function syncPaneRow({
    kind,
    row,
    item,
    inventory,
    counterInventory,
    counterHasSpace
  }: {
    kind: TradeMode
    row: TradeRowElements
    item: PlayerInventory['slots'][number]
    inventory: PlayerInventory
    counterInventory: PlayerInventory
    counterHasSpace: boolean
  }): boolean {
    const potionDefinition = item ? getPotionShopItemDefinitionById(item.id) : undefined
    const equipmentDefinition = item
      ? getPlayerEquipmentItemDefinitionById(item.id)
      : undefined
    const itemDefinition =
      kind === 'buy'
        ? potionDefinition
        : potionDefinition ?? equipmentDefinition

    if (
      !item ||
      !itemDefinition ||
      (kind === 'buy' ? !potionDefinition : false)
    ) {
      row.button.hidden = true
      return false
    }

    row.button.hidden = false
    if (potionDefinition) {
      renderPotionIcon(row.icon, potionDefinition.id, POISON_SHOP_ROW_ICON_SCALE)
    } else {
      setGenericIcon(row.icon)
    }
    row.name.textContent = itemDefinition.label
    row.meta.textContent =
      kind === 'buy'
        ? `구매 · x${item.quantity} · ${itemDefinition.description}`
        : equipmentDefinition
          ? `판매 · ${getPlayerEquipmentSlotLabelById(equipmentDefinition.slotId)} · ${equipmentDefinition.description}`
          : `판매 · 소비품 · ${itemDefinition.description}`

    if (kind === 'buy') {
      const price = getPotionShopBuyPriceById(itemDefinition.id)
      const canAfford = price !== undefined && inventory.gold >= price
      const canBuy = price !== undefined && canAfford && counterHasSpace

      row.price.textContent = price !== undefined ? formatGoldAmount(price) : '--'
      row.button.disabled = !canBuy
      row.button.classList.toggle('blacksmith-shop-overlay__row--disabled', !canBuy)
      row.button.title = canBuy
        ? `${itemDefinition.label} 구매`
        : price === undefined
          ? '구매할 수 없는 아이템입니다.'
          : !canAfford
            ? '돈이 부족합니다.'
            : '인벤토리가 가득 찼습니다.'
      return true
    }

    const price = getPotionShopSellPriceById(itemDefinition.id)
    const canSell = price !== undefined && counterInventory.gold >= price

    row.price.textContent = price !== undefined ? formatGoldAmount(price) : '--'
    row.button.disabled = !canSell
    row.button.classList.toggle('blacksmith-shop-overlay__row--disabled', !canSell)
    row.button.title = canSell
      ? `${itemDefinition.label} 판매`
      : price === undefined
        ? '판매할 수 없는 아이템입니다.'
        : '상점 보유금이 부족합니다.'
    return true
  }
}

const renderPotionIcon = (
  element: HTMLElement,
  itemId: keyof typeof POTION_ICON_FRAME_BY_ID,
  scale: number
) => {
  setBackgroundFrame(element, POTION_ICON_FRAME_BY_ID[itemId], scale)
}

const setGenericIcon = (element: HTMLElement) => {
  element.style.backgroundImage = 'none'
  element.style.backgroundPosition = '0 0'
  element.style.backgroundSize = 'auto'
  element.style.backgroundColor = 'rgba(141, 110, 49, 0.18)'
  element.style.border = '1px solid rgba(111, 89, 58, 0.42)'
  element.style.borderRadius = '50%'
  element.style.width = '16px'
  element.style.height = '16px'
}

const setPortraitFrame = (
  element: HTMLElement,
  frame: { x: number; y: number; width: number; height: number },
  scale: number
) => {
  setBackgroundFrame(
    element,
    {
      imageUrl: TINY_DUNGEON_TILESET_IMAGE_URL,
      imageWidth: TINY_DUNGEON_TILESET_WIDTH,
      imageHeight: TINY_DUNGEON_TILESET_HEIGHT,
      frame
    },
    scale
  )
}

const setBackgroundFrame = (
  element: HTMLElement,
  frame: {
    imageUrl: string
    imageWidth: number
    imageHeight: number
    frame: { x: number; y: number; width: number; height: number }
  },
  scale: number
) => {
  element.style.backgroundImage = `url(${frame.imageUrl})`
  element.style.backgroundRepeat = 'no-repeat'
  element.style.backgroundPosition = `-${frame.frame.x * scale}px -${frame.frame.y * scale}px`
  element.style.backgroundSize = `${frame.imageWidth * scale}px ${frame.imageHeight * scale}px`
  element.style.width = `${frame.frame.width * scale}px`
  element.style.height = `${frame.frame.height * scale}px`
  element.style.imageRendering = 'pixelated'
}

const formatGoldAmount = (gold: number): string =>
  `${gold.toLocaleString('ko-KR')}원`
