import {
  buyPotionShopItem,
  getPotionShopBuyPriceById,
  getPotionShopItemDefinitionById,
  getPotionShopSellPriceById,
  sellPotionShopItem
} from '../game/potionShop'
import {
  getPlayerEquipmentItemDefinitionById,
  getPlayerEquipmentSlotLabelById,
  type PlayerEquipmentIconKey,
  type PlayerEquipmentItemDefinition
} from '../game/playerEquipment'
import {
  getPlayerInventoryFilledSlotCount
} from '../game/playerInventory'
import type { PlayerInventory } from '../game/playerInventory'

type CreatePotionShopOverlayInput = {
  mountElement: HTMLElement
  getPlayerName: () => string
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
const UI_SPRITESHEET_IMAGE_URL = new URL(
  '../assets/spritesheets/uipack_rpg_sheet.png',
  import.meta.url
).href
const WEAPON_SWORD_IMAGE_URL = new URL(
  '../assets/weapons/weapon-sword.png',
  import.meta.url
).href
const WEAPON_AXE_IMAGE_URL = new URL(
  '../assets/weapons/weapon-axe.png',
  import.meta.url
).href
const WEAPON_SPEAR_IMAGE_URL = new URL(
  '../assets/weapons/weapon-spear.png',
  import.meta.url
).href
const WEAPON_DAGGER_IMAGE_URL = new URL(
  '../assets/weapons/weapon-dagger.png',
  import.meta.url
).href
const WEAPON_MACE_IMAGE_URL = new URL(
  '../assets/weapons/weapon-mace.png',
  import.meta.url
).href
const WEAPON_STAFF_IMAGE_URL = new URL(
  '../assets/weapons/weapon-staff.png',
  import.meta.url
).href
const TINY_DUNGEON_TILESET_WIDTH = 192
const TINY_DUNGEON_TILESET_HEIGHT = 176
const UI_SPRITESHEET_WIDTH = 512
const UI_SPRITESHEET_HEIGHT = 512
const WEAPON_SWORD_IMAGE_WIDTH = 337
const WEAPON_SWORD_IMAGE_HEIGHT = 344
const WEAPON_AXE_IMAGE_WIDTH = 355
const WEAPON_AXE_IMAGE_HEIGHT = 343
const WEAPON_SPEAR_IMAGE_WIDTH = 332
const WEAPON_SPEAR_IMAGE_HEIGHT = 342
const WEAPON_DAGGER_IMAGE_WIDTH = 219
const WEAPON_DAGGER_IMAGE_HEIGHT = 229
const WEAPON_MACE_IMAGE_WIDTH = 323
const WEAPON_MACE_IMAGE_HEIGHT = 325
const WEAPON_STAFF_IMAGE_WIDTH = 328
const WEAPON_STAFF_IMAGE_HEIGHT = 335
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
const ICON_CHECK_FRAME = {
  x: 369,
  y: 184,
  width: 16,
  height: 15
}
const ICON_CIRCLE_FRAME = {
  x: 356,
  y: 466,
  width: 17,
  height: 17
}
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
const TINY_DUNGEON_WEAPON_FRAME = {
  x: 144,
  y: 144,
  width: 16,
  height: 16
}
const EQUIPMENT_ICON_FRAME_BY_KEY: Record<
  PlayerEquipmentIconKey,
  {
    imageUrl: string
    imageWidth: number
    imageHeight: number
    frame: { x: number; y: number; width: number; height: number }
  }
> = {
  'tiny-dungeon-weapon': {
    imageUrl: TINY_DUNGEON_TILESET_IMAGE_URL,
    imageWidth: TINY_DUNGEON_TILESET_WIDTH,
    imageHeight: TINY_DUNGEON_TILESET_HEIGHT,
    frame: TINY_DUNGEON_WEAPON_FRAME
  },
  'tiny-knight-gray-helmet': {
    imageUrl: TINY_DUNGEON_TILESET_IMAGE_URL,
    imageWidth: TINY_DUNGEON_TILESET_WIDTH,
    imageHeight: TINY_DUNGEON_TILESET_HEIGHT,
    frame: {
      x: 48,
      y: 112,
      width: 16,
      height: 16
    }
  },
  'tiny-knight-open-helmet': {
    imageUrl: TINY_DUNGEON_TILESET_IMAGE_URL,
    imageWidth: TINY_DUNGEON_TILESET_WIDTH,
    imageHeight: TINY_DUNGEON_TILESET_HEIGHT,
    frame: {
      x: 16,
      y: 128,
      width: 16,
      height: 16
    }
  },
  'town-crate-sword-right': {
    imageUrl: new URL('../assets/tilesets/town-32.png', import.meta.url).href,
    imageWidth: 256,
    imageHeight: 2240,
    frame: {
      x: 64,
      y: 1504,
      width: 32,
      height: 32
    }
  },
  'weapon-sword': {
    imageUrl: WEAPON_SWORD_IMAGE_URL,
    imageWidth: WEAPON_SWORD_IMAGE_WIDTH,
    imageHeight: WEAPON_SWORD_IMAGE_HEIGHT,
    frame: {
      x: 0,
      y: 0,
      width: WEAPON_SWORD_IMAGE_WIDTH,
      height: WEAPON_SWORD_IMAGE_HEIGHT
    }
  },
  'weapon-axe': {
    imageUrl: WEAPON_AXE_IMAGE_URL,
    imageWidth: WEAPON_AXE_IMAGE_WIDTH,
    imageHeight: WEAPON_AXE_IMAGE_HEIGHT,
    frame: {
      x: 0,
      y: 0,
      width: WEAPON_AXE_IMAGE_WIDTH,
      height: WEAPON_AXE_IMAGE_HEIGHT
    }
  },
  'weapon-spear': {
    imageUrl: WEAPON_SPEAR_IMAGE_URL,
    imageWidth: WEAPON_SPEAR_IMAGE_WIDTH,
    imageHeight: WEAPON_SPEAR_IMAGE_HEIGHT,
    frame: {
      x: 0,
      y: 0,
      width: WEAPON_SPEAR_IMAGE_WIDTH,
      height: WEAPON_SPEAR_IMAGE_HEIGHT
    }
  },
  'weapon-dagger': {
    imageUrl: WEAPON_DAGGER_IMAGE_URL,
    imageWidth: WEAPON_DAGGER_IMAGE_WIDTH,
    imageHeight: WEAPON_DAGGER_IMAGE_HEIGHT,
    frame: {
      x: 0,
      y: 0,
      width: WEAPON_DAGGER_IMAGE_WIDTH,
      height: WEAPON_DAGGER_IMAGE_HEIGHT
    }
  },
  'weapon-mace': {
    imageUrl: WEAPON_MACE_IMAGE_URL,
    imageWidth: WEAPON_MACE_IMAGE_WIDTH,
    imageHeight: WEAPON_MACE_IMAGE_HEIGHT,
    frame: {
      x: 0,
      y: 0,
      width: WEAPON_MACE_IMAGE_WIDTH,
      height: WEAPON_MACE_IMAGE_HEIGHT
    }
  },
  'weapon-staff': {
    imageUrl: WEAPON_STAFF_IMAGE_URL,
    imageWidth: WEAPON_STAFF_IMAGE_WIDTH,
    imageHeight: WEAPON_STAFF_IMAGE_HEIGHT,
    frame: {
      x: 0,
      y: 0,
      width: WEAPON_STAFF_IMAGE_WIDTH,
      height: WEAPON_STAFF_IMAGE_HEIGHT
    }
  },
  'ui-circle-beige': {
    imageUrl: UI_SPRITESHEET_IMAGE_URL,
    imageWidth: UI_SPRITESHEET_WIDTH,
    imageHeight: UI_SPRITESHEET_HEIGHT,
    frame: ICON_CIRCLE_FRAME
  },
  'ui-check-beige': {
    imageUrl: UI_SPRITESHEET_IMAGE_URL,
    imageWidth: UI_SPRITESHEET_WIDTH,
    imageHeight: UI_SPRITESHEET_HEIGHT,
    frame: ICON_CHECK_FRAME
  }
}
const POISON_SHOP_ROW_ICON_SCALE = 0.85
const MIN_TRADE_SLOT_COUNT = 8
const PORTRAIT_SCALE = 3

export const createPotionShopOverlay = ({
  mountElement,
  getPlayerName,
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
  const modeBackButton = document.createElement('button')
  const closeButton = document.createElement('button')
  const playerCard = document.createElement('div')
  const playerPortrait = document.createElement('span')
  const playerCardText = document.createElement('div')
  const playerName = document.createElement('div')
  const playerGold = document.createElement('div')
  const panes = document.createElement('div')
  const statusElement = document.createElement('div')
  const quantityDialog = document.createElement('div')
  const quantityDialogCard = document.createElement('div')
  const quantityDialogTitle = document.createElement('div')
  const quantityDialogSubtitle = document.createElement('div')
  const quantityDialogField = document.createElement('label')
  const quantityDialogFieldLabel = document.createElement('span')
  const quantityInput = document.createElement('input')
  const quantityDialogActions = document.createElement('div')
  const quantityConfirmButton = document.createElement('button')
  const quantityCancelButton = document.createElement('button')
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
  let suppressRowClicks = false
  let suppressRowClicksFrameId = 0
  let pendingTrade:
    | {
        kind: TradeMode
        slotIndex: number
        itemLabel: string
      }
    | undefined

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
  subtitleElement.hidden = true

  modeBackButton.type = 'button'
  modeBackButton.className = 'blacksmith-shop-overlay__mode-back'
  modeBackButton.textContent = '메뉴로 돌아가기'
  modeBackButton.setAttribute('aria-label', '물약 상점 닫기')
  modeBackButton.title = '물약 상점 닫기'

  closeButton.type = 'button'
  closeButton.className = 'blacksmith-shop-overlay__close'
  closeButton.hidden = true
  closeButton.setAttribute('aria-label', '물약 상점 닫기')
  closeButton.title = '거래 닫기 (Esc)'
  closeButton.textContent = '×'

  centerCard.append(titleElement, subtitleElement, modeBackButton)

  playerCard.className = 'blacksmith-shop-overlay__portrait-card'
  playerPortrait.className = 'blacksmith-shop-overlay__portrait'
  playerCardText.className = 'blacksmith-shop-overlay__portrait-text'
  playerName.className = 'blacksmith-shop-overlay__portrait-name'
  playerName.textContent = getPlayerName()
  playerGold.className = 'blacksmith-shop-overlay__portrait-gold'
  playerCardText.append(playerName, playerGold)
  playerCard.append(playerPortrait, playerCardText)

  panes.className = 'blacksmith-shop-overlay__panes'
  statusElement.className = 'blacksmith-shop-overlay__status'
  statusElement.hidden = true
  quantityDialog.className = 'potion-shop-overlay__quantity-dialog'
  quantityDialog.hidden = true
  quantityDialog.setAttribute('role', 'dialog')
  quantityDialog.setAttribute('aria-modal', 'false')
  quantityDialogCard.className = 'potion-shop-overlay__quantity-dialog-card'
  quantityDialogTitle.className = 'potion-shop-overlay__quantity-dialog-title'
  quantityDialogTitle.textContent = '구매 수량'
  quantityDialogSubtitle.className =
    'potion-shop-overlay__quantity-dialog-subtitle'
  quantityDialogField.className = 'potion-shop-overlay__quantity-dialog-field'
  quantityDialogFieldLabel.className =
    'potion-shop-overlay__quantity-dialog-field-label'
  quantityDialogFieldLabel.textContent = '수량'
  quantityInput.className = 'pause-menu-overlay__volume-input potion-shop-overlay__quantity-input'
  quantityInput.type = 'number'
  quantityInput.min = '1'
  quantityInput.step = '1'
  quantityInput.value = '1'
  quantityInput.inputMode = 'numeric'
  quantityDialogActions.className = 'potion-shop-overlay__quantity-dialog-actions'
  quantityConfirmButton.type = 'button'
  quantityConfirmButton.className = 'blacksmith-shop-overlay__close potion-shop-overlay__quantity-confirm'
  quantityConfirmButton.textContent = '확인'
  quantityCancelButton.type = 'button'
  quantityCancelButton.className = 'blacksmith-shop-overlay__close potion-shop-overlay__quantity-cancel'
  quantityCancelButton.textContent = '취소'
  quantityDialogField.append(quantityDialogFieldLabel, quantityInput)
  quantityDialogActions.append(quantityConfirmButton, quantityCancelButton)
  quantityDialogCard.append(
    quantityDialogTitle,
    quantityDialogSubtitle,
    quantityDialogField,
    quantityDialogActions
  )
  quantityDialog.append(quantityDialogCard)
  footerElement.className = 'blacksmith-shop-overlay__footer'
  footerElement.hidden = true

  setPortraitFrame(merchantPortrait, POTION_MERCHANT_PORTRAIT_FRAME, PORTRAIT_SCALE)
  setPortraitFrame(playerPortrait, PLAYER_PORTRAIT_FRAME, PORTRAIT_SCALE)

  header.append(merchantCard, centerCard, playerCard)
  panes.append(merchantPane.root, playerPane.root)
  panelBody.style.position = 'relative'
  panelBody.append(header, panes, statusElement, footerElement, quantityDialog)
  panel.append(closeButton, panelBody)
  overlayRoot.append(backdropButton, panel)
  mountElement.append(overlayRoot)

  function syncLayout() {
    const isOpen = getIsOpen()
    const playerNameText = getPlayerName()
    const playerInventory = getPlayerInventory()

    if (isOpen && !wasOpen) {
      statusMessage = undefined
      closeQuantityDialog()
      panelBody.scrollTop = 0
      // Ignore the input that opened the shop so it cannot auto-click a row.
      suppressRowClicks = true
      suppressRowClicksFrameId += 1
      const frameId = suppressRowClicksFrameId

      window.requestAnimationFrame(() => {
        if (suppressRowClicksFrameId === frameId && getIsOpen()) {
          suppressRowClicks = false
        }
      })
    }

    overlayRoot.hidden = !isOpen
    overlayRoot.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      backdropButton.hidden = true
      panel.hidden = true
      closeButton.hidden = true
      statusMessage = undefined
      closeQuantityDialog()
      suppressRowClicks = false
      suppressRowClicksFrameId += 1
      wasOpen = false
      return
    }

    backdropButton.hidden = false
    closeButton.hidden = false
    panel.hidden = false

    titleElement.textContent = '물약 상점'
    statusElement.hidden = statusMessage === undefined
    statusElement.textContent = statusMessage ?? ''

    playerName.textContent = playerNameText
    playerGold.textContent = `${formatGoldAmount(playerInventory.gold)}`
    syncQuantityDialog()
    merchantPane.sync()
    playerPane.sync()

    wasOpen = true
  }

  function handleMerchantRowClick(slotIndex: number) {
    if (suppressRowClicks) {
      return
    }

    const item = getMerchantInventory().slots[slotIndex]

    if (!item) {
      return
    }

    openQuantityDialog({
      kind: 'buy',
      slotIndex,
      itemLabel: item.label
    })
  }

  function handlePlayerRowClick(slotIndex: number) {
    if (suppressRowClicks) {
      return
    }

    const item = getPlayerInventory().slots[slotIndex]

    if (!item) {
      return
    }

    openQuantityDialog({
      kind: 'sell',
      slotIndex,
      itemLabel: item.label
    })
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

  function handleModeBackButtonClick(event: MouseEvent) {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  function handleBackdropClick(event: MouseEvent) {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  function handleQuantityConfirmClick(event: MouseEvent) {
    event.preventDefault()
    event.stopPropagation()
    commitPendingTrade()
  }

  function handleQuantityCancelClick(event: MouseEvent) {
    event.preventDefault()
    event.stopPropagation()
    closeQuantityDialog()
    syncLayout()
  }

  function handleQuantityInputKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      event.preventDefault()
      commitPendingTrade()
      return
    }

    if (event.key === 'Escape') {
      event.preventDefault()
      closeQuantityDialog()
      syncLayout()
    }
  }

  closeButton.addEventListener('click', handleCloseButtonClick)
  closeButton.addEventListener('pointerdown', handleCloseButtonPointerDown)
  modeBackButton.addEventListener('click', handleModeBackButtonClick)
  backdropButton.addEventListener('click', handleBackdropClick)
  quantityConfirmButton.addEventListener('click', handleQuantityConfirmClick)
  quantityCancelButton.addEventListener('click', handleQuantityCancelClick)
  quantityInput.addEventListener('keydown', handleQuantityInputKeydown)

  const destroy = () => {
    closeButton.removeEventListener('click', handleCloseButtonClick)
    closeButton.removeEventListener('pointerdown', handleCloseButtonPointerDown)
    modeBackButton.removeEventListener('click', handleModeBackButtonClick)
    backdropButton.removeEventListener('click', handleBackdropClick)
    quantityConfirmButton.removeEventListener('click', handleQuantityConfirmClick)
    quantityCancelButton.removeEventListener('click', handleQuantityCancelClick)
    quantityInput.removeEventListener('keydown', handleQuantityInputKeydown)
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

    const initialSlots = Math.max(
      getInventory().slots.length,
      MIN_TRADE_SLOT_COUNT
    )

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
      const visibleRows = rows.map((row, index) => {
        const item = inventory.slots[index]

        return syncPaneRow({
          kind,
          slotIndex: index,
          row,
          item,
          inventory,
          counterInventory
        })
      })
      const visibleCount = visibleRows.filter(Boolean).length
      const visibleSlotCount = Math.max(
        inventory.slots.length,
        MIN_TRADE_SLOT_COUNT
      )

      summaryElement.textContent =
        kind === 'buy'
          ? `상점 재고 · ${getPlayerInventoryFilledSlotCount(inventory)} / ${visibleSlotCount} 칸`
          : `내 인벤토리 · ${getPlayerInventoryFilledSlotCount(inventory)} / ${visibleSlotCount} 칸`

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
    slotIndex,
    row,
    item,
    inventory,
    counterInventory
  }: {
    kind: TradeMode
    slotIndex: number
    row: TradeRowElements
    item: PlayerInventory['slots'][number]
    inventory: PlayerInventory
    counterInventory: PlayerInventory
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
      row.button.hidden = false
      row.button.disabled = true
      row.button.classList.remove('blacksmith-shop-overlay__row--selected')
      row.button.classList.add('blacksmith-shop-overlay__row--disabled')
      row.button.title =
        kind === 'buy' ? '빈 상품 칸입니다.' : '빈 판매 칸입니다.'
      row.name.textContent = ''
      row.meta.textContent = ''
      row.price.textContent = ''
      clearRowIcon(row.icon)
      return true
    }

    row.button.hidden = false
    row.button.disabled = false
    row.button.classList.remove('blacksmith-shop-overlay__row--disabled')
    row.button.classList.toggle(
      'blacksmith-shop-overlay__row--selected',
      pendingTrade?.kind === kind && pendingTrade.slotIndex === slotIndex
    )
    if (potionDefinition) {
      renderPotionIcon(row.icon, potionDefinition.id, POISON_SHOP_ROW_ICON_SCALE)
    } else if (equipmentDefinition) {
      renderEquipmentIcon(row.icon, equipmentDefinition, POISON_SHOP_ROW_ICON_SCALE)
    } else {
      setGenericIcon(row.icon)
    }
    row.name.textContent = itemDefinition.label
    row.meta.textContent =
      kind === 'buy'
        ? `구매 · 클릭 후 수량 입력 · ${itemDefinition.description}`
        : equipmentDefinition
          ? `판매 · 클릭 후 수량 입력 · ${getPlayerEquipmentSlotLabelById(equipmentDefinition.slotId)} · ${equipmentDefinition.description}`
          : `판매 · 클릭 후 수량 입력 · 소비품 · ${itemDefinition.description}`

    if (kind === 'buy') {
      const price = getPotionShopBuyPriceById(itemDefinition.id)
      const canAfford = price !== undefined && inventory.gold >= price
      const canBuy = price !== undefined && canAfford

      row.price.textContent = price !== undefined ? formatGoldAmount(price) : '--'
      row.button.disabled = !canBuy
      row.button.classList.toggle('blacksmith-shop-overlay__row--disabled', !canBuy)
      row.button.title = canBuy
        ? `${itemDefinition.label} 클릭 후 수량 입력`
        : price === undefined
          ? '구매할 수 없는 아이템입니다.'
          : !canAfford
            ? '돈이 부족합니다.'
            : '구매하려면 수량을 입력하세요.'
      return true
    }

    const price = getPotionShopSellPriceById(itemDefinition.id)
    const canSell = price !== undefined && counterInventory.gold >= price

    row.price.textContent = price !== undefined ? formatGoldAmount(price) : '--'
    row.button.disabled = !canSell
    row.button.classList.toggle('blacksmith-shop-overlay__row--disabled', !canSell)
    row.button.title = canSell
      ? `${itemDefinition.label} 클릭 후 수량 입력`
      : price === undefined
        ? '판매할 수 없는 아이템입니다.'
        : '상점 보유금이 부족합니다.'
    return true
  }

  function openQuantityDialog({
    kind,
    slotIndex,
    itemLabel
  }: {
    kind: TradeMode
    slotIndex: number
    itemLabel: string
  }) {
    pendingTrade = {
      kind,
      slotIndex,
      itemLabel
    }
    quantityInput.value = '1'
    syncLayout()
    quantityDialog.hidden = false
    quantityInput.focus()
    quantityInput.select()
  }

  function closeQuantityDialog() {
    pendingTrade = undefined
    quantityDialog.hidden = true
  }

  function syncQuantityDialog() {
    if (!pendingTrade) {
      quantityDialog.hidden = true
      return
    }

    quantityDialog.hidden = false
    quantityDialogTitle.textContent =
      pendingTrade.kind === 'buy' ? '구매 수량' : '판매 수량'
    quantityDialogSubtitle.textContent = pendingTrade.itemLabel
    quantityDialogFieldLabel.textContent = '수량'
    quantityConfirmButton.textContent =
      pendingTrade.kind === 'buy' ? '구매' : '판매'
  }

  function commitPendingTrade() {
    if (!pendingTrade) {
      return
    }

    const quantity = parseQuantityInput(quantityInput.value)

    if (quantity === undefined) {
      statusMessage = '수량은 1개 이상의 정수여야 합니다.'
      syncLayout()
      quantityInput.focus()
      quantityInput.select()
      return
    }

    const nextState =
      pendingTrade.kind === 'buy'
        ? buyPotionShopItem({
            playerInventory: getPlayerInventory(),
            merchantInventory: getMerchantInventory(),
            merchantSlotIndex: pendingTrade.slotIndex,
            quantity
          })
        : sellPotionShopItem({
            playerInventory: getPlayerInventory(),
            merchantInventory: getMerchantInventory(),
            playerSlotIndex: pendingTrade.slotIndex,
            quantity
          })

    statusMessage = nextState.message
    if (nextState.ok) {
      onRequestTradeStateChange(
        nextState.playerInventory,
        nextState.merchantInventory
      )
      closeQuantityDialog()
    }
    syncLayout()
    if (!nextState.ok) {
      quantityInput.focus()
      quantityInput.select()
    }
  }

  function parseQuantityInput(value: string): number | undefined {
    const normalizedValue = value.replace(/[\s,]/g, '')

    if (normalizedValue.length === 0) {
      return undefined
    }

    const quantity = Number(normalizedValue)

    return Number.isInteger(quantity) && quantity > 0 ? quantity : undefined
  }
}

const renderPotionIcon = (
  element: HTMLElement,
  itemId: keyof typeof POTION_ICON_FRAME_BY_ID,
  scale: number
) => {
  setBackgroundFrame(element, POTION_ICON_FRAME_BY_ID[itemId], scale)
}

const renderEquipmentIcon = (
  element: HTMLElement,
  definition: PlayerEquipmentItemDefinition,
  scale: number
) => {
  const frame = EQUIPMENT_ICON_FRAME_BY_KEY[definition.icon.key]

  setBackgroundFrame(element, frame, scale * definition.icon.scale)
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

const clearRowIcon = (element: HTMLElement) => {
  element.style.backgroundImage = 'none'
  element.style.backgroundPosition = '0 0'
  element.style.backgroundSize = 'auto'
  element.style.backgroundColor = 'transparent'
  element.style.border = '0'
  element.style.borderRadius = '0'
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
