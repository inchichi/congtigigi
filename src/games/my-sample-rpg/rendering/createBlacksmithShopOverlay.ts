import {
  getPlayerEquipmentItemDefinitionById,
  getPlayerEquipmentSlotLabelById,
  type PlayerEquipmentIconKey,
  type PlayerEquipmentSlotId
} from '../playerEquipment'
import { getPotionShopItemDefinitionById } from '../potionShop'
import {
  buyBlacksmithShopItem,
  getBlacksmithShopBuyPriceById,
  getBlacksmithShopSellPriceById,
  sellBlacksmithShopItem
} from '../blacksmithShop'
import {
  findFirstEmptyPlayerInventorySlotIndex,
  getPlayerInventoryFilledSlotCount
} from '../playerInventory'
import type { PlayerInventory } from '../playerInventory'

type CreateBlacksmithShopOverlayInput = {
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

export type BlacksmithShopOverlay = {
  syncFrame: () => void
  destroy: () => void
}

type TradeCategoryId = 'all' | PlayerEquipmentSlotId | 'other'
type TradeMode = 'buy' | 'sell'
type BlacksmithServiceMode = 'menu' | 'shop' | 'upgrade' | 'repair'

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
  tabBar: HTMLDivElement
  tabs: HTMLButtonElement[]
  list: HTMLDivElement
  emptyState: HTMLDivElement
  rows: TradeRowElements[]
  selectedCategory: TradeCategoryId
  sync: () => void
}

const UI_SPRITESHEET_IMAGE_URL = new URL(
  '../assets/spritesheets/uipack_rpg_sheet.png',
  import.meta.url
).href
const TINY_DUNGEON_TILESET_IMAGE_URL = new URL(
  '../assets/tilesets/tiny-dungeon-16.png',
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
const UI_SPRITESHEET_WIDTH = 512
const UI_SPRITESHEET_HEIGHT = 512
const TINY_DUNGEON_TILESET_WIDTH = 192
const TINY_DUNGEON_TILESET_HEIGHT = 176
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
const BLACKSMITH_PORTRAIT_FRAME = {
  x: 32,
  y: 112,
  width: 16,
  height: 16
}
const TINY_DUNGEON_WEAPON_FRAME = {
  x: 144,
  y: 144,
  width: 16,
  height: 16
}
const TRADE_CATEGORY_OPTIONS: Array<{
  id: TradeCategoryId
  label: string
}> = [
  { id: 'all', label: '전체' },
  { id: 'weapon', label: '무기' },
  { id: 'armor', label: '옷' },
  { id: 'boots', label: '신발' },
  { id: 'accessory', label: '장신구' },
  { id: 'other', label: '기타' }
]
const BLACKSMITH_SERVICE_OPTIONS: Array<{
  mode: Exclude<BlacksmithServiceMode, 'menu'>
  label: string
  description: string
}> = [
  {
    mode: 'shop',
    label: '상점',
    description: '아이템을 사고팔 수 있습니다.'
  },
  {
    mode: 'upgrade',
    label: '강화',
    description: '준비중입니다.'
  },
  {
    mode: 'repair',
    label: '수리',
    description: '준비중입니다.'
  }
]
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
const ROW_ICON_SCALE = 0.85
const PORTRAIT_SCALE = 3

export const createBlacksmithShopOverlay = ({
  mountElement,
  getPlayerName,
  getPlayerInventory,
  getMerchantInventory,
  getIsOpen,
  onRequestOpenChange,
  onRequestTradeStateChange
}: CreateBlacksmithShopOverlayInput): BlacksmithShopOverlay => {
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
  const modeBackButton = document.createElement('button')
  const serviceMenu = document.createElement('section')
  const serviceMenuHeader = document.createElement('div')
  const serviceMenuTitle = document.createElement('div')
  const serviceMenuDescription = document.createElement('div')
  const serviceMenuGrid = document.createElement('div')
  const selectionPanel = document.createElement('section')
  const selectionPanelBody = document.createElement('div')
  const workshopPanelWindow = document.createElement('section')
  const workshopPanelBody = document.createElement('div')
  const workshopPanel = document.createElement('section')
  const workshopBadge = document.createElement('div')
  const workshopTitle = document.createElement('div')
  const workshopDescription = document.createElement('div')
  const workshopBackButton = document.createElement('button')
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
  let currentServiceMode: BlacksmithServiceMode = 'menu'
  let wasOpen = false

  overlayRoot.className = 'blacksmith-shop-overlay'

  backdropButton.type = 'button'
  backdropButton.className = 'blacksmith-shop-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.tabIndex = -1
  backdropButton.setAttribute('aria-hidden', 'true')

  panel.className = 'blacksmith-shop-overlay__panel blacksmith-shop-overlay__panel--shop'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'false')
  panel.setAttribute('aria-labelledby', 'blacksmith-shop-title')

  panelBody.className = 'blacksmith-shop-overlay__panel-body'
  header.className = 'blacksmith-shop-overlay__header'

  merchantCard.className = 'blacksmith-shop-overlay__portrait-card'
  merchantPortrait.className = 'blacksmith-shop-overlay__portrait'
  merchantCardText.className = 'blacksmith-shop-overlay__portrait-text'
  merchantName.className = 'blacksmith-shop-overlay__portrait-name'
  merchantName.textContent = '대장장이'
  merchantGold.className = 'blacksmith-shop-overlay__portrait-gold'
  merchantGold.hidden = true
  merchantCardText.append(merchantName, merchantGold)
  merchantCard.append(merchantPortrait, merchantCardText)

  centerCard.className = 'blacksmith-shop-overlay__center-card'
  titleElement.id = 'blacksmith-shop-title'
  titleElement.className = 'blacksmith-shop-overlay__title'
  titleElement.textContent = '대장장이 상점'
  subtitleElement.className = 'blacksmith-shop-overlay__subtitle'
  subtitleElement.hidden = true

  closeButton.type = 'button'
  closeButton.className = 'blacksmith-shop-overlay__close'
  closeButton.hidden = true
  closeButton.setAttribute('aria-label', '대장장이 거래 닫기')
  closeButton.title = '거래 닫기 (Esc)'
  closeButton.textContent = '×'

  centerCard.append(titleElement, subtitleElement)

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
  footerElement.className = 'blacksmith-shop-overlay__footer'
  footerElement.hidden = true

  modeBackButton.type = 'button'
  modeBackButton.className = 'blacksmith-shop-overlay__mode-back'
  modeBackButton.textContent = '메뉴로 돌아가기'
  modeBackButton.setAttribute('aria-label', '대장장이 메뉴로 돌아가기')
  modeBackButton.title = '대장장이 메뉴로 돌아가기'

  serviceMenu.className = 'blacksmith-shop-overlay__service-menu'
  serviceMenuHeader.className = 'blacksmith-shop-overlay__service-menu-header'
  serviceMenuTitle.className = 'blacksmith-shop-overlay__service-menu-title'
  serviceMenuTitle.textContent = '대장장이와 상호작용'
  serviceMenuDescription.className =
    'blacksmith-shop-overlay__service-menu-description'
  serviceMenuDescription.textContent = '먼저 원하는 행동을 선택하세요.'
  serviceMenuGrid.className = 'blacksmith-shop-overlay__service-menu-grid'
  serviceMenuTitle.id = 'blacksmith-shop-selection-title'
  selectionPanel.className =
    'blacksmith-shop-overlay__panel blacksmith-shop-overlay__panel--selection'
  selectionPanel.hidden = true
  selectionPanel.setAttribute('role', 'dialog')
  selectionPanel.setAttribute('aria-modal', 'false')
  selectionPanel.setAttribute('aria-labelledby', 'blacksmith-shop-selection-title')
  selectionPanelBody.className =
    'blacksmith-shop-overlay__panel-body blacksmith-shop-overlay__panel-body--selection'

  workshopPanelWindow.className =
    'blacksmith-shop-overlay__panel blacksmith-shop-overlay__panel--workshop'
  workshopPanelWindow.hidden = true
  workshopPanelWindow.setAttribute('role', 'dialog')
  workshopPanelWindow.setAttribute('aria-modal', 'false')
  workshopPanelWindow.setAttribute('aria-labelledby', 'blacksmith-shop-workshop-title')
  workshopPanelBody.className =
    'blacksmith-shop-overlay__panel-body blacksmith-shop-overlay__panel-body--workshop'

  workshopPanel.className = 'blacksmith-shop-overlay__workshop-panel'
  workshopBadge.className = 'blacksmith-shop-overlay__workshop-badge'
  workshopBadge.textContent = '준비중입니다'
  workshopTitle.className = 'blacksmith-shop-overlay__workshop-title'
  workshopTitle.id = 'blacksmith-shop-workshop-title'
  workshopTitle.textContent = '준비중입니다'
  workshopDescription.className =
    'blacksmith-shop-overlay__workshop-description'
  workshopDescription.textContent = '강화와 수리는 아직 준비 중입니다.'

  workshopBackButton.type = 'button'
  workshopBackButton.className = 'blacksmith-shop-overlay__mode-back'
  workshopBackButton.textContent = '메뉴로 돌아가기'
  workshopBackButton.setAttribute('aria-label', '대장장이 메뉴로 돌아가기')
  workshopBackButton.title = '대장장이 메뉴로 돌아가기'

  for (const option of BLACKSMITH_SERVICE_OPTIONS) {
    const serviceCard = document.createElement('button')
    const serviceCardLabel = document.createElement('div')
    const serviceCardDescription = document.createElement('div')

    serviceCard.type = 'button'
    serviceCard.className = 'blacksmith-shop-overlay__service-card'
    serviceCard.setAttribute('aria-label', option.label)
    serviceCard.title = option.description

    serviceCardLabel.className = 'blacksmith-shop-overlay__service-card-label'
    serviceCardLabel.textContent = option.label
    serviceCardDescription.className =
      'blacksmith-shop-overlay__service-card-description'
    serviceCardDescription.textContent = option.description

    serviceCard.append(serviceCardLabel, serviceCardDescription)
    serviceCard.addEventListener('click', (event) => {
      event.preventDefault()
      event.stopPropagation()
      setServiceMode(option.mode)
    })

    serviceMenuGrid.append(serviceCard)
  }

  setPortraitFrame(merchantPortrait, BLACKSMITH_PORTRAIT_FRAME, PORTRAIT_SCALE)
  setPortraitFrame(playerPortrait, PLAYER_PORTRAIT_FRAME, PORTRAIT_SCALE)

  header.append(merchantCard, centerCard, playerCard)
  centerCard.append(titleElement, subtitleElement, modeBackButton)
  serviceMenuHeader.append(serviceMenuTitle, serviceMenuDescription)
  serviceMenu.append(serviceMenuHeader, serviceMenuGrid)
  workshopPanel.append(workshopBadge, workshopTitle, workshopDescription)
  panes.append(merchantPane.root, playerPane.root)
  selectionPanelBody.append(serviceMenu)
  selectionPanel.append(selectionPanelBody, closeButton)
  panelBody.append(header, panes, statusElement, footerElement)
  workshopPanelBody.append(workshopPanel, workshopBackButton)
  workshopPanelWindow.append(workshopPanelBody)
  panel.append(panelBody)
  overlayRoot.append(backdropButton, selectionPanel, panel, workshopPanelWindow)
  mountElement.append(overlayRoot)

  function syncLayout() {
    const isOpen = getIsOpen()
    const playerNameText = getPlayerName()
    const playerInventory = getPlayerInventory()

    if (isOpen && !wasOpen) {
      currentServiceMode = 'menu'
      statusMessage = undefined
    }

    const isMenuMode = currentServiceMode === 'menu'
    const isShopMode = currentServiceMode === 'shop'
    const isWorkshopMode =
      currentServiceMode === 'upgrade' || currentServiceMode === 'repair'

    overlayRoot.hidden = !isOpen
    overlayRoot.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      backdropButton.hidden = true
      selectionPanel.hidden = true
      panel.hidden = true
      workshopPanelWindow.hidden = true
      closeButton.hidden = true
      currentServiceMode = 'menu'
      statusMessage = undefined
      wasOpen = false
      return
    }

    backdropButton.hidden = false
    closeButton.hidden = false
    panel.hidden = false

    selectionPanel.hidden = !isMenuMode
    panel.hidden = !isShopMode
    workshopPanelWindow.hidden = !isWorkshopMode
    modeBackButton.hidden = !isShopMode
    workshopBackButton.hidden = !isWorkshopMode
    if (isMenuMode) {
      selectionPanel.append(closeButton)
    } else if (isShopMode) {
      panel.append(closeButton)
    } else {
      workshopPanelWindow.append(closeButton)
    }
    closeButton.hidden = false

    if (isMenuMode) {
      titleElement.textContent = '대장장이와 상호작용'
    } else if (isShopMode) {
      titleElement.textContent = '대장장이 상점'
    } else {
      const modeLabel = currentServiceMode === 'upgrade' ? '강화' : '수리'

      titleElement.textContent = `${modeLabel} 준비중입니다`
      workshopBadge.textContent = '준비중입니다'
      workshopTitle.textContent = `${modeLabel} 준비중입니다`
      workshopDescription.textContent = '아직 구현되지 않았습니다.'
    }

    statusElement.hidden = statusMessage === undefined
    statusElement.textContent = statusMessage ?? ''
    playerName.textContent = playerNameText
    playerGold.textContent = `${formatGoldAmount(playerInventory.gold)}`

    if (isShopMode) {
      merchantPane.sync()
      playerPane.sync()
    }

    wasOpen = true
  }

  const setServiceMode = (nextMode: BlacksmithServiceMode) => {
    if (currentServiceMode === nextMode) {
      return
    }

    currentServiceMode = nextMode

    if (nextMode !== 'shop') {
      statusMessage = undefined
    }

    syncLayout()
  }

  function handleMerchantRowClick(slotIndex: number) {
    const nextState = buyBlacksmithShopItem({
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
    const nextState = sellBlacksmithShopItem({
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
    const tabBar = document.createElement('div')
    const list = document.createElement('div')
    const emptyState = document.createElement('div')
    const rows: TradeRowElements[] = []
    const tabs: HTMLButtonElement[] = []
    let selectedCategory: TradeCategoryId = 'all'

    root.className = `blacksmith-shop-overlay__pane blacksmith-shop-overlay__pane--${kind}`
    header.className = 'blacksmith-shop-overlay__pane-header'
    titleElement.className = 'blacksmith-shop-overlay__pane-title'
    titleElement.textContent = title
    summaryElement.className = 'blacksmith-shop-overlay__pane-summary'
    summaryElement.textContent = subtitle
    tabBar.className = 'blacksmith-shop-overlay__tab-bar'
    list.className = 'blacksmith-shop-overlay__list'
    emptyState.className = 'blacksmith-shop-overlay__empty-state'
    emptyState.textContent =
      kind === 'buy'
        ? '상점 재고가 비어 있습니다'
        : '판매할 아이템이 없습니다'

    for (const category of TRADE_CATEGORY_OPTIONS) {
      const tabButton = document.createElement('button')

      tabButton.type = 'button'
      tabButton.className = 'blacksmith-shop-overlay__tab'
      tabButton.textContent = category.label
      tabButton.dataset.tradeCategoryId = category.id
      tabButton.addEventListener('click', (event) => {
        event.preventDefault()
        event.stopPropagation()
        selectedCategory = category.id
        syncPane()
      })

      tabBar.append(tabButton)
      tabs.push(tabButton)
    }

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
    root.append(header, tabBar, list, emptyState)

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
          selectedCategory,
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

      for (const tabButton of tabs) {
        const isActive = tabButton.dataset.tradeCategoryId === selectedCategory

        tabButton.classList.toggle('blacksmith-shop-overlay__tab--active', isActive)
        tabButton.setAttribute('aria-pressed', String(isActive))
      }
    }

    return {
      root,
      header,
      title: titleElement,
      summary: summaryElement,
      tabBar,
      tabs,
      list,
      emptyState,
      rows,
      selectedCategory,
      sync: syncPane
    }
  }

  function syncPaneRow({
    kind,
    row,
    item,
    selectedCategory,
    inventory,
    counterInventory,
    counterHasSpace
  }: {
    kind: TradeMode
    row: TradeRowElements
    item: PlayerInventory['slots'][number]
    selectedCategory: TradeCategoryId
    inventory: PlayerInventory
    counterInventory: PlayerInventory
    counterHasSpace: boolean
  }): boolean {
    const equipmentDefinition = item
      ? getPlayerEquipmentItemDefinitionById(item.id)
      : undefined
    const potionDefinition = item
      ? getPotionShopItemDefinitionById(item.id)
      : undefined
    const renderDefinition = equipmentDefinition ?? potionDefinition
    const itemCategory = equipmentDefinition?.slotId ?? 'other'

    if (
      !item ||
      !renderDefinition ||
      (selectedCategory !== 'all' && selectedCategory !== itemCategory)
    ) {
      row.button.hidden = true
      return false
    }

    row.button.hidden = false

    if (equipmentDefinition) {
      renderEquipmentIcon(row.icon, equipmentDefinition, ROW_ICON_SCALE)
      row.name.textContent = equipmentDefinition.label
      row.meta.textContent =
        kind === 'buy'
          ? `구매 · ${getPlayerEquipmentSlotLabelById(equipmentDefinition.slotId)} · 레벨 ${equipmentDefinition.level}`
          : `판매 · ${getPlayerEquipmentSlotLabelById(equipmentDefinition.slotId)} · 레벨 ${equipmentDefinition.level}`
    } else if (potionDefinition) {
      renderPotionIcon(row.icon, potionDefinition.id, ROW_ICON_SCALE)
      row.name.textContent = potionDefinition.label
      row.meta.textContent =
        kind === 'buy'
          ? '구매 · 거래 불가'
          : `판매 · 소비품 · ${potionDefinition.description}`
    } else {
      setGenericIcon(row.icon)
      row.name.textContent = item.label
      row.meta.textContent =
        kind === 'buy'
          ? '구매 · 거래 불가'
          : `판매 · 소비품 · ${item.label}`
    }

    if (kind === 'buy') {
      const price = getBlacksmithShopBuyPriceById(renderDefinition.id)
      const canAfford = price !== undefined && inventory.gold >= price
      const canBuy = price !== undefined && canAfford && counterHasSpace

      row.price.textContent = price !== undefined ? formatGoldAmount(price) : '--'
      row.button.disabled = !canBuy
      row.button.classList.toggle('blacksmith-shop-overlay__row--disabled', !canBuy)
      row.button.title = canBuy
        ? `${renderDefinition.label} 구매`
        : price === undefined
          ? '구매할 수 없는 아이템입니다.'
          : !canAfford
            ? '돈이 부족합니다.'
            : '인벤토리가 가득 찼습니다.'
      return true
    }

    const price = getBlacksmithShopSellPriceById(renderDefinition.id)
    const canSell = price !== undefined && counterInventory.gold >= price

    row.price.textContent = price !== undefined ? formatGoldAmount(price) : '--'
    row.button.disabled = !canSell
    row.button.classList.toggle('blacksmith-shop-overlay__row--disabled', !canSell)
    row.button.title = canSell
      ? `${renderDefinition.label} 판매`
      : price === undefined
        ? '판매할 수 없는 아이템입니다.'
        : '상점 보유금이 부족합니다.'
    return true
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
    setServiceMode('menu')
  }

  closeButton.addEventListener('click', handleCloseButtonClick)
  closeButton.addEventListener('pointerdown', handleCloseButtonPointerDown)
  modeBackButton.addEventListener('click', handleModeBackButtonClick)
  workshopBackButton.addEventListener('click', handleModeBackButtonClick)

  const destroy = () => {
    closeButton.removeEventListener('click', handleCloseButtonClick)
    closeButton.removeEventListener('pointerdown', handleCloseButtonPointerDown)
    modeBackButton.removeEventListener('click', handleModeBackButtonClick)
    workshopBackButton.removeEventListener('click', handleModeBackButtonClick)
    overlayRoot.remove()
  }

  syncLayout()

  return {
    syncFrame: syncLayout,
    destroy
  }
}

const renderEquipmentIcon = (
  element: HTMLElement,
  definition: NonNullable<ReturnType<typeof getPlayerEquipmentItemDefinitionById>>,
  scale: number
) => {
  const frame = EQUIPMENT_ICON_FRAME_BY_KEY[definition.icon.key]

  setBackgroundFrame(element, frame, scale * definition.icon.scale)
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
