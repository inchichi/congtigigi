import type {
  PlayerEquipment,
  PlayerEquipmentIconKey
} from '../game/playerEquipment'
import {
  getPlayerEquipmentItemDefinitionById,
  getPlayerEquipmentSlotLabelById
} from '../game/playerEquipment'
import { getPlayerInventoryFilledSlotCount } from '../game/playerInventory'
import type { PlayerInventory } from '../game/playerInventory'
import type { PlayerProfile } from '../game/playerProfile'
import {
  findPlayerQuickslotIndexByInventorySlotIndex,
  type PlayerQuickslots
} from '../game/playerQuickslots'
import { equipPlayerInventorySlot } from '../game/playerLoadout'
import { usePlayerInventoryConsumable } from '../game/playerConsumables'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreatePlayerInventoryOverlayInput = {
  mountElement: HTMLElement
  profile: PlayerProfile
  getInventory: () => PlayerInventory
  getEquipment: () => PlayerEquipment
  getQuickslots: () => PlayerQuickslots
  getIsOpen: () => boolean
  onRequestOpenChange: (isOpen: boolean) => void
  onRequestInventoryChange: (nextInventory: PlayerInventory) => void
  onRequestEquipmentChange: (nextEquipment: PlayerEquipment) => void
  onRequestProfileChange: (nextProfile: PlayerProfile) => void
  onConsumableUsed?: (itemId: string) => void
}

export type PlayerInventoryOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const UI_SPRITESHEET_IMAGE_URL = new URL(
  '../assets/spritesheets/uipack_rpg_sheet.png',
  import.meta.url
).href
const TINY_DUNGEON_TILESET_IMAGE_URL = new URL(
  '../assets/tilesets/tiny-dungeon-16.png',
  import.meta.url
).href
const TOWN_TILESET_IMAGE_URL = new URL(
  '../assets/tilesets/town-32.png',
  import.meta.url
).href
const UI_SPRITESHEET_WIDTH = 512
const UI_SPRITESHEET_HEIGHT = 512
const TINY_DUNGEON_TILESET_WIDTH = 192
const TINY_DUNGEON_TILESET_HEIGHT = 176
const TOWN_TILESET_WIDTH = 256
const TOWN_TILESET_HEIGHT = 2240
const OVERLAY_MARGIN = 16
const TINY_DUNGEON_WEAPON_FRAME = {
  x: 144,
  y: 144,
  width: 16,
  height: 16
}
const TINY_DUNGEON_CONSUMABLE_ICON_FRAMES = {
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
const BUTTON_SQUARE_FRAME = {
  x: 293,
  y: 294,
  width: 45,
  height: 49
}
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
    imageUrl: TOWN_TILESET_IMAGE_URL,
    imageWidth: TOWN_TILESET_WIDTH,
    imageHeight: TOWN_TILESET_HEIGHT,
    frame: {
      x: 64,
      y: 1504,
      width: 32,
      height: 32
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
const PANEL_MARGIN = 16
type InventoryCategoryId = 'all' | 'equipment' | 'consumable' | 'other'

const INVENTORY_CATEGORY_OPTIONS: Array<{
  id: InventoryCategoryId
  label: string
}> = [
  { id: 'equipment', label: '장비' },
  { id: 'consumable', label: '소비' },
  { id: 'other', label: '기타' }
]

export const createPlayerInventoryOverlay = ({
  mountElement,
  profile,
  getInventory,
  getQuickslots,
  getEquipment,
  getIsOpen,
  onRequestOpenChange,
  onRequestInventoryChange,
  onRequestEquipmentChange,
  onRequestProfileChange,
  onConsumableUsed
}: CreatePlayerInventoryOverlayInput): PlayerInventoryOverlay => {
  const overlayRoot = document.createElement('div')
  const backdropButton = document.createElement('button')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const headerRow = document.createElement('div')
  const titleGroup = document.createElement('div')
  const titleElement = document.createElement('div')
  const summaryElement = document.createElement('div')
  const closeButton = document.createElement('button')
  const closeIcon = document.createElement('span')
  const tabBar = document.createElement('div')
  const tabButtons: HTMLButtonElement[] = []
  const slotGrid = document.createElement('div')
  const detailsPanel = document.createElement('section')
  const detailsHeader = document.createElement('div')
  const detailsIcon = document.createElement('span')
  const detailsText = document.createElement('div')
  const detailsName = document.createElement('div')
  const detailsMeta = document.createElement('div')
  const detailsDescription = document.createElement('div')
  const detailsHint = document.createElement('div')
  const emptyStateElement = document.createElement('div')
  const slotButtons: HTMLButtonElement[] = []
  const slotIndexLabels: HTMLSpanElement[] = []
  const slotIcons: HTMLSpanElement[] = []
  const slotItemLabels: HTMLSpanElement[] = []
  const slotQuantityLabels: HTMLSpanElement[] = []
  const slotEmptyMarkers: HTMLSpanElement[] = []
  const initialInventory = getInventory()
  let selectedCategory: InventoryCategoryId = 'equipment'
  let wasOpen = false
  let hoveredSlotIndex: number | undefined
  let lastPointerPosition:
    | {
        x: number
        y: number
      }
    | undefined
  let tooltipAnchor = {
    x: 0,
    y: 0
  }
  let panelPosition = { left: 0, top: 0 }
  let hasPanelPosition = false
  let dragState:
    | {
        pointerId: number
        offsetX: number
        offsetY: number
        width: number
        height: number
      }
    | undefined

  overlayRoot.className = 'player-inventory-overlay'

  backdropButton.type = 'button'
  backdropButton.className = 'player-inventory-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.tabIndex = -1
  backdropButton.setAttribute('aria-hidden', 'true')

  panel.id = 'player-inventory-panel'
  panel.className = 'player-inventory-overlay__panel'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'false')
  panel.setAttribute('aria-labelledby', 'player-inventory-title')

  panelBody.className = 'player-inventory-overlay__panel-body'
  panelBody.style.gridTemplateAreas = '"header" "tabs" "slots"'
  panelBody.style.gridTemplateColumns = 'max-content'
  headerRow.className = 'player-inventory-overlay__header'
  titleGroup.className = 'player-inventory-overlay__title-group'
  titleElement.id = 'player-inventory-title'
  titleElement.className = 'player-inventory-overlay__title'
  titleElement.textContent = '가방'
  summaryElement.className = 'player-inventory-overlay__summary'
  summaryElement.textContent = `${getPlayerInventoryFilledSlotCount(initialInventory)} / ${initialInventory.slots.length}칸 · ${formatGoldAmount(initialInventory.gold)}`

  closeButton.type = 'button'
  closeButton.className = 'player-inventory-overlay__close player-stat-overlay__close'
  closeButton.setAttribute('aria-label', '가방 닫기')
  closeButton.title = '가방 닫기 (Esc)'

  closeIcon.className = 'player-inventory-overlay__close-icon player-stat-overlay__close-icon'
  closeIcon.textContent = '×'
  closeIcon.setAttribute('aria-hidden', 'true')

  tabBar.className = 'player-inventory-overlay__tab-bar'
  for (const option of INVENTORY_CATEGORY_OPTIONS) {
    const tabButton = document.createElement('button')

    tabButton.type = 'button'
    tabButton.className = 'player-inventory-overlay__tab'
    tabButton.textContent = option.label
    tabButton.dataset.playerInventoryCategoryId = option.id
    tabButton.setAttribute('aria-pressed', String(option.id === selectedCategory))
    tabButton.addEventListener('click', (event) => {
      event.preventDefault()
      event.stopPropagation()
      selectedCategory = option.id
      syncFrame()
    })

    tabBar.append(tabButton)
    tabButtons.push(tabButton)
  }

  slotGrid.className = 'player-inventory-overlay__slot-grid'
  slotGrid.style.gap = '1px'

  detailsPanel.className = 'player-inventory-overlay__tooltip'
  detailsPanel.setAttribute('role', 'tooltip')
  detailsHeader.className = 'player-inventory-overlay__tooltip-header'
  detailsIcon.className = 'player-inventory-overlay__tooltip-icon'
  detailsIcon.setAttribute('aria-hidden', 'true')
  detailsText.className = 'player-inventory-overlay__tooltip-text'
  detailsName.className = 'player-inventory-overlay__tooltip-name'
  detailsMeta.className = 'player-inventory-overlay__tooltip-meta'
  detailsDescription.className = 'player-inventory-overlay__tooltip-description'
  detailsHint.className = 'player-inventory-overlay__tooltip-hint'
  detailsText.append(detailsName, detailsMeta, detailsDescription, detailsHint)
  detailsHeader.append(detailsIcon, detailsText)
  detailsPanel.append(detailsHeader)
  detailsPanel.hidden = true
  detailsPanel.style.display = 'none'
  detailsPanel.setAttribute('aria-hidden', 'true')

  emptyStateElement.className = 'player-inventory-overlay__empty-state'
  emptyStateElement.textContent = '이 탭에는 아이템이 없습니다'

  for (let index = 0; index < initialInventory.slots.length; index += 1) {
    const slotButton = document.createElement('button')
    const slotIndexLabel = document.createElement('span')
    const slotIcon = document.createElement('span')
    const slotItemLabel = document.createElement('span')
    const slotQuantityLabel = document.createElement('span')
    const slotEmptyMarker = document.createElement('span')

    slotButton.type = 'button'
    slotButton.className = 'player-inventory-overlay__slot'
    slotButton.dataset.playerInventorySlotIndex = String(index)
    slotButton.setAttribute('aria-label', `빈 가방 칸 ${index + 1}`)
    slotButton.draggable = false

    slotIndexLabel.className = 'player-inventory-overlay__slot-index'
    slotIndexLabel.textContent = String(index + 1).padStart(2, '0')
    slotIndexLabel.hidden = true

    slotIcon.className = 'player-inventory-overlay__slot-icon'
    slotIcon.setAttribute('aria-hidden', 'true')

    slotItemLabel.className = 'player-inventory-overlay__slot-item'
    slotItemLabel.hidden = true

    slotQuantityLabel.className = 'player-inventory-overlay__slot-quantity'
    slotQuantityLabel.hidden = true

    slotEmptyMarker.className = 'player-inventory-overlay__slot-empty-marker'
    slotEmptyMarker.setAttribute('aria-hidden', 'true')
    slotEmptyMarker.hidden = true

    slotButton.append(
      slotEmptyMarker,
      slotIndexLabel,
      slotIcon,
      slotItemLabel,
      slotQuantityLabel
    )
    slotGrid.append(slotButton)

    slotButtons.push(slotButton)
    slotIndexLabels.push(slotIndexLabel)
    slotIcons.push(slotIcon)
    slotItemLabels.push(slotItemLabel)
    slotQuantityLabels.push(slotQuantityLabel)
    slotEmptyMarkers.push(slotEmptyMarker)
    slotButton.addEventListener('dragstart', handleSlotGridDragStart)
    slotButton.addEventListener('pointerenter', showTooltipForSlotButton)
    slotButton.addEventListener('pointerleave', hideTooltip)
  }

  titleGroup.append(titleElement, summaryElement)
  headerRow.append(titleGroup, closeButton)
  closeButton.append(closeIcon)
  panelBody.append(headerRow, tabBar, slotGrid, emptyStateElement)
  panel.append(panelBody)
  overlayRoot.append(backdropButton, panel, detailsPanel)
  mountElement.append(overlayRoot)

  const clampPanelPosition = (
    left: number,
    top: number,
    width: number,
    height: number
  ) => ({
    left: clamp(
      left,
      PANEL_MARGIN,
      Math.max(PANEL_MARGIN, window.innerWidth - width - PANEL_MARGIN)
    ),
    top: clamp(
      top,
      PANEL_MARGIN,
      Math.max(PANEL_MARGIN, window.innerHeight - height - PANEL_MARGIN)
    )
  })

  const stopDragging = (pointerId?: number) => {
    if (dragState && pointerId !== undefined && dragState.pointerId !== pointerId) {
      return
    }

    dragState = undefined
    headerRow.classList.remove('player-inventory-overlay__header--dragging')
  }

  const startDragging = (event: PointerEvent) => {
    if (event.button !== 0 || panel.hidden || !event.isPrimary) {
      return
    }

    const target = event.target

    if (!(target instanceof Element) || closeButton.contains(target)) {
      return
    }

    const rect = panel.getBoundingClientRect()

    dragState = {
      pointerId: event.pointerId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
      width: rect.width,
      height: rect.height
    }
    hasPanelPosition = true
    headerRow.classList.add('player-inventory-overlay__header--dragging')
    event.preventDefault()
  }

  const handleDocumentPointerMove = (event: PointerEvent) => {
    lastPointerPosition = {
      x: event.clientX,
      y: event.clientY
    }

    if (dragState && event.pointerId === dragState.pointerId) {
      panelPosition = clampPanelPosition(
        event.clientX - dragState.offsetX,
        event.clientY - dragState.offsetY,
        dragState.width,
        dragState.height
      )
      panel.style.left = `${panelPosition.left}px`
      panel.style.top = `${panelPosition.top}px`
    }

    syncHoveredSlotTooltip()
  }

  const handleDocumentPointerUp = (event: PointerEvent) => {
    stopDragging(event.pointerId)
  }

  const setEmptySlotAppearance = (element: HTMLElement) => {
    element.style.backgroundImage = 'none'
    element.style.backgroundRepeat = 'no-repeat'
    element.style.backgroundPosition = '0 0'
    element.style.backgroundSize = '100% 100%'
    element.style.backgroundColor = '#fff9ee'
    element.style.border = '1px solid rgba(111, 89, 58, 0.42)'
    element.style.boxShadow = '0 0 0 1px rgba(255, 255, 255, 0.12)'
  }

  const setCardDimensions = (
    element: HTMLElement,
    width: number,
    height: number
  ) => {
    element.style.width = `${width}px`
    element.style.height = `${height}px`
  }

  const isLikelyConsumableInventoryItem = (item: {
    id: string
    label: string
  }): boolean => {
    if (getPlayerEquipmentItemDefinitionById(item.id)) {
      return false
    }

    return /potion|elixir|antidote|herb|food|drink|scroll|물약|포션|회복|음료/.test(
      `${item.label} ${item.id}`.toLowerCase()
    )
  }

  const getInventoryItemCategory = (item: {
    id: string
    label: string
  }): InventoryCategoryId => {
    if (getPlayerEquipmentItemDefinitionById(item.id)) {
      return 'equipment'
    }

    if (isLikelyConsumableInventoryItem(item)) {
      return 'consumable'
    }

    return 'other'
  }

  const getInventoryCategoryLabel = (
    category: InventoryCategoryId
  ): string => {
    switch (category) {
      case 'all':
        return '전체'
      case 'equipment':
        return '장비'
      case 'consumable':
        return '소비'
      case 'other':
        return '기타'
    }
  }

  const getInventorySlotButton = (target: EventTarget | null) => {
    if (!(target instanceof Element)) {
      return undefined
    }

    return target.closest('button[data-player-inventory-slot-index]') as
      | HTMLButtonElement
      | undefined
  }

  const getHoveredInventorySlotButton = (
    clientX: number,
    clientY: number
  ) => {
    for (const slotButton of slotButtons) {
      if (slotButton.hidden) {
        continue
      }

      const rect = slotButton.getBoundingClientRect()

      if (
        clientX >= rect.left &&
        clientX < rect.right &&
        clientY >= rect.top &&
        clientY < rect.bottom
      ) {
        return slotButton
      }
    }

    return undefined
  }

  const updateTooltipForSlotButton = (
    slotButton: HTMLButtonElement,
    clientX: number,
    clientY: number
  ) => {
    const slotIndex = Number(slotButton.dataset.playerInventorySlotIndex)

    if (Number.isNaN(slotIndex)) {
      hideTooltip()
      return
    }

    hoveredSlotIndex = slotIndex
    lastPointerPosition = {
      x: clientX,
      y: clientY
    }
    tooltipAnchor = {
      x: clientX,
      y: clientY
    }
    syncTooltip()
  }

  function showTooltipForSlotButton(event: PointerEvent) {
    const slotButton = getInventorySlotButton(event.currentTarget)

    if (!slotButton) {
      hideTooltip()
      return
    }

    updateTooltipForSlotButton(slotButton, event.clientX, event.clientY)
  }

  const syncHoveredSlotTooltip = () => {
    if (!lastPointerPosition) {
      hideTooltip()
      return
    }

    const slotButton = getHoveredInventorySlotButton(
      lastPointerPosition.x,
      lastPointerPosition.y
    )

    if (!slotButton) {
      hideTooltip()
      return
    }

    updateTooltipForSlotButton(
      slotButton,
      lastPointerPosition.x,
      lastPointerPosition.y
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
    element.hidden = false
    element.style.backgroundImage = `url(${frame.imageUrl})`
    element.style.backgroundRepeat = 'no-repeat'
    element.style.backgroundPosition = `-${frame.frame.x * scale}px -${frame.frame.y * scale}px`
    element.style.backgroundSize = `${frame.imageWidth * scale}px ${frame.imageHeight * scale}px`
    element.style.width = `${frame.frame.width * scale}px`
    element.style.height = `${frame.frame.height * scale}px`
    element.style.imageRendering = 'pixelated'
  }

  const getConsumableIconFrame = (itemId: string) => {
    switch (itemId) {
      case 'health-potion':
        return TINY_DUNGEON_CONSUMABLE_ICON_FRAMES['health-potion']
      case 'mana-potion':
        return TINY_DUNGEON_CONSUMABLE_ICON_FRAMES['mana-potion']
      default:
        return undefined
    }
  }

  const renderConsumableIcon = (
    element: HTMLElement,
    itemId: string | undefined,
    scale: number
  ) => {
    const frame = itemId === undefined ? undefined : getConsumableIconFrame(itemId)

    if (frame) {
      setBackgroundFrame(element, frame, scale)
      return
    }

    setBackgroundFrame(
      element,
      EQUIPMENT_ICON_FRAME_BY_KEY['ui-circle-beige'],
      scale
    )
  }

  const clearFrame = (element: HTMLElement) => {
    element.hidden = true
    element.style.backgroundImage = 'none'
  }

  const renderEquipmentIcon = (
    element: HTMLElement,
    itemId: string | undefined,
    scale: number
  ) => {
    if (!itemId) {
      setBackgroundFrame(element, EQUIPMENT_ICON_FRAME_BY_KEY['ui-circle-beige'], scale)
      return
    }

    const definition = getPlayerEquipmentItemDefinitionById(itemId)

    if (!definition) {
      setBackgroundFrame(element, EQUIPMENT_ICON_FRAME_BY_KEY['ui-circle-beige'], scale)
      return
    }

    setBackgroundFrame(
      element,
      EQUIPMENT_ICON_FRAME_BY_KEY[definition.icon.key],
      scale * definition.icon.scale
    )
  }

  const renderInventorySlotIcon = (
    element: HTMLElement,
    item:
      | {
          id: string
          label: string
          quantity: number
        }
      | undefined,
    scale: number
  ) => {
    if (!item) {
      clearFrame(element)
      return
    }

    const definition = getPlayerEquipmentItemDefinitionById(item.id)

    if (definition) {
      renderEquipmentIcon(element, item.id, scale)
      return
    }

    if (isLikelyConsumableInventoryItem(item)) {
      renderConsumableIcon(element, item.id, scale * 0.96)
      return
    }

    setBackgroundFrame(
      element,
      EQUIPMENT_ICON_FRAME_BY_KEY['ui-check-beige'],
      scale * 0.96
    )
  }

  function hideTooltip() {
    hoveredSlotIndex = undefined
    detailsPanel.hidden = true
    detailsPanel.style.display = 'none'
    detailsPanel.setAttribute('aria-hidden', 'true')
    detailsIcon.hidden = true
  }

  const syncTooltip = () => {
    const inventory = getInventory()
    const quickslots = getQuickslots()

    if (
      hoveredSlotIndex === undefined ||
      hoveredSlotIndex < 0 ||
      hoveredSlotIndex >= inventory.slots.length
    ) {
      hideTooltip()
      return
    }

    const selectedSlot = inventory.slots[hoveredSlotIndex]

    if (
      !selectedSlot ||
      (selectedCategory !== 'all' &&
        getInventoryItemCategory(selectedSlot) !== selectedCategory)
    ) {
      hideTooltip()
      return
    }

    const selectedDefinition = getPlayerEquipmentItemDefinitionById(
      selectedSlot.id
    )
    const selectedCategoryLabel = getInventoryCategoryLabel(
      getInventoryItemCategory(selectedSlot)
    )
    const selectedQuickslotIndex = findPlayerQuickslotIndexByInventorySlotIndex(
      quickslots,
      hoveredSlotIndex
    )

    detailsName.textContent = selectedSlot.label
    detailsMeta.textContent = `${selectedCategoryLabel}${selectedSlot.quantity > 1 ? ` · x${selectedSlot.quantity}` : ''}${selectedQuickslotIndex !== undefined && isLikelyConsumableInventoryItem(selectedSlot) ? ` · 퀵슬롯 ${selectedQuickslotIndex + 1}` : ''}`
    detailsDescription.textContent =
      selectedDefinition?.description ??
      (isLikelyConsumableInventoryItem(selectedSlot)
        ? '회복 아이템입니다.'
        : '기타 아이템입니다.')
    detailsHint.textContent = selectedDefinition
      ? `장비창으로 드래그해서 ${getPlayerEquipmentSlotLabelById(selectedDefinition.slotId)}에 장착 · 더블클릭하면 장착`
      : isLikelyConsumableInventoryItem(selectedSlot)
        ? '퀵슬롯으로 드래그해서 등록 · 더블클릭하면 사용'
        : '드래그해서 이동할 수 있습니다'

    if (selectedDefinition) {
      renderEquipmentIcon(detailsIcon, selectedSlot.id, 1.02)
      detailsIcon.hidden = false
    } else if (isLikelyConsumableInventoryItem(selectedSlot)) {
      renderConsumableIcon(detailsIcon, selectedSlot.id, 1.02)
      detailsIcon.hidden = false
    } else {
      clearFrame(detailsIcon)
    }
    detailsPanel.hidden = false
    detailsPanel.style.display = 'grid'
    detailsPanel.setAttribute('aria-hidden', 'false')
    detailsPanel.style.visibility = 'hidden'
    detailsPanel.style.left = '0px'
    detailsPanel.style.top = '0px'

    const tooltipRect = detailsPanel.getBoundingClientRect()
    const maxLeft = Math.max(
      OVERLAY_MARGIN,
      window.innerWidth - tooltipRect.width - OVERLAY_MARGIN
    )
    const maxTop = Math.max(
      OVERLAY_MARGIN,
      window.innerHeight - tooltipRect.height - OVERLAY_MARGIN
    )

    detailsPanel.style.left = `${clamp(
      tooltipAnchor.x + 14,
      OVERLAY_MARGIN,
      maxLeft
    )}px`
    detailsPanel.style.top = `${clamp(
      tooltipAnchor.y + 12,
      OVERLAY_MARGIN,
      maxTop
    )}px`
    detailsPanel.style.visibility = 'visible'
  }

  const syncLayout = () => {
    const isOpen = getIsOpen()
    const uiScale = getResponsiveUiScale()

    if (isOpen && !wasOpen) {
      selectedCategory = 'equipment'
    }

    overlayRoot.hidden = !isOpen
    overlayRoot.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      stopDragging()
      backdropButton.hidden = true
      panel.hidden = true
      hideTooltip()
      wasOpen = false
      return
    }
    const inventorySlotScale = 0.92

    backdropButton.hidden = false
    panel.hidden = false
    panel.style.transformOrigin = 'left top'
    panel.style.transform = `scale(${uiScale})`
    panel.style.backgroundImage = 'none'
    panel.style.backgroundRepeat = 'no-repeat'
    panel.style.backgroundPosition = '0 0'
    panel.style.backgroundSize = '100% 100%'
    panel.style.width = 'max-content'
    panel.style.height = 'max-content'
    panelBody.style.width = 'max-content'
    panelBody.style.height = 'max-content'
    panelBody.style.gridTemplateColumns = 'max-content'

    const inventory = getInventory()
    const quickslots = getQuickslots()
    const filledSlotCount = getPlayerInventoryFilledSlotCount(inventory)

    let visibleSlotCount = 0

    for (let index = 0; index < slotButtons.length; index += 1) {
      const slotButton = slotButtons[index]
      const slotIndexLabel = slotIndexLabels[index]
      const slotIcon = slotIcons[index]
      const slotItemLabel = slotItemLabels[index]
      const slotQuantityLabel = slotQuantityLabels[index]
      const slotEmptyMarker = slotEmptyMarkers[index]
      const slot = inventory.slots[index]
      const slotDefinition = slot
        ? getPlayerEquipmentItemDefinitionById(slot.id)
        : undefined
      const slotCategory = slot ? getInventoryItemCategory(slot) : undefined
      const assignedQuickslotIndex = findPlayerQuickslotIndexByInventorySlotIndex(
        quickslots,
        index
      )
      const isDraggableSlot = Boolean(
        slot && (slotDefinition || isLikelyConsumableInventoryItem(slot))
      )
      const isConsumableSlot = Boolean(
        slot && !slotDefinition && isLikelyConsumableInventoryItem(slot)
      )
      const isVisible =
        slot === undefined ||
        selectedCategory === 'all' ||
        slotCategory === selectedCategory

      setEmptySlotAppearance(slotButton)
      setCardDimensions(
        slotButton,
        BUTTON_SQUARE_FRAME.width * inventorySlotScale,
        BUTTON_SQUARE_FRAME.height * inventorySlotScale
      )
      slotButton.hidden = !isVisible
      slotButton.style.display = isVisible ? '' : 'none'
      if (assignedQuickslotIndex !== undefined && isConsumableSlot) {
        slotButton.style.borderColor = 'rgba(91, 134, 214, 0.42)'
        slotButton.style.boxShadow =
          'inset 0 1px 0 rgba(255, 255, 255, 0.7), 0 0 0 1px rgba(91, 134, 214, 0.08)'
      } else {
        slotButton.style.borderColor = 'rgba(111, 89, 58, 0.34)'
        slotButton.style.boxShadow =
          'inset 0 1px 0 rgba(255, 255, 255, 0.7)'
      }
      slotButton.style.padding = `${2 * inventorySlotScale}px ${3 * inventorySlotScale}px ${3 * inventorySlotScale}px`
      slotButton.style.fontSize =
        inventorySlotScale >= 1.2 ? '0.66rem' : '0.62rem'
      slotButton.classList.toggle(
        'player-inventory-overlay__slot--equippable',
        Boolean(slotDefinition)
      )
      slotButton.classList.toggle(
        'player-inventory-overlay__slot--draggable',
        isDraggableSlot
      )
      slotButton.classList.toggle(
        'player-inventory-overlay__slot--assigned',
        assignedQuickslotIndex !== undefined && isConsumableSlot
      )
      slotButton.draggable = isDraggableSlot && isVisible
      const targetSlotLabel = slotDefinition
        ? getPlayerEquipmentSlotLabelById(slotDefinition.slotId)
        : undefined
      const quickslotHint =
        assignedQuickslotIndex !== undefined && isConsumableSlot
          ? ` · 퀵슬롯 ${assignedQuickslotIndex + 1}에 등록됨`
          : ''
      const equipmentDragHint = slotDefinition
        ? ` · 장비창으로 드래그해서 ${targetSlotLabel}에 장착`
        : ''
      const consumableDragHint = isConsumableSlot
        ? ' · 드래그해서 퀵슬롯에 등록'
        : ''
      const equipmentDoubleClickHint = slotDefinition
        ? ` · 더블클릭하면 ${targetSlotLabel}에 장착`
        : ''
      const consumableDoubleClickHint = isConsumableSlot
        ? ' · 더블클릭하면 사용'
        : ''
      const slotLabelText = slot
        ? slotDefinition
          ? `${slot.label}${slot.quantity > 1 ? ` x${slot.quantity}` : ''}${equipmentDragHint}${equipmentDoubleClickHint}`
          : `${slot.label}${slot.quantity > 1 ? ` x${slot.quantity}` : ''}${quickslotHint}${consumableDragHint}${consumableDoubleClickHint}`
        : `빈 가방 칸 ${index + 1}`

      slotIndexLabel.hidden = true
      slotButton.classList.toggle(
        'player-inventory-overlay__slot--filled',
        slot !== undefined
      )
      slotButton.classList.toggle(
        'player-inventory-overlay__slot--empty',
        slot === undefined
      )
      slotButton.setAttribute('aria-label', slotLabelText)
      slotButton.title = slotLabelText

      if (slot && isVisible) {
        visibleSlotCount += 1
      }

      if (slot) {
        slotQuantityLabel.hidden = slot.quantity <= 1
        slotQuantityLabel.textContent = slot.quantity > 1 ? `x${slot.quantity}` : ''
        slotItemLabel.hidden = true
        slotItemLabel.textContent = ''
        slotEmptyMarker.hidden = true
        renderInventorySlotIcon(slotIcon, slot, inventorySlotScale)
      } else {
        slotItemLabel.hidden = true
        slotItemLabel.textContent = ''
        slotQuantityLabel.hidden = true
        slotQuantityLabel.textContent = ''
        slotEmptyMarker.hidden = true
        clearFrame(slotIcon)
      }
    }

    panel.style.left = `${PANEL_MARGIN}px`
    panel.style.top = `${PANEL_MARGIN}px`
    const measuredRect = panel.getBoundingClientRect()
    const renderedWidth = measuredRect.width
    const renderedHeight = measuredRect.height
    const defaultPosition = clampPanelPosition(
      (window.innerWidth - renderedWidth) / 2,
      (window.innerHeight - renderedHeight) / 2,
      renderedWidth,
      renderedHeight
    )

    if (!hasPanelPosition) {
      panelPosition = defaultPosition
    } else {
      panelPosition = clampPanelPosition(
        panelPosition.left,
        panelPosition.top,
        renderedWidth,
        renderedHeight
      )
    }

    panel.style.left = `${panelPosition.left}px`
    panel.style.top = `${panelPosition.top}px`

    syncHoveredSlotTooltip()

    for (const tabButton of tabButtons) {
      const isActive =
        tabButton.dataset.playerInventoryCategoryId === selectedCategory

      tabButton.classList.toggle('player-inventory-overlay__tab--active', isActive)
      tabButton.setAttribute('aria-pressed', String(isActive))
    }

    summaryElement.textContent = `${filledSlotCount} / ${inventory.slots.length}칸 · ${formatGoldAmount(inventory.gold)}`
    emptyStateElement.hidden =
      selectedCategory === 'all' || visibleSlotCount > 0
    wasOpen = true
  }

  const handleInventorySlotDoubleClick = (slotIndex: number) => {
    const inventory = getInventory()
    const item = inventory.slots[slotIndex]

    if (!item) {
      return
    }

    const equipmentDefinition = getPlayerEquipmentItemDefinitionById(item.id)

    if (equipmentDefinition) {
      const nextState = equipPlayerInventorySlot({
        equipment: getEquipment(),
        inventory,
        slotIndex
      })

      if (!nextState) {
        syncFrame()
        return
      }

      onRequestEquipmentChange(nextState.equipment)
      onRequestInventoryChange(nextState.inventory)
      syncFrame()
      return
    }

    const nextState = usePlayerInventoryConsumable({
      profile,
      inventory,
      slotIndex
    })

    if (!nextState) {
      syncFrame()
      return
    }

    onRequestProfileChange(nextState.profile)
    onRequestInventoryChange(nextState.inventory)
    onConsumableUsed?.(item.id)
    syncFrame()
  }

  const handlePanelDoubleClick = (event: MouseEvent) => {
    const target = event.target

    if (!(target instanceof Element)) {
      return
    }

    const inventoryButton = target.closest(
      'button[data-player-inventory-slot-index]'
    ) as HTMLButtonElement | null

    if (inventoryButton) {
      event.preventDefault()
      event.stopPropagation()
      const slotIndex = Number(
        inventoryButton.dataset.playerInventorySlotIndex
      )

      if (!Number.isNaN(slotIndex)) {
        event.preventDefault()
        event.stopPropagation()
        handleInventorySlotDoubleClick(slotIndex)
      }

      return
    }
  }

  function handleSlotGridDragStart(event: DragEvent) {
    const slotButton =
      event.currentTarget instanceof HTMLButtonElement
        ? event.currentTarget
        : undefined

    if (!slotButton || !event.dataTransfer) {
      return
    }

    const slotIndex = Number(slotButton.dataset.playerInventorySlotIndex)
    const slot = getInventory().slots[slotIndex]
    const slotDefinition = slot ? getPlayerEquipmentItemDefinitionById(slot.id) : undefined
    const isConsumableSlot = Boolean(
      slot && !slotDefinition && isLikelyConsumableInventoryItem(slot)
    )

    if (!slot || (!slotDefinition && !isConsumableSlot)) {
      event.preventDefault()
      return
    }

    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('application/x-player-drag-origin', 'inventory')
    event.dataTransfer.setData(
      'application/x-player-inventory-slot-index',
      String(slotIndex)
    )
    event.dataTransfer.setData('text/plain', String(slotIndex))
  }

  const handleBackdropClick = (event: MouseEvent) => {
    event.preventDefault()
    event.stopPropagation()
    stopDragging()
    onRequestOpenChange(false)
  }

  const handleBackdropPointerDown = (event: PointerEvent) => {
    event.preventDefault()
    event.stopPropagation()
    stopDragging()
    onRequestOpenChange(false)
  }

  const handleCloseButtonClick = (event: MouseEvent) => {
    event.preventDefault()
    event.stopPropagation()
    stopDragging()
    onRequestOpenChange(false)
  }

  const handleCloseButtonPointerDown = (event: PointerEvent) => {
    event.preventDefault()
    event.stopPropagation()
    stopDragging()
    onRequestOpenChange(false)
  }

  backdropButton.addEventListener('click', handleBackdropClick)
  backdropButton.addEventListener('pointerdown', handleBackdropPointerDown)
  headerRow.addEventListener('pointerdown', startDragging)
  panel.addEventListener('pointerleave', hideTooltip)
  document.addEventListener('pointermove', handleDocumentPointerMove)
  document.addEventListener('pointerup', handleDocumentPointerUp)
  document.addEventListener('pointercancel', handleDocumentPointerUp)
  closeButton.addEventListener('click', handleCloseButtonClick)
  closeButton.addEventListener('pointerdown', handleCloseButtonPointerDown)
  panel.addEventListener('dblclick', handlePanelDoubleClick)

  const syncFrame = () => {
    syncLayout()
  }

  const destroy = () => {
    stopDragging()
    backdropButton.removeEventListener('click', handleBackdropClick)
    backdropButton.removeEventListener('pointerdown', handleBackdropPointerDown)
    headerRow.removeEventListener('pointerdown', startDragging)
    panel.removeEventListener('pointerleave', hideTooltip)
    for (const slotButton of slotButtons) {
      slotButton.removeEventListener('dragstart', handleSlotGridDragStart)
    }
    document.removeEventListener('pointermove', handleDocumentPointerMove)
    document.removeEventListener('pointerup', handleDocumentPointerUp)
    document.removeEventListener('pointercancel', handleDocumentPointerUp)
    closeButton.removeEventListener('click', handleCloseButtonClick)
    closeButton.removeEventListener('pointerdown', handleCloseButtonPointerDown)
    panel.removeEventListener('dblclick', handlePanelDoubleClick)
    overlayRoot.remove()
  }

  syncFrame()

  return {
    syncFrame,
    destroy
  }
}

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))

const formatGoldAmount = (gold: number): string =>
  `${gold.toLocaleString('ko-KR')}원`
