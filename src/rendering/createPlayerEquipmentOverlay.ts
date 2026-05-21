import type {
  PlayerEquipment,
  PlayerEquipmentIconKey,
  PlayerEquipmentSlotId
} from '../game/playerEquipment'
import {
  getPlayerEquipmentItemDefinitionById,
  getPlayerEquipmentSlotLabelById
} from '../game/playerEquipment'
import { PLAYER_CHARACTER_APPEARANCE_TYPE } from '../game/characterState'
import type { PlayerInventory } from '../game/playerInventory'
import type { PlayerProfile } from '../game/playerProfile'
import {
  PLAYER_MAX_LEVEL,
  getPlayerJobDisplayName
} from '../game/playerProfile'
import {
  equipPlayerInventorySlot,
  unequipPlayerEquipmentSlot
} from '../game/playerLoadout'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreatePlayerEquipmentOverlayInput = {
  mountElement: HTMLElement
  profile: PlayerProfile
  getInventory: () => PlayerInventory
  getEquipment: () => PlayerEquipment
  getIsOpen: () => boolean
  onRequestOpenChange: (isOpen: boolean) => void
  onRequestInventoryChange: (nextInventory: PlayerInventory) => void
  onRequestEquipmentChange: (nextEquipment: PlayerEquipment) => void
}

export type PlayerEquipmentOverlay = {
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
const PANEL_INSET_FRAME = {
  x: 200,
  y: 294,
  width: 93,
  height: 94
}
const PANEL_LIGHT_FRAME = {
  x: 190,
  y: 200,
  width: 93,
  height: 94
}
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
const PLAYER_PORTRAIT_FRAME_BY_APPEARANCE_TYPE: Record<
  string,
  {
    imageUrl: string
    imageWidth: number
    imageHeight: number
    frame: { x: number; y: number; width: number; height: number }
  }
> = {
  [PLAYER_CHARACTER_APPEARANCE_TYPE]: {
    imageUrl: TINY_DUNGEON_TILESET_IMAGE_URL,
    imageWidth: TINY_DUNGEON_TILESET_WIDTH,
    imageHeight: TINY_DUNGEON_TILESET_HEIGHT,
    frame: {
      x: 32,
      y: 128,
      width: 16,
      height: 16
    }
  }
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
const PANEL_MIN_SCALE = 3
const PANEL_MAX_SCALE = 4
const EQUIPMENT_SLOT_SCALE_THRESHOLD = 4
const PANEL_MARGIN = 16

const EQUIPMENT_SLOT_AREA_CLASS_BY_ID: Record<PlayerEquipmentSlotId, string> = {
  weapon: 'player-inventory-overlay__equipment-slot--weapon',
  armor: 'player-inventory-overlay__equipment-slot--armor',
  hat: 'player-inventory-overlay__equipment-slot--hat',
  boots: 'player-inventory-overlay__equipment-slot--boots',
  accessory: 'player-inventory-overlay__equipment-slot--accessory'
}

export const createPlayerEquipmentOverlay = ({
  mountElement,
  profile,
  getInventory,
  getEquipment,
  getIsOpen,
  onRequestOpenChange,
  onRequestInventoryChange,
  onRequestEquipmentChange
}: CreatePlayerEquipmentOverlayInput): PlayerEquipmentOverlay => {
  const overlayRoot = document.createElement('div')
  const backdropButton = document.createElement('button')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const headerRow = document.createElement('div')
  const titleGroup = document.createElement('div')
  const titleElement = document.createElement('div')
  const equipmentSection = document.createElement('section')
  const equipmentSectionHeader = document.createElement('div')
  const equipmentSectionTitle = document.createElement('div')
  const equipmentSectionSummary = document.createElement('div')
  const equipmentLayout = document.createElement('div')
  const equipmentCenterCard = document.createElement('div')
  const equipmentCenterPreview = document.createElement('div')
  const equipmentCenterPortrait = document.createElement('span')
  const equipmentCenterPortraitWeapon = document.createElement('span')
  const equipmentCenterPortraitAccessory = document.createElement('span')
  const equipmentCenterDetails = document.createElement('div')
  const equipmentCenterName = document.createElement('div')
  const equipmentCenterJob = document.createElement('div')
  const equipmentCenterLevel = document.createElement('div')
  const equipmentCenterSet = document.createElement('div')
  const closeButton = document.createElement('button')
  const closeIcon = document.createElement('span')
  const tooltipPanel = document.createElement('div')
  const tooltipHeader = document.createElement('div')
  const tooltipIcon = document.createElement('span')
  const tooltipText = document.createElement('div')
  const tooltipName = document.createElement('div')
  const tooltipMeta = document.createElement('div')
  const tooltipDescription = document.createElement('div')
  const tooltipHint = document.createElement('div')
  const equipmentSlotCards: HTMLButtonElement[] = []
  const equipmentSlotHeaderLabels: HTMLSpanElement[] = []
  const equipmentSlotItemLabels: HTMLDivElement[] = []
  const equipmentSlotDescriptionLabels: HTMLDivElement[] = []
  const equipmentSlotLevelBadges: HTMLDivElement[] = []
  const equipmentSlotIcons: HTMLSpanElement[] = []
  const initialEquipment = getEquipment()
  let hoveredEquipmentSlotIndex: number | undefined
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

  overlayRoot.className = 'player-equipment-overlay'

  backdropButton.type = 'button'
  backdropButton.className = 'player-equipment-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.tabIndex = -1
  backdropButton.setAttribute('aria-hidden', 'true')

  panel.id = 'player-equipment-panel'
  panel.className = 'player-inventory-overlay__panel'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'false')
  panel.setAttribute('aria-labelledby', 'player-equipment-title')

  panelBody.className = 'player-inventory-overlay__panel-body'
  panelBody.style.gridTemplateColumns = 'minmax(0, 1fr)'
  panelBody.style.gridTemplateAreas = '"header" "equipment" "footer"'
  panelBody.style.alignContent = 'start'
  panelBody.style.gap = '6px'
  panelBody.style.padding = '10px 12px 10px'

  headerRow.className = 'player-inventory-overlay__header'
  titleGroup.className = 'player-inventory-overlay__title-group'
  titleElement.id = 'player-equipment-title'
  titleElement.className = 'player-inventory-overlay__title'
  titleElement.textContent = '장비'

  equipmentSection.className = 'player-inventory-overlay__equipment-section'
  equipmentSection.style.gridArea = 'equipment'
  equipmentSectionHeader.className = 'player-inventory-overlay__equipment-header'
  equipmentSectionTitle.className = 'player-inventory-overlay__equipment-title'
  equipmentSectionTitle.textContent = '장비'
  equipmentSectionTitle.hidden = true
  equipmentSectionSummary.className =
    'player-inventory-overlay__equipment-summary'
  equipmentSectionSummary.textContent = '착용 중인 장비'

  equipmentLayout.className = 'player-inventory-overlay__equipment-layout'
  equipmentLayout.setAttribute('role', 'list')

  equipmentCenterCard.className = 'player-inventory-overlay__equipment-center'
  equipmentCenterPreview.className =
    'player-inventory-overlay__equipment-center-preview'
  equipmentCenterPortrait.className =
    'player-inventory-overlay__equipment-center-portrait'
  equipmentCenterPortraitWeapon.className =
    'player-inventory-overlay__equipment-center-gear player-inventory-overlay__equipment-center-gear--weapon'
  equipmentCenterPortraitAccessory.className =
    'player-inventory-overlay__equipment-center-gear player-inventory-overlay__equipment-center-gear--accessory'
  equipmentCenterDetails.className =
    'player-inventory-overlay__equipment-center-details'
  equipmentCenterName.className = 'player-inventory-overlay__equipment-center-name'
  equipmentCenterName.textContent = profile.name
  equipmentCenterJob.className = 'player-inventory-overlay__equipment-center-job'
  equipmentCenterJob.textContent = getPlayerJobDisplayName(profile)
  equipmentCenterLevel.className =
    'player-inventory-overlay__equipment-center-level'
  equipmentCenterLevel.textContent =
    profile.level >= PLAYER_MAX_LEVEL
      ? `레벨 ${profile.level} · MAX`
      : `레벨 ${profile.level}`
  equipmentCenterSet.className = 'player-inventory-overlay__equipment-center-set'
  equipmentCenterSet.textContent = initialEquipment.setName
  equipmentCenterSet.hidden = initialEquipment.setName === '기본 장비'

  closeButton.type = 'button'
  closeButton.className = 'player-equipment-overlay__close player-stat-overlay__close'
  closeButton.setAttribute('aria-label', '장비 창 닫기')
  closeButton.title = '장비 창 닫기 (Esc)'

  closeIcon.className = 'player-equipment-overlay__close-icon player-stat-overlay__close-icon'
  closeIcon.textContent = '×'
  closeIcon.setAttribute('aria-hidden', 'true')

  tooltipPanel.className = 'player-inventory-overlay__tooltip'
  tooltipPanel.hidden = true
  tooltipPanel.setAttribute('aria-hidden', 'true')
  tooltipPanel.style.display = 'none'

  tooltipHeader.className = 'player-inventory-overlay__tooltip-header'
  tooltipIcon.className = 'player-inventory-overlay__tooltip-icon'
  tooltipIcon.setAttribute('aria-hidden', 'true')
  tooltipText.className = 'player-inventory-overlay__tooltip-text'
  tooltipName.className = 'player-inventory-overlay__tooltip-name'
  tooltipMeta.className = 'player-inventory-overlay__tooltip-meta'
  tooltipDescription.className = 'player-inventory-overlay__tooltip-description'
  tooltipHint.className = 'player-inventory-overlay__tooltip-hint'
  tooltipText.append(tooltipName, tooltipMeta, tooltipDescription, tooltipHint)
  tooltipHeader.append(tooltipIcon, tooltipText)
  tooltipPanel.append(tooltipHeader)

  for (const slot of initialEquipment.slots) {
    const slotCard = document.createElement('button')
    const slotHeaderLabel = document.createElement('span')
    const slotItemLabel = document.createElement('div')
    const slotDescriptionLabel = document.createElement('div')
    const slotLevelBadge = document.createElement('div')
    const slotIcon = document.createElement('span')

    slotCard.className = 'player-inventory-overlay__equipment-slot'
    slotCard.classList.add(EQUIPMENT_SLOT_AREA_CLASS_BY_ID[slot.id])
    slotCard.type = 'button'
    slotCard.dataset.playerEquipmentSlotId = slot.id
    slotCard.setAttribute('role', 'listitem')
    slotCard.setAttribute('aria-label', `${slot.label} 장비칸`)

    slotHeaderLabel.className = 'player-inventory-overlay__equipment-slot-label'
    slotHeaderLabel.textContent = slot.label
    slotHeaderLabel.hidden = true

    slotItemLabel.className = 'player-inventory-overlay__equipment-slot-item'
    slotItemLabel.textContent = slot.item?.label ?? '비어 있음'

    slotDescriptionLabel.className =
      'player-inventory-overlay__equipment-slot-description'
    slotDescriptionLabel.textContent = slot.item?.description ?? '가방에서 장착하세요'
    slotDescriptionLabel.hidden = true

    slotLevelBadge.className = 'player-inventory-overlay__equipment-slot-level'
    slotLevelBadge.textContent = `레벨 ${slot.item?.level ?? '-'}`
    slotLevelBadge.hidden = true

    slotIcon.className = 'player-inventory-overlay__equipment-slot-icon'
    slotIcon.setAttribute('aria-hidden', 'true')

    slotCard.append(
      slotIcon,
      slotHeaderLabel,
      slotItemLabel,
      slotDescriptionLabel,
      slotLevelBadge
    )
    equipmentLayout.append(slotCard)

    equipmentSlotCards.push(slotCard)
    equipmentSlotHeaderLabels.push(slotHeaderLabel)
    equipmentSlotItemLabels.push(slotItemLabel)
    equipmentSlotDescriptionLabels.push(slotDescriptionLabel)
    equipmentSlotLevelBadges.push(slotLevelBadge)
    equipmentSlotIcons.push(slotIcon)
    slotCard.addEventListener('dragover', handleEquipmentLayoutDragOver)
    slotCard.addEventListener('drop', handleEquipmentLayoutDrop)
    slotCard.addEventListener('pointerenter', showTooltipForEquipmentSlot)
    slotCard.addEventListener('pointermove', showTooltipForEquipmentSlot)
    slotCard.addEventListener('pointerleave', hideTooltip)
  }

  equipmentLayout.append(equipmentCenterCard)
  equipmentCenterPreview.append(
    equipmentCenterPortrait,
    equipmentCenterPortraitWeapon,
    equipmentCenterPortraitAccessory
  )
  equipmentCenterDetails.append(
    equipmentCenterName,
    equipmentCenterJob,
    equipmentCenterLevel,
    equipmentCenterSet
  )
  equipmentCenterCard.append(equipmentCenterPreview, equipmentCenterDetails)
  equipmentSectionHeader.append(equipmentSectionSummary)
  equipmentSection.append(equipmentSectionHeader, equipmentLayout)

  const footerElement = document.createElement('div')
  footerElement.className = 'player-inventory-overlay__footer'
  footerElement.textContent = '클릭하면 해제 · 가방에서 드래그해서 장착 · U, Esc로 닫기'

  titleGroup.append(titleElement)
  headerRow.append(titleGroup, closeButton)
  closeButton.append(closeIcon)
  panelBody.append(headerRow, equipmentSection, footerElement)
  panel.append(panelBody)
  overlayRoot.append(backdropButton, panel, tooltipPanel)
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
    if (!dragState || event.pointerId !== dragState.pointerId) {
      return
    }

    panelPosition = clampPanelPosition(
      event.clientX - dragState.offsetX,
      event.clientY - dragState.offsetY,
      dragState.width,
      dragState.height
    )
    panel.style.left = `${panelPosition.left}px`
    panel.style.top = `${panelPosition.top}px`
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

  const hasInventoryDragData = (dataTransfer: DataTransfer): boolean =>
    Array.from(dataTransfer.types).some((type) =>
      type === 'application/x-player-drag-origin' ||
      type === 'application/x-player-inventory-slot-index' ||
      type === 'text/plain'
    )

  const setCardDimensions = (
    element: HTMLElement,
    width: number,
    height: number
  ) => {
    element.style.width = `${width}px`
    element.style.height = `${height}px`
  }

  const getEquipmentSlotButton = (target: EventTarget | null) => {
    if (!(target instanceof Element)) {
      return undefined
    }

    return target.closest('button[data-player-equipment-slot-id]') as
      | HTMLButtonElement
      | undefined
  }

  const getEquipmentSlotIndexFromButton = (
    button: HTMLButtonElement
  ): number => equipmentSlotCards.indexOf(button)

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

  const renderPlayerPortrait = (scale: number) => {
    setBackgroundFrame(
      equipmentCenterPortrait,
      PLAYER_PORTRAIT_FRAME_BY_APPEARANCE_TYPE[
        PLAYER_CHARACTER_APPEARANCE_TYPE
      ],
      scale
    )
  }

  const renderEquipmentPreviewIcon = (
    element: HTMLElement,
    itemId: string | undefined,
    scale: number
  ) => {
    if (!itemId) {
      clearFrame(element)
      return
    }

    const definition = getPlayerEquipmentItemDefinitionById(itemId)

    if (!definition) {
      clearFrame(element)
      return
    }

    setBackgroundFrame(
      element,
      EQUIPMENT_ICON_FRAME_BY_KEY[definition.icon.key],
      scale * definition.icon.scale
    )
  }

  function hideTooltip() {
    hoveredEquipmentSlotIndex = undefined
    tooltipPanel.hidden = true
    tooltipPanel.style.display = 'none'
    tooltipPanel.setAttribute('aria-hidden', 'true')
    tooltipIcon.hidden = true
  }

  const syncTooltip = () => {
    const equipment = getEquipment()

    if (
      hoveredEquipmentSlotIndex === undefined ||
      hoveredEquipmentSlotIndex < 0 ||
      hoveredEquipmentSlotIndex >= equipment.slots.length
    ) {
      hideTooltip()
      return
    }

    const slot = equipment.slots[hoveredEquipmentSlotIndex]
    const slotLabel = slot.label
    const slotItem = slot.item

    tooltipName.textContent = slotItem
      ? slotItem.label
      : `${slotLabel} 비어 있음`
    tooltipMeta.textContent = slotItem
      ? `${slotLabel} · 레벨 ${slotItem.level}`
      : slotLabel
    tooltipDescription.textContent = slotItem
      ? slotItem.description
      : '가방에서 장착하세요'
    tooltipHint.textContent = slotItem
      ? '클릭하면 해제'
      : `가방에서 드래그해서 ${slotLabel}에 장착`

    if (slotItem) {
      renderEquipmentIcon(tooltipIcon, slotItem.id, 1.1)
    } else {
      clearFrame(tooltipIcon)
    }

    tooltipPanel.hidden = false
    tooltipPanel.style.display = 'grid'
    tooltipPanel.setAttribute('aria-hidden', 'false')
    tooltipPanel.style.visibility = 'hidden'
    tooltipPanel.style.left = '0px'
    tooltipPanel.style.top = '0px'

    const tooltipRect = tooltipPanel.getBoundingClientRect()
    const maxLeft = Math.max(
      PANEL_MARGIN,
      window.innerWidth - tooltipRect.width - PANEL_MARGIN
    )
    const maxTop = Math.max(
      PANEL_MARGIN,
      window.innerHeight - tooltipRect.height - PANEL_MARGIN
    )

    tooltipPanel.style.left = `${clamp(
      tooltipAnchor.x + 14,
      PANEL_MARGIN,
      maxLeft
    )}px`
    tooltipPanel.style.top = `${clamp(
      tooltipAnchor.y + 12,
      PANEL_MARGIN,
      maxTop
    )}px`
    tooltipPanel.style.visibility = 'visible'
  }

  function showTooltipForEquipmentSlot(event: PointerEvent) {
    const slotButton =
      event.currentTarget instanceof HTMLButtonElement
        ? event.currentTarget
        : undefined

    if (!slotButton) {
      hideTooltip()
      return
    }

    const slotIndex = getEquipmentSlotIndexFromButton(slotButton)

    if (slotIndex < 0) {
      hideTooltip()
      return
    }

    hoveredEquipmentSlotIndex = slotIndex
    lastPointerPosition = {
      x: event.clientX,
      y: event.clientY
    }
    tooltipAnchor = {
      x: event.clientX,
      y: event.clientY
    }
    syncTooltip()
  }

  const setPreviewPosition = (
    element: HTMLElement,
    leftPercent: number,
    topPercent: number,
    rotation = 0
  ) => {
    element.style.left = `${leftPercent}%`
    element.style.top = `${topPercent}%`
    element.style.transform = `translate(-50%, -50%) rotate(${rotation}rad)`
  }

  const syncLayout = () => {
    const isOpen = getIsOpen()
    const uiScale = getResponsiveUiScale()

    overlayRoot.hidden = !isOpen
    overlayRoot.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      stopDragging()
      backdropButton.hidden = true
      panel.hidden = true
      hideTooltip()
      return
    }

    const availableWidth = Math.max(
      1,
      window.innerWidth - OVERLAY_MARGIN * 2
    )
    const availableHeight = Math.max(
      1,
      window.innerHeight - OVERLAY_MARGIN * 2
    )
    const panelScale = clamp(
      Math.floor(
        Math.min(
          availableWidth / PANEL_INSET_FRAME.width,
          availableHeight / PANEL_INSET_FRAME.height
        )
      ),
      PANEL_MIN_SCALE,
      PANEL_MAX_SCALE
    )
    const renderedWidth = PANEL_INSET_FRAME.width * panelScale * uiScale
    const renderedHeight = PANEL_INSET_FRAME.height * panelScale * uiScale
    const defaultPosition = clampPanelPosition(
      window.innerWidth - renderedWidth - OVERLAY_MARGIN,
      (window.innerHeight - renderedHeight) / 2,
      renderedWidth,
      renderedHeight
    )
    const equipmentSlotScale =
      panelScale >= EQUIPMENT_SLOT_SCALE_THRESHOLD ? 1.45 : 1.2
    const centerCardScale =
      panelScale >= EQUIPMENT_SLOT_SCALE_THRESHOLD ? 1.45 : 1.2

    backdropButton.hidden = false
    panel.hidden = false
    panel.style.transformOrigin = 'left top'
    panel.style.transform = `scale(${uiScale})`

    setCardDimensions(
      panel,
      PANEL_INSET_FRAME.width * panelScale,
      PANEL_INSET_FRAME.height * panelScale
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

    const equipment = getEquipment()
    const getEquippedItemIdBySlotId = (slotId: PlayerEquipmentSlotId) =>
      equipment.slots.find((slot) => slot.id === slotId)?.item?.id

    setEmptySlotAppearance(equipmentCenterCard)
    setCardDimensions(
      equipmentCenterCard,
      PANEL_LIGHT_FRAME.width * centerCardScale,
      PANEL_LIGHT_FRAME.height * centerCardScale
    )
    equipmentCenterPreview.style.height = `${Math.round(
      centerCardScale >= 1.45 ? 60 : 52
    )}px`
    renderPlayerPortrait(centerCardScale >= 1.45 ? 3.5 : 3)
    setPreviewPosition(equipmentCenterPortrait, 50, 52)
    renderEquipmentPreviewIcon(
      equipmentCenterPortraitWeapon,
      getEquippedItemIdBySlotId('weapon'),
      centerCardScale >= 1.45 ? 1.6 : 1.35
    )
    renderEquipmentPreviewIcon(
      equipmentCenterPortraitAccessory,
      getEquippedItemIdBySlotId('accessory'),
      centerCardScale >= 1.45 ? 1.25 : 1.05
    )
    setPreviewPosition(equipmentCenterPortraitWeapon, 32, 62, -0.55)
    setPreviewPosition(equipmentCenterPortraitAccessory, 41, 54)

    equipmentCenterName.textContent = profile.name
    equipmentCenterJob.textContent = getPlayerJobDisplayName(profile)
    equipmentCenterLevel.textContent =
      profile.level >= PLAYER_MAX_LEVEL
        ? `레벨 ${profile.level} · MAX`
        : `레벨 ${profile.level}`
    equipmentCenterSet.textContent = equipment.setName
    equipmentCenterSet.hidden = equipment.setName === '기본 장비'

    for (let index = 0; index < equipmentSlotCards.length; index += 1) {
      const slot = equipment.slots[index]
      const slotLabel = getPlayerEquipmentSlotLabelById(slot.id)
      const slotCard = equipmentSlotCards[index]
      const slotHeaderLabel = equipmentSlotHeaderLabels[index]
      const slotItemLabel = equipmentSlotItemLabels[index]
      const slotDescriptionLabel = equipmentSlotDescriptionLabels[index]
      const slotLevelBadge = equipmentSlotLevelBadges[index]
      const slotIcon = equipmentSlotIcons[index]
      const slotItem = slot.item

      setEmptySlotAppearance(slotCard)
      setCardDimensions(
        slotCard,
        BUTTON_SQUARE_FRAME.width * equipmentSlotScale,
        BUTTON_SQUARE_FRAME.height * equipmentSlotScale
      )
      slotCard.style.padding = `${6 * equipmentSlotScale}px ${8 * equipmentSlotScale}px ${7 * equipmentSlotScale}px`
      slotCard.style.fontSize =
        equipmentSlotScale >= 1.45 ? '0.68rem' : '0.64rem'
      slotCard.classList.toggle(
        'player-inventory-overlay__equipment-slot--empty',
        slotItem === undefined
      )
      slotCard.classList.toggle(
        'player-inventory-overlay__equipment-slot--filled',
        slotItem !== undefined
      )
      slotCard.setAttribute(
        'aria-label',
        slotItem
          ? `${slotLabel} ${slotItem.label}. 클릭하면 해제`
          : `${slotLabel} 비어 있음. 가방에서 드래그해서 장착`
      )
      slotCard.title = slotItem
        ? `${slotLabel} ${slotItem.label}. 클릭하면 해제`
        : `${slotLabel} 비어 있음. 가방에서 드래그해서 장착`

      slotHeaderLabel.hidden = slotItem !== undefined
      slotDescriptionLabel.hidden = true
      slotLevelBadge.hidden = true

      if (slotItem) {
        slotItemLabel.hidden = true
        slotItemLabel.textContent = ''
        renderEquipmentIcon(slotIcon, slotItem.id, equipmentSlotScale)
      } else {
        slotItemLabel.hidden = false
        slotItemLabel.textContent = '비어 있음'
        clearFrame(slotIcon)
      }
    }

    if (hoveredEquipmentSlotIndex !== undefined && lastPointerPosition) {
      syncTooltip()
    }
  }

  const handleEquipmentSlotClick = (slotId: PlayerEquipmentSlotId) => {
    const nextState = unequipPlayerEquipmentSlot({
      equipment: getEquipment(),
      inventory: getInventory(),
      slotId
    })

    if (!nextState) {
      return
    }

    onRequestEquipmentChange(nextState.equipment)
    onRequestInventoryChange(nextState.inventory)
  }

  const handlePanelClick = (event: MouseEvent) => {
    const target = event.target

    if (!(target instanceof Element)) {
      return
    }

    const equipmentButton = getEquipmentSlotButton(target)

    if (!equipmentButton) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    const slotId = equipmentButton.dataset.playerEquipmentSlotId as
      | PlayerEquipmentSlotId
      | undefined

    if (slotId) {
      handleEquipmentSlotClick(slotId)
    }
  }

  function handleEquipmentLayoutDragOver(event: DragEvent) {
    if (!event.dataTransfer) {
      return
    }

    if (!hasInventoryDragData(event.dataTransfer)) {
      return
    }

    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }

  function handleEquipmentLayoutDrop(event: DragEvent) {
    if (!event.dataTransfer) {
      return
    }

    if (event.dataTransfer.getData('application/x-player-drag-origin') !== 'inventory') {
      return
    }

    const slotIndexText =
      event.dataTransfer.getData('application/x-player-inventory-slot-index') ||
      event.dataTransfer.getData('text/plain')

    if (slotIndexText === '') {
      return
    }

    const slotIndex = Number(slotIndexText)

    if (Number.isNaN(slotIndex)) {
      return
    }

    const inventoryItem = getInventory().slots[slotIndex]
    const inventoryDefinition = inventoryItem
      ? getPlayerEquipmentItemDefinitionById(inventoryItem.id)
      : undefined

    if (!inventoryDefinition) {
      return
    }

    event.preventDefault()
    event.stopPropagation()

    const nextState = equipPlayerInventorySlot({
      equipment: getEquipment(),
      inventory: getInventory(),
      slotIndex
    })

    if (!nextState) {
      return
    }

    onRequestEquipmentChange(nextState.equipment)
    onRequestInventoryChange(nextState.inventory)
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
  document.addEventListener('pointermove', handleDocumentPointerMove)
  document.addEventListener('pointerup', handleDocumentPointerUp)
  document.addEventListener('pointercancel', handleDocumentPointerUp)
  closeButton.addEventListener('click', handleCloseButtonClick)
  closeButton.addEventListener('pointerdown', handleCloseButtonPointerDown)
  panel.addEventListener('click', handlePanelClick)
  panel.addEventListener('pointerleave', hideTooltip)

  const syncFrame = () => {
    syncLayout()
  }

  const destroy = () => {
    stopDragging()
    backdropButton.removeEventListener('click', handleBackdropClick)
    backdropButton.removeEventListener('pointerdown', handleBackdropPointerDown)
    headerRow.removeEventListener('pointerdown', startDragging)
    document.removeEventListener('pointermove', handleDocumentPointerMove)
    document.removeEventListener('pointerup', handleDocumentPointerUp)
    document.removeEventListener('pointercancel', handleDocumentPointerUp)
    closeButton.removeEventListener('click', handleCloseButtonClick)
    closeButton.removeEventListener('pointerdown', handleCloseButtonPointerDown)
    panel.removeEventListener('click', handlePanelClick)
    panel.removeEventListener('pointerleave', hideTooltip)
    for (const slotCard of equipmentSlotCards) {
      slotCard.removeEventListener('dragover', handleEquipmentLayoutDragOver)
      slotCard.removeEventListener('drop', handleEquipmentLayoutDrop)
      slotCard.removeEventListener('pointerenter', showTooltipForEquipmentSlot)
      slotCard.removeEventListener('pointermove', showTooltipForEquipmentSlot)
      slotCard.removeEventListener('pointerleave', hideTooltip)
    }
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
