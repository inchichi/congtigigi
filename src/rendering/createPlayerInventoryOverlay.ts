import type {
  PlayerEquipment,
  PlayerEquipmentIconKey,
  PlayerEquipmentSlotId
} from '../game/playerEquipment'
import {
  getPlayerEquipmentItemDefinitionById,
  getPlayerEquipmentSlotLabelById
} from '../game/playerEquipment'
import { getPlayerInventoryFilledSlotCount } from '../game/playerInventory'
import { PLAYER_CHARACTER_APPEARANCE_TYPE } from '../game/characterState'
import type { PlayerInventory } from '../game/playerInventory'
import type { PlayerProfile } from '../game/playerProfile'
import {
  equipPlayerInventorySlot,
  unequipPlayerEquipmentSlot
} from '../game/playerLoadout'
import {
  PLAYER_MAX_LEVEL,
  getPlayerJobDisplayName
} from '../game/playerProfile'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreatePlayerInventoryOverlayInput = {
  mountElement: HTMLElement
  profile: PlayerProfile
  getInventory: () => PlayerInventory
  getEquipment: () => PlayerEquipment
  getIsOpen: () => boolean
  onRequestOpenChange: (isOpen: boolean) => void
  onRequestInventoryChange: (nextInventory: PlayerInventory) => void
  onRequestEquipmentChange: (nextEquipment: PlayerEquipment) => void
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
const ICON_CROSS_FRAME = {
  x: 369,
  y: 169,
  width: 16,
  height: 15
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
const PANEL_MIN_SCALE = 5
const PANEL_MAX_SCALE = 7
const INVENTORY_SLOT_SCALE_THRESHOLD = 7
const EQUIPMENT_SLOT_SCALE_THRESHOLD = 6

const EQUIPMENT_SLOT_AREA_CLASS_BY_ID: Record<PlayerEquipmentSlotId, string> = {
  weapon: 'player-inventory-overlay__equipment-slot--weapon',
  armor: 'player-inventory-overlay__equipment-slot--armor',
  boots: 'player-inventory-overlay__equipment-slot--boots',
  accessory: 'player-inventory-overlay__equipment-slot--accessory'
}

export const createPlayerInventoryOverlay = ({
  mountElement,
  profile,
  getInventory,
  getEquipment,
  getIsOpen,
  onRequestOpenChange,
  onRequestInventoryChange,
  onRequestEquipmentChange
}: CreatePlayerInventoryOverlayInput): PlayerInventoryOverlay => {
  const overlayRoot = document.createElement('div')
  const backdropButton = document.createElement('button')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const headerRow = document.createElement('div')
  const titleGroup = document.createElement('div')
  const titleElement = document.createElement('div')
  const summaryElement = document.createElement('div')
  const equipmentSection = document.createElement('section')
  const equipmentSectionHeader = document.createElement('div')
  const equipmentSectionTitle = document.createElement('div')
  const equipmentSectionSummary = document.createElement('div')
  const equipmentLayout = document.createElement('div')
  const equipmentCenterCard = document.createElement('div')
  const equipmentCenterPreview = document.createElement('div')
  const equipmentCenterPortrait = document.createElement('span')
  const equipmentCenterPortraitWeapon = document.createElement('span')
  const equipmentCenterPortraitArmor = document.createElement('span')
  const equipmentCenterPortraitBoots = document.createElement('span')
  const equipmentCenterPortraitAccessory = document.createElement('span')
  const equipmentCenterDetails = document.createElement('div')
  const equipmentCenterName = document.createElement('div')
  const equipmentCenterJob = document.createElement('div')
  const equipmentCenterLevel = document.createElement('div')
  const equipmentCenterSet = document.createElement('div')
  const closeButton = document.createElement('button')
  const closeIcon = document.createElement('span')
  const slotGrid = document.createElement('div')
  const footerElement = document.createElement('div')
  const emptyStateElement = document.createElement('div')
  const equipmentSlotCards: HTMLButtonElement[] = []
  const equipmentSlotHeaderLabels: HTMLSpanElement[] = []
  const equipmentSlotItemLabels: HTMLDivElement[] = []
  const equipmentSlotDescriptionLabels: HTMLDivElement[] = []
  const equipmentSlotLevelBadges: HTMLDivElement[] = []
  const equipmentSlotIcons: HTMLSpanElement[] = []
  const slotButtons: HTMLButtonElement[] = []
  const slotIndexLabels: HTMLSpanElement[] = []
  const slotIcons: HTMLSpanElement[] = []
  const slotItemLabels: HTMLSpanElement[] = []
  const slotQuantityLabels: HTMLSpanElement[] = []
  const slotEmptyMarkers: HTMLSpanElement[] = []
  const initialInventory = getInventory()
  const initialEquipment = getEquipment()

  overlayRoot.className = 'player-inventory-overlay'

  backdropButton.type = 'button'
  backdropButton.className = 'player-inventory-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.setAttribute('aria-label', '가방 닫기')

  panel.id = 'player-inventory-panel'
  panel.className = 'player-inventory-overlay__panel'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'true')
  panel.setAttribute('aria-labelledby', 'player-inventory-title')

  panelBody.className = 'player-inventory-overlay__panel-body'
  headerRow.className = 'player-inventory-overlay__header'
  titleGroup.className = 'player-inventory-overlay__title-group'
  titleElement.id = 'player-inventory-title'
  titleElement.className = 'player-inventory-overlay__title'
  titleElement.textContent = '가방'
  summaryElement.className = 'player-inventory-overlay__summary'
  summaryElement.textContent = `${getPlayerInventoryFilledSlotCount(initialInventory)} / ${initialInventory.slots.length}칸 · ${formatGoldAmount(initialInventory.gold)}`

  equipmentSection.className = 'player-inventory-overlay__equipment-section'
  equipmentSectionHeader.className = 'player-inventory-overlay__equipment-header'
  equipmentSectionTitle.className = 'player-inventory-overlay__equipment-title'
  equipmentSectionTitle.textContent = '장비'
  equipmentSectionSummary.className =
    'player-inventory-overlay__equipment-summary'
  equipmentSectionSummary.textContent = '기본 장비'

  equipmentLayout.className = 'player-inventory-overlay__equipment-layout'
  equipmentLayout.setAttribute('role', 'list')

  equipmentCenterCard.className = 'player-inventory-overlay__equipment-center'
  equipmentCenterPreview.className =
    'player-inventory-overlay__equipment-center-preview'
  equipmentCenterPortrait.className =
    'player-inventory-overlay__equipment-center-portrait'
  equipmentCenterPortraitWeapon.className =
    'player-inventory-overlay__equipment-center-gear player-inventory-overlay__equipment-center-gear--weapon'
  equipmentCenterPortraitArmor.className =
    'player-inventory-overlay__equipment-center-gear player-inventory-overlay__equipment-center-gear--armor'
  equipmentCenterPortraitBoots.className =
    'player-inventory-overlay__equipment-center-gear player-inventory-overlay__equipment-center-gear--boots'
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

  closeButton.type = 'button'
  closeButton.className = 'player-inventory-overlay__close'
  closeButton.setAttribute('aria-label', '가방 닫기')
  closeButton.title = '가방 닫기 (Esc)'

  closeIcon.className = 'player-inventory-overlay__close-icon'
  closeIcon.setAttribute('aria-hidden', 'true')

  slotGrid.className = 'player-inventory-overlay__slot-grid'
  slotGrid.style.gap = '12px'

  footerElement.className = 'player-inventory-overlay__footer'
  footerElement.textContent = 'I, Esc로 닫기 · 클릭으로 착용/해제'

  emptyStateElement.className = 'player-inventory-overlay__empty-state'
  emptyStateElement.textContent = '아직 아이템이 없습니다'

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

    slotItemLabel.className = 'player-inventory-overlay__equipment-slot-item'
    slotItemLabel.textContent = slot.item?.label ?? '비어 있음'

    slotDescriptionLabel.className =
      'player-inventory-overlay__equipment-slot-description'
    slotDescriptionLabel.textContent = slot.item?.description ?? '가방에서 장착하세요'

    slotLevelBadge.className = 'player-inventory-overlay__equipment-slot-level'
    slotLevelBadge.textContent = `레벨 ${slot.item?.level ?? '-'}`

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
  }

  equipmentLayout.append(equipmentCenterCard)
  equipmentCenterPreview.append(
    equipmentCenterPortrait,
    equipmentCenterPortraitWeapon,
    equipmentCenterPortraitArmor,
    equipmentCenterPortraitBoots,
    equipmentCenterPortraitAccessory
  )
  equipmentCenterDetails.append(
    equipmentCenterName,
    equipmentCenterJob,
    equipmentCenterLevel,
    equipmentCenterSet
  )
  equipmentCenterCard.append(equipmentCenterPreview, equipmentCenterDetails)
  equipmentSectionHeader.append(equipmentSectionTitle, equipmentSectionSummary)
  equipmentSection.append(equipmentSectionHeader, equipmentLayout)

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

    slotIndexLabel.className = 'player-inventory-overlay__slot-index'
    slotIndexLabel.textContent = String(index + 1).padStart(2, '0')

    slotIcon.className = 'player-inventory-overlay__slot-icon'
    slotIcon.setAttribute('aria-hidden', 'true')

    slotItemLabel.className = 'player-inventory-overlay__slot-item'
    slotItemLabel.hidden = true

    slotQuantityLabel.className = 'player-inventory-overlay__slot-quantity'
    slotQuantityLabel.hidden = true

    slotEmptyMarker.className = 'player-inventory-overlay__slot-empty-marker'
    slotEmptyMarker.setAttribute('aria-hidden', 'true')

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
  }

  titleGroup.append(titleElement, summaryElement)
  headerRow.append(titleGroup, closeButton)
  closeButton.append(closeIcon)
  panelBody.append(
    headerRow,
    equipmentSection,
    slotGrid,
    emptyStateElement,
    footerElement
  )
  panel.append(panelBody)
  overlayRoot.append(backdropButton, panel)
  mountElement.append(overlayRoot)

  const setSpriteFrame = (
    element: HTMLElement,
    frame: { x: number; y: number; width: number; height: number },
    scale: number
  ) => {
    element.style.backgroundImage = `url(${UI_SPRITESHEET_IMAGE_URL})`
    element.style.backgroundRepeat = 'no-repeat'
    element.style.backgroundPosition = `-${frame.x * scale}px -${frame.y * scale}px`
    element.style.backgroundSize = `${UI_SPRITESHEET_WIDTH * scale}px ${UI_SPRITESHEET_HEIGHT * scale}px`
    element.style.width = `${frame.width * scale}px`
    element.style.height = `${frame.height * scale}px`
    element.style.imageRendering = 'pixelated'
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

  setSpriteFrame(closeButton, BUTTON_SQUARE_FRAME, 1.2)
  setSpriteFrame(closeIcon, ICON_CROSS_FRAME, 2)

  const syncLayout = () => {
    const isOpen = getIsOpen()
    const uiScale = getResponsiveUiScale()

    overlayRoot.hidden = !isOpen
    overlayRoot.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      backdropButton.hidden = true
      panel.hidden = true
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
    const inventorySlotScale = panelScale >= INVENTORY_SLOT_SCALE_THRESHOLD ? 2 : 1
    const equipmentSlotScale = panelScale >= EQUIPMENT_SLOT_SCALE_THRESHOLD ? 2 : 1.75
    const centerCardScale = panelScale >= EQUIPMENT_SLOT_SCALE_THRESHOLD ? 2 : 1.5

    backdropButton.hidden = false
    panel.hidden = false
    panel.style.transformOrigin = 'center center'
    panel.style.transform = `translate(-50%, -50%) scale(${uiScale})`

    const inventory = getInventory()
    const equipment = getEquipment()
    const getEquippedItemIdBySlotId = (slotId: PlayerEquipmentSlotId) =>
      equipment.slots.find((slot) => slot.id === slotId)?.item?.id
    setSpriteFrame(panel, PANEL_INSET_FRAME, panelScale)
    panel.style.width = `${PANEL_INSET_FRAME.width * panelScale}px`
    panel.style.height = `${PANEL_INSET_FRAME.height * panelScale}px`

    setSpriteFrame(equipmentCenterCard, PANEL_LIGHT_FRAME, centerCardScale)
    equipmentCenterCard.style.width = `${PANEL_LIGHT_FRAME.width * centerCardScale}px`
    equipmentCenterCard.style.height = `${PANEL_LIGHT_FRAME.height * centerCardScale}px`
    equipmentCenterPreview.style.height = `${Math.round(
      (centerCardScale >= 2 ? 108 : 96)
    )}px`
    renderPlayerPortrait(centerCardScale >= 2 ? 4 : 3.5)
    setPreviewPosition(equipmentCenterPortrait, 50, 52)
    renderEquipmentPreviewIcon(
      equipmentCenterPortraitWeapon,
      getEquippedItemIdBySlotId('weapon'),
      centerCardScale >= 2 ? 1.8 : 1.6
    )
    renderEquipmentPreviewIcon(
      equipmentCenterPortraitArmor,
      getEquippedItemIdBySlotId('armor'),
      centerCardScale >= 2 ? 1.6 : 1.4
    )
    renderEquipmentPreviewIcon(
      equipmentCenterPortraitBoots,
      getEquippedItemIdBySlotId('boots'),
      centerCardScale >= 2 ? 1.5 : 1.25
    )
    renderEquipmentPreviewIcon(
      equipmentCenterPortraitAccessory,
      getEquippedItemIdBySlotId('accessory'),
      centerCardScale >= 2 ? 1.5 : 1.25
    )
    // Front-facing portrait reads better with the weapon on the viewer-left side.
    setPreviewPosition(equipmentCenterPortraitWeapon, 32, 62, -0.55)
    setPreviewPosition(equipmentCenterPortraitArmor, 50, 29)
    setPreviewPosition(equipmentCenterPortraitBoots, 50, 84)
    setPreviewPosition(equipmentCenterPortraitAccessory, 41, 54)

    equipmentSectionSummary.textContent = `${equipment.setName} • 레벨 ${equipment.level}`
    equipmentCenterName.textContent = profile.name
    equipmentCenterJob.textContent = getPlayerJobDisplayName(profile)
    equipmentCenterLevel.textContent =
      profile.level >= PLAYER_MAX_LEVEL
        ? `레벨 ${profile.level} · MAX`
        : `레벨 ${profile.level}`
    equipmentCenterSet.textContent = equipment.setName

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

      setSpriteFrame(slotCard, BUTTON_SQUARE_FRAME, equipmentSlotScale)
      slotCard.style.padding = `${8 * equipmentSlotScale}px ${10 * equipmentSlotScale}px ${9 * equipmentSlotScale}px`
      slotCard.style.fontSize = equipmentSlotScale === 2 ? '0.74rem' : '0.7rem'
      slotCard.classList.toggle(
        'player-inventory-overlay__equipment-slot--empty',
        slotItem === undefined
      )
      slotCard.setAttribute(
        'aria-label',
        slotItem
          ? `${slotLabel} ${slotItem.label}. 클릭하면 해제`
          : `${slotLabel} 비어 있음. 가방에서 장착`
      )
      slotCard.title = slotItem
        ? `${slotLabel} ${slotItem.label}. 클릭하면 해제`
        : `${slotLabel} 비어 있음. 가방에서 장착`

      slotHeaderLabel.textContent = slotLabel
      slotItemLabel.textContent = slotItem ? slotItem.label : '비어 있음'
      slotDescriptionLabel.textContent = slotItem
        ? slotItem.description
        : '가방에서 장착하세요'
      slotLevelBadge.textContent = slotItem ? `레벨 ${slotItem.level}` : '--'
      renderEquipmentIcon(slotIcon, slotItem?.id, equipmentSlotScale)
    }

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

      setSpriteFrame(slotButton, BUTTON_SQUARE_FRAME, inventorySlotScale)
      setSpriteFrame(slotEmptyMarker, ICON_CIRCLE_FRAME, inventorySlotScale)
      slotButton.style.padding = `${8 * inventorySlotScale}px ${10 * inventorySlotScale}px ${9 * inventorySlotScale}px`
      slotButton.style.fontSize = inventorySlotScale === 2 ? '0.9rem' : '0.78rem'
      slotButton.classList.toggle(
        'player-inventory-overlay__slot--equippable',
        Boolean(slotDefinition)
      )
      const targetSlotLabel = slotDefinition
        ? getPlayerEquipmentSlotLabelById(slotDefinition.slotId)
        : undefined

      slotIndexLabel.textContent = String(index + 1).padStart(2, '0')
      slotButton.classList.toggle(
        'player-inventory-overlay__slot--filled',
        slot !== undefined
      )
      slotButton.classList.toggle(
        'player-inventory-overlay__slot--empty',
        slot === undefined
      )
      slotButton.setAttribute(
        'aria-label',
        slot
          ? slotDefinition
            ? `${slot.label}${slot.quantity > 1 ? ` x${slot.quantity}` : ''}. 클릭하면 ${targetSlotLabel}에 장착`
            : `${slot.label}${slot.quantity > 1 ? ` x${slot.quantity}` : ''}`
          : `빈 가방 칸 ${index + 1}`
      )
      slotButton.title = slot
        ? slotDefinition
          ? `${slot.label}${slot.quantity > 1 ? ` x${slot.quantity}` : ''}. 클릭하면 ${targetSlotLabel}에 장착`
          : `${slot.label}${slot.quantity > 1 ? ` x${slot.quantity}` : ''}`
        : `빈 가방 칸 ${index + 1}`

      if (slot) {
        slotItemLabel.hidden = false
        slotItemLabel.textContent = slot.label
        slotQuantityLabel.hidden = slot.quantity <= 1
        slotQuantityLabel.textContent = slot.quantity > 1 ? `x${slot.quantity}` : ''
        slotEmptyMarker.hidden = true
        if (slotDefinition) {
          renderEquipmentIcon(slotIcon, slot.id, inventorySlotScale)
        } else {
          clearFrame(slotIcon)
        }
      } else {
        slotItemLabel.hidden = true
        slotItemLabel.textContent = ''
        slotQuantityLabel.hidden = true
        slotQuantityLabel.textContent = ''
        slotEmptyMarker.hidden = false
        clearFrame(slotIcon)
      }
    }
    const filledSlotCount = getPlayerInventoryFilledSlotCount(inventory)

    summaryElement.textContent = `${filledSlotCount} / ${inventory.slots.length}칸 · ${formatGoldAmount(inventory.gold)}`
    emptyStateElement.hidden = filledSlotCount > 0
  }

  const handleInventorySlotClick = (slotIndex: number) => {
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
        handleInventorySlotClick(slotIndex)
      }

      return
    }

    const equipmentButton = target.closest(
      'button[data-player-equipment-slot-id]'
    ) as HTMLButtonElement | null

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

  const handleBackdropClick = (event: MouseEvent) => {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  const handleBackdropPointerDown = (event: PointerEvent) => {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  const handleCloseButtonClick = (event: MouseEvent) => {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  const handleCloseButtonPointerDown = (event: PointerEvent) => {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  backdropButton.addEventListener('click', handleBackdropClick)
  backdropButton.addEventListener('pointerdown', handleBackdropPointerDown)
  closeButton.addEventListener('click', handleCloseButtonClick)
  closeButton.addEventListener('pointerdown', handleCloseButtonPointerDown)
  panel.addEventListener('click', handlePanelClick)

  const handleGlobalCloseIntent = (event: Event) => {
    if (!getIsOpen()) {
      return
    }

    const target = event.target

    if (!(target instanceof Node)) {
      return
    }

    if (!panel.contains(target) || closeButton.contains(target)) {
      event.preventDefault()
      event.stopPropagation()
      onRequestOpenChange(false)
    }
  }

  document.addEventListener('pointerdown', handleGlobalCloseIntent, true)
  document.addEventListener('click', handleGlobalCloseIntent, true)

  const syncFrame = () => {
    syncLayout()
  }

  const destroy = () => {
    backdropButton.removeEventListener('click', handleBackdropClick)
    backdropButton.removeEventListener('pointerdown', handleBackdropPointerDown)
    closeButton.removeEventListener('click', handleCloseButtonClick)
    closeButton.removeEventListener('pointerdown', handleCloseButtonPointerDown)
    panel.removeEventListener('click', handlePanelClick)
    document.removeEventListener('pointerdown', handleGlobalCloseIntent, true)
    document.removeEventListener('click', handleGlobalCloseIntent, true)
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
