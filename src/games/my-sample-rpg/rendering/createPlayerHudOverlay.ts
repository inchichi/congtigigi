import {
  PLAYER_MAX_LEVEL,
  getPlayerJobDisplayName,
  type PlayerProfile
} from '../playerProfile'
import { getPlayerExperienceToNextLevel } from '../playerExperience'
import { getPlayerEquipmentItemDefinitionById } from '../playerEquipment'
import type { PlayerInventory } from '../playerInventory'
import {
  clearPlayerQuickslotAssignment,
  setPlayerQuickslotAssignment,
  type PlayerQuickslots
} from '../playerQuickslots'
import {
  PLAYER_SKILL_DRAG_MIME_TYPE,
  clearPlayerSkillSlotAssignment,
  PLAYER_SKILL_SLOT_COUNT,
  PLAYER_SKILL_SLOT_HOTKEY_LABELS,
  setPlayerSkillSlotAssignment,
  type PlayerSkillSlots
} from '../playerSkillSlots'
import { getPlayerSkillDisplayInfoById } from '../playerSkills'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreatePlayerHudOverlayInput = {
  mountElement: HTMLElement
  profile: PlayerProfile
  getInventory: () => PlayerInventory
  getQuickslots: () => PlayerQuickslots
  getSkillSlots: () => PlayerSkillSlots
  onRequestQuickslotChange: (nextQuickslots: PlayerQuickslots) => void
  onRequestSkillSlotsChange: (nextSkillSlots: PlayerSkillSlots) => void
}

export type PlayerHudOverlay = {
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
const BUTTON_SQUARE_FRAME = {
  x: 293,
  y: 294,
  width: 45,
  height: 49
}
const ICON_CIRCLE_FRAME = {
  x: 356,
  y: 466,
  width: 17,
  height: 17
}
const TINY_DUNGEON_TILESET_WIDTH = 192
const TINY_DUNGEON_TILESET_HEIGHT = 176
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
const HUD_MARGIN = 16
const HUD_PANEL_MIN_WIDTH = 780
const HUD_PANEL_MAX_WIDTH = 920
const HUD_PANEL_SCALE_MULTIPLIER = 0.82
const HEALTH_BAR_COLOR = '#d06b5d'
const MANA_BAR_COLOR = '#5b86d6'
const QUICK_CONSUMABLE_HOTKEYS = ['1', '2', '3', '4', '5', '6'] as const
const QUICK_CONSUMABLE_SLOT_COUNT = QUICK_CONSUMABLE_HOTKEYS.length
type HudHoverTarget =
  | {
      kind: 'consumable'
      index: number
    }
  | {
      kind: 'skill'
      index: number
    }

export const createPlayerHudOverlay = ({
  mountElement,
  profile,
  getInventory,
  getQuickslots,
  getSkillSlots,
  onRequestQuickslotChange,
  onRequestSkillSlotsChange
}: CreatePlayerHudOverlayInput): PlayerHudOverlay => {
  const overlayRoot = document.createElement('div')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const infoSection = document.createElement('div')
  const infoGrid = document.createElement('div')
  const nicknameRow = createInfoRow('닉네임')
  const levelRow = createInfoRow('레벨')
  const jobRow = createInfoRow('직업')
  const statusSection = document.createElement('div')
  const resourceGrid = document.createElement('div')
  const hpRow = createResourceRow('체력')
  const mpRow = createResourceRow('마나')
  const expRow = createResourceRow('경험치')
  const skillSection = document.createElement('div')
  const skillSectionTitle = document.createElement('div')
  const skillGrid = document.createElement('div')
  const consumableSection = document.createElement('div')
  const consumableSectionTitle = document.createElement('div')
  const consumableGrid = document.createElement('div')
  const tooltipPanel = document.createElement('div')
  const tooltipHeader = document.createElement('div')
  const tooltipIcon = document.createElement('span')
  const tooltipText = document.createElement('div')
  const tooltipName = document.createElement('div')
  const tooltipMeta = document.createElement('div')
  const tooltipDescription = document.createElement('div')
  const tooltipHint = document.createElement('div')
  const skillButtons: HTMLButtonElement[] = []
  const skillIcons: HTMLSpanElement[] = []
  const skillHotkeyLabels: HTMLSpanElement[] = []
  const skillNameLabels: HTMLSpanElement[] = []
  const skillDescriptionLabels: HTMLSpanElement[] = []
  const consumableButtons: HTMLButtonElement[] = []
  const consumableHotkeyLabels: HTMLSpanElement[] = []
  const consumableIcons: HTMLSpanElement[] = []
  const consumableNameLabels: HTMLSpanElement[] = []
  const consumableQuantityLabels: HTMLSpanElement[] = []
  const skillSlotCount = PLAYER_SKILL_SLOT_COUNT
  let hoveredHudTarget: HudHoverTarget | undefined
  let tooltipAnchor = {
    x: 0,
    y: 0
  }

  overlayRoot.className = 'player-hud-overlay'

  panel.className = 'player-hud-overlay__panel'
  panel.setAttribute('aria-label', '캐릭터 상태')

  panelBody.className = 'player-hud-overlay__panel-body'

  infoSection.className = 'player-hud-overlay__info-section'
  infoGrid.className = 'player-hud-overlay__info-grid'

  statusSection.className = 'player-hud-overlay__status-section'
  resourceGrid.className = 'player-hud-overlay__resource-grid'

  skillSection.className = 'player-hud-overlay__skill-section'
  skillSectionTitle.className = 'player-hud-overlay__skill-title'
  skillSectionTitle.textContent = '스킬'
  skillGrid.className = 'player-hud-overlay__skill-grid'
  consumableSection.className = 'player-hud-overlay__consumable-section'
  consumableSectionTitle.className = 'player-hud-overlay__section-title'
  consumableSectionTitle.textContent = '퀵슬롯'
  consumableGrid.className = 'player-hud-overlay__consumable-grid'
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

  for (let index = 0; index < QUICK_CONSUMABLE_SLOT_COUNT; index += 1) {
    const consumableButton = document.createElement('button')
    const hotkeyLabel = document.createElement('span')
    const icon = document.createElement('span')
    const nameLabel = document.createElement('span')
    const quantityLabel = document.createElement('span')

    consumableButton.type = 'button'
    consumableButton.className = 'player-hud-overlay__consumable-slot'
    consumableButton.dataset.playerQuickslotIndex = String(index)
    consumableButton.setAttribute(
      'aria-label',
      `퀵슬롯 ${index + 1} 비어있음`
    )
    consumableButton.title = `퀵슬롯 ${index + 1} 비어있음`

    hotkeyLabel.className = 'player-hud-overlay__consumable-hotkey'
    hotkeyLabel.textContent = QUICK_CONSUMABLE_HOTKEYS[index]

    icon.className = 'player-hud-overlay__consumable-icon'
    icon.setAttribute('aria-hidden', 'true')
    icon.hidden = true

    nameLabel.className = 'player-hud-overlay__consumable-name'
    nameLabel.textContent = '비어있음'

    quantityLabel.className = 'player-hud-overlay__consumable-quantity'
    quantityLabel.hidden = true

    consumableButton.append(hotkeyLabel, icon, nameLabel, quantityLabel)
    consumableGrid.append(consumableButton)

    consumableButtons.push(consumableButton)
    consumableHotkeyLabels.push(hotkeyLabel)
    consumableIcons.push(icon)
    consumableNameLabels.push(nameLabel)
    consumableQuantityLabels.push(quantityLabel)
    consumableButton.addEventListener('dragstart', handleQuickslotDragStart)
    consumableButton.addEventListener('dragover', handleQuickslotDragOver)
    consumableButton.addEventListener('drop', handleQuickslotDrop)
    consumableButton.addEventListener('pointerenter', showTooltipForConsumable)
    consumableButton.addEventListener('pointermove', showTooltipForConsumable)
    consumableButton.addEventListener('pointerleave', hideTooltip)
  }

  for (let index = 0; index < skillSlotCount; index += 1) {
    const skillButton = document.createElement('button')
    const skillIcon = document.createElement('span')
    const hotkeyLabel = document.createElement('span')
    const nameLabel = document.createElement('span')
    const descriptionLabel = document.createElement('span')

    skillButton.type = 'button'
    skillButton.className = 'player-hud-overlay__skill-slot'
    skillButton.dataset.playerSkillIndex = String(index)
    skillButton.setAttribute('aria-label', `스킬 슬롯 ${index + 1} 비어있음`)
    skillButton.title = `스킬 슬롯 ${index + 1} 비어있음`
    skillButton.draggable = false

    skillIcon.className = 'player-hud-overlay__skill-icon'
    skillIcon.setAttribute('aria-hidden', 'true')
    skillIcon.hidden = true

    hotkeyLabel.className = 'player-hud-overlay__skill-hotkey'
    hotkeyLabel.textContent = PLAYER_SKILL_SLOT_HOTKEY_LABELS[index]

    nameLabel.className = 'player-hud-overlay__skill-name'
    nameLabel.textContent = '비어있음'

    descriptionLabel.className = 'player-hud-overlay__skill-description'
    descriptionLabel.textContent = ''

    skillButton.append(hotkeyLabel, skillIcon, nameLabel, descriptionLabel)
    skillGrid.append(skillButton)

    skillButtons.push(skillButton)
    skillIcons.push(skillIcon)
    skillHotkeyLabels.push(hotkeyLabel)
    skillNameLabels.push(nameLabel)
    skillDescriptionLabels.push(descriptionLabel)
    skillButton.addEventListener('dragstart', handleSkillSlotDragStart)
    skillButton.addEventListener('dragover', handleSkillSlotDragOver)
    skillButton.addEventListener('drop', handleSkillSlotDrop)
    skillButton.addEventListener('pointerenter', showTooltipForSkill)
    skillButton.addEventListener('pointermove', showTooltipForSkill)
    skillButton.addEventListener('pointerleave', hideTooltip)
  }

  infoSection.append(infoGrid)
  infoGrid.append(nicknameRow.row, levelRow.row, jobRow.row)
  statusSection.append(resourceGrid)
  resourceGrid.append(hpRow.row, mpRow.row, expRow.row)
  skillSection.append(skillSectionTitle, skillGrid)
  consumableSection.append(consumableSectionTitle, consumableGrid)
  panelBody.append(infoSection, statusSection, skillSection, consumableSection)
  panel.append(panelBody)
  overlayRoot.append(panel, tooltipPanel)
  mountElement.append(overlayRoot)

  const setSpriteFrame = (
    element: HTMLElement,
    frame: { x: number; y: number; width: number; height: number },
    scaleX = 1,
    scaleY = scaleX
  ) => {
    element.style.backgroundImage = `url(${UI_SPRITESHEET_IMAGE_URL})`
    element.style.backgroundRepeat = 'no-repeat'
    element.style.backgroundPosition = `-${frame.x * scaleX}px -${frame.y * scaleY}px`
    element.style.backgroundSize = `${512 * scaleX}px ${512 * scaleY}px`
    element.style.width = `${frame.width * scaleX}px`
    element.style.height = `${frame.height * scaleY}px`
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

  const setEmptySlotAppearance = (element: HTMLElement) => {
    element.style.backgroundImage = 'none'
    element.style.backgroundRepeat = 'no-repeat'
    element.style.backgroundPosition = '0 0'
    element.style.backgroundSize = '100% 100%'
    element.style.backgroundColor = 'rgba(255, 249, 238, 0.9)'
    element.style.border = '1px solid rgba(111, 89, 58, 0.34)'
    element.style.boxShadow = 'inset 0 1px 0 rgba(255, 255, 255, 0.7)'
  }

  const clearFrame = (element: HTMLElement) => {
    element.hidden = true
    element.style.backgroundImage = 'none'
  }

  const renderSkillIcon = (
    element: HTMLElement,
    iconUrl: string | undefined,
    scale = 0.92
  ) => {
    if (!iconUrl) {
      clearFrame(element)
      return
    }

    element.hidden = false
    element.style.backgroundImage = `url(${iconUrl})`
    element.style.backgroundRepeat = 'no-repeat'
    element.style.backgroundPosition = 'center'
    element.style.backgroundSize = 'contain'
    element.style.width = `${Math.round(34 * scale)}px`
    element.style.height = `${Math.round(34 * scale)}px`
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
    scale = 0.96
  ) => {
    if (!itemId) {
      clearFrame(element)
      return
    }

    const frame = getConsumableIconFrame(itemId)

    if (frame) {
      setBackgroundFrame(element, frame, scale)
      return
    }

    element.hidden = false
    setSpriteFrame(element, ICON_CIRCLE_FRAME, scale)
  }

  const getSkillButtonIndexFromButton = (button: HTMLButtonElement): number =>
    Number(button.dataset.playerSkillIndex)

  const getSkillSlotButtonIndexFromButton = (
    button: HTMLButtonElement
  ): number => Number(button.dataset.playerSkillIndex)

  const hasSkillDragData = (dataTransfer: DataTransfer): boolean =>
    Array.from(dataTransfer.types).some((type) =>
      type === PLAYER_SKILL_DRAG_MIME_TYPE || type === 'text/plain'
    )

  const getDraggedSkillId = (event: DragEvent): string | undefined => {
    const rawValue =
      event.dataTransfer?.getData(PLAYER_SKILL_DRAG_MIME_TYPE) ||
      event.dataTransfer?.getData('text/plain')

    if (!rawValue) {
      return undefined
    }

    return rawValue.length > 0 ? rawValue : undefined
  }

  function hideTooltip() {
    hoveredHudTarget = undefined
    tooltipPanel.hidden = true
    tooltipPanel.style.display = 'none'
    tooltipPanel.setAttribute('aria-hidden', 'true')
    tooltipIcon.hidden = true
  }

  const syncTooltip = () => {
    if (!hoveredHudTarget) {
      hideTooltip()
      return
    }

    if (hoveredHudTarget.kind === 'consumable') {
      const index = hoveredHudTarget.index

      if (index < 0 || index >= consumableButtons.length) {
        hideTooltip()
        return
      }

      const quickslots = getQuickslots()
      const inventory = getInventory()
      const quickslot = quickslots.slots[index]
      const item = quickslot ? inventory.slots[quickslot.inventorySlotIndex] : undefined
      const isQuickslotItem = item ? isLikelyConsumableInventoryItem(item) : false
      const quickslotItem = item && isQuickslotItem ? item : undefined

      tooltipName.textContent = quickslotItem
        ? quickslotItem.label
        : `퀵슬롯 ${index + 1} 비어 있음`
      tooltipMeta.textContent = quickslotItem
        ? `퀵슬롯 ${index + 1}${quickslotItem.quantity > 1 ? ` · x${quickslotItem.quantity}` : ''}`
        : `퀵슬롯 ${index + 1}`
      tooltipDescription.textContent = quickslotItem
        ? '소비 아이템입니다.'
        : '소비 아이템을 등록하세요.'
      tooltipHint.textContent = quickslotItem
        ? '드래그해서 다른 칸으로 이동 · 더블클릭하면 사용'
        : '소비 아이템을 드래그해서 등록'

      renderConsumableIcon(tooltipIcon, quickslotItem?.id, 1.02)
    } else {
      const index = hoveredHudTarget.index
      const skillSlot = getSkillSlots().slots[index]
      const skill = skillSlot
        ? getPlayerSkillDisplayInfoById(skillSlot.skillId)
        : undefined

      if (!skill) {
        tooltipName.textContent = `스킬 슬롯 ${index + 1} 비어 있음`
        tooltipMeta.textContent = `스킬 슬롯 ${index + 1}`
        tooltipDescription.textContent = '스킬을 장착하세요.'
        tooltipHint.textContent = '스킬 슬롯을 채우면 여기서 확인할 수 있습니다'
      } else {
        tooltipName.textContent = skill.label
        tooltipMeta.textContent = `스킬 슬롯 ${index + 1}`
        tooltipDescription.textContent = skill.description
        tooltipHint.textContent = '스킬 슬롯에 장착된 기술입니다'
      }
      renderSkillIcon(tooltipIcon, skill?.iconUrl, 1.02)
    }

    tooltipPanel.hidden = false
    tooltipPanel.style.display = 'grid'
    tooltipPanel.setAttribute('aria-hidden', 'false')
    tooltipPanel.style.visibility = 'hidden'
    tooltipPanel.style.left = '0px'
    tooltipPanel.style.top = '0px'

    const tooltipRect = tooltipPanel.getBoundingClientRect()
    const maxLeft = Math.max(
      HUD_MARGIN,
      window.innerWidth - tooltipRect.width - HUD_MARGIN
    )
    const maxTop = Math.max(
      HUD_MARGIN,
      window.innerHeight - tooltipRect.height - HUD_MARGIN
    )

    tooltipPanel.style.left = `${clamp(
      tooltipAnchor.x + 14,
      HUD_MARGIN,
      maxLeft
    )}px`
    tooltipPanel.style.top = `${clamp(
      tooltipAnchor.y + 12,
      HUD_MARGIN,
      maxTop
    )}px`
    tooltipPanel.style.visibility = 'visible'
  }

  function showTooltipForConsumable(event: PointerEvent) {
    const button =
      event.currentTarget instanceof HTMLButtonElement
        ? event.currentTarget
        : undefined

    if (!button) {
      hideTooltip()
      return
    }

    const index = Number(button.dataset.playerQuickslotIndex)

    if (Number.isNaN(index)) {
      hideTooltip()
      return
    }

    hoveredHudTarget = {
      kind: 'consumable',
      index
    }
    tooltipAnchor = {
      x: event.clientX,
      y: event.clientY
    }
    syncTooltip()
  }

  function showTooltipForSkill(event: PointerEvent) {
    const button =
      event.currentTarget instanceof HTMLButtonElement
        ? event.currentTarget
        : undefined

    if (!button) {
      hideTooltip()
      return
    }

    const index = getSkillButtonIndexFromButton(button)

    if (Number.isNaN(index)) {
      hideTooltip()
      return
    }

    hoveredHudTarget = {
      kind: 'skill',
      index
    }
    tooltipAnchor = {
      x: event.clientX,
      y: event.clientY
    }
    syncTooltip()
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

  const getQuickslotButton = (target: EventTarget | null) => {
    if (!(target instanceof Element)) {
      return undefined
    }

    return target.closest('button[data-player-quickslot-index]') as
      | HTMLButtonElement
      | undefined
  }

  const getSkillButton = (target: EventTarget | null) => {
    if (!(target instanceof Element)) {
      return undefined
    }

    return target.closest('button[data-player-skill-index]') as
      | HTMLButtonElement
      | undefined
  }

  const getQuickslotIndexFromButton = (button: HTMLButtonElement): number =>
    Number(button.dataset.playerQuickslotIndex)

  const hasInventoryDragData = (dataTransfer: DataTransfer): boolean =>
    Array.from(dataTransfer.types).some((type) =>
      type === 'application/x-player-drag-origin' ||
      type === 'application/x-player-inventory-slot-index' ||
      type === 'text/plain'
    )

  const getDraggedInventorySlotIndex = (event: DragEvent): number | undefined => {
    const rawValue =
      event.dataTransfer?.getData('application/x-player-inventory-slot-index') ||
      event.dataTransfer?.getData('text/plain')

    if (!rawValue) {
      return undefined
    }

    const slotIndex = Number(rawValue)

    return Number.isNaN(slotIndex) ? undefined : slotIndex
  }

  function handleQuickslotDragStart(event: DragEvent) {
    const quickslotButton = getQuickslotButton(event.target)

    if (!quickslotButton || !event.dataTransfer) {
      return
    }

    const quickslotIndex = getQuickslotIndexFromButton(quickslotButton)
    const quickslot = getQuickslots().slots[quickslotIndex]
    const inventorySlotIndex = quickslot?.inventorySlotIndex
    const inventory = getInventory()
    const item =
      inventorySlotIndex === undefined
        ? undefined
        : inventory.slots[inventorySlotIndex]

    if (!item || !isLikelyConsumableInventoryItem(item)) {
      event.preventDefault()
      return
    }

    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData(
      'application/x-player-inventory-slot-index',
      String(inventorySlotIndex)
    )
    event.dataTransfer.setData('text/plain', String(inventorySlotIndex))
  }

  function handleQuickslotDragOver(event: DragEvent) {
    const quickslotButton = getQuickslotButton(event.target)

    if (!quickslotButton || !event.dataTransfer) {
      return
    }

    if (!hasInventoryDragData(event.dataTransfer)) {
      return
    }

    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }

  function handleQuickslotDrop(event: DragEvent) {
    const quickslotButton = getQuickslotButton(event.target)

    if (!quickslotButton) {
      return
    }

    const quickslotIndex = getQuickslotIndexFromButton(quickslotButton)
    const inventorySlotIndex = getDraggedInventorySlotIndex(event)
    const inventory = getInventory()

    if (inventorySlotIndex === undefined) {
      return
    }

    const item = inventory.slots[inventorySlotIndex]

    if (!item || !isLikelyConsumableInventoryItem(item)) {
      return
    }

    event.preventDefault()
    onRequestQuickslotChange(
      setPlayerQuickslotAssignment({
        quickslots: getQuickslots(),
        quickslotIndex,
        inventorySlotIndex
      })
    )
  }

  const handleQuickslotContextMenu = (event: MouseEvent) => {
    const quickslotButton = getQuickslotButton(event.target)

    if (!quickslotButton) {
      return
    }

    event.preventDefault()
    onRequestQuickslotChange(
      clearPlayerQuickslotAssignment({
        quickslots: getQuickslots(),
        quickslotIndex: getQuickslotIndexFromButton(quickslotButton)
      })
    )
  }

  function handleSkillSlotDragStart(event: DragEvent) {
    const skillButton =
      event.currentTarget instanceof HTMLButtonElement
        ? event.currentTarget
        : undefined

    if (!skillButton || !event.dataTransfer) {
      return
    }

    const skillSlotIndex = getSkillSlotButtonIndexFromButton(skillButton)
    const skillSlot = getSkillSlots().slots[skillSlotIndex]

    if (!skillSlot) {
      event.preventDefault()
      return
    }

    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData(PLAYER_SKILL_DRAG_MIME_TYPE, skillSlot.skillId)
    event.dataTransfer.setData('text/plain', skillSlot.skillId)

    const skillIcon = skillIcons[skillSlotIndex]

    if (skillIcon) {
      event.dataTransfer.setDragImage(
        skillIcon,
        Math.round(skillIcon.offsetWidth / 2),
        Math.round(skillIcon.offsetHeight / 2)
      )
    }
  }

  function handleSkillSlotDragOver(event: DragEvent) {
    const skillButton =
      event.currentTarget instanceof HTMLButtonElement
        ? event.currentTarget
        : undefined

    if (!skillButton || !event.dataTransfer) {
      return
    }

    if (!hasSkillDragData(event.dataTransfer)) {
      return
    }

    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }

  function handleSkillSlotDrop(event: DragEvent) {
    const skillButton =
      event.currentTarget instanceof HTMLButtonElement
        ? event.currentTarget
        : undefined

    if (!skillButton) {
      return
    }

    const skillId = getDraggedSkillId(event)

    if (!skillId) {
      return
    }

    event.preventDefault()
    onRequestSkillSlotsChange(
      setPlayerSkillSlotAssignment({
        skillSlots: getSkillSlots(),
        skillSlotIndex: getSkillSlotButtonIndexFromButton(skillButton),
        skillId
      })
    )
  }

  const handleSkillSlotContextMenu = (event: MouseEvent) => {
    const skillButton = getSkillButton(event.target)

    if (!skillButton) {
      return
    }

    event.preventDefault()
    onRequestSkillSlotsChange(
      clearPlayerSkillSlotAssignment({
        skillSlots: getSkillSlots(),
        skillSlotIndex: getSkillSlotButtonIndexFromButton(skillButton)
      })
    )
  }

  const syncResourceRow = (
    row: HTMLDivElement,
    value: { current: number; max: number },
    color: string,
    valueText?: string
  ) => {
    const fill = row.querySelector<HTMLElement>('.player-hud-overlay__resource-fill')
    const valueLabel = row.querySelector<HTMLElement>('.player-hud-overlay__resource-value')

    if (!fill || !valueLabel) {
      return
    }

    const ratio = value.max === 0 ? 1 : value.current / value.max

    fill.style.width = `${clamp(ratio * 100, 0, 100)}%`
    fill.style.background = color
    valueLabel.textContent = valueText ?? `${value.current} / ${value.max}`
  }

  const syncLayout = () => {
    const uiScale = getResponsiveUiScale()
    const inventory = getInventory()
    const panelWidth = clamp(
      window.innerWidth - HUD_MARGIN * 2,
      HUD_PANEL_MIN_WIDTH,
      HUD_PANEL_MAX_WIDTH
    )

    panel.style.width = `${panelWidth}px`
    panel.style.height = 'auto'
    panel.style.left = '50%'
    panel.style.bottom = `${Math.round(HUD_MARGIN * uiScale)}px`
    panel.style.transformOrigin = 'center bottom'
    panel.style.transform = `translateX(-50%) scale(${
      uiScale * HUD_PANEL_SCALE_MULTIPLIER
    })`

    const nextExperienceToLevelUp = getPlayerExperienceToNextLevel(profile.level)
    const jobDisplayName = getPlayerJobDisplayName({
      job: profile.job,
      level: profile.level
    })

    nicknameRow.value.textContent = profile.name
    levelRow.value.textContent = `Lv ${profile.level}`
    jobRow.value.textContent = jobDisplayName
    syncResourceRow(hpRow.row, profile.hp, HEALTH_BAR_COLOR)
    syncResourceRow(mpRow.row, profile.mp, MANA_BAR_COLOR)
    syncResourceRow(
      expRow.row,
      {
        current: profile.experience.current,
        max: nextExperienceToLevelUp
      },
      '#d7b24d',
      profile.level >= PLAYER_MAX_LEVEL
        ? 'MAX'
        : `${profile.experience.current} / ${nextExperienceToLevelUp}`
    )

    for (let index = 0; index < skillButtons.length; index += 1) {
      const skillSlot = getSkillSlots().slots[index]
      const skill = skillSlot
        ? getPlayerSkillDisplayInfoById(skillSlot.skillId)
        : undefined
      const skillButton = skillButtons[index]
      const skillIcon = skillIcons[index]
      const hotkeyLabel = skillHotkeyLabels[index]
      const nameLabel = skillNameLabels[index]
      const descriptionLabel = skillDescriptionLabels[index]

      setSpriteFrame(skillButton, BUTTON_SQUARE_FRAME)
      setEmptySlotAppearance(skillButton)
      skillButton.draggable = Boolean(skill)
      skillButton.classList.toggle(
        'player-hud-overlay__skill-slot--draggable',
        Boolean(skill)
      )
      skillButton.classList.toggle(
        'player-hud-overlay__skill-slot--filled',
        Boolean(skill)
      )
      skillButton.setAttribute(
        'aria-label',
        skill
          ? `${skill.label} ${skill.description}`
          : `스킬 슬롯 ${index + 1} 비어있음`
      )
      skillButton.title = skill
        ? `${skill.label} · ${skill.description}`
        : `스킬 슬롯 ${index + 1} 비어있음`
      hotkeyLabel.textContent = PLAYER_SKILL_SLOT_HOTKEY_LABELS[index]
      nameLabel.textContent = skill ? skill.label : '비어있음'
      descriptionLabel.textContent = skill ? skill.description : ''
      nameLabel.hidden = Boolean(skill)
      descriptionLabel.hidden = Boolean(skill)
      renderSkillIcon(skillIcon, skill?.iconUrl, skill ? 1 : 0.92)
    }

    const quickslots = getQuickslots()

    for (let index = 0; index < consumableButtons.length; index += 1) {
      const consumableButton = consumableButtons[index]
      const hotkeyLabel = consumableHotkeyLabels[index]
      const icon = consumableIcons[index]
      const nameLabel = consumableNameLabels[index]
      const quantityLabel = consumableQuantityLabels[index]
      const quickslot = quickslots.slots[index]
      const item = quickslot ? inventory.slots[quickslot.inventorySlotIndex] : undefined
      const isQuickslotItem = item ? isLikelyConsumableInventoryItem(item) : false
      const hasQuickslotItem = Boolean(item && isQuickslotItem)
      const quickslotItem = hasQuickslotItem ? item : undefined

      setSpriteFrame(consumableButton, BUTTON_SQUARE_FRAME)
      setEmptySlotAppearance(consumableButton)
      consumableButton.classList.toggle(
        'player-hud-overlay__consumable-slot--filled',
        hasQuickslotItem
      )
      consumableButton.classList.toggle(
        'player-hud-overlay__consumable-slot--empty',
        !hasQuickslotItem
      )
      consumableButton.draggable = hasQuickslotItem
      if (hasQuickslotItem) {
        consumableButton.style.borderColor = 'rgba(91, 134, 214, 0.42)'
        consumableButton.style.boxShadow =
          'inset 0 1px 0 rgba(255, 255, 255, 0.7), 0 0 0 1px rgba(91, 134, 214, 0.08)'
      } else {
        consumableButton.style.borderColor = 'rgba(111, 89, 58, 0.34)'
        consumableButton.style.boxShadow =
          'inset 0 1px 0 rgba(255, 255, 255, 0.7)'
      }
      consumableButton.setAttribute(
        'aria-label',
        quickslotItem
          ? `퀵슬롯 ${index + 1} ${quickslotItem.label}${
              quickslotItem.quantity > 1 ? ` x${quickslotItem.quantity}` : ''
            }. 드래그해서 다른 칸으로 이동`
          : `퀵슬롯 ${index + 1} 비어있음`
      )
      consumableButton.title = quickslotItem
        ? `${quickslotItem.label}${quickslotItem.quantity > 1 ? ` x${quickslotItem.quantity}` : ''}. 드래그해서 다른 칸으로 이동`
        : `퀵슬롯 ${index + 1} 비어있음`
      hotkeyLabel.textContent = QUICK_CONSUMABLE_HOTKEYS[index]

      if (!icon) {
        continue
      }

      renderConsumableIcon(icon, quickslotItem?.id)
      nameLabel.hidden = Boolean(quickslotItem)
      nameLabel.textContent = quickslotItem ? '' : '비어있음'
      quantityLabel.hidden = !(quickslotItem && quickslotItem.quantity > 1)
      quantityLabel.textContent =
        quickslotItem && quickslotItem.quantity > 1
          ? `x${quickslotItem.quantity}`
          : ''
    }

    if (hoveredHudTarget) {
      syncTooltip()
    }
  }

  consumableGrid.addEventListener('contextmenu', handleQuickslotContextMenu)
  skillGrid.addEventListener('contextmenu', handleSkillSlotContextMenu)
  panel.addEventListener('pointerleave', hideTooltip)

  const destroy = () => {
    overlayRoot.remove()
    for (const consumableButton of consumableButtons) {
      consumableButton.removeEventListener('dragstart', handleQuickslotDragStart)
      consumableButton.removeEventListener('dragover', handleQuickslotDragOver)
      consumableButton.removeEventListener('drop', handleQuickslotDrop)
      consumableButton.removeEventListener('pointerenter', showTooltipForConsumable)
      consumableButton.removeEventListener('pointermove', showTooltipForConsumable)
      consumableButton.removeEventListener('pointerleave', hideTooltip)
    }
    for (const skillButton of skillButtons) {
      skillButton.removeEventListener('dragstart', handleSkillSlotDragStart)
      skillButton.removeEventListener('dragover', handleSkillSlotDragOver)
      skillButton.removeEventListener('drop', handleSkillSlotDrop)
      skillButton.removeEventListener('pointerenter', showTooltipForSkill)
      skillButton.removeEventListener('pointermove', showTooltipForSkill)
      skillButton.removeEventListener('pointerleave', hideTooltip)
    }
    consumableGrid.removeEventListener('contextmenu', handleQuickslotContextMenu)
    skillGrid.removeEventListener('contextmenu', handleSkillSlotContextMenu)
    panel.removeEventListener('pointerleave', hideTooltip)
  }

  syncLayout()

  return {
    syncFrame: syncLayout,
    destroy
  }
}

type ResourceRow = {
  row: HTMLDivElement
}

type InfoRow = {
  row: HTMLDivElement
  value: HTMLDivElement
}

const createInfoRow = (label: string): InfoRow => {
  const row = document.createElement('div')
  const labelElement = document.createElement('div')
  const value = document.createElement('div')

  row.className = 'player-hud-overlay__info-row'
  labelElement.className = 'player-hud-overlay__info-label'
  labelElement.textContent = label
  value.className = 'player-hud-overlay__info-value'

  row.append(labelElement, value)

  return {
    row,
    value
  }
}

const createResourceRow = (label: string): ResourceRow => {
  const row = document.createElement('div')
  const labelElement = document.createElement('div')
  const track = document.createElement('div')
  const fill = document.createElement('div')
  const value = document.createElement('div')

  row.className = 'player-hud-overlay__resource-row'
  labelElement.className = 'player-hud-overlay__resource-label'
  labelElement.textContent = label
  track.className = 'player-hud-overlay__resource-track'
  fill.className = 'player-hud-overlay__resource-fill'
  value.className = 'player-hud-overlay__resource-value'

  track.append(fill)
  row.append(labelElement, track, value)

  return {
    row
  }
}

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))
