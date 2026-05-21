import type { PlayerProfile } from '../game/playerProfile'
import {
  getPlayerSkillLevelLabel,
  getPlayerSkillPointCost,
  getPlayerSkillUserLevel,
  spendPlayerSkillPoint
} from '../game/playerProgression'
import {
  PLAYER_SKILL_DRAG_MIME_TYPE
} from '../game/playerSkillSlots'
import {
  PLAYER_PROTECT_SKILL_ID,
  getPlayerSkillDisplayInfoById,
  isPlayerSkillUnlockedInProfile
} from '../game/playerSkills'
import { PLAYER_SMASH_SKILL_ID } from '../game/playerSmashSkill'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreatePlayerSkillOverlayInput = {
  mountElement: HTMLElement
  profile: PlayerProfile
  getIsOpen: () => boolean
  onRequestOpenChange: (isOpen: boolean) => void
  onRequestProfileChange: (nextProfile: PlayerProfile) => void
}

export type PlayerSkillOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const OVERLAY_MARGIN = 16
const PANEL_MIN_WIDTH = 440
const PANEL_MAX_WIDTH = 620
const PANEL_MIN_HEIGHT = 340
const PANEL_MAX_HEIGHT = 480
const PANEL_MARGIN = 16

export const createPlayerSkillOverlay = ({
  mountElement,
  profile,
  getIsOpen,
  onRequestOpenChange,
  onRequestProfileChange
}: CreatePlayerSkillOverlayInput): PlayerSkillOverlay => {
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
  const infoCard = document.createElement('div')
  const infoName = document.createElement('div')
  const infoJob = document.createElement('div')
  const infoLevel = document.createElement('div')
  const infoHint = document.createElement('div')
  const skillGrid = document.createElement('div')
  const footerElement = document.createElement('div')
  const skillRows: HTMLButtonElement[] = []
  const skillIcons: HTMLSpanElement[] = []
  const skillHotkeys: HTMLSpanElement[] = []
  const skillNames: HTMLSpanElement[] = []
  const skillLevels: HTMLSpanElement[] = []
  const skillDescriptions: HTMLSpanElement[] = []
  const skillActions: HTMLSpanElement[] = []
  const visibleSkillEntries = [
    {
      profileSkillIndex: 0,
      skillId: PLAYER_SMASH_SKILL_ID
    },
    {
      profileSkillIndex: 1,
      skillId: PLAYER_PROTECT_SKILL_ID
    }
  ] as const
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

  overlayRoot.className = 'player-skill-overlay'

  backdropButton.type = 'button'
  backdropButton.className = 'player-skill-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.tabIndex = -1
  backdropButton.setAttribute('aria-hidden', 'true')

  panel.className = 'player-skill-overlay__panel'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'false')
  panel.setAttribute('aria-labelledby', 'player-skill-overlay-title')

  panelBody.className = 'player-skill-overlay__panel-body'
  headerRow.className = 'player-skill-overlay__header'
  titleGroup.className = 'player-skill-overlay__title-group'
  titleElement.id = 'player-skill-overlay-title'
  titleElement.className = 'player-skill-overlay__title'
  titleElement.textContent = '스킬'
  summaryElement.className = 'player-skill-overlay__summary'
  summaryElement.textContent = `남은 포인트 ${profile.availableSkillPoints}`

  closeButton.type = 'button'
  closeButton.className = 'player-skill-overlay__close'
  closeButton.setAttribute('aria-label', '스킬 창 닫기')
  closeButton.title = '스킬 창 닫기 (Esc)'

  closeIcon.className = 'player-skill-overlay__close-icon'
  closeIcon.textContent = '×'
  closeIcon.setAttribute('aria-hidden', 'true')

  infoCard.className = 'player-skill-overlay__info-card'
  infoName.className = 'player-skill-overlay__info-name'
  infoName.textContent = profile.name
  infoJob.className = 'player-skill-overlay__info-job'
  infoJob.textContent = profile.job
  infoLevel.className = 'player-skill-overlay__info-level'
  infoLevel.textContent = `사용자 레벨 ${getPlayerSkillUserLevel(
    profile.totalSkillPointsEarned
  )}`
  infoHint.className = 'player-skill-overlay__info-hint'
  infoHint.textContent = '레벨업마다 스킬 포인트를 사용해 기술을 성장시킵니다'

  skillGrid.className = 'player-skill-overlay__skill-grid'
  footerElement.className = 'player-skill-overlay__footer'
  footerElement.textContent = '클릭해서 스킬 강화 · Esc로 닫기'

  for (const skillEntry of visibleSkillEntries) {
    const skill = profile.skills[skillEntry.profileSkillIndex]
    const displaySkill = getPlayerSkillDisplayInfoById(skillEntry.skillId)
    const row = document.createElement('button')
    const icon = document.createElement('span')
    const hotkey = document.createElement('span')
    const content = document.createElement('div')
    const name = document.createElement('span')
    const description = document.createElement('span')
    const level = document.createElement('span')
    const action = document.createElement('span')

    row.type = 'button'
    row.className = 'player-skill-overlay__skill-row'
    row.dataset.playerSkillIndex = String(skillEntry.profileSkillIndex)
    row.dataset.playerSkillId = skillEntry.skillId
    row.setAttribute('aria-label', `${skill.label} 강화`)
    row.draggable = isPlayerSkillUnlockedInProfile(profile, skillEntry.skillId)

    icon.className = 'player-skill-overlay__skill-icon'
    icon.setAttribute('aria-hidden', 'true')

    hotkey.className = 'player-skill-overlay__skill-hotkey'
    hotkey.textContent = skill.hotkey

    content.className = 'player-skill-overlay__skill-content'
    name.className = 'player-skill-overlay__skill-name'
    name.textContent = skill.label
    description.className = 'player-skill-overlay__skill-description'
    description.textContent = skill.description
    level.className = 'player-skill-overlay__skill-level'
    level.textContent = getPlayerSkillLevelLabel(skill)
    action.className = 'player-skill-overlay__skill-action'
    action.textContent = '강화'

    content.append(name, description)
    row.append(icon, hotkey, content, level, action)
    renderSkillIcon(icon, displaySkill.iconUrl)
    skillGrid.append(row)

    skillRows.push(row)
    skillIcons.push(icon)
    skillHotkeys.push(hotkey)
    skillNames.push(name)
    skillLevels.push(level)
    skillDescriptions.push(description)
    skillActions.push(action)
  }

  titleGroup.append(titleElement, summaryElement)
  headerRow.append(titleGroup, closeButton)
  closeButton.append(closeIcon)
  infoCard.append(infoName, infoJob, infoLevel, infoHint)
  panelBody.append(headerRow, infoCard, skillGrid, footerElement)
  panel.append(panelBody)
  overlayRoot.append(backdropButton, panel)
  mountElement.append(overlayRoot)

  const clampPanelPosition = (
    left: number,
    top: number,
    width: number,
    height: number
  ) => ({
    left: clamp(left, PANEL_MARGIN, Math.max(PANEL_MARGIN, window.innerWidth - width - PANEL_MARGIN)),
    top: clamp(top, PANEL_MARGIN, Math.max(PANEL_MARGIN, window.innerHeight - height - PANEL_MARGIN))
  })

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
    headerRow.classList.add('player-skill-overlay__header--dragging')
    event.preventDefault()
  }

  const stopDragging = (pointerId?: number) => {
    if (dragState && pointerId !== undefined && dragState.pointerId !== pointerId) {
      return
    }

    dragState = undefined
    headerRow.classList.remove('player-skill-overlay__header--dragging')
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
      return
    }

    const availableWidth = Math.max(1, window.innerWidth - OVERLAY_MARGIN * 2)
    const availableHeight = Math.max(1, window.innerHeight - OVERLAY_MARGIN * 2)
    const panelWidth = clamp(availableWidth, PANEL_MIN_WIDTH, PANEL_MAX_WIDTH)
    const panelHeight = clamp(availableHeight, PANEL_MIN_HEIGHT, PANEL_MAX_HEIGHT)
    const renderedWidth = panelWidth * uiScale
    const renderedHeight = panelHeight * uiScale
    const defaultPosition = clampPanelPosition(
      (window.innerWidth - renderedWidth) / 2,
      (window.innerHeight - renderedHeight) / 2,
      renderedWidth,
      renderedHeight
    )

    backdropButton.hidden = false
    panel.hidden = false
    panel.style.width = `${panelWidth}px`
    panel.style.height = `${panelHeight}px`
    panel.style.transformOrigin = 'left top'
    panel.style.transform = `scale(${uiScale})`

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

    infoName.textContent = profile.name
    infoJob.textContent = profile.job
    infoLevel.textContent = `사용자 레벨 ${getPlayerSkillUserLevel(
      profile.totalSkillPointsEarned
    )}`
    summaryElement.textContent = `남은 포인트 ${profile.availableSkillPoints}`

    for (let index = 0; index < visibleSkillEntries.length; index += 1) {
      const skillEntry = visibleSkillEntries[index]
      const skill = profile.skills[skillEntry.profileSkillIndex]

      if (!skill) {
        continue
      }

      const displaySkill = getPlayerSkillDisplayInfoById(skillEntry.skillId)
      const row = skillRows[index]
      const icon = skillIcons[index]
      const hotkey = skillHotkeys[index]
      const name = skillNames[index]
      const level = skillLevels[index]
      const description = skillDescriptions[index]
      const action = skillActions[index]
    const skillPointCost = getPlayerSkillPointCost(skill)
    const canUpgrade =
      profile.availableSkillPoints >= skillPointCost && skillPointCost > 0
    const actionLabel =
      skill.level <= 0
        ? '잠김'
        : skill.level >= skill.maxLevel
          ? 'MAX'
          : '강화'

      row.classList.toggle('player-skill-overlay__skill-row--locked', !canUpgrade)
      row.classList.toggle(
        'player-skill-overlay__skill-row--draggable',
        isPlayerSkillUnlockedInProfile(profile, skillEntry.skillId)
      )
      row.draggable = isPlayerSkillUnlockedInProfile(profile, skillEntry.skillId)
      row.title = canUpgrade
        ? `${skill.label}을 ${skillPointCost} 포인트로 올립니다`
        : skill.level >= skill.maxLevel
          ? `${skill.label}은 이미 최대 레벨입니다`
          : '스킬 포인트가 부족합니다'
      row.setAttribute(
        'aria-label',
        canUpgrade
          ? `${skill.label} 강화`
          : skill.level >= skill.maxLevel
            ? `${skill.label} 최대 레벨`
            : `${skill.label} 강화 불가. 스킬 포인트가 부족합니다`
      )
      hotkey.textContent = skill.hotkey
      name.textContent = skill.label
      description.textContent = skill.description
      level.textContent = getPlayerSkillLevelLabel(skill)
      action.textContent = actionLabel
      renderSkillIcon(icon, displaySkill.iconUrl)
    }
  }

  const handleSkillClick = (skillIndex: number) => {
    const nextProfile = spendPlayerSkillPoint(profile, skillIndex)

    if (!nextProfile) {
      return
    }

    onRequestProfileChange(nextProfile)
  }

  const handleSkillRowDragStart = (event: DragEvent) => {
    const row =
      event.currentTarget instanceof HTMLButtonElement
        ? event.currentTarget
        : undefined

    if (!row || !event.dataTransfer) {
      return
    }

    const skillId = row.dataset.playerSkillId

    if (!skillId) {
      event.preventDefault()
      return
    }

    const skillIndex = Number(row.dataset.playerSkillIndex)
    const skill = Number.isNaN(skillIndex)
      ? undefined
      : profile.skills[skillIndex]

    if (!skill || !isPlayerSkillUnlockedInProfile(profile, skillId)) {
      event.preventDefault()
      return
    }

    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData(PLAYER_SKILL_DRAG_MIME_TYPE, skillId)
    event.dataTransfer.setData('text/plain', skillId)
  }

  const handlePanelClick = (event: MouseEvent) => {
    const target = event.target

    if (!(target instanceof Element)) {
      return
    }

    const skillButton = target.closest('button[data-player-skill-index]') as
      | HTMLButtonElement
      | null

    if (!skillButton) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    const skillIndex = Number(skillButton.dataset.playerSkillIndex)

    if (!Number.isNaN(skillIndex)) {
      handleSkillClick(skillIndex)
    }
  }

  const handleBackdropPointerDown = (event: PointerEvent) => {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  const handleBackdropClick = (event: MouseEvent) => {
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

  const handleCloseButtonClick = (event: MouseEvent) => {
    event.preventDefault()
    event.stopPropagation()
    stopDragging()
    onRequestOpenChange(false)
  }

  const handleGlobalKeyDown = (event: KeyboardEvent) => {
    if (!getIsOpen() || event.key !== 'Escape') {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  }

  backdropButton.addEventListener('pointerdown', handleBackdropPointerDown)
  backdropButton.addEventListener('click', handleBackdropClick)
  headerRow.addEventListener('pointerdown', startDragging)
  document.addEventListener('pointermove', handleDocumentPointerMove)
  document.addEventListener('pointerup', handleDocumentPointerUp)
  document.addEventListener('pointercancel', handleDocumentPointerUp)
  closeButton.addEventListener('pointerdown', handleCloseButtonPointerDown)
  closeButton.addEventListener('click', handleCloseButtonClick)
  panel.addEventListener('click', handlePanelClick)
  document.addEventListener('keydown', handleGlobalKeyDown, true)
  for (const skillRow of skillRows) {
    skillRow.addEventListener('dragstart', handleSkillRowDragStart)
  }

  const syncFrame = () => {
    syncLayout()
  }

  const destroy = () => {
    backdropButton.removeEventListener('pointerdown', handleBackdropPointerDown)
    backdropButton.removeEventListener('click', handleBackdropClick)
    headerRow.removeEventListener('pointerdown', startDragging)
    document.removeEventListener('pointermove', handleDocumentPointerMove)
    document.removeEventListener('pointerup', handleDocumentPointerUp)
    document.removeEventListener('pointercancel', handleDocumentPointerUp)
    closeButton.removeEventListener('pointerdown', handleCloseButtonPointerDown)
    closeButton.removeEventListener('click', handleCloseButtonClick)
    panel.removeEventListener('click', handlePanelClick)
    document.removeEventListener('keydown', handleGlobalKeyDown, true)
    for (const skillRow of skillRows) {
      skillRow.removeEventListener('dragstart', handleSkillRowDragStart)
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

const renderSkillIcon = (
  element: HTMLSpanElement,
  iconUrl: string | undefined
) => {
  if (!iconUrl) {
    element.hidden = true
    element.style.backgroundImage = 'none'
    element.style.backgroundRepeat = 'no-repeat'
    element.style.backgroundPosition = 'center'
    element.style.backgroundSize = 'contain'
    element.style.imageRendering = 'pixelated'
    return
  }

  element.hidden = false
  element.style.backgroundImage = `url(${iconUrl})`
  element.style.backgroundRepeat = 'no-repeat'
  element.style.backgroundPosition = 'center'
  element.style.backgroundSize = 'contain'
  element.style.imageRendering = 'pixelated'
}
