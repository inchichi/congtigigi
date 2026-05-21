import type { PlayerProfile, PlayerStatId } from '../game/playerProfile'
import {
  PLAYER_MAX_LEVEL,
  getPlayerJobPrimaryStatLabel
} from '../game/playerProfile'
import { spendPlayerStatPoint } from '../game/playerProgression'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreatePlayerStatOverlayInput = {
  mountElement: HTMLElement
  profile: PlayerProfile
  getIsOpen: () => boolean
  onRequestOpenChange: (isOpen: boolean) => void
  onRequestProfileChange: (nextProfile: PlayerProfile) => void
}

export type PlayerStatOverlay = {
  syncFrame: () => void
  destroy: () => void
}

type StatRowConfig = {
  id: PlayerStatId
  label: string
  description: string
}

const STAT_ROWS: StatRowConfig[] = [
  {
    id: 'strength',
    label: '힘',
    description: '물리 공격력이 올라갑니다'
  },
  {
    id: 'agility',
    label: '민첩',
    description: '이동 속도가 점점 빨라집니다'
  },
  {
    id: 'intelligence',
    label: '지력',
    description: '최대 마나가 올라갑니다'
  },
  {
    id: 'luck',
    label: '행운',
    description: '회피율이 올라갑니다'
  }
]

const OVERLAY_MARGIN = 16
const PANEL_MIN_WIDTH = 280
const PANEL_MAX_WIDTH = 340
const PANEL_MIN_HEIGHT = 300
const PANEL_MAX_HEIGHT = 380

export const createPlayerStatOverlay = ({
  mountElement,
  profile,
  getIsOpen,
  onRequestOpenChange,
  onRequestProfileChange
}: CreatePlayerStatOverlayInput): PlayerStatOverlay => {
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
  const statGrid = document.createElement('div')
  const footerElement = document.createElement('div')
  const statRows: HTMLButtonElement[] = []
  const statContents: HTMLDivElement[] = []
  const statLabelElements: HTMLSpanElement[] = []
  const statValues: HTMLSpanElement[] = []
  const statDescriptions: HTMLSpanElement[] = []
  const statActions: HTMLSpanElement[] = []
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

  overlayRoot.className = 'player-stat-overlay player-stat-overlay--sidebar'

  backdropButton.type = 'button'
  backdropButton.className = 'player-stat-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.tabIndex = -1
  backdropButton.setAttribute('aria-hidden', 'true')

  panel.className = 'player-stat-overlay__panel'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'false')
  panel.setAttribute('aria-labelledby', 'player-stat-overlay-title')

  panelBody.className = 'player-stat-overlay__panel-body'
  headerRow.className = 'player-stat-overlay__header'
  titleGroup.className = 'player-stat-overlay__title-group'
  titleElement.id = 'player-stat-overlay-title'
  titleElement.className = 'player-stat-overlay__title'
  titleElement.textContent = '스텟'
  summaryElement.className = 'player-stat-overlay__summary'
  summaryElement.textContent = `남은 포인트 ${profile.statPoints}`

  closeButton.type = 'button'
  closeButton.className = 'player-stat-overlay__close'
  closeButton.setAttribute('aria-label', '스텟 창 닫기')
  closeButton.title = '스텟 창 닫기 (Esc)'

  closeIcon.className = 'player-stat-overlay__close-icon'
  closeIcon.textContent = '×'
  closeIcon.setAttribute('aria-hidden', 'true')

  infoCard.className = 'player-stat-overlay__info-card'
  infoName.className = 'player-stat-overlay__info-name'
  infoName.textContent = profile.name
  infoJob.className = 'player-stat-overlay__info-job'
  infoJob.textContent = `${profile.job}`
  infoLevel.className = 'player-stat-overlay__info-level'
  infoLevel.textContent =
    profile.level >= PLAYER_MAX_LEVEL
      ? `레벨 ${profile.level} · MAX`
      : `레벨 ${profile.level}`
  infoHint.className = 'player-stat-overlay__info-hint'
  infoHint.textContent = '전사=힘 · 궁수=민첩 · 마법사=지력 · 도적=행운'

  statGrid.className = 'player-stat-overlay__stat-grid'
  footerElement.className = 'player-stat-overlay__footer'
  footerElement.textContent = '클릭해서 스텟 강화 · Esc로 닫기'

  for (const stat of STAT_ROWS) {
    const row = document.createElement('button')
    const content = document.createElement('div')
    const label = document.createElement('span')
    const description = document.createElement('span')
    const value = document.createElement('span')
    const action = document.createElement('span')

    row.type = 'button'
    row.className = 'player-stat-overlay__stat-row'
    row.dataset.playerStatId = stat.id
    row.setAttribute('aria-label', `${stat.label} 강화`)

    content.className = 'player-stat-overlay__stat-content'
    label.className = 'player-stat-overlay__stat-label'
    label.textContent = stat.label
    description.className = 'player-stat-overlay__stat-description'
    description.textContent = stat.description
    value.className = 'player-stat-overlay__stat-value'
    value.textContent = '0'
    action.className = 'player-stat-overlay__stat-action'
    action.textContent = '+1'

    content.append(label, description)
    row.append(content, value, action)
    statGrid.append(row)

    statRows.push(row)
    statContents.push(content)
    statLabelElements.push(label)
    statValues.push(value)
    statDescriptions.push(description)
    statActions.push(action)
  }

  titleGroup.append(titleElement, summaryElement)
  headerRow.append(titleGroup, closeButton)
  closeButton.append(closeIcon)
  infoCard.append(infoName, infoJob, infoLevel, infoHint)
  panelBody.append(headerRow, infoCard, statGrid, footerElement)
  panel.append(panelBody)
  overlayRoot.append(backdropButton, panel)
  mountElement.append(overlayRoot)

  const clampPanelPosition = (
    left: number,
    top: number,
    width: number,
    height: number
  ) => ({
    left: clamp(
      left,
      OVERLAY_MARGIN,
      Math.max(OVERLAY_MARGIN, window.innerWidth - width - OVERLAY_MARGIN)
    ),
    top: clamp(
      top,
      OVERLAY_MARGIN,
      Math.max(OVERLAY_MARGIN, window.innerHeight - height - OVERLAY_MARGIN)
    )
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
    headerRow.classList.add('player-stat-overlay__header--dragging')
    event.preventDefault()
  }

  const stopDragging = (pointerId?: number) => {
    if (dragState && pointerId !== undefined && dragState.pointerId !== pointerId) {
      return
    }

    dragState = undefined
    headerRow.classList.remove('player-stat-overlay__header--dragging')
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
      backdropButton.hidden = true
      panel.hidden = true
      return
    }

    const availableWidth = Math.max(1, window.innerWidth - OVERLAY_MARGIN * 2)
    const availableHeight = Math.max(1, window.innerHeight - OVERLAY_MARGIN * 2)
    const panelWidth = clamp(
      Math.round(window.innerWidth * 0.24),
      PANEL_MIN_WIDTH,
      Math.min(PANEL_MAX_WIDTH, availableWidth)
    )
    const panelHeight = clamp(
      Math.round(window.innerHeight * 0.42),
      PANEL_MIN_HEIGHT,
      Math.min(PANEL_MAX_HEIGHT, availableHeight)
    )
    const renderedWidth = panelWidth * uiScale
    const renderedHeight = panelHeight * uiScale
    const defaultPosition = clampPanelPosition(
      window.innerWidth - renderedWidth - OVERLAY_MARGIN,
      OVERLAY_MARGIN,
      renderedWidth,
      renderedHeight
    )

    backdropButton.hidden = false
    panel.hidden = false
    panel.style.left = 'auto'
    panel.style.right = 'auto'
    panel.style.top = 'auto'
    panel.style.bottom = 'auto'
    panel.style.width = `${panelWidth}px`
    panel.style.height = `${panelHeight}px`
    panel.style.transformOrigin = 'left top'
    panel.style.transform = `scale(${uiScale})`
    panelBody.style.height = '100%'
    panelBody.style.overflowY = 'auto'

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
    infoLevel.textContent =
      profile.level >= PLAYER_MAX_LEVEL
        ? `레벨 ${profile.level} · MAX`
        : `레벨 ${profile.level}`
    summaryElement.textContent = `남은 포인트 ${profile.statPoints}`
    const recommendedStatLabel = getPlayerJobPrimaryStatLabel(profile.job)

    infoHint.textContent = recommendedStatLabel
      ? `${profile.job}의 주력 스텟은 ${recommendedStatLabel}입니다`
      : '전사=힘 · 궁수=민첩 · 마법사=지력 · 도적=행운'

    for (let index = 0; index < STAT_ROWS.length; index += 1) {
      const stat = STAT_ROWS[index]
      const row = statRows[index]
      const content = statContents[index]
      const label = statLabelElements[index]
      const value = statValues[index]
      const description = statDescriptions[index]
      const action = statActions[index]
      const canUpgrade = profile.statPoints > 0

      row.disabled = !canUpgrade
      row.classList.toggle('player-stat-overlay__stat-row--locked', !canUpgrade)
      row.classList.toggle(
        'player-stat-overlay__stat-row--recommended',
        stat.label === recommendedStatLabel
      )
      content.classList.toggle('player-stat-overlay__stat-content--locked', !canUpgrade)
      row.title = canUpgrade
        ? `${stat.label}을 1 올립니다`
        : '스텟 포인트가 부족합니다'
      row.setAttribute(
        'aria-label',
        canUpgrade
          ? `${stat.label} 강화`
          : `${stat.label} 강화 불가. 스텟 포인트가 부족합니다`
      )
      label.textContent = stat.label
      value.textContent = String(profile.stats[stat.id])
      description.textContent = stat.description
      action.textContent = canUpgrade ? '+1' : '잠김'
    }
  }

  const handleStatClick = (statId: PlayerStatId) => {
    const nextProfile = spendPlayerStatPoint(profile, statId)

    if (!nextProfile) {
      return
    }

    onRequestProfileChange(nextProfile)
  }

  const handlePanelClick = (event: MouseEvent) => {
    const target = event.target

    if (!(target instanceof Element)) {
      return
    }

    const statButton = target.closest('button[data-player-stat-id]') as
      | HTMLButtonElement
      | null

    if (!statButton) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    const statId = statButton.dataset.playerStatId as PlayerStatId | undefined

    if (statId) {
      handleStatClick(statId)
    }
  }

  const handleBackdropPointerDown = (event: PointerEvent) => {
    event.preventDefault()
    event.stopPropagation()
    stopDragging()
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
