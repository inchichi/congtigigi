import type { PlayerProfile } from '../game/playerProfile'

type CreatePlayerHudOverlayInput = {
  mountElement: HTMLElement
  profile: PlayerProfile
  getIsInventoryOpen: () => boolean
  onRequestInventoryOpenChange: (nextIsOpen: boolean) => void
}

export type PlayerHudOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const UI_SPRITESHEET_IMAGE_URL = new URL(
  '../assets/spritesheets/uipack_rpg_sheet.png',
  import.meta.url
).href
const HUD_PANEL_FRAME = {
  x: 190,
  y: 100,
  width: 100,
  height: 100
}
const BUTTON_SQUARE_FRAME = {
  x: 293,
  y: 294,
  width: 45,
  height: 49
}
const HUD_MARGIN = 16
const HUD_PANEL_MIN_WIDTH = 420
const HUD_PANEL_MAX_WIDTH = 560
const HUD_PANEL_MIN_HEIGHT = 208
const HUD_PANEL_HEIGHT = 236
const HEALTH_BAR_COLOR = '#d06b5d'
const MANA_BAR_COLOR = '#5b86d6'

export const createPlayerHudOverlay = ({
  mountElement,
  profile,
  getIsInventoryOpen,
  onRequestInventoryOpenChange
}: CreatePlayerHudOverlayInput): PlayerHudOverlay => {
  const overlayRoot = document.createElement('div')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const headerRow = document.createElement('div')
  const titleGroup = document.createElement('div')
  const nameElement = document.createElement('div')
  const jobElement = document.createElement('div')
  const levelBadge = document.createElement('div')
  const attackBadge = document.createElement('div')
  const bagButton = document.createElement('button')
  const bagIcon = createBagIconSvg()
  const resourceGrid = document.createElement('div')
  const hpRow = createResourceRow('체력')
  const mpRow = createResourceRow('마나')
  const statGrid = document.createElement('div')
  const skillSection = document.createElement('div')
  const skillSectionTitle = document.createElement('div')
  const skillGrid = document.createElement('div')
  const skillButtons: HTMLButtonElement[] = []
  const skillHotkeyLabels: HTMLSpanElement[] = []
  const skillNameLabels: HTMLSpanElement[] = []
  const skillDescriptionLabels: HTMLSpanElement[] = []
  const statChips: HTMLDivElement[] = []
  const skillSlotCount = profile.skills.length

  overlayRoot.className = 'player-hud-overlay'

  panel.className = 'player-hud-overlay__panel'
  panel.setAttribute('aria-label', '캐릭터 상태')

  panelBody.className = 'player-hud-overlay__panel-body'
  headerRow.className = 'player-hud-overlay__header'
  titleGroup.className = 'player-hud-overlay__title-group'
  nameElement.className = 'player-hud-overlay__name'
  nameElement.textContent = profile.name
  jobElement.className = 'player-hud-overlay__job'
  jobElement.textContent = profile.job
  levelBadge.className = 'player-hud-overlay__level'
  levelBadge.textContent = `레벨 ${profile.level}`
  attackBadge.className = 'player-hud-overlay__attack-badge'
  attackBadge.textContent = '공격 A'

  bagButton.type = 'button'
  bagButton.className = 'player-hud-overlay__bag-button'
  bagButton.setAttribute('aria-label', '가방 열기')
  bagButton.setAttribute('aria-haspopup', 'dialog')
  bagButton.setAttribute('aria-controls', 'player-inventory-panel')
  bagButton.setAttribute('aria-expanded', 'false')
  bagButton.title = '가방 열기'

  bagIcon.classList.add('player-hud-overlay__bag-icon')
  bagIcon.setAttribute('aria-hidden', 'true')

  resourceGrid.className = 'player-hud-overlay__resource-grid'

  statGrid.className = 'player-hud-overlay__stat-grid'
  skillSection.className = 'player-hud-overlay__skill-section'
  skillSectionTitle.className = 'player-hud-overlay__skill-title'
  skillSectionTitle.textContent = '스킬'
  skillGrid.className = 'player-hud-overlay__skill-grid'

  for (const stat of [
    ['공격', profile.stats.attack],
    ['방어', profile.stats.defense],
    ['민첩', profile.stats.agility]
  ] as const) {
    const chip = document.createElement('div')

    chip.className = 'player-hud-overlay__stat-chip'
    chip.textContent = `${stat[0]} ${stat[1]}`
    statChips.push(chip)
    statGrid.append(chip)
  }

  for (let index = 0; index < skillSlotCount; index += 1) {
    const skill = profile.skills[index]
    const skillButton = document.createElement('button')
    const hotkeyLabel = document.createElement('span')
    const nameLabel = document.createElement('span')
    const descriptionLabel = document.createElement('span')

    skillButton.type = 'button'
    skillButton.className = 'player-hud-overlay__skill-slot'
    skillButton.setAttribute('aria-label', skill.label)
    skillButton.title = skill.description

    hotkeyLabel.className = 'player-hud-overlay__skill-hotkey'
    hotkeyLabel.textContent = skill.hotkey

    nameLabel.className = 'player-hud-overlay__skill-name'
    nameLabel.textContent = skill.label

    descriptionLabel.className = 'player-hud-overlay__skill-description'
    descriptionLabel.textContent = skill.description

    skillButton.append(hotkeyLabel, nameLabel, descriptionLabel)
    skillGrid.append(skillButton)

    skillButtons.push(skillButton)
    skillHotkeyLabels.push(hotkeyLabel)
    skillNameLabels.push(nameLabel)
    skillDescriptionLabels.push(descriptionLabel)
  }

  const headerIdentity = document.createElement('div')
  const headerMeta = document.createElement('div')

  headerIdentity.className = 'player-hud-overlay__header-identity'
  headerMeta.className = 'player-hud-overlay__header-meta'

  headerIdentity.append(nameElement, jobElement)
  headerMeta.append(levelBadge, attackBadge, bagButton)
  bagButton.append(bagIcon)
  headerRow.append(headerIdentity, headerMeta)

  resourceGrid.append(hpRow.row, mpRow.row)
  skillSection.append(skillSectionTitle, skillGrid)
  panelBody.append(headerRow, resourceGrid, statGrid, skillSection)
  panel.append(panelBody)
  overlayRoot.append(panel)
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

  const syncResourceRow = (
    row: HTMLDivElement,
    value: { current: number; max: number },
    color: string
  ) => {
    const fill = row.querySelector<HTMLElement>('.player-hud-overlay__resource-fill')
    const valueLabel = row.querySelector<HTMLElement>('.player-hud-overlay__resource-value')

    if (!fill || !valueLabel) {
      return
    }

    const ratio = value.max === 0 ? 0 : value.current / value.max

    fill.style.width = `${clamp(ratio * 100, 0, 100)}%`
    fill.style.background = color
    valueLabel.textContent = `${value.current} / ${value.max}`
  }

  const syncLayout = () => {
    const panelWidth = clamp(
      window.innerWidth - HUD_MARGIN * 2,
      HUD_PANEL_MIN_WIDTH,
      HUD_PANEL_MAX_WIDTH
    )
    const panelHeight = clamp(
      window.innerHeight - HUD_MARGIN * 2,
      HUD_PANEL_MIN_HEIGHT,
      HUD_PANEL_HEIGHT
    )

    setSpriteFrame(
      panel,
      HUD_PANEL_FRAME,
      panelWidth / HUD_PANEL_FRAME.width,
      panelHeight / HUD_PANEL_FRAME.height
    )
    panel.style.width = `${panelWidth}px`
    panel.style.height = `${panelHeight}px`
    panel.style.left = '50%'
    panel.style.bottom = `${HUD_MARGIN}px`
    panel.style.transform = 'translateX(-50%)'

    setSpriteFrame(bagButton, BUTTON_SQUARE_FRAME)
    bagButton.classList.toggle(
      'player-hud-overlay__bag-button--active',
      getIsInventoryOpen()
    )
    bagButton.setAttribute('aria-expanded', String(getIsInventoryOpen()))
    bagButton.setAttribute(
      'aria-label',
      getIsInventoryOpen() ? '가방 닫기' : '가방 열기'
    )
    bagButton.title = getIsInventoryOpen() ? '가방 닫기' : '가방 열기'

    syncResourceRow(hpRow.row, profile.hp, HEALTH_BAR_COLOR)
    syncResourceRow(mpRow.row, profile.mp, MANA_BAR_COLOR)

    for (let index = 0; index < skillButtons.length; index += 1) {
      const skill = profile.skills[index]
      const skillButton = skillButtons[index]

      setSpriteFrame(skillButton, BUTTON_SQUARE_FRAME)
      skillButton.setAttribute('aria-label', skill.label)
      skillButton.title = skill.description
      skillHotkeyLabels[index].textContent = skill.hotkey
      skillNameLabels[index].textContent = skill.label
      skillDescriptionLabels[index].textContent = skill.description
    }
  }

  const handleBagButtonClick = (event: MouseEvent) => {
    event.preventDefault()
    event.stopPropagation()

    onRequestInventoryOpenChange(!getIsInventoryOpen())
  }

  bagButton.addEventListener('click', handleBagButtonClick)

  const destroy = () => {
    bagButton.removeEventListener('click', handleBagButtonClick)
    overlayRoot.remove()
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

const createBagIconSvg = (): SVGSVGElement => {
  const namespace = 'http://www.w3.org/2000/svg'
  const svg = document.createElementNS(namespace, 'svg')
  const body = document.createElementNS(namespace, 'path')
  const tie = document.createElementNS(namespace, 'path')
  const handle = document.createElementNS(namespace, 'path')

  svg.setAttribute('viewBox', '0 0 24 24')
  svg.setAttribute('aria-hidden', 'true')
  svg.setAttribute('focusable', 'false')

  body.setAttribute('d', 'M7 8h10l2 4v8H5v-8l2-4z')
  body.setAttribute('fill', 'currentColor')

  tie.setAttribute('d', 'M9 7c0-1.7 1.3-3 3-3s3 1.3 3 3h-2c0-.6-.4-1-1-1s-1 .4-1 1H9z')
  tie.setAttribute('fill', 'currentColor')

  handle.setAttribute('d', 'M8 8h8v2H8z')
  handle.setAttribute('fill', '#f4e7c5')
  handle.setAttribute('opacity', '0.55')

  svg.append(body, tie, handle)

  return svg
}

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value))
