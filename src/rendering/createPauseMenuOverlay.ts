import { getResponsiveUiScale } from './getResponsiveUiScale'

export type AudioSettings = {
  bgmVolume: number
  sfxVolume: number
}

type CreatePauseMenuOverlayInput = {
  mountElement: HTMLElement
  getIsOpen: () => boolean
  getAudioSettings: () => AudioSettings
  onRequestOpenChange: (isOpen: boolean) => void
  onAudioSettingsChange: (nextAudioSettings: AudioSettings) => void
}

export type PauseMenuOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const OVERLAY_MARGIN = 16
const PANEL_WIDTH = 360
const PANEL_HEIGHT = 300

export const createPauseMenuOverlay = ({
  mountElement,
  getIsOpen,
  getAudioSettings,
  onRequestOpenChange,
  onAudioSettingsChange
}: CreatePauseMenuOverlayInput): PauseMenuOverlay => {
  const overlayRoot = document.createElement('div')
  const backdropButton = document.createElement('button')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const titleElement = document.createElement('div')
  const resumeButton = document.createElement('button')
  const bgmRow = createVolumeRow('BGM')
  const sfxRow = createVolumeRow('효과음')

  overlayRoot.className = 'pause-menu-overlay'
  overlayRoot.setAttribute('aria-hidden', 'true')

  backdropButton.type = 'button'
  backdropButton.className = 'pause-menu-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.setAttribute('aria-label', '메뉴 닫기')

  panel.className = 'pause-menu-overlay__panel'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'true')
  panel.setAttribute('aria-labelledby', 'pause-menu-title')

  panelBody.className = 'pause-menu-overlay__panel-body'
  titleElement.id = 'pause-menu-title'
  titleElement.className = 'pause-menu-overlay__title'
  titleElement.textContent = '일시정지'

  resumeButton.type = 'button'
  resumeButton.className = 'pause-menu-overlay__resume'
  resumeButton.textContent = '계속하기'

  panelBody.append(titleElement, bgmRow.row, sfxRow.row, resumeButton)
  panel.append(panelBody)
  overlayRoot.append(backdropButton, panel)
  mountElement.append(overlayRoot)

  const syncRow = (
    row: VolumeRow,
    value: number
  ) => {
    const percentage = Math.round(value * 100)

    row.input.value = String(percentage)
    row.value.textContent = `${percentage}%`
  }

  const syncFrame = () => {
    const isOpen = getIsOpen()
    const audioSettings = getAudioSettings()
    const uiScale = getResponsiveUiScale()
    const availableWidth = Math.max(1, window.innerWidth - OVERLAY_MARGIN * 2)
    const availableHeight = Math.max(1, window.innerHeight - OVERLAY_MARGIN * 2)

    overlayRoot.hidden = !isOpen
    overlayRoot.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      backdropButton.hidden = true
      panel.hidden = true
      return
    }

    backdropButton.hidden = false
    panel.hidden = false
    panel.style.width = `${Math.min(PANEL_WIDTH, availableWidth)}px`
    panel.style.height = `${Math.min(PANEL_HEIGHT, availableHeight)}px`
    panel.style.transformOrigin = 'center center'
    panel.style.transform = `translate(-50%, -50%) scale(${uiScale})`
    syncRow(bgmRow, audioSettings.bgmVolume)
    syncRow(sfxRow, audioSettings.sfxVolume)
  }

  const handleResumeClick = (event: MouseEvent) => {
    event.preventDefault()
    onRequestOpenChange(false)
  }

  const handleBackdropClick = (event: MouseEvent) => {
    event.preventDefault()
    onRequestOpenChange(false)
  }

  const handleVolumeInput = (
    setting: keyof AudioSettings,
    input: HTMLInputElement
  ) => {
    const audioSettings = getAudioSettings()
    const nextVolume = clamp(Number(input.value) / 100, 0, 1)

    onAudioSettingsChange({
      ...audioSettings,
      [setting]: nextVolume
    })
    syncFrame()
  }
  const handleBgmVolumeInput = () => {
    handleVolumeInput('bgmVolume', bgmRow.input)
  }
  const handleSfxVolumeInput = () => {
    handleVolumeInput('sfxVolume', sfxRow.input)
  }

  resumeButton.addEventListener('click', handleResumeClick)
  backdropButton.addEventListener('click', handleBackdropClick)
  bgmRow.input.addEventListener('input', handleBgmVolumeInput)
  sfxRow.input.addEventListener('input', handleSfxVolumeInput)

  syncFrame()

  return {
    syncFrame,
    destroy: () => {
      resumeButton.removeEventListener('click', handleResumeClick)
      backdropButton.removeEventListener('click', handleBackdropClick)
      bgmRow.input.removeEventListener('input', handleBgmVolumeInput)
      sfxRow.input.removeEventListener('input', handleSfxVolumeInput)
      overlayRoot.remove()
    }
  }
}

type VolumeRow = {
  row: HTMLLabelElement
  input: HTMLInputElement
  value: HTMLSpanElement
}

const createVolumeRow = (label: string): VolumeRow => {
  const row = document.createElement('label')
  const name = document.createElement('span')
  const input = document.createElement('input')
  const value = document.createElement('span')

  row.className = 'pause-menu-overlay__volume-row'
  name.className = 'pause-menu-overlay__volume-label'
  name.textContent = label
  input.className = 'pause-menu-overlay__volume-input'
  input.type = 'range'
  input.min = '0'
  input.max = '100'
  input.step = '1'
  value.className = 'pause-menu-overlay__volume-value'

  row.append(name, input, value)

  return {
    row,
    input,
    value
  }
}

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(value, max))
