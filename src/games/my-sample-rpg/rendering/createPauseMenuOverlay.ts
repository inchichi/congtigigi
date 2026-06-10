import { getResponsiveUiScale } from './getResponsiveUiScale'
import {
  getPlayerControlBindingDefinitions,
  getPlayerControlBindingDisplayText,
  type PlayerControlBindingId,
  type PlayerControlBindings
} from '../playerControls'

export type AudioSettings = {
  bgmVolume: number
  sfxVolume: number
}

type CreatePauseMenuOverlayInput = {
  mountElement: HTMLElement
  getIsOpen: () => boolean
  getAudioSettings: () => AudioSettings
  getControlBindings: () => PlayerControlBindings
  getControlBindingCaptureTarget: () => PlayerControlBindingId | undefined
  onRequestOpenChange: (isOpen: boolean) => void
  onAudioSettingsChange: (nextAudioSettings: AudioSettings) => void
  onRequestControlBindingCapture: (
    bindingId: PlayerControlBindingId | undefined
  ) => void
  onRequestControlBindingsReset: () => void
}

export type PauseMenuOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const OVERLAY_MARGIN = 16
const PANEL_WIDTH = 420
const MAIN_PANEL_HEIGHT = 300
const CONTROLS_PANEL_HEIGHT = 420

export const createPauseMenuOverlay = ({
  mountElement,
  getIsOpen,
  getAudioSettings,
  getControlBindings,
  getControlBindingCaptureTarget,
  onRequestOpenChange,
  onAudioSettingsChange,
  onRequestControlBindingCapture,
  onRequestControlBindingsReset
}: CreatePauseMenuOverlayInput): PauseMenuOverlay => {
  type PauseMenuScreen = 'main' | 'controls'

  const overlayRoot = document.createElement('div')
  const backdropButton = document.createElement('button')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const titleElement = document.createElement('div')
  const screenHost = document.createElement('div')
  const mainScreen = document.createElement('section')
  const controlsScreen = document.createElement('section')
  const controlsButton = document.createElement('button')
  const keySection = document.createElement('section')
  const keySectionTitle = document.createElement('div')
  const keySectionHint = document.createElement('div')
  const keySectionActions = document.createElement('div')
  const keyBindingGrid = document.createElement('div')
  const resetButton = document.createElement('button')
  const resumeButton = document.createElement('button')
  const backButton = document.createElement('button')
  const bgmRow = createVolumeRow('BGM')
  const sfxRow = createVolumeRow('효과음')
  const bindingRows = getPlayerControlBindingDefinitions().map(
    (definition) => {
      const button = document.createElement('button')
      const label = document.createElement('span')
      const value = document.createElement('span')

      button.type = 'button'
      button.className = 'pause-menu-overlay__binding-button'
      button.setAttribute('aria-label', definition.label)

      label.className = 'pause-menu-overlay__binding-label'
      label.textContent = definition.label

      value.className = 'pause-menu-overlay__binding-value'

      button.append(label, value)
      keyBindingGrid.append(button)

      button.addEventListener('click', (event) => {
        event.preventDefault()
        onRequestControlBindingCapture(definition.id)
      })

      return {
        button,
        label,
        value,
        definition
      }
    }
  )
  let currentScreen: PauseMenuScreen = 'main'

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

  screenHost.className = 'pause-menu-overlay__screen-host'

  mainScreen.className = 'pause-menu-overlay__screen pause-menu-overlay__screen--main'
  controlsScreen.className =
    'pause-menu-overlay__screen pause-menu-overlay__screen--controls'

  controlsButton.type = 'button'
  controlsButton.className = 'pause-menu-overlay__resume'
  controlsButton.textContent = '단축키 설정'

  keySection.className = 'pause-menu-overlay__key-section'
  keySectionTitle.className = 'pause-menu-overlay__key-section-title'
  keySectionTitle.textContent = '현재 단축키'
  keySectionHint.className = 'pause-menu-overlay__key-section-hint'
  keySectionHint.textContent = '항목을 클릭하고 새 키를 누르세요.'
  keySectionActions.className = 'pause-menu-overlay__key-section-actions'
  keyBindingGrid.className = 'pause-menu-overlay__key-grid'

  resetButton.type = 'button'
  resetButton.className = 'pause-menu-overlay__resume'
  resetButton.textContent = '초기 설정으로'

  resumeButton.type = 'button'
  resumeButton.className = 'pause-menu-overlay__resume'
  resumeButton.textContent = '계속하기'

  backButton.type = 'button'
  backButton.className = 'pause-menu-overlay__resume'
  backButton.textContent = '뒤로가기'

  keySectionActions.append(resetButton, backButton)
  keySection.append(
    keySectionTitle,
    keySectionHint,
    keySectionActions,
    keyBindingGrid
  )
  mainScreen.append(controlsButton, bgmRow.row, sfxRow.row, resumeButton)
  controlsScreen.append(keySection)
  screenHost.append(mainScreen, controlsScreen)
  panelBody.append(titleElement, screenHost)
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
    const controlBindings = getControlBindings()
    const controlBindingCaptureTarget = getControlBindingCaptureTarget()
    const uiScale = getResponsiveUiScale()
    const availableWidth = Math.max(1, window.innerWidth - OVERLAY_MARGIN * 2)
    const availableHeight = Math.max(1, window.innerHeight - OVERLAY_MARGIN * 2)
    const panelHeight =
      currentScreen === 'controls'
        ? CONTROLS_PANEL_HEIGHT
        : MAIN_PANEL_HEIGHT

    overlayRoot.hidden = !isOpen
    overlayRoot.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      backdropButton.hidden = true
      panel.hidden = true
      currentScreen = 'main'
      return
    }

    backdropButton.hidden = false
    panel.hidden = false
    panel.style.width = `${Math.min(PANEL_WIDTH, availableWidth)}px`
    panel.style.height = `${Math.min(panelHeight, availableHeight)}px`
    panel.style.transformOrigin = 'center center'
    panel.style.transform = `translate(-50%, -50%) scale(${uiScale})`
    titleElement.textContent =
      currentScreen === 'controls' ? '단축키 설정' : '일시정지'
    mainScreen.hidden = currentScreen !== 'main'
    controlsScreen.hidden = currentScreen !== 'controls'
    for (const bindingRow of bindingRows) {
      const isListening =
        controlBindingCaptureTarget === bindingRow.definition.id

      bindingRow.button.classList.toggle(
        'pause-menu-overlay__binding-button--listening',
        isListening
      )
      bindingRow.button.setAttribute(
        'aria-label',
        isListening
          ? `${bindingRow.definition.label}, 새 키를 누르세요`
          : `${bindingRow.definition.label}, 현재 키 ${getPlayerControlBindingDisplayText(controlBindings[bindingRow.definition.id])}`
      )
      bindingRow.value.textContent = isListening
        ? '입력 대기'
        : getPlayerControlBindingDisplayText(
            controlBindings[bindingRow.definition.id]
          )
    }
    syncRow(bgmRow, audioSettings.bgmVolume)
    syncRow(sfxRow, audioSettings.sfxVolume)
  }

  const handleControlsClick = (event: MouseEvent) => {
    event.preventDefault()
    currentScreen = 'controls'
    syncFrame()
  }

  const handleResumeClick = (event: MouseEvent) => {
    event.preventDefault()
    onRequestOpenChange(false)
  }

  const handleBackClick = (event: MouseEvent) => {
    event.preventDefault()
    onRequestControlBindingCapture(undefined)
    currentScreen = 'main'
    syncFrame()
  }

  const handleResetClick = (event: MouseEvent) => {
    event.preventDefault()
    onRequestControlBindingsReset()
  }

  const handleBackdropClick = (event: MouseEvent) => {
    event.preventDefault()
    onRequestControlBindingCapture(undefined)
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

  controlsButton.addEventListener('click', handleControlsClick)
  resumeButton.addEventListener('click', handleResumeClick)
  resetButton.addEventListener('click', handleResetClick)
  backButton.addEventListener('click', handleBackClick)
  backdropButton.addEventListener('click', handleBackdropClick)
  bgmRow.input.addEventListener('input', handleBgmVolumeInput)
  sfxRow.input.addEventListener('input', handleSfxVolumeInput)

  syncFrame()

  return {
    syncFrame,
    destroy: () => {
      controlsButton.removeEventListener('click', handleControlsClick)
      resumeButton.removeEventListener('click', handleResumeClick)
      resetButton.removeEventListener('click', handleResetClick)
      backButton.removeEventListener('click', handleBackClick)
      backdropButton.removeEventListener('click', handleBackdropClick)
      bgmRow.input.removeEventListener('input', handleBgmVolumeInput)
      sfxRow.input.removeEventListener('input', handleSfxVolumeInput)
      for (const bindingRow of bindingRows) {
        bindingRow.button.remove()
      }
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
