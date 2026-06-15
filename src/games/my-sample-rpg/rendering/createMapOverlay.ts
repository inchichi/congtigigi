import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreateMapOverlayInput = {
  mountElement: HTMLElement
  cameraElement: HTMLElement
  sourceCanvas: HTMLCanvasElement
  mapPixelWidth: number
  mapPixelHeight: number
  getSceneScale: () => number
  getFocusPoint: () => {
    x: number
    y: number
  }
  onExpandedChange?: (isExpanded: boolean) => void
}

export type MapOverlay = {
  syncFrame: () => void
  getIsExpanded: () => boolean
  getIsVisible: () => boolean
  setExpanded: (isExpanded: boolean) => void
  setVisible: (isVisible: boolean) => void
  toggleExpanded: () => void
  toggleVisible: () => void
  destroy: () => void
}

const COLLAPSED_BASE_MAX_SIZE = 220
const COLLAPSED_SIZE_SCALE = 0.8
const COLLAPSED_MAX_SIZE = Math.round(COLLAPSED_BASE_MAX_SIZE * COLLAPSED_SIZE_SCALE)
const COLLAPSED_FOCUS_WORLD_SIZE = 640
const OVERLAY_MARGIN = 16
const DISPLAY_BORDER_SHADOW = '0 0 0 2px rgba(244, 231, 197, 0.92), 0 16px 32px rgba(0, 0, 0, 0.45)'
const FOCUS_MARKER_SIZE = 14

export const createMapOverlay = ({
  mountElement,
  cameraElement,
  sourceCanvas,
  mapPixelWidth,
  mapPixelHeight,
  getSceneScale,
  getFocusPoint,
  onExpandedChange
}: CreateMapOverlayInput): MapOverlay => {
  const overlayRoot = document.createElement('div')
  const backdropButton = document.createElement('button')
  const panelButton = document.createElement('button')
  const mapFrame = document.createElement('div')
  const previewCanvas = document.createElement('canvas')
  const viewportFrame = document.createElement('div')
  const badgeElement = document.createElement('div')
  const previewContext = previewCanvas.getContext('2d')

  if (!previewContext) {
    throw new Error('Missing 2D context for the map overlay')
  }

  let isExpanded = false
  let isVisible = true
  let displayWidth = 0
  let displayHeight = 0
  let backingWidth = 0
  let backingHeight = 0

  const syncMapFocusMode = () => {
    mountElement.classList.toggle('game-root--map-focused', isExpanded)
    document.body.classList.toggle('game-root--map-focused', isExpanded)
  }

  overlayRoot.className = 'world-map-overlay'
  overlayRoot.setAttribute('aria-hidden', 'false')

  backdropButton.type = 'button'
  backdropButton.className = 'world-map-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.setAttribute('aria-label', '월드맵 닫기')
  backdropButton.tabIndex = -1

  panelButton.type = 'button'
  panelButton.className = 'world-map-overlay__panel'
  panelButton.setAttribute('aria-label', '월드맵 열기')
  panelButton.setAttribute('aria-expanded', 'false')

  mapFrame.className = 'world-map-overlay__frame'
  mapFrame.setAttribute('aria-hidden', 'true')

  previewCanvas.className = 'world-map-overlay__canvas'
  viewportFrame.className = 'world-map-overlay__viewport'
  badgeElement.className = 'world-map-overlay__badge'
  badgeElement.textContent = '월드맵'

  previewCanvas.setAttribute('aria-hidden', 'true')
  viewportFrame.setAttribute('aria-hidden', 'true')
  badgeElement.setAttribute('aria-hidden', 'true')

  mapFrame.append(previewCanvas, viewportFrame)
  panelButton.append(mapFrame, badgeElement)
  overlayRoot.append(backdropButton, panelButton)
  mountElement.append(overlayRoot)

  const shouldShowOverlay = () => isVisible || isExpanded

  const syncFrame = () => {
    if (!shouldShowOverlay() || displayWidth === 0 || displayHeight === 0) {
      return
    }

    const uiScale = getResponsiveUiScale()
    const focusPoint = getFocusPoint()
    const sourceScaleX = sourceCanvas.width / mapPixelWidth
    const sourceScaleY = sourceCanvas.height / mapPixelHeight
    previewContext.setTransform(1, 0, 0, 1, 0, 0)
    previewContext.clearRect(0, 0, backingWidth, backingHeight)
    previewContext.imageSmoothingEnabled = false

    if (isExpanded) {
      previewContext.drawImage(
        sourceCanvas,
        0,
        0,
        sourceCanvas.width,
        sourceCanvas.height,
        0,
        0,
        backingWidth,
        backingHeight
      )

      const sceneScale = getSceneScale()
      const scaleX = displayWidth / (mapPixelWidth * sceneScale)
      const scaleY = displayHeight / (mapPixelHeight * sceneScale)
      const viewportWidth = Math.max(
        1,
        Math.min(displayWidth, Math.round(cameraElement.clientWidth * scaleX))
      )
      const viewportHeight = Math.max(
        1,
        Math.min(displayHeight, Math.round(cameraElement.clientHeight * scaleY))
      )
      const maxViewportLeft = Math.max(0, displayWidth - viewportWidth)
      const maxViewportTop = Math.max(0, displayHeight - viewportHeight)
      const viewportLeft = Math.min(
        maxViewportLeft,
        Math.max(0, Math.round(cameraElement.scrollLeft * scaleX))
      )
      const viewportTop = Math.min(
        maxViewportTop,
        Math.max(0, Math.round(cameraElement.scrollTop * scaleY))
      )

      viewportFrame.style.left = `${viewportLeft}px`
      viewportFrame.style.top = `${viewportTop}px`
      viewportFrame.style.width = `${viewportWidth}px`
      viewportFrame.style.height = `${viewportHeight}px`
      viewportFrame.style.borderRadius = '0'
      viewportFrame.style.background = 'transparent'
      viewportFrame.style.border = '2px solid rgba(244, 231, 197, 0.95)'
      viewportFrame.style.boxShadow = '0 0 0 1px rgba(23, 19, 17, 0.78)'
      viewportFrame.style.transform = 'none'
      mapFrame.style.borderRadius = '0'
      mapFrame.style.background = 'rgba(16, 16, 14, 0.98)'
      badgeElement.textContent = '월드맵'
      badgeElement.style.left = '8px'
      badgeElement.style.top = '8px'
      badgeElement.style.transform = 'none'
      return
    }

    const focusWindowSize = Math.min(
      mapPixelWidth,
      mapPixelHeight,
      COLLAPSED_FOCUS_WORLD_SIZE
    )
    const sourceLeft = clamp(
      focusPoint.x - focusWindowSize / 2,
      0,
      mapPixelWidth - focusWindowSize
    )
    const sourceTop = clamp(
      focusPoint.y - focusWindowSize / 2,
      0,
      mapPixelHeight - focusWindowSize
    )

    previewContext.drawImage(
      sourceCanvas,
      sourceLeft * sourceScaleX,
      sourceTop * sourceScaleY,
      focusWindowSize * sourceScaleX,
      focusWindowSize * sourceScaleY,
      0,
      0,
      backingWidth,
      backingHeight
    )

    const focusMarkerLeft = clamp(
      Math.round(((focusPoint.x - sourceLeft) / focusWindowSize) * displayWidth),
      0,
      displayWidth
    )
    const focusMarkerTop = clamp(
      Math.round(((focusPoint.y - sourceTop) / focusWindowSize) * displayHeight),
      0,
      displayHeight
    )

    viewportFrame.style.left = `${Math.max(
      0,
      focusMarkerLeft - Math.round(FOCUS_MARKER_SIZE / 2)
    )}px`
    viewportFrame.style.top = `${Math.max(
      0,
      focusMarkerTop - Math.round(FOCUS_MARKER_SIZE / 2)
    )}px`
    viewportFrame.style.width = `${FOCUS_MARKER_SIZE}px`
    viewportFrame.style.height = `${FOCUS_MARKER_SIZE}px`
    viewportFrame.style.borderRadius = '999px'
    viewportFrame.style.background = 'rgba(244, 231, 197, 0.16)'
    viewportFrame.style.border = '2px solid rgba(244, 231, 197, 0.98)'
    viewportFrame.style.boxShadow = '0 0 0 1px rgba(23, 19, 17, 0.78)'
    viewportFrame.style.transform = 'none'
    mapFrame.style.borderRadius = '50%'
    mapFrame.style.background = '#10100e'
    badgeElement.textContent = '월드맵'
    badgeElement.style.left = '50%'
    badgeElement.style.top = `${displayHeight + Math.max(6, Math.round(8 * uiScale))}px`
    badgeElement.style.transform = 'translateX(-50%)'
  }

  const syncLayout = () => {
    const shouldShow = shouldShowOverlay()

    overlayRoot.hidden = !shouldShow
    overlayRoot.style.display = shouldShow ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!shouldShow))
    panelButton.hidden = !shouldShow

    if (!shouldShow) {
      backdropButton.hidden = true
      return
    }

    const uiScale = getResponsiveUiScale()
    const availableWidth = isExpanded
      ? Math.max(1, window.innerWidth - OVERLAY_MARGIN * 2)
      : Math.max(
          1,
          Math.min(COLLAPSED_MAX_SIZE, window.innerWidth - OVERLAY_MARGIN * 2)
        )
    const availableHeight = isExpanded
      ? Math.max(1, window.innerHeight - OVERLAY_MARGIN * 2)
      : Math.max(
          1,
          Math.min(COLLAPSED_MAX_SIZE, window.innerHeight - OVERLAY_MARGIN * 2)
        )
    const scale = isExpanded
      ? Math.min(availableWidth / mapPixelWidth, availableHeight / mapPixelHeight)
      : Math.min(1, Math.min(availableWidth, availableHeight) / COLLAPSED_MAX_SIZE)
    const nextDisplayWidth = isExpanded
      ? Math.max(1, Math.round(mapPixelWidth * scale))
      : Math.max(1, Math.round(COLLAPSED_MAX_SIZE * scale))
    const nextDisplayHeight = isExpanded
      ? Math.max(1, Math.round(mapPixelHeight * scale))
      : Math.max(1, Math.round(COLLAPSED_MAX_SIZE * scale))
    const nextBackingWidth = Math.max(
      1,
      Math.round(nextDisplayWidth * (window.devicePixelRatio || 1))
    )
    const nextBackingHeight = Math.max(
      1,
      Math.round(nextDisplayHeight * (window.devicePixelRatio || 1))
    )

    displayWidth = nextDisplayWidth
    displayHeight = nextDisplayHeight
    backingWidth = nextBackingWidth
    backingHeight = nextBackingHeight

    panelButton.classList.toggle(
      'world-map-overlay__panel--expanded',
      isExpanded
    )
    panelButton.classList.toggle(
      'world-map-overlay__panel--collapsed',
      !isExpanded
    )
    const scaledMargin = Math.round(OVERLAY_MARGIN * uiScale)
    panelButton.style.left = isExpanded ? '50%' : `${scaledMargin}px`
    panelButton.style.top = isExpanded ? '50%' : `${scaledMargin}px`
    panelButton.style.transformOrigin = isExpanded ? 'center center' : 'top left'
    panelButton.style.transform = isExpanded
      ? 'translate(-50%, -50%)'
      : `scale(${uiScale})`
    panelButton.style.width = `${displayWidth}px`
    panelButton.style.height = `${displayHeight}px`
    panelButton.style.cursor = isExpanded ? 'zoom-out' : 'zoom-in'
    panelButton.style.boxShadow = 'none'
    panelButton.style.background = 'transparent'
    panelButton.setAttribute(
      'aria-label',
      isExpanded ? '월드맵 닫기' : '월드맵 열기'
    )
    panelButton.setAttribute('aria-expanded', String(isExpanded))
    panelButton.title = isExpanded
      ? '월드맵 닫기'
      : '월드맵 열기'
    backdropButton.hidden = !isExpanded

    previewCanvas.width = backingWidth
    previewCanvas.height = backingHeight
    previewCanvas.style.width = '100%'
    previewCanvas.style.height = '100%'
    mapFrame.style.width = '100%'
    mapFrame.style.height = '100%'
    mapFrame.style.boxShadow = DISPLAY_BORDER_SHADOW

    syncFrame()
  }

  const setExpanded = (nextExpanded: boolean) => {
    if (isExpanded === nextExpanded) {
      return
    }

    isExpanded = nextExpanded
    syncMapFocusMode()
    syncLayout()
    onExpandedChange?.(isExpanded)
  }

  const setVisible = (nextVisible: boolean) => {
    if (isVisible === nextVisible) {
      return
    }

    isVisible = nextVisible

    if (!isVisible && isExpanded) {
      isExpanded = false
      syncLayout()
      onExpandedChange?.(isExpanded)
      return
    }

    syncLayout()
  }

  const toggleExpanded = () => {
    setExpanded(!isExpanded)
  }

  const toggleVisible = () => {
    setVisible(!isVisible)
  }

  const handlePanelClick = (event: MouseEvent) => {
    event.preventDefault()
    setExpanded(!isExpanded)
  }

  const handleBackdropClick = (event: MouseEvent) => {
    event.preventDefault()
    setExpanded(false)
  }

  const handleWindowResize = () => {
    syncLayout()
  }

  panelButton.addEventListener('click', handlePanelClick)
  backdropButton.addEventListener('click', handleBackdropClick)
  window.addEventListener('resize', handleWindowResize)

  syncMapFocusMode()
  syncLayout()

  return {
    syncFrame,
    getIsExpanded: () => isExpanded,
    getIsVisible: () => isVisible,
    setExpanded,
    setVisible,
    toggleExpanded,
    toggleVisible,
    destroy: () => {
      mountElement.classList.remove('game-root--map-focused')
      document.body.classList.remove('game-root--map-focused')
      panelButton.removeEventListener('click', handlePanelClick)
      backdropButton.removeEventListener('click', handleBackdropClick)
      window.removeEventListener('resize', handleWindowResize)
      overlayRoot.remove()
    }
  }
}

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(value, max))
