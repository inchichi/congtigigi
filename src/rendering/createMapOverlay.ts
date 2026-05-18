import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreateMapOverlayInput = {
  mountElement: HTMLElement
  sourceCanvas: HTMLCanvasElement
  mapPixelWidth: number
  mapPixelHeight: number
  sceneScale: number
  getFocusPoint: () => {
    x: number
    y: number
  }
}

export type MapOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const COLLAPSED_MAX_SIZE = 220
const COLLAPSED_FOCUS_WORLD_SIZE = 640
const OVERLAY_MARGIN = 16
const DISPLAY_BORDER_SHADOW = '0 0 0 2px rgba(244, 231, 197, 0.92), 0 16px 32px rgba(0, 0, 0, 0.45)'
const FOCUS_MARKER_SIZE = 14

export const createMapOverlay = ({
  mountElement,
  sourceCanvas,
  mapPixelWidth,
  mapPixelHeight,
  sceneScale,
  getFocusPoint
}: CreateMapOverlayInput): MapOverlay => {
  const overlayRoot = document.createElement('div')
  const backdropButton = document.createElement('button')
  const panelButton = document.createElement('button')
  const previewCanvas = document.createElement('canvas')
  const viewportFrame = document.createElement('div')
  const badgeElement = document.createElement('div')
  const previewContext = previewCanvas.getContext('2d')

  if (!previewContext) {
    throw new Error('Missing 2D context for the map overlay')
  }

  let isExpanded = false
  let displayWidth = 0
  let displayHeight = 0
  let backingWidth = 0
  let backingHeight = 0

  overlayRoot.className = 'world-map-overlay'
  overlayRoot.setAttribute('aria-hidden', 'false')

  backdropButton.type = 'button'
  backdropButton.className = 'world-map-overlay__backdrop'
  backdropButton.hidden = true
  backdropButton.setAttribute('aria-label', 'Close the world map')
  backdropButton.tabIndex = -1

  panelButton.type = 'button'
  panelButton.className = 'world-map-overlay__panel'
  panelButton.setAttribute('aria-label', 'Open the zoomed world map')
  panelButton.setAttribute('aria-expanded', 'false')

  previewCanvas.className = 'world-map-overlay__canvas'
  viewportFrame.className = 'world-map-overlay__viewport'
  badgeElement.className = 'world-map-overlay__badge'
  badgeElement.textContent = 'ZOOM MAP'

  previewCanvas.setAttribute('aria-hidden', 'true')
  viewportFrame.setAttribute('aria-hidden', 'true')
  badgeElement.setAttribute('aria-hidden', 'true')

  panelButton.append(previewCanvas, viewportFrame, badgeElement)
  overlayRoot.append(backdropButton, panelButton)
  mountElement.append(overlayRoot)

  const syncFrame = () => {
    if (displayWidth === 0 || displayHeight === 0) {
      return
    }

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

      const scaleX = displayWidth / (mapPixelWidth * sceneScale)
      const scaleY = displayHeight / (mapPixelHeight * sceneScale)
      const viewportWidth = Math.max(
        1,
        Math.min(displayWidth, Math.round(mountElement.clientWidth * scaleX))
      )
      const viewportHeight = Math.max(
        1,
        Math.min(displayHeight, Math.round(mountElement.clientHeight * scaleY))
      )
      const maxViewportLeft = Math.max(0, displayWidth - viewportWidth)
      const maxViewportTop = Math.max(0, displayHeight - viewportHeight)
      const viewportLeft = Math.min(
        maxViewportLeft,
        Math.max(0, Math.round(mountElement.scrollLeft * scaleX))
      )
      const viewportTop = Math.min(
        maxViewportTop,
        Math.max(0, Math.round(mountElement.scrollTop * scaleY))
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
      badgeElement.textContent = 'WORLD MAP'
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
    badgeElement.textContent = 'ZOOM MAP'
  }

  const syncLayout = () => {
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
      ? `translate(-50%, -50%) scale(${uiScale})`
      : `scale(${uiScale})`
    panelButton.style.width = `${displayWidth}px`
    panelButton.style.height = `${displayHeight}px`
    panelButton.style.cursor = isExpanded ? 'zoom-out' : 'zoom-in'
    panelButton.style.boxShadow = DISPLAY_BORDER_SHADOW
    panelButton.setAttribute(
      'aria-label',
      isExpanded ? 'Close the world map' : 'Open the zoomed world map'
    )
    panelButton.setAttribute('aria-expanded', String(isExpanded))
    panelButton.title = isExpanded
      ? 'Click to close the world map'
      : 'Click to open the zoomed world map'
    backdropButton.hidden = !isExpanded

    previewCanvas.width = backingWidth
    previewCanvas.height = backingHeight
    previewCanvas.style.width = '100%'
    previewCanvas.style.height = '100%'

    syncFrame()
  }

  const setExpanded = (nextExpanded: boolean) => {
    if (isExpanded === nextExpanded) {
      return
    }

    isExpanded = nextExpanded
    syncLayout()
  }

  const handlePanelClick = (event: MouseEvent) => {
    event.preventDefault()
    setExpanded(!isExpanded)
  }

  const handleBackdropClick = (event: MouseEvent) => {
    event.preventDefault()
    setExpanded(false)
  }

  const handleWindowKeyDown = (event: KeyboardEvent) => {
    if (!isExpanded || event.key !== 'Escape') {
      return
    }

    event.preventDefault()
    setExpanded(false)
  }

  const handleWindowResize = () => {
    syncLayout()
  }

  panelButton.addEventListener('click', handlePanelClick)
  backdropButton.addEventListener('click', handleBackdropClick)
  window.addEventListener('keydown', handleWindowKeyDown)
  window.addEventListener('resize', handleWindowResize)

  syncLayout()

  return {
    syncFrame,
    destroy: () => {
      panelButton.removeEventListener('click', handlePanelClick)
      backdropButton.removeEventListener('click', handleBackdropClick)
      window.removeEventListener('keydown', handleWindowKeyDown)
      window.removeEventListener('resize', handleWindowResize)
      overlayRoot.remove()
    }
  }
}

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(value, max))
