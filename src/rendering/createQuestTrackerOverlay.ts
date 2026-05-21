import type { FirstSlimeHuntQuestState } from '../game/firstSlimeHuntQuest'
import { getFirstSlimeHuntTrackerText } from '../game/firstSlimeHuntQuest'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreateQuestTrackerOverlayInput = {
  mountElement: HTMLElement
  getQuest: () => FirstSlimeHuntQuestState
}

export type QuestTrackerOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const OVERLAY_MARGIN = 16

export const createQuestTrackerOverlay = ({
  mountElement,
  getQuest
}: CreateQuestTrackerOverlayInput): QuestTrackerOverlay => {
  const overlayRoot = document.createElement('div')
  const panel = document.createElement('div')
  const title = document.createElement('div')
  const objective = document.createElement('div')

  overlayRoot.className = 'quest-tracker-overlay'
  overlayRoot.setAttribute('aria-hidden', 'true')
  panel.className = 'quest-tracker-overlay__panel'
  title.className = 'quest-tracker-overlay__title'
  title.textContent = '퀘스트'
  objective.className = 'quest-tracker-overlay__objective'

  panel.append(title, objective)
  overlayRoot.append(panel)
  mountElement.append(overlayRoot)

  const syncFrame = () => {
    const trackerText = getFirstSlimeHuntTrackerText(getQuest())
    const isVisible = trackerText !== undefined
    const uiScale = getResponsiveUiScale()

    overlayRoot.hidden = !isVisible
    overlayRoot.style.display = isVisible ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isVisible))

    if (!trackerText) {
      objective.textContent = ''
      return
    }

    objective.textContent = trackerText
    panel.style.right = `${Math.round(OVERLAY_MARGIN * uiScale)}px`
    panel.style.top = `${Math.round(OVERLAY_MARGIN * uiScale)}px`
    panel.style.transformOrigin = 'top right'
    panel.style.transform = `scale(${uiScale})`
  }

  syncFrame()

  return {
    syncFrame,
    destroy: () => {
      overlayRoot.remove()
    }
  }
}
