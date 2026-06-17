import {
  type QuestLogState
} from '../questLog'
import {
  getVisibleQuestTrackers,
  hideVisibleQuestTrackers
} from '../lua/luaGameLogic'
import { getResponsiveUiScale } from './getResponsiveUiScale'
import { createQuestTargetLink } from './questObjectiveTargetView'

type CreateQuestTrackerOverlayInput = {
  mountElement: HTMLElement
  getQuestLog: () => QuestLogState
  onQuestLogChange: (nextQuestLog: QuestLogState) => void
}

export type QuestTrackerOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const OVERLAY_MARGIN = 16

export const createQuestTrackerOverlay = ({
  mountElement,
  getQuestLog,
  onQuestLogChange
}: CreateQuestTrackerOverlayInput): QuestTrackerOverlay => {
  const overlayRoot = document.createElement('div')
  const panel = document.createElement('div')
  const header = document.createElement('div')
  const title = document.createElement('div')
  const closeButton = document.createElement('button')
  const closeIcon = document.createElement('span')
  const trackerList = document.createElement('div')
  let previousRenderSignature = ''

  overlayRoot.className = 'quest-tracker-overlay'
  overlayRoot.setAttribute('aria-hidden', 'true')
  panel.className = 'quest-tracker-overlay__panel'
  header.className = 'quest-tracker-overlay__header'
  title.className = 'quest-tracker-overlay__title'
  title.textContent = '퀘스트 알림 창'
  closeButton.type = 'button'
  closeButton.className = 'quest-tracker-overlay__close'
  closeButton.setAttribute('aria-label', '퀘스트 알림 창 닫기')
  closeButton.title = '퀘스트 알림 창 닫기'
  closeIcon.className = 'quest-tracker-overlay__close-icon'
  closeIcon.textContent = '×'
  closeIcon.setAttribute('aria-hidden', 'true')
  closeButton.append(closeIcon)
  trackerList.className = 'quest-tracker-overlay__list'

  header.append(title, closeButton)
  panel.append(header, trackerList)
  overlayRoot.append(panel)
  mountElement.append(overlayRoot)

  const syncFrame = () => {
    const trackerItems = getVisibleQuestTrackers(getQuestLog())
    const isVisible = trackerItems.length > 0
    const uiScale = getResponsiveUiScale()
    const renderSignature = isVisible
      ? JSON.stringify({
          uiScale,
          items: trackerItems.map((item) => item.text)
        })
      : 'hidden'

    if (previousRenderSignature === renderSignature) {
      return
    }

    previousRenderSignature = renderSignature

    overlayRoot.hidden = !isVisible
    overlayRoot.style.display = isVisible ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isVisible))

    if (!isVisible) {
      trackerList.replaceChildren()
      return
    }

    trackerList.replaceChildren(
      ...trackerItems.map((item) => {
        const objective = document.createElement('div')

        objective.className = 'quest-tracker-overlay__objective'
        objective.append(document.createTextNode(item.text))
        // 활성 목표의 대상(몬스터/아이템)을 파란 밑줄 링크로 — 클릭 시 이미지 팝업.
        const targetLink = item.objective
          ? createQuestTargetLink(item.objective)
          : undefined
        if (targetLink) {
          objective.append(document.createTextNode(' '), targetLink)
        }
        return objective
      })
    )
    panel.style.right = `${Math.round(OVERLAY_MARGIN * uiScale)}px`
    panel.style.top = `${Math.round(OVERLAY_MARGIN * uiScale)}px`
    panel.style.transformOrigin = 'top right'
    panel.style.transform = `scale(${uiScale})`
  }

  closeButton.addEventListener('click', (event) => {
    event.preventDefault()
    event.stopPropagation()
    onQuestLogChange(hideVisibleQuestTrackers(getQuestLog()))
  })

  syncFrame()

  return {
    syncFrame,
    destroy: () => {
      overlayRoot.remove()
    }
  }
}
