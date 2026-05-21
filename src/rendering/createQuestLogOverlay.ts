import {
  QUEST_DEFINITIONS,
  abandonQuest,
  getQuestDefinition,
  getQuestProgress,
  setQuestTrackerVisible,
  type QuestDefinition,
  type QuestLogState,
  type QuestProgress
} from '../game/questLog'
import { getResponsiveUiScale } from './getResponsiveUiScale'

type CreateQuestLogOverlayInput = {
  mountElement: HTMLElement
  getIsOpen: () => boolean
  getQuestLog: () => QuestLogState
  onRequestOpenChange: (isOpen: boolean) => void
  onQuestLogChange: (nextQuestLog: QuestLogState) => void
}

export type QuestLogOverlay = {
  syncFrame: () => void
  destroy: () => void
}

const TINY_DUNGEON_TILESET_IMAGE_URL = new URL(
  '../assets/tilesets/tiny-dungeon-16.png',
  import.meta.url
).href
const TINY_DUNGEON_TILESET_WIDTH = 192
const TINY_DUNGEON_TILESET_HEIGHT = 176
const WIZARD_PORTRAIT_FRAME = {
  x: 0,
  y: 112,
  width: 16,
  height: 16
}
const PORTRAIT_SCALE = 4

export const createQuestLogOverlay = ({
  mountElement,
  getIsOpen,
  getQuestLog,
  onRequestOpenChange,
  onQuestLogChange
}: CreateQuestLogOverlayInput): QuestLogOverlay => {
  const overlayRoot = document.createElement('div')
  const panel = document.createElement('div')
  const panelBody = document.createElement('div')
  const header = document.createElement('div')
  const titleGroup = document.createElement('div')
  const title = document.createElement('div')
  const summary = document.createElement('div')
  const closeButton = document.createElement('button')
  const closeIcon = document.createElement('span')
  const regionPane = document.createElement('div')
  const tabBar = document.createElement('div')
  const regionTab = document.createElement('button')
  const questList = document.createElement('div')
  const emptyState = document.createElement('div')
  const detailPane = document.createElement('div')
  const detailEmptyState = document.createElement('div')
  const detailHeader = document.createElement('div')
  const portrait = document.createElement('div')
  const detailTitleGroup = document.createElement('div')
  const detailTitle = document.createElement('div')
  const detailMeta = document.createElement('div')
  const requestText = document.createElement('p')
  const guideText = document.createElement('p')
  const objectiveList = document.createElement('div')
  const actionRow = document.createElement('div')
  const trackerToggleButton = document.createElement('button')
  const abandonButton = document.createElement('button')
  const confirmPanel = document.createElement('div')
  const confirmText = document.createElement('div')
  const confirmActionRow = document.createElement('div')
  const confirmAbandonButton = document.createElement('button')
  const cancelAbandonButton = document.createElement('button')
  let selectedQuestId: string | undefined
  let isAbandonConfirmVisible = false

  overlayRoot.className = 'quest-log-overlay'
  panel.className = 'quest-log-overlay__panel'
  panel.hidden = true
  panel.setAttribute('role', 'dialog')
  panel.setAttribute('aria-modal', 'false')
  panel.setAttribute('aria-labelledby', 'quest-log-overlay-title')
  panelBody.className = 'quest-log-overlay__panel-body'
  header.className = 'quest-log-overlay__header'
  titleGroup.className = 'quest-log-overlay__title-group'
  title.id = 'quest-log-overlay-title'
  title.className = 'quest-log-overlay__title'
  title.textContent = '퀘스트 창'
  summary.className = 'quest-log-overlay__summary'

  closeButton.type = 'button'
  closeButton.className = 'quest-log-overlay__close'
  closeButton.setAttribute('aria-label', '퀘스트 창 닫기')
  closeButton.title = '퀘스트 창 닫기 (Esc)'
  closeIcon.className = 'quest-log-overlay__close-icon'
  closeIcon.textContent = '×'
  closeIcon.setAttribute('aria-hidden', 'true')
  closeButton.append(closeIcon)

  regionPane.className = 'quest-log-overlay__region-pane'
  tabBar.className = 'quest-log-overlay__tab-bar'
  regionTab.type = 'button'
  regionTab.className = 'quest-log-overlay__tab quest-log-overlay__tab--active'
  regionTab.textContent = '티르코네일 마을'
  regionTab.setAttribute('aria-pressed', 'true')
  questList.className = 'quest-log-overlay__quest-list'
  emptyState.className = 'quest-log-overlay__empty-state'
  emptyState.textContent = '진행 중인 퀘스트가 없습니다'

  detailPane.className = 'quest-log-overlay__detail-pane'
  detailEmptyState.className = 'quest-log-overlay__detail-empty'
  detailEmptyState.textContent = '왼쪽에서 퀘스트를 선택하세요'
  detailHeader.className = 'quest-log-overlay__detail-header'
  portrait.className = 'quest-log-overlay__portrait'
  portrait.setAttribute('aria-hidden', 'true')
  detailTitleGroup.className = 'quest-log-overlay__detail-title-group'
  detailTitle.className = 'quest-log-overlay__detail-title'
  detailMeta.className = 'quest-log-overlay__detail-meta'
  requestText.className = 'quest-log-overlay__description'
  guideText.className = 'quest-log-overlay__description'
  objectiveList.className = 'quest-log-overlay__objective-list'
  actionRow.className = 'quest-log-overlay__action-row'

  trackerToggleButton.type = 'button'
  trackerToggleButton.className = 'quest-log-overlay__tracker-toggle'
  trackerToggleButton.textContent = '퀘스트 알림'
  abandonButton.type = 'button'
  abandonButton.className = 'quest-log-overlay__abandon'
  abandonButton.textContent = '포기하기'

  confirmPanel.className = 'quest-log-overlay__confirm'
  confirmText.className = 'quest-log-overlay__confirm-text'
  confirmText.textContent = '정말 이 퀘스트를 포기할까요?'
  confirmActionRow.className = 'quest-log-overlay__confirm-actions'
  confirmAbandonButton.type = 'button'
  confirmAbandonButton.className = 'quest-log-overlay__confirm-abandon'
  confirmAbandonButton.textContent = '포기'
  cancelAbandonButton.type = 'button'
  cancelAbandonButton.className = 'quest-log-overlay__confirm-cancel'
  cancelAbandonButton.textContent = '취소'

  titleGroup.append(title, summary)
  header.append(titleGroup, closeButton)
  tabBar.append(regionTab)
  regionPane.append(tabBar, questList, emptyState)
  detailTitleGroup.append(detailTitle, detailMeta)
  detailHeader.append(portrait, detailTitleGroup)
  actionRow.append(trackerToggleButton, abandonButton)
  confirmActionRow.append(confirmAbandonButton, cancelAbandonButton)
  confirmPanel.append(confirmText, confirmActionRow)
  detailPane.append(
    detailEmptyState,
    detailHeader,
    requestText,
    guideText,
    objectiveList,
    actionRow,
    confirmPanel
  )
  panelBody.append(header, regionPane, detailPane)
  panel.append(panelBody)
  overlayRoot.append(panel)
  mountElement.append(overlayRoot)

  const syncFrame = () => {
    const isOpen = getIsOpen()

    panel.hidden = !isOpen
    panel.style.display = isOpen ? '' : 'none'
    overlayRoot.setAttribute('aria-hidden', String(!isOpen))

    if (!isOpen) {
      return
    }

    const questLog = getQuestLog()
    const visibleDefinitions = getVisibleQuestDefinitions(questLog)

    if (
      selectedQuestId === undefined ||
      !visibleDefinitions.some((definition) => definition.id === selectedQuestId)
    ) {
      selectedQuestId = visibleDefinitions[0]?.id
      isAbandonConfirmVisible = false
    }

    const selectedDefinition = selectedQuestId
      ? getQuestDefinition(selectedQuestId)
      : undefined
    const selectedQuest = selectedQuestId
      ? getQuestProgress(questLog, selectedQuestId)
      : undefined
    const uiScale = getResponsiveUiScale()

    panel.style.transform = `translate(-50%, -50%) scale(${uiScale})`
    summary.textContent = `${visibleDefinitions.length}개 진행 중`
    renderQuestList(visibleDefinitions)
    renderQuestDetail(selectedDefinition, selectedQuest)
  }

  const renderQuestList = (definitions: QuestDefinition[]) => {
    questList.replaceChildren()
    emptyState.hidden = definitions.length > 0
    emptyState.style.display = definitions.length > 0 ? 'none' : ''

    for (const definition of definitions) {
      const questButton = document.createElement('button')
      const isSelected = definition.id === selectedQuestId

      questButton.type = 'button'
      questButton.className = 'quest-log-overlay__quest-row'
      questButton.classList.toggle('quest-log-overlay__quest-row--active', isSelected)
      questButton.textContent = `[${definition.giverName}] ${definition.title}`
      questButton.setAttribute('aria-pressed', String(isSelected))
      questButton.addEventListener('click', (event) => {
        event.preventDefault()
        event.stopPropagation()
        selectedQuestId = definition.id
        isAbandonConfirmVisible = false
        syncFrame()
      })

      questList.append(questButton)
    }
  }

  const renderQuestDetail = (
    definition: QuestDefinition | undefined,
    quest: QuestProgress | undefined
  ) => {
    const hasQuest = definition !== undefined && quest !== undefined

    detailPane.classList.toggle('quest-log-overlay__detail-pane--empty', !hasQuest)
    detailEmptyState.hidden = hasQuest
    detailEmptyState.style.display = hasQuest ? 'none' : ''
    detailHeader.hidden = !hasQuest
    detailHeader.style.display = hasQuest ? '' : 'none'
    requestText.hidden = !hasQuest
    requestText.style.display = hasQuest ? '' : 'none'
    guideText.hidden = !hasQuest
    guideText.style.display = hasQuest ? '' : 'none'
    objectiveList.hidden = !hasQuest
    objectiveList.style.display = hasQuest ? '' : 'none'
    actionRow.hidden = !hasQuest
    actionRow.style.display = hasQuest ? '' : 'none'
    confirmPanel.hidden = !hasQuest || !isAbandonConfirmVisible
    confirmPanel.style.display = hasQuest && isAbandonConfirmVisible ? '' : 'none'

    if (!hasQuest) {
      detailTitle.textContent = '선택된 퀘스트가 없습니다'
      detailMeta.textContent = ''
      requestText.textContent = ''
      guideText.textContent = ''
      objectiveList.replaceChildren()
      return
    }

    setPortraitFrame(portrait)
    detailTitle.textContent = definition.title
    detailMeta.textContent = `[${definition.giverName}] ${definition.regionName}`
    requestText.textContent = definition.requestText
    guideText.textContent = definition.guideText
    trackerToggleButton.classList.toggle(
      'quest-log-overlay__tracker-toggle--active',
      quest.trackerVisible
    )
    trackerToggleButton.setAttribute('aria-pressed', String(quest.trackerVisible))
    objectiveList.replaceChildren(
      ...definition.objectives.map((objective) => {
        const objectiveRow = document.createElement('div')

        objectiveRow.className = 'quest-log-overlay__objective-row'
        objectiveRow.textContent = `${objective.label}: ${quest.objectives[objective.id] ?? 0}/${objective.required}`
        return objectiveRow
      })
    )
  }

  trackerToggleButton.addEventListener('click', (event) => {
    event.preventDefault()
    event.stopPropagation()

    if (!selectedQuestId) {
      return
    }

    const quest = getQuestProgress(getQuestLog(), selectedQuestId)

    onQuestLogChange(
      setQuestTrackerVisible(getQuestLog(), selectedQuestId, !quest.trackerVisible)
    )
    isAbandonConfirmVisible = false
    syncFrame()
  })

  abandonButton.addEventListener('click', (event) => {
    event.preventDefault()
    event.stopPropagation()
    isAbandonConfirmVisible = true
    syncFrame()
  })

  confirmAbandonButton.addEventListener('click', (event) => {
    event.preventDefault()
    event.stopPropagation()

    if (!selectedQuestId) {
      return
    }

    onQuestLogChange(abandonQuest(getQuestLog(), selectedQuestId))
    selectedQuestId = undefined
    isAbandonConfirmVisible = false
    syncFrame()
  })

  cancelAbandonButton.addEventListener('click', (event) => {
    event.preventDefault()
    event.stopPropagation()
    isAbandonConfirmVisible = false
    syncFrame()
  })

  closeButton.addEventListener('click', (event) => {
    event.preventDefault()
    event.stopPropagation()
    onRequestOpenChange(false)
  })

  syncFrame()

  return {
    syncFrame,
    destroy: () => {
      overlayRoot.remove()
    }
  }
}

const getVisibleQuestDefinitions = (
  questLog: QuestLogState
): QuestDefinition[] =>
  QUEST_DEFINITIONS.filter((definition) => {
    const quest = getQuestProgress(questLog, definition.id)

    return quest.status === 'active' || quest.status === 'ready-to-turn-in'
  })

const setPortraitFrame = (element: HTMLElement) => {
  element.style.backgroundImage = `url(${TINY_DUNGEON_TILESET_IMAGE_URL})`
  element.style.backgroundRepeat = 'no-repeat'
  element.style.backgroundPosition = `-${WIZARD_PORTRAIT_FRAME.x * PORTRAIT_SCALE}px -${WIZARD_PORTRAIT_FRAME.y * PORTRAIT_SCALE}px`
  element.style.backgroundSize = `${TINY_DUNGEON_TILESET_WIDTH * PORTRAIT_SCALE}px ${TINY_DUNGEON_TILESET_HEIGHT * PORTRAIT_SCALE}px`
  element.style.width = `${WIZARD_PORTRAIT_FRAME.width * PORTRAIT_SCALE}px`
  element.style.height = `${WIZARD_PORTRAIT_FRAME.height * PORTRAIT_SCALE}px`
}
