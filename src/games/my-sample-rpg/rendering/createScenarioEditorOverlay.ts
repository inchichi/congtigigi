import {
  SCENARIO_EDITOR_SYSTEM_PROMPT,
  buildScenarioEditorUserPrompt,
  extractScenarioEditorResponseText,
  parseScenarioEditorResult,
  type ScenarioEditorNpcDialogueDraft,
  type ScenarioEditorQuestDraft,
  type ScenarioEditorResult
} from './scenarioEditorProtocol'

type CreateScenarioEditorOverlayInput = {
  mountElement: HTMLElement
}

export type ScenarioEditorOverlay = {
  syncFrame: () => void
  destroy: () => void
}

type StoredScenarioEditorState = {
  isOpen: boolean
  apiKey: string
  scenario: string
  rawResponse: string
}

type ScenarioEditorFieldElements = {
  field: HTMLLabelElement
  input: HTMLInputElement | HTMLTextAreaElement
}

const STORAGE_KEY = 'my-sample-rpg:scenario-editor'
const OVERLAY_MARGIN = 16
const PANEL_WIDTH = 520
const DEFAULT_LLM_ENDPOINT = 'https://api.openai.com/v1/responses'
const DEFAULT_LLM_MODEL = 'gpt-5.4-mini'
const AVAILABLE_NPC_NAMES = ['마법사', '대장장이', '물약상인'] as const
const DEFAULT_SCENARIO_TEXT = [
  '티르코네일 마을 외곽에서 밤마다 슬라임이 늘어난다.',
  '마을 사람들은 버려진 우물에서 푸른 빛이 샌다고 말한다.',
  '초보 모험가는 마법사, 대장장이, 물약상인을 도와 원인을 조사하고,',
  '마지막에는 동굴 입구의 슬라임 우두머리를 처치해 마을을 지켜야 한다.'
].join(' ')

export const createScenarioEditorOverlay = ({
  mountElement
}: CreateScenarioEditorOverlayInput): ScenarioEditorOverlay => {
  const storedState = readStoredState()
  const overlayRoot = document.createElement('div')
  const launcherButton = document.createElement('button')
  const launcherBadge = document.createElement('span')
  const launcherLabel = document.createElement('span')
  const panel = document.createElement('section')
  const panelBody = document.createElement('div')
  const header = document.createElement('header')
  const headerCopy = document.createElement('div')
  const eyebrow = document.createElement('div')
  const title = document.createElement('div')
  const subtitle = document.createElement('div')
  const headerActions = document.createElement('div')
  const statusPill = document.createElement('div')
  const closeButton = document.createElement('button')
  const closeIcon = document.createElement('span')
  const configGrid = document.createElement('div')
  const apiKeyField = createField({
    label: 'API 키',
    type: 'password',
    placeholder: 'sk-...',
    autocomplete: 'off'
  })
  const scenarioSection = document.createElement('section')
  const scenarioHeader = document.createElement('div')
  const scenarioTitle = document.createElement('div')
  const scenarioHint = document.createElement('div')
  const scenarioTextarea = document.createElement('textarea')
  const actionRow = document.createElement('div')
  const generateButton = document.createElement('button')
  const stopButton = document.createElement('button')
  const sampleButton = document.createElement('button')
  const clearButton = document.createElement('button')
  const resultSection = document.createElement('section')
  const resultHeader = document.createElement('div')
  const resultHeaderTitle = document.createElement('div')
  const resultHeaderMeta = document.createElement('div')
  const errorBanner = document.createElement('div')
  const resultSummaryCard = document.createElement('div')
  const resultSummaryLabel = document.createElement('div')
  const resultSummaryText = document.createElement('div')
  const resultGrid = document.createElement('div')
  const questColumn = document.createElement('section')
  const questColumnHeader = document.createElement('div')
  const questColumnTitle = document.createElement('div')
  const questColumnCount = document.createElement('div')
  const questList = document.createElement('div')
  const dialogueColumn = document.createElement('section')
  const dialogueColumnHeader = document.createElement('div')
  const dialogueColumnTitle = document.createElement('div')
  const dialogueColumnCount = document.createElement('div')
  const dialogueList = document.createElement('div')
  const rawDetails = document.createElement('details')
  const rawSummary = document.createElement('summary')
  const rawPre = document.createElement('pre')
  const footer = document.createElement('div')
  const footerHint = document.createElement('div')
  const footerState = document.createElement('div')
  let isOpen = storedState?.isOpen ?? true
  let apiKey = storedState?.apiKey ?? ''
  let scenario = storedState?.scenario ?? DEFAULT_SCENARIO_TEXT
  let rawResponse = storedState?.rawResponse ?? ''
  let parsedResult: ScenarioEditorResult | undefined = rawResponse
    ? parseScenarioEditorResult(rawResponse)
    : undefined
  let errorMessage = ''
  let isGenerating = false
  let activeRequest: AbortController | undefined
  let isDestroyed = false

  overlayRoot.className = 'scenario-editor-overlay'

  launcherButton.type = 'button'
  launcherButton.className = 'scenario-editor-overlay__launcher'
  launcherButton.title = '시나리오 에디터 열기'
  launcherBadge.className = 'scenario-editor-overlay__launcher-badge'
  launcherBadge.textContent = 'AI'
  launcherLabel.className = 'scenario-editor-overlay__launcher-label'
  launcherLabel.textContent = '시나리오 에디터'
  launcherButton.append(launcherBadge, launcherLabel)

  panel.className = 'scenario-editor-overlay__panel'
  panel.setAttribute('role', 'region')
  panel.setAttribute('aria-labelledby', 'scenario-editor-title')

  panelBody.className = 'scenario-editor-overlay__panel-body'

  header.className = 'scenario-editor-overlay__header'
  headerCopy.className = 'scenario-editor-overlay__header-copy'
  eyebrow.className = 'scenario-editor-overlay__eyebrow'
  eyebrow.textContent = 'LLM WORKBENCH'
  title.id = 'scenario-editor-title'
  title.className = 'scenario-editor-overlay__title'
  title.textContent = '시나리오 에디터'
  subtitle.className = 'scenario-editor-overlay__subtitle'
  subtitle.textContent = 'API 키만 넣고 자연어 시나리오를 퀘스트와 NPC 대사 초안으로 바꿉니다.'

  headerActions.className = 'scenario-editor-overlay__header-actions'
  statusPill.className = 'scenario-editor-overlay__status-pill'
  statusPill.setAttribute('role', 'status')
  statusPill.setAttribute('aria-live', 'polite')
  closeButton.type = 'button'
  closeButton.className = 'scenario-editor-overlay__close'
  closeButton.setAttribute('aria-label', '시나리오 에디터 닫기')
  closeButton.title = '시나리오 에디터 닫기'
  closeIcon.className = 'scenario-editor-overlay__close-icon'
  closeIcon.textContent = '×'
  closeIcon.setAttribute('aria-hidden', 'true')
  closeButton.append(closeIcon)
  headerActions.append(statusPill, closeButton)
  headerCopy.append(eyebrow, title, subtitle)
  header.append(headerCopy, headerActions)

  configGrid.className = 'scenario-editor-overlay__config-grid'
  configGrid.append(apiKeyField.field)

  scenarioSection.className = 'scenario-editor-overlay__scenario-section'
  scenarioHeader.className = 'scenario-editor-overlay__section-header'
  scenarioTitle.className = 'scenario-editor-overlay__section-title'
  scenarioTitle.textContent = '시나리오 입력'
  scenarioHint.className = 'scenario-editor-overlay__section-hint'
  scenarioHint.textContent = 'Ctrl/Cmd + Enter로 바로 생성할 수 있습니다.'
  scenarioTextarea.className = 'scenario-editor-overlay__scenario-input'
  scenarioTextarea.spellcheck = false
  scenarioTextarea.value = scenario
  scenarioHeader.append(scenarioTitle, scenarioHint)
  scenarioSection.append(scenarioHeader, scenarioTextarea)

  actionRow.className = 'scenario-editor-overlay__action-row'
  generateButton.type = 'button'
  generateButton.className = 'scenario-editor-overlay__button scenario-editor-overlay__button--primary'
  generateButton.textContent = '생성'
  stopButton.type = 'button'
  stopButton.className = 'scenario-editor-overlay__button'
  stopButton.textContent = '중단'
  sampleButton.type = 'button'
  sampleButton.className = 'scenario-editor-overlay__button'
  sampleButton.textContent = '샘플'
  clearButton.type = 'button'
  clearButton.className = 'scenario-editor-overlay__button'
  clearButton.textContent = '비우기'
  actionRow.append(generateButton, stopButton, sampleButton, clearButton)

  resultSection.className = 'scenario-editor-overlay__result-section'
  resultHeader.className = 'scenario-editor-overlay__result-header'
  resultHeaderTitle.className = 'scenario-editor-overlay__result-title'
  resultHeaderTitle.textContent = 'LLM 결과'
  resultHeaderMeta.className = 'scenario-editor-overlay__result-meta'
  errorBanner.className = 'scenario-editor-overlay__error'
  errorBanner.hidden = true
  resultSummaryCard.className = 'scenario-editor-overlay__summary-card'
  resultSummaryLabel.className = 'scenario-editor-overlay__summary-label'
  resultSummaryLabel.textContent = '시나리오 요약'
  resultSummaryText.className = 'scenario-editor-overlay__summary-text'
  resultSummaryCard.append(resultSummaryLabel, resultSummaryText)

  resultGrid.className = 'scenario-editor-overlay__result-grid'

  questColumn.className = 'scenario-editor-overlay__column'
  questColumnHeader.className = 'scenario-editor-overlay__column-header'
  questColumnTitle.className = 'scenario-editor-overlay__column-title'
  questColumnTitle.textContent = '퀘스트'
  questColumnCount.className = 'scenario-editor-overlay__column-count'
  questColumnHeader.append(questColumnTitle, questColumnCount)
  questList.className = 'scenario-editor-overlay__card-list'
  questColumn.append(questColumnHeader, questList)

  dialogueColumn.className = 'scenario-editor-overlay__column'
  dialogueColumnHeader.className = 'scenario-editor-overlay__column-header'
  dialogueColumnTitle.className = 'scenario-editor-overlay__column-title'
  dialogueColumnTitle.textContent = 'NPC 대사'
  dialogueColumnCount.className = 'scenario-editor-overlay__column-count'
  dialogueColumnHeader.append(dialogueColumnTitle, dialogueColumnCount)
  dialogueList.className = 'scenario-editor-overlay__card-list'
  dialogueColumn.append(dialogueColumnHeader, dialogueList)

  resultGrid.append(questColumn, dialogueColumn)

  rawDetails.className = 'scenario-editor-overlay__raw'
  rawSummary.className = 'scenario-editor-overlay__raw-summary'
  rawSummary.textContent = '원문 JSON'
  rawPre.className = 'scenario-editor-overlay__raw-pre'
  rawDetails.append(rawSummary, rawPre)

  resultHeader.append(resultHeaderTitle, resultHeaderMeta)
  footer.className = 'scenario-editor-overlay__footer'
  footerHint.className = 'scenario-editor-overlay__footer-hint'
  footerHint.textContent = '시나리오를 바꾸면 이전 결과는 지워집니다.'
  footerState.className = 'scenario-editor-overlay__footer-state'
  footer.append(footerHint, footerState)

  resultSection.append(
    resultHeader,
    errorBanner,
    resultSummaryCard,
    resultGrid,
    rawDetails,
    footer
  )

  panelBody.append(
    header,
    configGrid,
    scenarioSection,
    actionRow,
    resultSection
  )
  panel.append(panelBody)
  overlayRoot.append(launcherButton, panel)
  mountElement.append(overlayRoot)

  apiKeyField.input.value = apiKey
  const syncView = () => {
    const resultQuests = parsedResult?.quests ?? []
    const resultDialogues = parsedResult?.npcDialogues ?? []
    const isReady = parsedResult !== undefined
    const isIdle = !isGenerating && !isReady && errorMessage.length === 0
    const isErrored = errorMessage.length > 0

    launcherButton.hidden = isOpen
    panel.hidden = !isOpen
    panel.setAttribute('aria-hidden', String(!isOpen))

    statusPill.dataset.state = isGenerating
      ? 'generating'
      : isErrored
        ? 'error'
        : isReady
          ? 'ready'
          : 'idle'
    statusPill.textContent = isGenerating
      ? '생성 중'
      : isErrored
        ? '오류'
        : isReady
          ? '완료'
          : '대기 중'

    resultHeaderMeta.textContent = isReady
      ? `${resultQuests.length} 퀘스트 · ${resultDialogues.length} NPC 대사`
      : '결과 없음'

    errorBanner.hidden = !isErrored
    errorBanner.textContent = errorMessage
    resultSummaryText.textContent = isReady
      ? parsedResult?.summary || '요약이 비어 있습니다.'
      : '시나리오를 입력하고 생성 버튼을 누르면 결과가 여기에 표시됩니다.'
    resultSummaryCard.classList.toggle(
      'scenario-editor-overlay__summary-card--empty',
      !isReady
    )
    questColumnCount.textContent = `${resultQuests.length}개`
    dialogueColumnCount.textContent = `${resultDialogues.length}개`
    footerState.textContent = isGenerating
      ? '모델 응답을 기다리는 중입니다.'
      : isReady
        ? '최근 결과가 저장되었습니다.'
        : isIdle
          ? '아직 생성되지 않았습니다.'
          : '오류를 확인해 주세요.'

    generateButton.disabled = isGenerating || !canGenerate()
    stopButton.disabled = !isGenerating
    sampleButton.disabled = isGenerating
    clearButton.disabled = isGenerating

    questList.replaceChildren(
      ...(resultQuests.length > 0
        ? resultQuests.map((quest, index) =>
            createQuestCard(quest, index)
          )
        : [createEmptyState('생성된 퀘스트가 없습니다.')])
    )
    dialogueList.replaceChildren(
      ...(resultDialogues.length > 0
        ? resultDialogues.map((dialogue, index) =>
            createDialogueCard(dialogue, index)
          )
        : [createEmptyState('생성된 NPC 대사가 없습니다.')])
    )

    if (rawResponse.length > 0) {
      rawDetails.hidden = false
      rawDetails.open = isErrored || !isReady
      rawPre.textContent = rawResponse
    } else {
      rawDetails.hidden = isErrored ? false : true
      rawDetails.open = false
      rawPre.textContent = '원문 JSON이 아직 없습니다.'
    }
  }

  const persistState = () => {
    const nextState: StoredScenarioEditorState = {
      isOpen,
      apiKey,
      scenario,
      rawResponse
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(nextState))
    } catch {
      // Ignore storage failures in restricted browsers.
    }
  }

  const setScenario = (nextScenario: string) => {
    scenario = nextScenario
    scenarioTextarea.value = nextScenario
    rawResponse = ''
    parsedResult = undefined
    errorMessage = ''
    persistState()
    syncView()
  }

  const handleLauncherClick = (event: MouseEvent) => {
    event.preventDefault()
    isOpen = true
    persistState()
    syncView()
  }

  const handleCloseClick = (event: MouseEvent) => {
    event.preventDefault()
    isOpen = false
    persistState()
    syncView()
  }

  const handleScenarioInput = (event: Event) => {
    const nextTarget = event.currentTarget

    if (!(nextTarget instanceof HTMLTextAreaElement)) {
      return
    }

    scenario = nextTarget.value
    rawResponse = ''
    parsedResult = undefined
    errorMessage = ''
    persistState()
    syncView()
  }

  const handleApiKeyInput = (event: Event) => {
    const nextTarget = event.currentTarget

    if (!(nextTarget instanceof HTMLInputElement)) {
      return
    }

    apiKey = nextTarget.value
    persistState()
    syncView()
  }

  const handleGenerateClick = (event: MouseEvent) => {
    event.preventDefault()
    void generateScenarioDraft()
  }

  const handleStopClick = (event: MouseEvent) => {
    event.preventDefault()
    activeRequest?.abort()
  }

  const handleSampleClick = (event: MouseEvent) => {
    event.preventDefault()
    setScenario(DEFAULT_SCENARIO_TEXT)
  }

  const handleClearClick = (event: MouseEvent) => {
    event.preventDefault()
    setScenario('')
  }

  const handleScenarioKeyDown = (event: KeyboardEvent) => {
    if (!(event.ctrlKey || event.metaKey) || event.key !== 'Enter') {
      return
    }

    event.preventDefault()
    void generateScenarioDraft()
  }

  const canGenerate = (): boolean => {
    return (
      apiKey.trim().length > 0 &&
      scenario.trim().length > 0
    )
  }

  const generateScenarioDraft = async (): Promise<void> => {
    if (isGenerating) {
      return
    }

    if (!canGenerate()) {
      errorMessage = 'API 키와 시나리오를 모두 채워주세요.'
      syncView()
      return
    }

    const request = new AbortController()

    if (activeRequest) {
      activeRequest.abort()
    }

    activeRequest = request
    isGenerating = true
    errorMessage = ''
    syncView()

    try {
      const response = await fetch(DEFAULT_LLM_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey.trim()}`
        },
        body: JSON.stringify({
          model: DEFAULT_LLM_MODEL,
          temperature: 0.35,
          input: [
            {
              role: 'system',
              content: SCENARIO_EDITOR_SYSTEM_PROMPT
            },
            {
              role: 'user',
              content: buildScenarioEditorUserPrompt(scenario, AVAILABLE_NPC_NAMES)
            }
          ]
        }),
        signal: request.signal
      })

      if (!response.ok) {
        const errorText = (await response.text()).trim()

        throw new Error(
          errorText.length > 0
            ? errorText
            : `요청 실패: ${response.status} ${response.statusText}`
        )
      }

      const responseBody = (await response.json()) as unknown
      const responseText = extractScenarioEditorResponseText(responseBody)

      if (!responseText) {
        throw new Error('응답 본문에서 텍스트를 찾지 못했습니다.')
      }

      const nextResult = parseScenarioEditorResult(responseText)

      if (!nextResult) {
        rawResponse = responseText.trim()
        parsedResult = undefined
        throw new Error('응답을 퀘스트와 NPC 대사 JSON으로 해석하지 못했습니다.')
      }

      rawResponse = responseText.trim()
      parsedResult = nextResult
      errorMessage = ''
      persistState()
      syncView()
    } catch (error) {
      if (!request.signal.aborted) {
        errorMessage =
          error instanceof Error ? error.message : '시나리오 생성에 실패했습니다.'
      }
      syncView()
    } finally {
      if (activeRequest === request) {
        activeRequest = undefined
      }

      isGenerating = false
      persistState()
      syncView()
    }
  }

  launcherButton.addEventListener('click', handleLauncherClick)
  closeButton.addEventListener('click', handleCloseClick)
  apiKeyField.input.addEventListener('input', handleApiKeyInput)
  scenarioTextarea.addEventListener('input', handleScenarioInput)
  scenarioTextarea.addEventListener('keydown', handleScenarioKeyDown)
  generateButton.addEventListener('click', handleGenerateClick)
  stopButton.addEventListener('click', handleStopClick)
  sampleButton.addEventListener('click', handleSampleClick)
  clearButton.addEventListener('click', handleClearClick)

  const syncFrame = () => {
    if (isDestroyed) {
      return
    }

    const availableWidth = Math.max(1, window.innerWidth - OVERLAY_MARGIN * 2)

    launcherButton.style.right = `${OVERLAY_MARGIN}px`
    launcherButton.style.top = `${OVERLAY_MARGIN}px`
    panel.style.right = `${OVERLAY_MARGIN}px`
    panel.style.top = `${OVERLAY_MARGIN}px`
    panel.style.bottom = `${OVERLAY_MARGIN}px`
    panel.style.width = `${Math.min(PANEL_WIDTH, availableWidth)}px`
  }

  const destroy = () => {
    if (isDestroyed) {
      return
    }

    isDestroyed = true
    activeRequest?.abort()
    launcherButton.removeEventListener('click', handleLauncherClick)
    closeButton.removeEventListener('click', handleCloseClick)
    apiKeyField.input.removeEventListener('input', handleApiKeyInput)
    scenarioTextarea.removeEventListener('input', handleScenarioInput)
    scenarioTextarea.removeEventListener('keydown', handleScenarioKeyDown)
    generateButton.removeEventListener('click', handleGenerateClick)
    stopButton.removeEventListener('click', handleStopClick)
    sampleButton.removeEventListener('click', handleSampleClick)
    clearButton.removeEventListener('click', handleClearClick)
    overlayRoot.remove()
  }

  syncView()
  syncFrame()

  return {
    syncFrame,
    destroy
  }
}

const createField = ({
  label,
  type,
  placeholder,
  autocomplete
}: {
  label: string
  type: 'password' | 'text'
  placeholder: string
  autocomplete: string
}): ScenarioEditorFieldElements => {
  const field = document.createElement('label')
  const labelElement = document.createElement('span')
  const input = document.createElement('input')

  field.className = 'scenario-editor-overlay__field'
  labelElement.className = 'scenario-editor-overlay__field-label'
  labelElement.textContent = label
  input.className = 'scenario-editor-overlay__field-input'
  input.type = type
  input.placeholder = placeholder
  input.setAttribute('autocomplete', autocomplete)
  input.spellcheck = false

  field.append(labelElement, input)

  return {
    field,
    input
  }
}

const createQuestCard = (
  quest: ScenarioEditorQuestDraft,
  index: number
): HTMLElement => {
  const card = document.createElement('article')
  const header = document.createElement('div')
  const indexLabel = document.createElement('div')
  const title = document.createElement('div')
  const meta = document.createElement('div')
  const goal = document.createElement('div')
  const objectiveList = document.createElement('div')
  const reward = document.createElement('div')

  card.className = 'scenario-editor-overlay__card scenario-editor-overlay__card--quest'
  header.className = 'scenario-editor-overlay__card-header'
  indexLabel.className = 'scenario-editor-overlay__card-index'
  indexLabel.textContent = `Q${index + 1}`
  title.className = 'scenario-editor-overlay__card-title'
  title.textContent = quest.title || '제목 없음'
  meta.className = 'scenario-editor-overlay__card-meta'
  meta.textContent = [quest.giver, quest.reward]
    .filter((item) => item.length > 0)
    .join(' · ')
  goal.className = 'scenario-editor-overlay__card-body'
  goal.textContent = quest.goal || '목표가 비어 있습니다.'
  objectiveList.className = 'scenario-editor-overlay__card-list-block'
  objectiveList.append(
    ...(quest.objectives.length > 0
      ? quest.objectives.map((objective, objectiveIndex) => {
          const row = document.createElement('div')

          row.className = 'scenario-editor-overlay__card-list-row'
          row.textContent = `${objectiveIndex + 1}. ${objective}`
          return row
        })
      : [createEmptyState('세부 목표 없음')])
  )
  reward.className = 'scenario-editor-overlay__card-reward'
  reward.textContent = quest.reward || '보상 정보 없음'

  header.append(indexLabel, title)
  card.append(header, meta, goal, objectiveList, reward)

  return card
}

const createDialogueCard = (
  dialogue: ScenarioEditorNpcDialogueDraft,
  index: number
): HTMLElement => {
  const card = document.createElement('article')
  const header = document.createElement('div')
  const indexLabel = document.createElement('div')
  const title = document.createElement('div')
  const context = document.createElement('div')
  const lineList = document.createElement('div')

  card.className = 'scenario-editor-overlay__card scenario-editor-overlay__card--dialogue'
  header.className = 'scenario-editor-overlay__card-header'
  indexLabel.className = 'scenario-editor-overlay__card-index'
  indexLabel.textContent = `N${index + 1}`
  title.className = 'scenario-editor-overlay__card-title'
  title.textContent = dialogue.npc || 'NPC'
  context.className = 'scenario-editor-overlay__card-context'
  context.textContent = dialogue.context || '대화 맥락 없음'
  lineList.className = 'scenario-editor-overlay__card-list-block'
  lineList.append(
    ...(dialogue.lines.length > 0
      ? dialogue.lines.map((line, lineIndex) => {
          const row = document.createElement('div')

          row.className = 'scenario-editor-overlay__card-list-row'
          row.textContent = `${lineIndex + 1}. ${line}`
          return row
        })
      : [createEmptyState('대사 없음')])
  )

  header.append(indexLabel, title)
  card.append(header, context, lineList)

  return card
}

const createEmptyState = (text: string): HTMLElement => {
  const emptyState = document.createElement('div')

  emptyState.className = 'scenario-editor-overlay__empty-state'
  emptyState.textContent = text

  return emptyState
}

const readStoredState = (): StoredScenarioEditorState | undefined => {
  const rawState = window.localStorage.getItem(STORAGE_KEY)

  if (!rawState) {
    return undefined
  }

  try {
    const parsedState = JSON.parse(rawState) as unknown

    if (!isRecord(parsedState)) {
      return undefined
    }

    return {
      isOpen: toBoolean(parsedState.isOpen, true),
      apiKey: toText(parsedState.apiKey),
      scenario: toText(parsedState.scenario),
      rawResponse: toText(parsedState.rawResponse)
    }
  } catch {
    return undefined
  }
}

const toBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === 'boolean') {
    return value
  }

  return fallback
}

const toText = (value: unknown): string => {
  if (typeof value === 'string') {
    return value.trim()
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  return ''
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null
}
