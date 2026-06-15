import {
  analyzeCurrentGameStructure,
  type GameStructureAnalysisProgress
} from './gameStructureAnalyzer'
import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import {
  buildGameStructureProfile,
  type EditorMapSource
} from './buildGameStructureProfile'
import { registerDynamicEventDefinition } from './DynamicEventManager'
import {
  createGeneratedEventJsonValidationIssues,
  type GeneratedEventJson
} from './eventJsonSchema'
import {
  createHolidayDialogueEventSpecFromGeneratedEventJson,
  generateEventCodePreview
} from './eventCodeGenerator'
import { generateEventJsonDraftWithOpenAi } from './openaiEventJsonGenerator'
import {
  appendEventEvaluation,
  createEvaluationAcceptanceStats,
  loadEventEvaluations,
  type EventEvaluationVerdict
} from './eventEvaluator'
import type {
  GameStructureAnalysisResult
} from './gameStructureProfile'
import type { HolidayDialogueEventSpec } from '../games/my-sample-rpg/eventGeneration'
import type { ApplyEventDraftResult } from '../games/my-sample-rpg/rendering/createPixiTiledMapView'

type SceneRenderer = {
  applyEventDraft: (
    draft: HolidayDialogueEventSpec,
    input?: { targetCharacterId?: string }
  ) => ApplyEventDraftResult
}

export type CreateLlmPanelInput = {
  mountElement: HTMLElement
  getSceneRenderer: () => SceneRenderer | undefined
  // 게임이 이미 파싱한 맵들. 주어지면 Validator/프롬프트의 ground-truth를 실제 .tmx에서 만든다.
  mapSources?: EditorMapSource[]
}

export type LlmPanelController = {
  open: () => void
  close: () => void
  toggle: () => void
  isOpen: () => boolean
  refresh: () => void
  destroy: () => void
}

type EditableFieldDescriptor = {
  id: string
  label: string
  kind: 'text' | 'number' | 'textarea'
  getValue: (json: GeneratedEventJson) => string | number | boolean
  setValue: (json: GeneratedEventJson, value: string | number | boolean) => GeneratedEventJson
}

type FieldRowController = {
  label: string
  root: HTMLDivElement
  input: HTMLInputElement | HTMLTextAreaElement
  sync: (json: GeneratedEventJson) => void
  setExpanded: (expanded: boolean) => void
  isExpanded: () => boolean
}

const DEFAULT_EVENT_PROMPT =
  '크리스마스 이벤트를 만들어줘. 산타 NPC가 등장하고 대화하면 선물을 주도록 해줘.'
const OPENAI_API_KEY_STORAGE_KEY = 'my-sample-rpg:openai-api-key'
const PANEL_POSITION_STORAGE_KEY = 'my-sample-rpg:llm-panel-position'

type PanelBounds = {
  left: number
  top: number
  width: number
  height: number
}

export const createLlmPanel = ({
  mountElement,
  getSceneRenderer,
  mapSources
}: CreateLlmPanelInput): LlmPanelController => {
  const baseProfile =
    mapSources && mapSources.length > 0
      ? buildGameStructureProfile(mapSources, CURRENT_GAME_PROJECT_PROFILE)
      : CURRENT_GAME_PROJECT_PROFILE
  const overlayRoot = document.createElement('section')
  const panel = document.createElement('div')
  const titleBar = document.createElement('div')
  const titleText = document.createElement('h2')
  const titleActions = document.createElement('div')
  const minimizeButton = document.createElement('button')
  const maximizeButton = document.createElement('button')
  const closeButton = document.createElement('button')
  const contentArea = document.createElement('div')
  const header = document.createElement('header')
  const hint = document.createElement('p')
  const pageTabs = document.createElement('div')
  const analysisTab = document.createElement('button')
  const draftTab = document.createElement('button')
  const codeTab = document.createElement('button')
  const apiKeyGate = document.createElement('section')
  const apiKeyLabel = document.createElement('label')
  const apiKeyInput = document.createElement('input')
  const apiKeyHint = document.createElement('p')
  const apiKeyConfirmButton = document.createElement('button')
  const toggleButton = document.createElement('button')
  const analysisSummary = document.createElement('p')
  const progressWrap = document.createElement('div')
  const progressBar = document.createElement('div')
  const progressBarFill = document.createElement('div')
  const progressLabel = document.createElement('p')
  const gusCard = document.createElement('section')
  const gusTitle = document.createElement('p')
  const gusScore = document.createElement('p')
  const gusMeta = document.createElement('p')
  const gusThreshold = document.createElement('p')
  const gusDetailList = document.createElement('ul')
  const lockingNotice = document.createElement('p')
  const analysisNotesTitle = document.createElement('p')
  const analysisNotesList = document.createElement('ul')
  const profileTitle = document.createElement('p')
  const profilePreview = document.createElement('pre')
  const draftPage = document.createElement('section')
  const codePage = document.createElement('section')
  const draftTitle = document.createElement('p')
  const promptLabel = document.createElement('label')
  const promptInput = document.createElement('textarea')
  const draftActionRow = document.createElement('div')
  const generateButton = document.createElement('button')
  const draftPreviewTitle = document.createElement('p')
  const draftPreview = document.createElement('pre')
  const editTitle = document.createElement('p')
  const editActionRow = document.createElement('div')
  const confirmButton = document.createElement('button')
  const codeTitle = document.createElement('p')
  const codeActionRow = document.createElement('div')
  const codeButton = document.createElement('button')
  const applyTitle = document.createElement('p')
  const applyActionRow = document.createElement('div')
  const applyButton = document.createElement('button')
  const jsonStatus = document.createElement('p')
  const fieldEditorTitle = document.createElement('p')
  const fieldEditorList = document.createElement('div')
  const fieldHint = document.createElement('p')
  const codePreviewTitle = document.createElement('p')
  const codePreview = document.createElement('pre')
  const generatedEventTitle = document.createElement('p')
  const generatedEventHistory = document.createElement('ul')
  const analysisPage = document.createElement('section')
  const draftPageBody = document.createElement('section')
  const codePageBody = document.createElement('section')
  const leftResizeHandle = document.createElement('div')
  const rightResizeHandle = document.createElement('div')
  const topResizeHandle = document.createElement('div')
  const bottomResizeHandle = document.createElement('div')
  const evaluatorTitle = document.createElement('p')
  const evaluatorHint = document.createElement('p')
  const evaluatorReasonInput = document.createElement('input')
  const evaluatorActionRow = document.createElement('div')
  const evaluatorAcceptButton = document.createElement('button')
  const evaluatorRejectButton = document.createElement('button')
  const evaluatorStats = document.createElement('p')

  let isOpen = false
  let isAnalyzing = false
  let isGenerating = false
  let isMinimized = false
  let isMaximized = false
  let apiKeyConfirmed = false
  let openAiApiKey = loadOpenAiApiKey()
  let panelBounds = loadPanelBounds()
  let restoreBounds: PanelBounds | undefined
  let currentPage: 'analysis' | 'draft' | 'code' = 'analysis'
  let analysisResult: GameStructureAnalysisResult | undefined
  let currentDraft: GeneratedEventJson | undefined
  let confirmedDraft: GeneratedEventJson | undefined
  let generatedCode = ''
  let analysisProgress: GameStructureAnalysisProgress | undefined
  let fieldRows: FieldRowController[] = []
  let evaluations = loadEventEvaluations()

  overlayRoot.className = 'event-draft-panel llm-panel'
  panel.className = 'event-draft-panel__window llm-panel__window'
  titleBar.className = 'llm-panel__titlebar'
  titleText.className = 'llm-panel__window-title'
  titleActions.className = 'llm-panel__window-actions'
  minimizeButton.className = 'llm-panel__window-button'
  maximizeButton.className = 'llm-panel__window-button'
  closeButton.className = 'llm-panel__window-button llm-panel__window-button--close'
  contentArea.className = 'llm-panel__content'
  header.className = 'event-draft-panel__header llm-panel__header'
  hint.className = 'event-draft-panel__hint'
  pageTabs.className = 'llm-panel__tabs'
  analysisTab.className = 'llm-panel__tab'
  draftTab.className = 'llm-panel__tab'
  codeTab.className = 'llm-panel__tab'
  apiKeyGate.className = 'event-draft-panel__api-key'
  apiKeyLabel.className = 'event-draft-panel__api-key-label'
  apiKeyInput.className = 'event-draft-panel__api-key-input'
  apiKeyHint.className = 'event-draft-panel__hint'
  apiKeyConfirmButton.className = 'event-draft-panel__button'
  toggleButton.className = 'event-draft-panel__mode-button'
  analysisSummary.className = 'event-draft-panel__status'
  progressWrap.className = 'llm-panel__progress'
  progressBar.className = 'llm-panel__progress-bar'
  progressBarFill.className = 'llm-panel__progress-bar-fill'
  progressLabel.className = 'event-draft-panel__hint'
  gusCard.className = 'llm-panel__gus-card'
  gusTitle.className = 'llm-panel__section-title'
  gusScore.className = 'llm-panel__gus-score'
  gusMeta.className = 'event-draft-panel__status'
  gusThreshold.className = 'event-draft-panel__hint'
  gusDetailList.className = 'llm-panel__detail-list'
  lockingNotice.className = 'event-draft-panel__status'
  analysisNotesTitle.className = 'event-draft-panel__section-title'
  analysisNotesList.className = 'llm-panel__detail-list'
  profileTitle.className = 'event-draft-panel__section-title'
  profilePreview.className = 'event-draft-panel__output'
  draftPage.className = 'llm-panel__page'
  codePage.className = 'llm-panel__page'
  draftPageBody.className = 'llm-panel__page-body'
  codePageBody.className = 'llm-panel__page-body'
  leftResizeHandle.className = 'llm-panel__resize-handle llm-panel__resize-handle--left'
  rightResizeHandle.className = 'llm-panel__resize-handle llm-panel__resize-handle--right'
  topResizeHandle.className = 'llm-panel__resize-handle llm-panel__resize-handle--top'
  bottomResizeHandle.className = 'llm-panel__resize-handle llm-panel__resize-handle--bottom'
  draftTitle.className = 'event-draft-panel__section-title'
  promptLabel.className = 'event-draft-panel__section-title'
  promptInput.className = 'event-draft-panel__textarea'
  draftActionRow.className = 'event-draft-panel__actions llm-panel__actions'
  generateButton.className = 'event-draft-panel__button'
  draftPreviewTitle.className = 'event-draft-panel__section-title'
  draftPreview.className = 'event-draft-panel__output'
  editTitle.className = 'event-draft-panel__section-title'
  editActionRow.className = 'event-draft-panel__actions llm-panel__actions'
  confirmButton.className = 'event-draft-panel__button'
  codeTitle.className = 'event-draft-panel__section-title'
  codeActionRow.className = 'event-draft-panel__actions llm-panel__actions'
  codeButton.className = 'event-draft-panel__button'
  applyTitle.className = 'event-draft-panel__section-title'
  applyActionRow.className = 'event-draft-panel__actions llm-panel__actions'
  applyButton.className = 'event-draft-panel__button'
  jsonStatus.className = 'event-draft-panel__status'
  fieldEditorTitle.className = 'event-draft-panel__section-title'
  fieldEditorList.className = 'llm-panel__field-list'
  fieldHint.className = 'event-draft-panel__hint'
  codePreviewTitle.className = 'event-draft-panel__section-title'
  codePreview.className = 'event-draft-panel__output'
  generatedEventTitle.className = 'event-draft-panel__section-title'
  generatedEventHistory.className = 'llm-panel__detail-list'

  titleText.textContent = 'LLM Game Analyzer'
  minimizeButton.type = 'button'
  minimizeButton.textContent = '_'
  maximizeButton.type = 'button'
  maximizeButton.textContent = '□'
  closeButton.type = 'button'
  closeButton.textContent = 'X'
  hint.textContent =
    'L 키로 열고 닫는다. 먼저 게임 구조를 분석해서 GUS가 기준을 넘으면 이벤트 생성 영역이 활성화된다.'
  toggleButton.type = 'button'
  toggleButton.textContent = '패널 닫기'
  toggleButton.hidden = true
  analysisTab.type = 'button'
  analysisTab.textContent = '현 게임 분석'
  draftTab.type = 'button'
  draftTab.textContent = 'JSON 생성'
  codeTab.type = 'button'
  codeTab.textContent = '코드 생성'
  apiKeyLabel.htmlFor = 'llm-panel-openai-api-key'
  apiKeyLabel.textContent = 'OpenAI API 키'
  apiKeyInput.id = 'llm-panel-openai-api-key'
  apiKeyInput.type = 'password'
  apiKeyInput.placeholder = 'sk-...'
  apiKeyInput.autocomplete = 'off'
  apiKeyInput.spellcheck = false
  apiKeyInput.value = openAiApiKey
  apiKeyHint.textContent = 'API 키를 입력하고 확인을 누르면 구조 분석을 시작한다.'
  apiKeyConfirmButton.type = 'button'
  apiKeyConfirmButton.textContent = '확인 후 분석 시작'
  promptLabel.textContent = '자연어 입력'
  promptInput.value = DEFAULT_EVENT_PROMPT
  promptInput.rows = 5
  promptInput.spellcheck = false
  generateButton.type = 'button'
  generateButton.textContent = 'JSON 생성'
  confirmButton.type = 'button'
  confirmButton.textContent = 'JSON 저장/확정'
  codeButton.type = 'button'
  codeButton.textContent = '코드 생성'
  applyButton.type = 'button'
  applyButton.textContent = '게임 적용'
  gusTitle.textContent = 'GUS'
  gusThreshold.textContent = '기준 점수는 75점이다.'
  lockingNotice.textContent = '분석이 끝나지 않았거나 GUS가 기준 미만이면 이벤트 생성 영역은 잠긴다.'
  analysisNotesTitle.textContent = '분석 메모'
  profileTitle.textContent = 'Game Structure Profile'
  draftTitle.textContent = '이벤트 생성'
  draftPreviewTitle.textContent = 'JSON 초안'
  editTitle.textContent = 'JSON 내용 수정'
  fieldEditorTitle.textContent = 'JSON 내용 수정'
  fieldHint.textContent =
    '수정할 필드를 토글로 열고 닫는다. 여러 필드를 동시에 펼쳐서 수정할 수 있다.'
  codeTitle.textContent = '코드 미리보기'
  codePreviewTitle.textContent = '코드 미리보기'
  generatedEventTitle.textContent = '생성 기록'
  evaluatorTitle.className = 'event-draft-panel__section-title'
  evaluatorTitle.textContent = '품질 평가 (Evaluator · 사람 이진 판정)'
  evaluatorHint.className = 'event-draft-panel__hint'
  evaluatorHint.textContent =
    '생성 결과가 쓸 만한지 사람이 acceptable / not으로 판정한다. 자동 검증(Validator)과 분리된 단계다.'
  evaluatorReasonInput.type = 'text'
  evaluatorReasonInput.className = 'event-draft-panel__api-key-input'
  evaluatorReasonInput.placeholder = '사유 (선택)'
  evaluatorReasonInput.spellcheck = false
  evaluatorActionRow.className = 'event-draft-panel__actions llm-panel__actions'
  evaluatorAcceptButton.type = 'button'
  evaluatorAcceptButton.textContent = 'Acceptable'
  evaluatorAcceptButton.className =
    'event-draft-panel__button event-draft-panel__button--accept'
  evaluatorRejectButton.type = 'button'
  evaluatorRejectButton.textContent = 'Not acceptable'
  evaluatorRejectButton.className =
    'event-draft-panel__button event-draft-panel__button--reject'
  evaluatorStats.className = 'event-draft-panel__status'
  evaluatorActionRow.append(evaluatorAcceptButton, evaluatorRejectButton)

  progressWrap.append(progressBar, progressBarFill)
  gusCard.append(gusTitle, gusScore, gusMeta, gusThreshold, gusDetailList)
  draftActionRow.append(generateButton, confirmButton)
  codeActionRow.append(codeButton, applyButton)
  apiKeyGate.append(apiKeyLabel, apiKeyInput, apiKeyHint, apiKeyConfirmButton)
  analysisPage.append(
    analysisSummary,
    progressWrap,
    progressLabel,
    gusCard,
    lockingNotice,
    analysisNotesTitle,
    analysisNotesList,
    profileTitle,
    profilePreview
  )
  draftPageBody.append(
    draftTitle,
    promptLabel,
    promptInput,
    jsonStatus,
    draftActionRow,
    draftPreviewTitle,
    draftPreview,
    editTitle,
    fieldEditorTitle,
    fieldHint,
    fieldEditorList,
    evaluatorTitle,
    evaluatorHint,
    evaluatorReasonInput,
    evaluatorActionRow,
    evaluatorStats
  )
  codePageBody.append(
    codeTitle,
    codePreviewTitle,
    codePreview,
    codeActionRow,
    generatedEventTitle,
    generatedEventHistory
  )
  pageTabs.append(analysisTab, draftTab, codeTab)
  draftPage.append(draftPageBody)
  codePage.append(codePageBody)
  titleActions.append(minimizeButton, maximizeButton, closeButton)
  titleBar.append(titleText, titleActions)
  contentArea.append(header, apiKeyGate, pageTabs, analysisPage, draftPage, codePage)
  panel.append(titleBar, contentArea, leftResizeHandle, rightResizeHandle, topResizeHandle, bottomResizeHandle)
  overlayRoot.append(panel)
  mountElement.append(overlayRoot)

  const refresh = () => {
    overlayRoot.hidden = !isOpen
    toggleButton.textContent = isOpen ? '패널 닫기' : '패널 열기'
    apiKeyInput.value = openAiApiKey
    if (!isMaximized) {
      panelBounds = normalizePanelBounds(panelBounds)
    }
    applyPanelBounds(panel, panelBounds)
    contentArea.hidden = isMinimized || !isOpen
    leftResizeHandle.hidden = isMinimized || !isOpen || isMaximized
    rightResizeHandle.hidden = isMinimized || !isOpen || isMaximized
    topResizeHandle.hidden = isMinimized || !isOpen || isMaximized
    bottomResizeHandle.hidden = isMinimized || !isOpen || isMaximized
    minimizeButton.textContent = isMinimized ? '▢' : '_'
    maximizeButton.textContent = isMaximized ? '❐' : '□'
    maximizeButton.title = isMaximized ? '복원' : '최대화'
    minimizeButton.title = isMinimized ? '복원' : '최소화'
    closeButton.title = '닫기'
    syncPageUi()
    renderAnalysisUi()
    renderGenerationGate()
    renderDraftUi()
    renderCodeUi()
    renderEvaluatorUi()
  }

  const syncPageUi = () => {
    const shouldShowGate = isOpen && !apiKeyConfirmed

    apiKeyGate.hidden = !shouldShowGate
    pageTabs.hidden = shouldShowGate
    analysisPage.hidden = shouldShowGate || currentPage !== 'analysis'
    draftPage.hidden = shouldShowGate || currentPage !== 'draft'
    codePage.hidden = shouldShowGate || currentPage !== 'code'

    analysisTab.dataset.active = String(!shouldShowGate && currentPage === 'analysis')
    draftTab.dataset.active = String(!shouldShowGate && currentPage === 'draft')
    codeTab.dataset.active = String(!shouldShowGate && currentPage === 'code')
  }

  const renderAnalysisUi = () => {
    if (isAnalyzing && analysisProgress) {
      analysisSummary.textContent = `${analysisProgress.step}: ${analysisProgress.detail}`
      progressLabel.textContent = `${analysisProgress.progress}%`
      progressBarFill.style.width = `${analysisProgress.progress}%`
    } else if (analysisResult) {
      analysisSummary.textContent = '분석 완료'
      progressLabel.textContent = '100%'
      progressBarFill.style.width = '100%'
    } else {
      analysisSummary.textContent = '패널을 열고 분석 시작을 누르면 구조 분석을 시작한다.'
      progressLabel.textContent = '0%'
      progressBarFill.style.width = '0%'
    }

    if (analysisResult) {
      const gus = analysisResult.gus
      gusScore.textContent = `${gus.gus_score.toFixed(1)} / ${gus.threshold}`
      gusMeta.textContent = `상태: ${gus.status}`

      gusDetailList.replaceChildren(
        ...Object.entries(gus.details).map(([key, value]) => {
          const item = document.createElement('li')
          item.textContent = `${key}: ${value.toFixed(1)}`
          return item
        })
      )

      analysisNotesList.replaceChildren(
        ...analysisResult.analysis_notes.map((note) => {
          const item = document.createElement('li')
          item.textContent = note
          return item
        })
      )
      profilePreview.textContent = JSON.stringify(analysisResult.profile, null, 2)
    } else {
      gusScore.textContent = '--'
      gusMeta.textContent = '상태: 분석 전'
      gusDetailList.replaceChildren()
      analysisNotesList.replaceChildren()
      profilePreview.textContent = JSON.stringify(baseProfile, null, 2)
    }
  }

  const renderGenerationGate = () => {
    const analysisReady = analysisResult?.gus.status === 'passed'

    lockingNotice.hidden = false
    generateButton.disabled = !analysisReady || isAnalyzing || isGenerating
    generateButton.textContent = isGenerating ? '생성 중...' : 'JSON 생성'
    confirmButton.disabled = !analysisReady || !currentDraft || hasDraftValidationErrors()
    codeButton.disabled = !analysisReady || !confirmedDraft || hasDraftValidationErrors()
    applyButton.disabled = !analysisReady || !confirmedDraft || hasDraftValidationErrors()

    if (!analysisReady && analysisResult) {
      const missingItems = analysisResult.gus.missing_items

      lockingNotice.textContent =
        missingItems.length > 0
          ? `GUS 미달로 잠김: ${missingItems.join(' ')}`
          : 'GUS 미달로 잠김'
    } else if (analysisReady) {
      lockingNotice.textContent = 'GUS 통과. 이벤트 생성 버튼을 사용할 수 있다.'
    }
  }

  const renderDraftUi = () => {
    if (!currentDraft) {
      draftPreview.textContent = ''
      jsonStatus.textContent = analysisResult
        ? '자연어를 입력한 뒤 JSON 생성 버튼을 누르라.'
        : '분석이 먼저 필요하다.'
      fieldEditorList.replaceChildren()
      fieldEditorTitle.textContent = 'JSON 내용 수정'
      return
    }

    draftPreview.textContent = JSON.stringify(currentDraft, null, 2)
    jsonStatus.textContent = getValidationSummary(currentDraft)

    if (fieldRows.length === 0) {
      rebuildFieldEditors(currentDraft)
    }

    renderFieldEditors()
  }

  const renderCodeUi = () => {
    codePreview.textContent = generatedCode || '코드 생성 버튼을 누르면 미리보기가 표시된다.'
  }

  const renderEvaluatorUi = () => {
    const canEvaluate = Boolean(currentDraft) && !isGenerating

    evaluatorAcceptButton.disabled = !canEvaluate
    evaluatorRejectButton.disabled = !canEvaluate
    evaluatorReasonInput.disabled = !canEvaluate

    const stats = createEvaluationAcceptanceStats(evaluations)

    evaluatorStats.textContent =
      stats.total === 0
        ? '아직 평가 기록 없음 (acceptance rate = accepted / total)'
        : `acceptance rate: ${stats.accepted}/${stats.total} = ${Math.round(stats.acceptance_rate * 100)}% (목표 60~70%+)`
  }

  const runEvaluation = (verdict: EventEvaluationVerdict) => {
    if (!currentDraft) {
      return
    }

    evaluations = appendEventEvaluation({
      event_id: currentDraft.event_id,
      event_name: currentDraft.event_name,
      verdict,
      reason: evaluatorReasonInput.value.trim(),
      evaluated_at: Date.now()
    })
    evaluatorReasonInput.value = ''
    generatedEventHistory.prepend(
      createEventHistoryItem(
        `평가 기록: ${verdict === 'acceptable' ? 'Acceptable' : 'Not acceptable'} (${currentDraft.event_name})`
      )
    )
    refresh()
  }

  const hasDraftValidationErrors = (): boolean =>
    currentDraft
      ? createGeneratedEventJsonValidationIssues(currentDraft, analysisResult?.profile ?? baseProfile).length > 0
      : true

  const getValidationSummary = (draft: GeneratedEventJson): string => {
    const issues = createGeneratedEventJsonValidationIssues(
      draft,
      analysisResult?.profile ?? baseProfile
    )

    if (issues.length === 0) {
      return 'Validator (자동 검증): 통과'
    }

    return `Validator (자동 검증) 실패: ${issues.map((issue) => `${issue.path} - ${issue.message}`).join(' / ')}`
  }

  const rebuildFieldEditors = (json: GeneratedEventJson) => {
    fieldRows = createFieldRows({
      json,
      onChange: (nextJson) => {
        currentDraft = nextJson
        confirmedDraft = undefined
        generatedCode = ''
        refresh()
      }
    })
    renderFieldEditors()
  }

  const renderFieldEditors = () => {
    fieldEditorList.replaceChildren()

    if (fieldRows.length === 0) {
      return
    }

    fieldEditorList.append(...fieldRows.map((row) => row.root))

    const currentJson = currentDraft

    if (currentJson) {
      fieldRows.forEach((row) => row.sync(currentJson))
    }
  }

  const createFieldRows = ({
    json,
    onChange
  }: {
    json: GeneratedEventJson
    onChange: (nextJson: GeneratedEventJson) => void
  }): FieldRowController[] => {
    const descriptors: EditableFieldDescriptor[] = [
      {
        id: 'event_id',
        label: 'event_id',
        kind: 'text',
        getValue: (draft) => draft.event_id,
        setValue: (draft, value) => ({ ...draft, event_id: String(value) })
      },
      {
        id: 'event_name',
        label: 'event_name',
        kind: 'text',
        getValue: (draft) => draft.event_name,
        setValue: (draft, value) => ({ ...draft, event_name: String(value) })
      },
      {
        id: 'description',
        label: 'description',
        kind: 'text',
        getValue: (draft) => draft.description,
        setValue: (draft, value) => ({ ...draft, description: String(value) })
      },
      {
        id: 'trigger.type',
        label: 'trigger.type',
        kind: 'text',
        getValue: (draft) => draft.trigger.type,
        setValue: (draft, value) => ({
          ...draft,
          trigger: { ...draft.trigger, type: String(value) }
        })
      },
      {
        id: 'trigger.target',
        label: 'trigger.target',
        kind: 'text',
        getValue: (draft) => draft.trigger.target,
        setValue: (draft, value) => ({
          ...draft,
          trigger: { ...draft.trigger, target: String(value) }
        })
      },
      {
        id: 'trigger.condition',
        label: 'trigger.condition',
        kind: 'text',
        getValue: (draft) => draft.trigger.condition,
        setValue: (draft, value) => ({
          ...draft,
          trigger: { ...draft.trigger, condition: String(value) }
        })
      },
      {
        id: 'location.map_id',
        label: 'location.map_id',
        kind: 'text',
        getValue: (draft) => draft.location.map_id,
        setValue: (draft, value) => ({
          ...draft,
          location: { ...draft.location, map_id: String(value) }
        })
      },
      {
        id: 'location.x',
        label: 'location.x',
        kind: 'number',
        getValue: (draft) => draft.location.x,
        setValue: (draft, value) => ({
          ...draft,
          location: { ...draft.location, x: Number(value) }
        })
      },
      {
        id: 'location.y',
        label: 'location.y',
        kind: 'number',
        getValue: (draft) => draft.location.y,
        setValue: (draft, value) => ({
          ...draft,
          location: { ...draft.location, y: Number(value) }
        })
      },
      {
        id: 'npc.id',
        label: 'npc.id',
        kind: 'text',
        getValue: (draft) => draft.npc.id,
        setValue: (draft, value) => ({
          ...draft,
          npc: { ...draft.npc, id: String(value) }
        })
      },
      {
        id: 'npc.name',
        label: 'npc.name',
        kind: 'text',
        getValue: (draft) => draft.npc.name,
        setValue: (draft, value) => ({
          ...draft,
          npc: { ...draft.npc, name: String(value) }
        })
      },
      {
        id: 'npc.dialogue_id',
        label: 'npc.dialogue_id',
        kind: 'text',
        getValue: (draft) => draft.npc.dialogue_id,
        setValue: (draft, value) => ({
          ...draft,
          npc: { ...draft.npc, dialogue_id: String(value) }
        })
      },
      {
        id: 'reward.item_id',
        label: 'reward.item_id',
        kind: 'text',
        getValue: (draft) => draft.reward.item_id,
        setValue: (draft, value) => ({
          ...draft,
          reward: { ...draft.reward, item_id: String(value) }
        })
      },
      {
        id: 'reward.amount',
        label: 'reward.amount',
        kind: 'number',
        getValue: (draft) => draft.reward.amount,
        setValue: (draft, value) => ({
          ...draft,
          reward: { ...draft.reward, amount: Number(value) }
        })
      },
      {
        id: 'duration.start',
        label: 'duration.start',
        kind: 'text',
        getValue: (draft) => draft.duration.start,
        setValue: (draft, value) => ({
          ...draft,
          duration: { ...draft.duration, start: String(value) }
        })
      },
      {
        id: 'duration.end',
        label: 'duration.end',
        kind: 'text',
        getValue: (draft) => draft.duration.end,
        setValue: (draft, value) => ({
          ...draft,
          duration: { ...draft.duration, end: String(value) }
        })
      }
    ]

    const rows: FieldRowController[] = descriptors.map((descriptor, index) =>
      createFieldRow({
        descriptor,
        json,
        onChange,
        defaultExpanded: index === 0
      })
    )

    const dialogueRow = createDialogueRow({
      json,
      onChange,
      defaultExpanded: false
    })

    rows.push(dialogueRow)

    return rows
  }

  const createFieldRow = ({
    descriptor,
    json,
    onChange,
    defaultExpanded
  }: {
    descriptor: EditableFieldDescriptor
    json: GeneratedEventJson
    onChange: (nextJson: GeneratedEventJson) => void
    defaultExpanded: boolean
  }): FieldRowController => {
    const root = document.createElement('div')
    const header = document.createElement('button')
    const labelText = document.createElement('span')
    const previewText = document.createElement('span')
    const body = document.createElement('div')
    const input =
      descriptor.kind === 'textarea'
        ? document.createElement('textarea')
        : document.createElement('input')
    let expanded = defaultExpanded

    root.className = 'llm-panel__field-row'
    root.dataset.expanded = String(expanded)
    header.className = 'llm-panel__field-toggle'
    header.type = 'button'
    labelText.textContent = descriptor.label
    previewText.className = 'llm-panel__field-toggle-preview'
    previewText.textContent = formatFieldPreview(descriptor.getValue(json))
    body.className = 'llm-panel__field-body'

    if (descriptor.kind !== 'textarea') {
      const inputElement = input as HTMLInputElement
      inputElement.type = descriptor.kind === 'number' ? 'number' : 'text'
      inputElement.className = 'llm-panel__field-input'
      inputElement.value = String(descriptor.getValue(json))
      inputElement.disabled = false
      inputElement.addEventListener('input', () => {
        onChange(
          descriptor.setValue(
            structuredClone(currentDraft ?? json),
            parseFieldValue(descriptor.kind, inputElement.value)
          )
        )
      })
      inputElement.addEventListener('keydown', stopPropagationWhenEditing)
      inputElement.addEventListener('keyup', stopPropagationWhenEditing)
      body.append(inputElement)
    } else {
      const textarea = input as HTMLTextAreaElement
      textarea.className = 'llm-panel__field-textarea'
      textarea.value = String(descriptor.getValue(json))
      textarea.addEventListener('input', () => {
        onChange(
          descriptor.setValue(structuredClone(currentDraft ?? json), textarea.value)
        )
      })
      textarea.addEventListener('keydown', stopPropagationWhenEditing)
      textarea.addEventListener('keyup', stopPropagationWhenEditing)
      body.append(textarea)
    }

    const setExpanded = (nextExpanded: boolean) => {
      expanded = nextExpanded
      root.dataset.expanded = String(expanded)
      header.dataset.expanded = String(expanded)
      header.setAttribute('aria-expanded', String(expanded))
      header.textContent = ''
      labelText.textContent = descriptor.label
      header.append(labelText, previewText)
      body.hidden = !expanded
    }

    header.addEventListener('click', () => {
      setExpanded(!expanded)
    })

    setExpanded(expanded)
    root.append(header, body)

    return {
      label: descriptor.label,
      root,
      input,
      sync: (nextJson) => {
        const nextValue = descriptor.getValue(nextJson)
        previewText.textContent = formatFieldPreview(nextValue)

        if (input instanceof HTMLInputElement) {
          input.value = String(nextValue)
        } else {
          input.value = String(nextValue)
        }
      },
      setExpanded,
      isExpanded: () => expanded
    }
  }

  const createDialogueRow = ({
    json,
    onChange,
    defaultExpanded
  }: {
    json: GeneratedEventJson
    onChange: (nextJson: GeneratedEventJson) => void
    defaultExpanded: boolean
  }): FieldRowController => {
    const root = document.createElement('div')
    const header = document.createElement('button')
    const labelText = document.createElement('span')
    const previewText = document.createElement('span')
    const body = document.createElement('div')
    const textarea = document.createElement('textarea')
    let expanded = defaultExpanded

    root.className = 'llm-panel__field-row'
    root.dataset.expanded = String(expanded)
    header.className = 'llm-panel__field-toggle'
    header.type = 'button'
    labelText.textContent = 'dialogue'
    previewText.className = 'llm-panel__field-toggle-preview'
    previewText.textContent = formatDialoguePreview(json)
    body.className = 'llm-panel__field-body'
    textarea.className = 'llm-panel__field-textarea'
    textarea.value = formatDialogueText(json)

    textarea.addEventListener('input', () => {
      onChange({
        ...structuredClone(currentDraft ?? json),
        dialogue: parseDialogueText(textarea.value, json.npc.name)
      })
    })
    textarea.addEventListener('keydown', stopPropagationWhenEditing)
    textarea.addEventListener('keyup', stopPropagationWhenEditing)
    body.append(textarea)

    const setExpanded = (nextExpanded: boolean) => {
      expanded = nextExpanded
      root.dataset.expanded = String(expanded)
      header.dataset.expanded = String(expanded)
      header.setAttribute('aria-expanded', String(expanded))
      header.textContent = ''
      labelText.textContent = 'dialogue'
      header.append(labelText, previewText)
      body.hidden = !expanded
    }

    header.addEventListener('click', () => {
      setExpanded(!expanded)
    })

    setExpanded(expanded)
    root.append(header, body)

    return {
      label: 'dialogue',
      root,
      input: textarea,
      sync: (nextJson) => {
        textarea.value = formatDialogueText(nextJson)
        previewText.textContent = formatDialoguePreview(nextJson)
      },
      setExpanded,
      isExpanded: () => expanded
    }
  }

  const runAnalysis = async () => {
    if (isAnalyzing) {
      return
    }

    isAnalyzing = true
    analysisResult = undefined
    analysisProgress = {
      step: 'start',
      progress: 0,
      detail: '분석을 시작한다.'
    }
    refresh()

    try {
      analysisResult = await analyzeCurrentGameStructure({
        profile: baseProfile,
        onProgress: (progress) => {
          analysisProgress = progress
          refresh()
        }
      })
      currentPage = 'draft'
    } finally {
      isAnalyzing = false
      analysisProgress = undefined
      refresh()
    }
  }

  const runGeneration = async () => {
    if (!analysisResult || isGenerating) {
      return
    }

    const apiKey = openAiApiKey.trim()

    if (apiKey.length === 0) {
      generatedEventHistory.prepend(
        createEventHistoryItem('생성 실패: OpenAI API 키가 필요하다.')
      )
      refresh()
      return
    }

    isGenerating = true
    generatedEventHistory.prepend(createEventHistoryItem('OpenAI로 JSON 초안 생성 중...'))
    refresh()

    try {
      const eventJson = await generateEventJsonDraftWithOpenAi({
        apiKey,
        userPrompt: promptInput.value,
        profile: analysisResult.profile
      })
      currentDraft = eventJson
      confirmedDraft = undefined
      generatedCode = ''
      rebuildFieldEditors(currentDraft)
      generatedEventHistory.prepend(createEventHistoryItem('JSON 초안 생성 완료 (LLM)'))
      currentPage = 'draft'
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      generatedEventHistory.prepend(createEventHistoryItem(`LLM 생성 실패: ${message}`))
    } finally {
      isGenerating = false
      refresh()
    }
  }

  const runConfirm = () => {
    if (!currentDraft || !analysisResult) {
      return
    }

    confirmedDraft = structuredClone(currentDraft)
    generatedEventHistory.prepend(createEventHistoryItem('JSON 저장/확정 완료'))
    currentPage = 'code'
    refresh()
  }

  const runCodeGeneration = () => {
    if (!analysisResult) {
      return
    }

    const sourceJson = confirmedDraft ?? currentDraft

    if (!sourceJson) {
      return
    }

    const preview = generateEventCodePreview(sourceJson, analysisResult.profile)
    generatedCode = preview.code
    registerDynamicEventDefinition({
      event_json: sourceJson,
      generated_code: preview.code,
      created_at: Date.now()
    })
    generatedEventHistory.prepend(createEventHistoryItem('코드 미리보기 생성 완료'))
    if (preview.warnings.length > 0) {
      generatedEventHistory.prepend(
        createEventHistoryItem(`경고: ${preview.warnings.join(' / ')}`)
      )
    }
    currentPage = 'code'
    refresh()
  }

  const runApply = () => {
    const sourceJson = confirmedDraft ?? currentDraft

    if (!sourceJson) {
      return
    }

    const sceneRenderer = getSceneRenderer()

    if (!sceneRenderer) {
      generatedEventHistory.prepend(createEventHistoryItem('적용 실패: 씬이 준비되지 않았다.'))
      refresh()
      return
    }

    const holidaySpec = createHolidayDialogueEventSpecFromGeneratedEventJson(
      sourceJson
    )

    if (!holidaySpec) {
      generatedEventHistory.prepend(createEventHistoryItem('적용 실패: 대화 라인이 비어 있다.'))
      refresh()
      return
    }

    const applyResult = sceneRenderer.applyEventDraft(holidaySpec, {
      targetCharacterId: holidaySpec.npc.id
    })

    generatedEventHistory.prepend(
      createEventHistoryItem(
        applyResult.didApply
          ? `게임 적용 완료: ${applyResult.targetCharacterId ?? holidaySpec.npc.id}`
          : '게임 적용 실패: 대상 NPC를 찾지 못했다.'
      )
    )
    currentPage = 'code'
    refresh()
  }

  const open = () => {
    isOpen = true
    refresh()
  }

  const close = () => {
    isOpen = false
    refresh()
  }

  const toggle = () => {
    if (isOpen) {
      close()
      return
    }

    open()
  }

  const destroy = () => {
    overlayRoot.remove()
  }

  const handlePanelKeydown = (event: KeyboardEvent) => {
    if (event.key === 'Escape' && isOpen) {
      event.preventDefault()
      close()
    }
  }

  toggleButton.addEventListener('click', toggle)
  analysisTab.addEventListener('click', () => {
    currentPage = 'analysis'
    syncPageUi()
  })
  draftTab.addEventListener('click', () => {
    currentPage = 'draft'
    syncPageUi()
  })
  codeTab.addEventListener('click', () => {
    currentPage = 'code'
    syncPageUi()
  })
  generateButton.addEventListener('click', () => {
    void runGeneration()
  })
  confirmButton.addEventListener('click', runConfirm)
  codeButton.addEventListener('click', runCodeGeneration)
  applyButton.addEventListener('click', runApply)
  apiKeyInput.addEventListener('input', () => {
    openAiApiKey = apiKeyInput.value
    saveOpenAiApiKey(openAiApiKey)
  })
  apiKeyInput.addEventListener('keydown', stopPropagationWhenEditing)
  apiKeyInput.addEventListener('keyup', stopPropagationWhenEditing)
  apiKeyConfirmButton.addEventListener('click', () => {
    const nextApiKey = apiKeyInput.value.trim()

    if (nextApiKey.length === 0) {
      apiKeyHint.textContent = 'OpenAI API 키를 입력한 뒤 확인을 눌러라.'
      apiKeyInput.focus()
      return
    }

    openAiApiKey = nextApiKey
    saveOpenAiApiKey(openAiApiKey)
    apiKeyConfirmed = true
    currentPage = 'analysis'
    apiKeyHint.textContent = 'API 키가 저장되었다. 구조 분석을 시작한다.'
    refresh()
    void runAnalysis()
  })
  promptInput.addEventListener('keydown', stopPropagationWhenEditing)
  promptInput.addEventListener('keyup', stopPropagationWhenEditing)
  evaluatorAcceptButton.addEventListener('click', () => runEvaluation('acceptable'))
  evaluatorRejectButton.addEventListener('click', () => runEvaluation('not_acceptable'))
  evaluatorReasonInput.addEventListener('keydown', stopPropagationWhenEditing)
  evaluatorReasonInput.addEventListener('keyup', stopPropagationWhenEditing)

  header.append(hint)
  refresh()
  window.addEventListener('keydown', handlePanelKeydown)
  enableWindowDragging({
    handle: titleBar,
    panel,
    getPosition: () => panelBounds,
    setPosition: (nextPosition) => {
      panelBounds = nextPosition
      savePanelBounds(nextPosition)
    },
    isEnabled: () => isOpen && !isMinimized && !isMaximized
  })
  enableWindowEdgeResizing({
    handle: leftResizeHandle,
    panel,
    getBounds: () => panelBounds,
    setBounds: (nextBounds) => {
      panelBounds = nextBounds
      savePanelBounds(nextBounds)
    },
    edge: 'left',
    isEnabled: () => isOpen && !isMinimized && !isMaximized
  })
  enableWindowEdgeResizing({
    handle: rightResizeHandle,
    panel,
    getBounds: () => panelBounds,
    setBounds: (nextBounds) => {
      panelBounds = nextBounds
      savePanelBounds(nextBounds)
    },
    edge: 'right',
    isEnabled: () => isOpen && !isMinimized && !isMaximized
  })
  enableWindowEdgeResizing({
    handle: topResizeHandle,
    panel,
    getBounds: () => panelBounds,
    setBounds: (nextBounds) => {
      panelBounds = nextBounds
      savePanelBounds(nextBounds)
    },
    edge: 'top',
    isEnabled: () => isOpen && !isMinimized && !isMaximized
  })
  enableWindowEdgeResizing({
    handle: bottomResizeHandle,
    panel,
    getBounds: () => panelBounds,
    setBounds: (nextBounds) => {
      panelBounds = nextBounds
      savePanelBounds(nextBounds)
    },
    edge: 'bottom',
    isEnabled: () => isOpen && !isMinimized && !isMaximized
  })
  minimizeButton.addEventListener('click', () => {
    if (isMinimized) {
      isMinimized = false
      refresh()
      return
    }

    isMinimized = true
    refresh()
  })
  titleBar.addEventListener('click', (event) => {
    const target = event.target as HTMLElement | null

    if (target?.closest('button')) {
      return
    }

    if (isMinimized) {
      isMinimized = false
      refresh()
    }
  })
  maximizeButton.addEventListener('click', () => {
    if (isMaximized) {
      isMaximized = false
      if (restoreBounds) {
        panelBounds = restoreBounds
      }
      refresh()
      return
    }

    isMinimized = false
    restoreBounds = { ...panelBounds }
    isMaximized = true
    panelBounds = createMaximizedBounds()
    refresh()
  })
  closeButton.addEventListener('click', close)

  return {
    open,
    close,
    toggle,
    isOpen: () => isOpen,
    refresh,
    destroy: () => {
      window.removeEventListener('keydown', handlePanelKeydown)
      destroy()
    }
  }
}

const stopPropagationWhenEditing = (event: KeyboardEvent) => {
  event.stopPropagation()
}

const enableWindowDragging = ({
  handle,
  panel,
  getPosition,
  setPosition,
  isEnabled
}: {
  handle: HTMLElement
  panel: HTMLDivElement
  getPosition: () => PanelBounds
  setPosition: (nextPosition: PanelBounds) => void
  isEnabled: () => boolean
}) => {
  let pointerOffsetX = 0
  let pointerOffsetY = 0
  let activePointerId: number | undefined

  const onPointerMove = (event: PointerEvent) => {
    if (activePointerId !== event.pointerId) {
      return
    }

    const nextLeft = clamp(
      event.clientX - pointerOffsetX,
      8,
      Math.max(window.innerWidth - panel.offsetWidth - 8, 8)
    )
    const nextTop = clamp(
      event.clientY - pointerOffsetY,
      8,
      Math.max(window.innerHeight - panel.offsetHeight - 8, 8)
    )

    const nextBounds = { ...getPosition(), left: nextLeft, top: nextTop }
    setPosition(nextBounds)
    applyPanelBounds(panel, nextBounds)
  }

  const onPointerUp = (event: PointerEvent) => {
    if (activePointerId !== event.pointerId) {
      return
    }

    activePointerId = undefined
    window.removeEventListener('pointermove', onPointerMove)
    window.removeEventListener('pointerup', onPointerUp)
    window.removeEventListener('pointercancel', onPointerUp)
  }

  handle.addEventListener('pointerdown', (event) => {
    const target = event.target as HTMLElement | null

    if (!isEnabled() || event.button !== 0 || target?.closest('button')) {
      return
    }

    event.preventDefault()
    activePointerId = event.pointerId
    const rect = panel.getBoundingClientRect()
    pointerOffsetX = event.clientX - rect.left
    pointerOffsetY = event.clientY - rect.top

    handle.setPointerCapture(event.pointerId)
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
    window.addEventListener('pointercancel', onPointerUp)
  })
}

const enableWindowEdgeResizing = ({
  handle,
  panel,
  getBounds,
  setBounds,
  edge,
  isEnabled
}: {
  handle: HTMLElement
  panel: HTMLDivElement
  getBounds: () => PanelBounds
  setBounds: (nextBounds: PanelBounds) => void
  edge: 'left' | 'right' | 'top' | 'bottom'
  isEnabled: () => boolean
}) => {
  let activePointerId: number | undefined
  let pointerStartX = 0
  let pointerStartY = 0
  let startBounds: PanelBounds | undefined

  const onPointerMove = (event: PointerEvent) => {
    if (activePointerId !== event.pointerId || !startBounds) {
      return
    }

    const minWidth = 420
    const minHeight = 300
    const maxWidth = Math.max(window.innerWidth - 16, minWidth)
    const maxHeight = Math.max(window.innerHeight - 16, minHeight)
    const rightEdge = startBounds.left + startBounds.width
    const bottomEdge = startBounds.top + startBounds.height

    const nextBounds =
      edge === 'right'
        ? {
            ...startBounds,
            width: clamp(
              startBounds.width + (event.clientX - pointerStartX),
              minWidth,
              Math.max(window.innerWidth - startBounds.left - 8, minWidth)
            )
          }
        : edge === 'left'
          ? {
              ...startBounds,
              left: clamp(
                startBounds.left + (event.clientX - pointerStartX),
                8,
                rightEdge - minWidth - 8
              ),
              width: clamp(
                rightEdge - clamp(
                  startBounds.left + (event.clientX - pointerStartX),
                  8,
                  rightEdge - minWidth - 8
                ),
                minWidth,
                maxWidth
              )
            }
          : edge === 'bottom'
            ? {
                ...startBounds,
                height: clamp(
                  startBounds.height + (event.clientY - pointerStartY),
                  minHeight,
                  Math.max(window.innerHeight - startBounds.top - 8, minHeight)
                )
              }
            : {
                ...startBounds,
                top: clamp(
                  startBounds.top + (event.clientY - pointerStartY),
                  8,
                  bottomEdge - minHeight - 8
                ),
                height: clamp(
                  bottomEdge - clamp(
                    startBounds.top + (event.clientY - pointerStartY),
                    8,
                    bottomEdge - minHeight - 8
                  ),
                  minHeight,
                  maxHeight
                )
              }

    setBounds(nextBounds)
    applyPanelBounds(panel, nextBounds)
  }

  const onPointerUp = (event: PointerEvent) => {
    if (activePointerId !== event.pointerId) {
      return
    }

    activePointerId = undefined
    startBounds = undefined
    window.removeEventListener('pointermove', onPointerMove)
    window.removeEventListener('pointerup', onPointerUp)
    window.removeEventListener('pointercancel', onPointerUp)
  }

  handle.addEventListener('pointerdown', (event) => {
    if (!isEnabled() || event.button !== 0) {
      return
    }

    event.preventDefault()
    activePointerId = event.pointerId
    startBounds = getBounds()
    pointerStartX = event.clientX
    pointerStartY = event.clientY
    handle.setPointerCapture(event.pointerId)
    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
    window.addEventListener('pointercancel', onPointerUp)
  })
}

const loadPanelBounds = (): PanelBounds => {
  if (typeof window === 'undefined') {
    return { left: 16, top: 16, width: 560, height: 720 }
  }

  const stored = window.localStorage.getItem(PANEL_POSITION_STORAGE_KEY)

  if (stored) {
    try {
      const parsed = JSON.parse(stored) as Partial<PanelBounds>
      if (typeof parsed.left === 'number' && typeof parsed.top === 'number') {
        return {
          left: parsed.left,
          top: parsed.top,
          width: typeof parsed.width === 'number' ? parsed.width : 560,
          height: typeof parsed.height === 'number' ? parsed.height : 720
        }
      }
    } catch {
      // ignore invalid storage
    }
  }

  return {
    left: Math.max(window.innerWidth - 576 - 16, 16),
    top: 18,
    width: 560,
    height: 720
  }
}

const savePanelBounds = (position: PanelBounds): void => {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.setItem(PANEL_POSITION_STORAGE_KEY, JSON.stringify(position))
}

const applyPanelBounds = (panel: HTMLDivElement, position: PanelBounds): void => {
  panel.style.left = `${position.left}px`
  panel.style.top = `${position.top}px`
  panel.style.width = `${position.width}px`
  panel.style.height = `${position.height}px`
  panel.style.right = 'auto'
  panel.style.bottom = 'auto'
}

const normalizePanelBounds = (bounds: PanelBounds): PanelBounds => {
  const width = clamp(bounds.width, 420, Math.max(window.innerWidth - 16, 420))
  const height = clamp(bounds.height, 300, Math.max(window.innerHeight - 16, 300))
  const left = clamp(bounds.left, 8, Math.max(window.innerWidth - width - 8, 8))
  const top = clamp(bounds.top, 8, Math.max(window.innerHeight - height - 8, 8))

  return { left, top, width, height }
}

const createMaximizedBounds = (): PanelBounds => ({
  left: 16,
  top: 16,
  width: Math.max(window.innerWidth - 32, 420),
  height: Math.max(window.innerHeight - 32, 300)
})

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value))

const loadOpenAiApiKey = (): string => {
  if (typeof window === 'undefined') {
    return ''
  }

  return window.localStorage.getItem(OPENAI_API_KEY_STORAGE_KEY) ?? ''
}

const saveOpenAiApiKey = (value: string): void => {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.setItem(OPENAI_API_KEY_STORAGE_KEY, value)
}

const parseFieldValue = (
  kind: EditableFieldDescriptor['kind'],
  rawValue: string
): string | number | boolean => {
  if (kind === 'number') {
    return Number(rawValue)
  }

  return rawValue
}

const formatFieldPreview = (value: string | number | boolean): string => {
  const text = String(value).replace(/\s+/g, ' ').trim()

  if (text.length === 0) {
    return '비어 있음'
  }

  return text.length > 26 ? `${text.slice(0, 26)}…` : text
}

const formatDialoguePreview = (json: GeneratedEventJson): string => {
  const lineCount = json.dialogue.length

  if (lineCount === 0) {
    return '대사 없음'
  }

  const firstLine = json.dialogue[0]
  const preview = `${firstLine.speaker}: ${firstLine.text}`.replace(/\s+/g, ' ').trim()

  if (preview.length > 26) {
    return `${preview.slice(0, 26)}… · ${lineCount}줄`
  }

  return `${preview} · ${lineCount}줄`
}

const formatDialogueText = (json: GeneratedEventJson): string =>
  json.dialogue.map((line) => `${line.speaker}: ${line.text}`).join('\n')

const parseDialogueText = (
  value: string,
  fallbackSpeaker: string
): Array<{ speaker: string; text: string }> =>
  value
    .split(/\r?\n/u)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => {
      const separatorIndex = line.indexOf(':')

      if (separatorIndex < 0) {
        return {
          speaker: fallbackSpeaker,
          text: line
        }
      }

      const speaker = line.slice(0, separatorIndex).trim()
      const text = line.slice(separatorIndex + 1).trim()

      return {
        speaker: speaker.length > 0 ? speaker : fallbackSpeaker,
        text
      }
    })

const createEventHistoryItem = (message: string): HTMLLIElement => {
  const item = document.createElement('li')
  item.textContent = `${new Date().toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })} - ${message}`
  return item
}
