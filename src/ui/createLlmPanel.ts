import {
  analyzeCurrentGameStructure,
  type GameStructureAnalysisProgress
} from '../llm/gameStructureAnalyzer'
import { CURRENT_GAME_PROJECT_PROFILE } from '../llm/currentGameProjectSnapshot'
import { registerDynamicEventDefinition } from '../events/DynamicEventManager'
import {
  createGeneratedEventJsonValidationIssues,
  type GeneratedEventJson
} from '../llm/eventJsonSchema'
import {
  createHolidayDialogueEventSpecFromGeneratedEventJson,
  generateEventCodePreview
} from '../llm/eventCodeGenerator'
import {
  generateMockEventJsonDraft,
} from '../llm/eventJsonGenerator'
import type {
  GameStructureAnalysisResult
} from '../llm/gameStructureProfile'
import type { HolidayDialogueEventSpec } from '../game/eventGeneration'
import type { ApplyEventDraftResult } from '../rendering/createPixiTiledMapView'

type SceneRenderer = {
  applyEventDraft: (
    draft: HolidayDialogueEventSpec,
    input?: { targetCharacterId?: string }
  ) => ApplyEventDraftResult
}

export type CreateLlmPanelInput = {
  mountElement: HTMLElement
  getSceneRenderer: () => SceneRenderer | undefined
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
  root: HTMLDivElement
  input: HTMLInputElement | HTMLTextAreaElement
  sync: (json: GeneratedEventJson) => void
}

const DEFAULT_EVENT_PROMPT =
  '크리스마스 이벤트를 만들어줘. 산타 NPC가 등장하고 대화하면 선물을 주도록 해줘.'

export const createLlmPanel = ({
  mountElement,
  getSceneRenderer
}: CreateLlmPanelInput): LlmPanelController => {
  const overlayRoot = document.createElement('section')
  const panel = document.createElement('div')
  const header = document.createElement('header')
  const title = document.createElement('h2')
  const hint = document.createElement('p')
  const topBar = document.createElement('div')
  const analyzeButton = document.createElement('button')
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
  const generatorSection = document.createElement('section')
  const generatorTitle = document.createElement('p')
  const promptLabel = document.createElement('label')
  const promptInput = document.createElement('textarea')
  const actionRow = document.createElement('div')
  const generateButton = document.createElement('button')
  const confirmButton = document.createElement('button')
  const codeButton = document.createElement('button')
  const applyButton = document.createElement('button')
  const jsonStatus = document.createElement('p')
  const fieldEditorTitle = document.createElement('p')
  const fieldEditorList = document.createElement('div')
  const fieldHint = document.createElement('p')
  const jsonPreviewTitle = document.createElement('p')
  const jsonPreview = document.createElement('pre')
  const codePreviewTitle = document.createElement('p')
  const codePreview = document.createElement('pre')
  const generatedEventTitle = document.createElement('p')
  const generatedEventHistory = document.createElement('ul')

  let isOpen = false
  let isAnalyzing = false
  let analysisResult: GameStructureAnalysisResult | undefined
  let currentDraft: GeneratedEventJson | undefined
  let confirmedDraft: GeneratedEventJson | undefined
  let generatedCode = ''
  let analysisProgress: GameStructureAnalysisProgress | undefined
  let fieldRows: FieldRowController[] = []

  overlayRoot.className = 'event-draft-panel llm-panel'
  panel.className = 'event-draft-panel__window llm-panel__window'
  header.className = 'event-draft-panel__header llm-panel__header'
  title.className = 'event-draft-panel__title'
  hint.className = 'event-draft-panel__hint'
  topBar.className = 'llm-panel__top-bar'
  analyzeButton.className = 'event-draft-panel__button'
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
  generatorSection.className = 'llm-panel__generator'
  generatorTitle.className = 'event-draft-panel__section-title'
  promptLabel.className = 'event-draft-panel__section-title'
  promptInput.className = 'event-draft-panel__textarea'
  actionRow.className = 'event-draft-panel__actions llm-panel__actions'
  generateButton.className = 'event-draft-panel__button'
  confirmButton.className = 'event-draft-panel__button'
  codeButton.className = 'event-draft-panel__button'
  applyButton.className = 'event-draft-panel__button'
  jsonStatus.className = 'event-draft-panel__status'
  fieldEditorTitle.className = 'event-draft-panel__section-title'
  fieldEditorList.className = 'llm-panel__field-list'
  fieldHint.className = 'event-draft-panel__hint'
  jsonPreviewTitle.className = 'event-draft-panel__section-title'
  jsonPreview.className = 'event-draft-panel__output'
  codePreviewTitle.className = 'event-draft-panel__section-title'
  codePreview.className = 'event-draft-panel__output'
  generatedEventTitle.className = 'event-draft-panel__section-title'
  generatedEventHistory.className = 'llm-panel__detail-list'

  title.textContent = 'LLM 분석 패널'
  hint.textContent =
    'L 키로 열고 닫는다. 먼저 게임 구조를 분석해서 GUS가 기준을 넘으면 이벤트 생성 영역이 활성화된다.'
  analyzeButton.type = 'button'
  analyzeButton.textContent = '분석 시작'
  toggleButton.type = 'button'
  toggleButton.textContent = '패널 닫기'
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
  generatorTitle.textContent = '이벤트 생성'
  fieldEditorTitle.textContent = 'JSON 내용 수정'
  fieldHint.textContent =
    '보이는 값을 바로 수정한다. dialogue는 줄 단위로 수정한다. 형식: speaker: text'
  jsonPreviewTitle.textContent = 'JSON 초안'
  codePreviewTitle.textContent = '코드 미리보기'
  generatedEventTitle.textContent = '생성 기록'

  progressWrap.append(progressBar, progressBarFill)
  gusCard.append(gusTitle, gusScore, gusMeta, gusThreshold, gusDetailList)
  actionRow.append(generateButton, confirmButton, codeButton, applyButton)
  generatorSection.append(
    generatorTitle,
    promptLabel,
    promptInput,
    actionRow,
    jsonStatus,
    fieldEditorTitle,
    fieldHint,
    fieldEditorList,
    jsonPreviewTitle,
    jsonPreview,
    codePreviewTitle,
    codePreview,
    generatedEventTitle,
    generatedEventHistory
  )
  panel.append(
    header,
    topBar,
    analysisSummary,
    progressWrap,
    progressLabel,
    gusCard,
    lockingNotice,
    analysisNotesTitle,
    analysisNotesList,
    profileTitle,
    profilePreview,
    generatorSection
  )
  overlayRoot.append(panel)
  mountElement.append(overlayRoot)

  const refresh = () => {
    overlayRoot.hidden = !isOpen
    analyzeButton.disabled = isAnalyzing
    toggleButton.textContent = isOpen ? '패널 닫기' : '패널 열기'
    renderAnalysisUi()
    renderGenerationGate()
    renderDraftUi()
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
      profilePreview.textContent = JSON.stringify(CURRENT_GAME_PROJECT_PROFILE, null, 2)
    }
  }

  const renderGenerationGate = () => {
    const analysisReady = analysisResult?.gus.status === 'passed'

    generatorSection.hidden = false
    lockingNotice.hidden = false
    generateButton.disabled = !analysisReady || isAnalyzing
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
      jsonPreview.textContent = ''
      codePreview.textContent = ''
      jsonStatus.textContent = analysisResult
        ? '자연어를 입력한 뒤 JSON 생성 버튼을 누르라.'
        : '분석이 먼저 필요하다.'
      fieldEditorList.replaceChildren()
      generatedEventHistory.replaceChildren()
      return
    }

    jsonPreview.textContent = JSON.stringify(currentDraft, null, 2)
    jsonStatus.textContent = getValidationSummary(currentDraft)

    if (fieldRows.length === 0) {
      rebuildFieldEditors(currentDraft)
    }

    codePreview.textContent = generatedCode || '코드 생성 버튼을 누르면 미리보기가 표시된다.'
  }

  const hasDraftValidationErrors = (): boolean =>
    currentDraft
      ? createGeneratedEventJsonValidationIssues(currentDraft, analysisResult?.profile ?? CURRENT_GAME_PROJECT_PROFILE).length > 0
      : true

  const getValidationSummary = (draft: GeneratedEventJson): string => {
    const issues = createGeneratedEventJsonValidationIssues(
      draft,
      analysisResult?.profile ?? CURRENT_GAME_PROJECT_PROFILE
    )

    if (issues.length === 0) {
      return 'JSON 검증 통과'
    }

    return `검증 실패: ${issues.map((issue) => `${issue.path} - ${issue.message}`).join(' / ')}`
  }

  const rebuildFieldEditors = (json: GeneratedEventJson) => {
    fieldEditorList.replaceChildren()
    fieldRows = createFieldRows({
      json,
      onChange: (nextJson) => {
        currentDraft = nextJson
        confirmedDraft = undefined
        generatedCode = ''
        refresh()
      }
    })

    for (const row of fieldRows) {
      fieldEditorList.append(row.root)
      row.sync(json)
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

    const rows: FieldRowController[] = descriptors.map((descriptor) =>
      createFieldRow({
        descriptor,
        json,
        onChange
      })
    )

    const dialogueRow = createDialogueRow({
      json,
      onChange
    })

    rows.push(dialogueRow)

    return rows
  }

  const createFieldRow = ({
    descriptor,
    json,
    onChange
  }: {
    descriptor: EditableFieldDescriptor
    json: GeneratedEventJson
    onChange: (nextJson: GeneratedEventJson) => void
  }): FieldRowController => {
    const root = document.createElement('div')
    const header = document.createElement('label')
    const labelText = document.createElement('span')
    const input =
      descriptor.kind === 'textarea'
        ? document.createElement('textarea')
        : document.createElement('input')

    root.className = 'llm-panel__field-row'
    header.className = 'llm-panel__field-toggle'
    labelText.textContent = descriptor.label

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
    }

    header.append(labelText)
    root.append(header, input)

    return {
      root,
      input,
      sync: (nextJson) => {
        const nextValue = descriptor.getValue(nextJson)

        if (input instanceof HTMLInputElement) {
          input.value = String(nextValue)
        } else {
          input.value = String(nextValue)
        }
      }
    }
  }

  const createDialogueRow = ({
    json,
    onChange
  }: {
    json: GeneratedEventJson
    onChange: (nextJson: GeneratedEventJson) => void
  }): FieldRowController => {
    const root = document.createElement('div')
    const header = document.createElement('label')
    const labelText = document.createElement('span')
    const textarea = document.createElement('textarea')

    root.className = 'llm-panel__field-row'
    header.className = 'llm-panel__field-toggle'
    labelText.textContent = 'dialogue'
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

    header.append(labelText)
    root.append(header, textarea)

    return {
      root,
      input: textarea,
      sync: (nextJson) => {
        textarea.value = formatDialogueText(nextJson)
      }
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
        onProgress: (progress) => {
          analysisProgress = progress
          refresh()
        }
      })
    } finally {
      isAnalyzing = false
      analysisProgress = undefined
      refresh()
    }
  }

  const runMockGeneration = () => {
    if (!analysisResult) {
      return
    }

    const generateResult = generateMockEventJsonDraft(
      promptInput.value,
      analysisResult.profile
    )
    currentDraft = generateResult.eventJson
    confirmedDraft = undefined
    generatedCode = ''
    rebuildFieldEditors(currentDraft)
    generatedEventHistory.prepend(createEventHistoryItem('JSON 초안 생성 완료'))
    refresh()
  }

  const runConfirm = () => {
    if (!currentDraft || !analysisResult) {
      return
    }

    confirmedDraft = structuredClone(currentDraft)
    generatedEventHistory.prepend(createEventHistoryItem('JSON 저장/확정 완료'))
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
    refresh()
  }

  const open = () => {
    isOpen = true
    refresh()

    if (!analysisResult && !isAnalyzing) {
      void runAnalysis()
    }
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

  analyzeButton.addEventListener('click', () => {
    void runAnalysis()
  })
  toggleButton.addEventListener('click', toggle)
  generateButton.addEventListener('click', runMockGeneration)
  confirmButton.addEventListener('click', runConfirm)
  codeButton.addEventListener('click', runCodeGeneration)
  applyButton.addEventListener('click', runApply)
  promptInput.addEventListener('keydown', stopPropagationWhenEditing)
  promptInput.addEventListener('keyup', stopPropagationWhenEditing)

  topBar.append(analyzeButton, toggleButton)
  refresh()
  window.addEventListener('keydown', handlePanelKeydown)

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

const parseFieldValue = (
  kind: EditableFieldDescriptor['kind'],
  rawValue: string
): string | number | boolean => {
  if (kind === 'number') {
    return Number(rawValue)
  }

  return rawValue
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
