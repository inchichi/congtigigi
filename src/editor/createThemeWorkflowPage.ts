import {
  createUnifiedThemeValidationIssues,
  applyUnifiedThemeStyles,
  generateUnifiedThemeDirection,
  generateUnifiedThemeQuest,
  generateUnifiedThemeStyleTargets,
  generateUnifiedThemeRewardItem,
  applyUnifiedThemeRewardItem,
  type UnifiedThemeRewardItem,
  loadStyleCatalog,
  MY_SAMPLE_RPG_GAME_ID,
  MY_SAMPLE_RPG_THEME_SCHEMA_VERSION,
  type StyleCatalog,
  type UnifiedThemeDirection,
  type UnifiedThemePlan,
  type UnifiedThemeStyleTargetDraft
} from './unifiedThemePipeline'
import { QWEN_LOCAL_TOKEN } from './qwenGenerate'
import { loadGame, type GameFile } from './loadGame'
import type { GameEntity } from './gameAdapter'
import type { GeneratedQuestJson } from './questJsonSchema'
import { dryRunQuestApply } from './dryRunQuestApply'
import { convertGeneratedQuestToDefinition } from './questCodeGenerator'
import { replacePendingQuests } from './pendingQuests'
import { createBeforeAfterView } from './createBeforeAfterView'
import {
  appendThemeWorkLog,
  clearThemeWorkLog,
  loadThemeWorkLog,
  type ThemeWorkLogEntry
} from './themeWorkLog'

type ThemeStage = 0 | 1 | 2 | 3

type CreateThemeWorkflowPageInput = {
  mountElement: HTMLElement
  initialFiles: GameFile[]
  initialPrompt?: string
}

const el = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string
): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag)
  node.className = className
  if (text !== undefined) {
    node.textContent = text
  }
  return node
}

const PREVIEW =
  'rounded-2xl border border-[#d9a85c]/25 bg-[#19191b] p-4 shadow-[0_12px_40px_rgba(0,0,0,0.2)]'
const SECONDARY_BUTTON =
  'rounded-xl border border-[#d9a85c]/35 bg-[#29292c] px-4 py-3 text-[13px] font-semibold text-[#e8d5a5] transition hover:border-[#d9a85c] hover:bg-[#343438] disabled:cursor-not-allowed disabled:opacity-45'
const PRIMARY_BUTTON =
  'rounded-xl border-2 border-[#9c6b2c] bg-gradient-to-b from-[#e8b96a] to-[#c9883a] px-5 py-3 text-[13px] font-bold text-[#241608] shadow-[0_0_18px_rgba(225,178,100,0.25)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45'

const jsonBlock = (value: unknown, className = ''): HTMLElement => {
  const block = el(
    'pre',
    `max-h-[360px] overflow-auto whitespace-pre-wrap break-words rounded-xl border border-[#d9a85c]/15 bg-[#121214] p-3 text-[11px] leading-[1.65] text-[#d4d4d4] ${className}`
  )
  block.textContent = JSON.stringify(value, null, 2)
  return block
}

const makeTag = (text: string, active = false): HTMLElement =>
  el(
    'span',
    active
      ? 'rounded-full border border-[#d9a85c]/60 bg-[#d9a85c]/15 px-2.5 py-1 text-[10px] text-[#f0ca7b]'
      : 'rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-1 text-[10px] text-[#9d9d9d]',
    text
  )

const createSectionTitle = (eyebrow: string, title: string, detail: string): HTMLElement => {
  const wrap = el('div', 'flex flex-col gap-1')
  wrap.append(
    el('div', 'text-[10px] font-semibold uppercase tracking-[0.18em] text-[#d9a85c]', eyebrow),
    el('h2', 'text-[20px] font-bold text-[#f1dfb5]', title),
    el('p', 'text-[12px] leading-relaxed text-[#9d9d9d]', detail)
  )
  return wrap
}

export const createThemeWorkflowPage = ({
  mountElement,
  initialFiles,
  initialPrompt = ''
}: CreateThemeWorkflowPageInput): void => {
  const game = loadGame(initialFiles)
  const profile = game.profile
  if (!profile) {
    mountElement.textContent = 'My Sample RPG 프로필을 불러오지 못했습니다.'
    return
  }

  let stage: ThemeStage = 0
  let direction: UnifiedThemeDirection | undefined
  let quest: GeneratedQuestJson | undefined
  let styleDraft: UnifiedThemeStyleTargetDraft | undefined
  let directionNote = ''
  let questNote = ''
  let styleNote = ''
  let rewardItem: UnifiedThemeRewardItem | undefined
  let rewardItemNote = ''
  let questCandidates: Array<{ quest: GeneratedQuestJson; note: string }> = []
  let catalog: StyleCatalog | undefined
  let isGenerating = false
  let isApplying = false
  let applied = false
  let statusText = '테마 설명을 입력하면 1단계부터 시작합니다.'
  let errorText = ''

  const shell = el('div', 'settings-game-font flex min-h-screen flex-col bg-[#171719] text-[#d4d4d4]')

  const header = el(
    'header',
    'flex flex-wrap items-center justify-between gap-3 border-b border-[#d9a85c]/20 bg-[#202022] px-5 py-4 lg:px-8'
  )
  const brand = el('div', 'flex items-center gap-3')
  const back = el('a', 'rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-[12px] text-[#c9c9c9] transition hover:border-[#d9a85c]/60 hover:text-[#f1dfb5]', '← 에디터') as HTMLAnchorElement
  back.href = '/editor.html'
  const brandText = el('div', 'flex flex-col gap-0.5')
  brandText.append(
    el('div', 'text-[18px] font-bold text-[#f1dfb5]', '테마 통합 작업실'),
    el('div', 'text-[11px] text-[#8f8f92]', 'Qwen 기획 · 퀘스트 설계 · FLUX 스타일 적용')
  )
  brand.append(back, brandText)
  const badges = el('div', 'flex flex-wrap items-center gap-2')
  const workLogButton = el(
    'button',
    'rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-[12px] text-[#c9c9c9] transition hover:border-[#d9a85c]/60 hover:text-[#f1dfb5]',
    '📜 작업 로그'
  ) as HTMLButtonElement
  workLogButton.type = 'button'
  badges.append(makeTag('My Sample RPG', true), makeTag('Qwen 연결됨', true), makeTag('단계별 승인'), workLogButton)
  header.append(brand, badges)

  const steps = [
    { label: '테마 방향', detail: '분위기와 아트 디렉션' },
    { label: '퀘스트', detail: '실행 가능한 목표' },
    { label: '스타일 대상', detail: 'FLUX 변경 후보' },
    { label: '최종 적용', detail: '검증 후 게임 반영' }
  ]
  const stepBar = el('div', 'grid grid-cols-2 gap-2 border-b border-[#d9a85c]/15 bg-[#1d1d1f] px-5 py-3 md:grid-cols-4 lg:px-8')
  const stepItems = steps.map((step, index) => {
    const item = el('div', 'flex items-center gap-2 rounded-xl border border-white/5 bg-white/[0.025] px-3 py-2')
    const number = el('span', 'flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[#2c2c30] text-[12px] text-[#9d9d9d]', String(index + 1))
    const text = el('div', 'min-w-0')
    text.append(el('div', 'truncate text-[12px] font-semibold text-[#bdbdbf]', step.label), el('div', 'truncate text-[10px] text-[#77777b]', step.detail))
    item.append(number, text)
    stepBar.append(item)
    return { item, number, text }
  })

  const promptPanel = el('section', `${PREVIEW} flex min-h-0 flex-col gap-4`)
  promptPanel.append(createSectionTitle('INPUT', '무엇을 만들까요?', '한 문장으로 세계의 테마와 원하는 변화를 설명하세요. 단계마다 결과를 확인한 뒤 다음 단계로 넘어갑니다.'))
  const promptInput = el('textarea', 'min-h-[180px] w-full resize-y rounded-xl border-2 border-[#dca14b]/45 bg-[#111113] px-4 py-3 text-[13px] leading-[1.7] text-[#e4e4e6] outline-none placeholder:text-[#737377] focus:border-[#d09a4c] focus:shadow-[0_0_18px_rgba(208,154,76,0.18)]') as HTMLTextAreaElement
  promptInput.placeholder = '예) 보랏빛 달빛이 비치는 어두운 마을로 바꾸고, 마법사가 동굴의 달빛석을 되찾는 퀘스트를 의뢰하게 해줘.'
  promptInput.value = initialPrompt
  const promptLabel = el('label', 'flex flex-col gap-2')
  promptLabel.append(el('div', 'text-[11px] font-semibold text-[#b59458]', '자연어 테마 프롬프트'), promptInput)

  const giverLabel = el('label', 'flex flex-col gap-2')
  const giverSelect = el('select', 'w-full rounded-xl border border-white/10 bg-[#252527] px-3 py-3 text-[12px] text-[#d4d4d4] outline-none focus:border-[#d09a4c]') as HTMLSelectElement
  giverSelect.append(new Option('자동 선택 — Qwen이 현재 게임 NPC 중 선택', ''))
  profile.npcs.forEach((npc) => giverSelect.append(new Option(`${npc.name} (${npc.id})`, npc.id)))
  giverLabel.append(el('div', 'text-[11px] font-semibold text-[#b59458]', '퀘스트 기버'), giverSelect)

  const rewardItemLabel = el('label', 'flex cursor-pointer items-center gap-2 rounded-xl border border-white/8 bg-white/[0.025] px-3 py-2.5')
  const rewardItemCheckbox = el('input', 'h-4 w-4 accent-[#d9a85c]') as HTMLInputElement
  rewardItemCheckbox.type = 'checkbox'
  rewardItemLabel.append(
    rewardItemCheckbox,
    el('span', 'text-[12px] text-[#cbb27b]', '🗡 신규 보상 아이템도 생성 (Qwen 데이터 + FLUX 아이콘)')
  )

  const quickPrompts = el('div', 'flex flex-wrap gap-2')
  const quickPromptValues = [
    '보랏빛 달빛이 비치는 다크 판타지 마을과 달빛석을 되찾는 퀘스트',
    '눈보라가 몰아치는 겨울 마을과 사냥터의 불씨를 지키는 퀘스트',
    '붉은 노을과 고대 유적의 저주를 풀어내는 모험'
  ]
  quickPromptValues.forEach((value) => {
    const button = el('button', SECONDARY_BUTTON, value) as HTMLButtonElement
    button.className = 'rounded-full border border-[#d9a85c]/25 bg-[#242426] px-3 py-2 text-[10px] text-[#cbb27b] transition hover:border-[#d9a85c] hover:bg-[#303034]'
    button.type = 'button'
    button.addEventListener('click', () => {
      promptInput.value = value
      promptInput.focus()
      render()
    })
    quickPrompts.append(button)
  })

  const catalogStatus = el('div', 'rounded-xl border border-white/8 bg-white/[0.025] p-3 text-[11px] leading-relaxed text-[#8f8f92]', 'FLUX 카탈로그를 확인하는 중…')
  const startButton = el('button', PRIMARY_BUTTON, '✨ 테마 방향 생성') as HTMLButtonElement
  startButton.type = 'button'
  promptPanel.append(promptLabel, giverLabel, rewardItemLabel, quickPrompts, catalogStatus, startButton)

  const status = el('div', 'rounded-xl border border-[#d9a85c]/20 bg-[#252527] px-3 py-2 text-[11px] leading-relaxed text-[#cdb783]', statusText)
  const error = el('div', 'hidden rounded-xl border border-[#e06c6c]/35 bg-[#3a2022] px-3 py-2 text-[11px] leading-relaxed text-[#f2a7a7]', '')

  const resultPanel = el('section', `${PREVIEW} flex min-h-0 flex-col gap-4`)
  resultPanel.append(createSectionTitle('RESULT BOARD', '단계별 결과 보드', '생성된 결과를 검토하고 “이 결과 사용” 또는 “새로 만들기”를 선택하세요.'))
  const currentStageLabel = el('div', 'rounded-lg bg-[#d9a85c]/10 px-3 py-2 text-[11px] font-semibold text-[#e0bd72]', '')
  const resultContent = el('div', 'min-h-[320px] flex-1')
  const actionRow = el('div', 'flex flex-wrap items-center gap-2 border-t border-white/8 pt-4')
  const beforeAfter = createBeforeAfterView()
  beforeAfter.view.className = 'hidden flex-col gap-3'
  resultPanel.append(currentStageLabel, status, error, resultContent, actionRow, beforeAfter.view)

  const layout = el('main', 'grid min-h-0 flex-1 gap-4 overflow-auto p-5 lg:grid-cols-[minmax(300px,0.75fr)_minmax(520px,1.5fr)] lg:p-8')
  layout.append(promptPanel, resultPanel)
  shell.append(header, stepBar, layout)
  mountElement.append(shell)

  const logOverlay = el('div', 'fixed inset-0 z-50 items-center justify-center bg-black/60 p-6')
  logOverlay.style.display = 'none'
  const logCard = el('div', 'flex max-h-[80vh] w-full max-w-[760px] flex-col gap-3 rounded-2xl border border-[#d9a85c]/30 bg-[#1b1b1d] p-5')
  const logHeader = el('div', 'flex items-center justify-between gap-2')
  const logHeaderButtons = el('div', 'flex items-center gap-2')
  const logClearButton = el('button', 'rounded-lg border border-[#e06c6c]/35 bg-[#2b1d1f] px-3 py-2 text-[11px] text-[#f2a7a7] transition hover:border-[#e06c6c]', '전체 삭제') as HTMLButtonElement
  logClearButton.type = 'button'
  const logCloseButton = el('button', SECONDARY_BUTTON, '닫기') as HTMLButtonElement
  logCloseButton.type = 'button'
  logHeaderButtons.append(logClearButton, logCloseButton)
  logHeader.append(el('div', 'text-[16px] font-bold text-[#f1dfb5]', '테마 작업 로그'), logHeaderButtons)
  const logList = el('div', 'flex flex-col gap-3 overflow-y-auto pr-1')
  logCard.append(logHeader, logList)
  logOverlay.append(logCard)
  shell.append(logOverlay)

  const selectedEntity = (): GameEntity | undefined => {
    if (!giverSelect.value) return undefined
    const npc = profile.npcs.find((candidate) => candidate.id === giverSelect.value)
    return npc
      ? { id: npc.id, name: npc.name, kind: 'npc', mapId: npc.map }
      : undefined
  }

  const buildPlan = (): UnifiedThemePlan | undefined =>
    direction && quest && styleDraft
      ? {
          schema_version: MY_SAMPLE_RPG_THEME_SCHEMA_VERSION,
          game_id: MY_SAMPLE_RPG_GAME_ID,
          theme: direction.theme,
          art_direction: direction.art_direction,
          quest,
          style_targets: styleDraft.style_targets
        }
      : undefined

  const setError = (message: string): void => {
    errorText = message
    error.textContent = message
  }

  const runDirection = async (): Promise<void> => {
    if (isGenerating || promptInput.value.trim().length === 0) return
    isGenerating = true
    errorText = ''
    setError('')
    statusText = 'Qwen이 테마 방향을 설계하는 중…'
    render()
    try {
      const { explanation, ...directionResult } = await generateUnifiedThemeDirection({
        apiKey: QWEN_LOCAL_TOKEN,
        userPrompt: promptInput.value
      })
      direction = directionResult
      directionNote = explanation?.trim() ?? ''
      questCandidates = []
      quest = undefined
      styleDraft = undefined
      questNote = ''
      styleNote = ''
      stage = 0
      statusText = '테마 방향을 확인하세요. 사용하면 퀘스트 단계로 넘어갑니다.'
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
      statusText = '테마 방향 생성에 실패했습니다.'
    } finally {
      isGenerating = false
      render()
    }
  }

  const runQuest = async (): Promise<void> => {
    if (isGenerating || !direction) return
    isGenerating = true
    errorText = ''
    statusText = 'Qwen이 승인된 방향에 맞는 퀘스트를 설계하는 중…'
    render()
    try {
      const base = {
        apiKey: QWEN_LOCAL_TOKEN,
        userPrompt: promptInput.value,
        profile,
        direction,
        entity: selectedEntity()
      }
      const [first, second] = await Promise.all([
        generateUnifiedThemeQuest({
          ...base,
          angle: 'Candidate A: propose the most direct, thematic quest for the request.'
        }),
        generateUnifiedThemeQuest({
          ...base,
          angle:
            'Candidate B: propose a clearly different alternative from candidate thinking — prefer a different objective type, location, or story angle.'
        })
      ])
      questCandidates = [first, second].map(({ explanation, ...candidate }) => ({
        quest: candidate,
        note: explanation?.trim() ?? ''
      }))
      quest = undefined
      questNote = ''
      styleDraft = undefined
      styleNote = ''
      rewardItem = undefined
      rewardItemNote = ''
      stage = 1
      statusText = '두 퀘스트 후보 중 하나를 선택하세요.'
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
      statusText = '퀘스트 생성에 실패했습니다.'
    } finally {
      isGenerating = false
      render()
    }
  }

  const runStyles = async (): Promise<void> => {
    if (isGenerating || !direction || !quest || !catalog) return
    isGenerating = true
    errorText = ''
    statusText = 'Qwen이 퀘스트와 테마에 맞는 FLUX 대상을 고르는 중…'
    render()
    try {
      const { explanation, ...styleResult } = await generateUnifiedThemeStyleTargets({
        apiKey: QWEN_LOCAL_TOKEN,
        userPrompt: promptInput.value,
        direction,
        quest,
        catalog
      })
      styleDraft = styleResult
      styleNote = explanation?.trim() ?? ''
      stage = 2
      statusText = 'FLUX 대상과 프롬프트를 확인하세요. 사용하면 최종 검토로 넘어갑니다.'
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
      statusText = 'FLUX 대상 생성에 실패했습니다.'
    } finally {
      isGenerating = false
      render()
    }
  }

  const questReadableText = (candidate: GeneratedQuestJson): string => {
    const giverName =
      profile.npcs.find((npc) => npc.id === candidate.giver_npc_id)?.name ?? candidate.giver_npc_id
    const objective = candidate.objectives[0]
    const rewardParts = [
      candidate.rewards.gold > 0 ? `골드 ${candidate.rewards.gold}` : '',
      candidate.rewards.experience > 0 ? `경험치 ${candidate.rewards.experience}` : '',
      ...candidate.rewards.items.map((item) => `${item.item_id} ×${item.quantity}`)
    ].filter(Boolean)
    return [
      `Q. ${candidate.title}`,
      `${giverName} NPC : "${candidate.start_dialogue_lines[0] ?? ''}"`,
      `내용 : ${candidate.request_text}`,
      `목표 : ${objective ? `${objective.label} ×${objective.required}` : '없음'}`,
      `안내 : ${candidate.guide_text}`,
      `보상 : ${rewardParts.join(' · ') || '없음'}`
    ].join('\n')
  }

  const selectQuestCandidate = async (index: number): Promise<void> => {
    const candidate = questCandidates[index]
    if (!candidate || isGenerating || isApplying || !direction) return
    quest = candidate.quest
    questNote = candidate.note
    styleDraft = undefined
    styleNote = ''
    rewardItem = undefined
    rewardItemNote = ''
    if (rewardItemCheckbox.checked && catalog) {
      isGenerating = true
      statusText = 'Qwen이 선택한 퀘스트의 보상 아이템을 설계하는 중…'
      render()
      try {
        const { explanation, ...itemResult } = await generateUnifiedThemeRewardItem({
          apiKey: QWEN_LOCAL_TOKEN,
          userPrompt: promptInput.value,
          direction,
          quest: candidate.quest,
          catalog
        })
        rewardItem = itemResult
        rewardItemNote = explanation?.trim() ?? ''
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught))
      } finally {
        isGenerating = false
      }
    }
    statusText = '퀘스트를 확인하세요. 사용하면 FLUX 대상 단계로 넘어갑니다.'
    render()
  }

  const acceptStage = (): void => {
    if (isGenerating) return
    if (stage === 0 && direction) void runQuest()
    else if (stage === 1 && quest) void runStyles()
    else if (stage === 2 && styleDraft) {
      stage = 3
      statusText = '최종 결과를 검증했습니다. 적용 전에 전체 설계를 확인하세요.'
      render()
    }
  }

  const regenerateStage = (): void => {
    if (stage === 0) void runDirection()
    else if (stage === 1) void runQuest()
    else if (stage === 2) void runStyles()
  }

  const resetWorkflow = (): void => {
    stage = 0
    direction = undefined
    questCandidates = []
    quest = undefined
    styleDraft = undefined
    directionNote = ''
    questNote = ''
    styleNote = ''
    rewardItem = undefined
    rewardItemNote = ''
    applied = false
    beforeAfter.view.className = 'hidden flex-col gap-3'
    statusText = '테마 설명을 입력하면 1단계부터 시작합니다.'
    errorText = ''
    render()
  }

  const downloadJson = (fileName: string, payload: unknown): void => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = fileName
    link.click()
    URL.revokeObjectURL(url)
  }

  const exportPlan = (): void => {
    const plan = buildPlan()
    if (!plan) return
    downloadJson(`${plan.theme || 'my-sample-rpg-theme'}.json`, plan)
  }

  const timestampToken = (): string =>
    new Date().toISOString().slice(0, 19).replace(/:/g, '').replace('T', '-')

  // 단계 결과를 프롬프트·기버와 함께 저장해 실행 간 결과 비교에 쓸 수 있게 한다.
  const downloadStageResult = (): void => {
    let token: string | undefined
    let result: unknown
    let explanation = ''
    if (stage === 0 && direction) {
      token = '1-direction'
      result = direction
      explanation = directionNote
    } else if (stage === 1 && quest) {
      token = '2-quest'
      result = quest
      explanation = questNote
    } else if (stage === 2 && styleDraft) {
      token = '3-style-targets'
      result = styleDraft
      explanation = styleNote
    }
    if (!token) return
    downloadJson(`theme-${token}-${timestampToken()}.json`, {
      stage: steps[stage].label,
      created_at: new Date().toISOString(),
      user_prompt: promptInput.value.trim(),
      giver_npc_id: selectedEntity()?.id ?? null,
      explanation,
      result
    })
  }

  const applyPlan = async (): Promise<void> => {
    const plan = buildPlan()
    if (!plan || !catalog || !quest) return
    const issues = createUnifiedThemeValidationIssues(plan, profile, catalog, selectedEntity())
    const dryRun = dryRunQuestApply(quest, profile, {
      selectedEntityId: selectedEntity()?.id
    })
    if (issues.length > 0 || !dryRun.ok) {
      setError([...issues.map((issue) => `${issue.path} - ${issue.message}`), ...dryRun.jsonIssues].join('\n'))
      statusText = '검증에 실패해 적용할 수 없습니다. 해당 단계로 돌아가 다시 생성하세요.'
      render()
      return
    }

    isApplying = true
    errorText = ''
    statusText = 'Qwen이 FLUX용 영어 프롬프트로 변환한 뒤 스타일을 적용하는 중…'
    render()
    try {
      const result = await applyUnifiedThemeStyles(plan, catalog)
      if (result.failed.length > 0) {
        throw new Error(result.failed.map((failure) => `${failure.targetRef}: ${failure.error}`).join('\n'))
      }
      let rewardIconPath: string | undefined
      if (rewardItem) {
        statusText = 'FLUX가 신규 아이템 아이콘을 생성하는 중… (첫 실행은 모델 로딩으로 수 분 걸릴 수 있음)'
        render()
        const iconResult = await applyUnifiedThemeRewardItem(rewardItem, plan)
        rewardIconPath = iconResult.iconPath
      }
      const definition = convertGeneratedQuestToDefinition(quest, profile)
      if (rewardItem) {
        definition.rewards.items = [
          ...definition.rewards.items,
          { id: rewardItem.item_id, label: rewardItem.label, quantity: 1 }
        ]
      }
      replacePendingQuests([definition])
      const logEntry: ThemeWorkLogEntry = {
        id: `${Date.now()}-${Math.floor(Math.random() * 1_000_000)}`,
        created_at: new Date().toISOString(),
        user_prompt: promptInput.value.trim(),
        giver_npc_id: selectedEntity()?.id ?? null,
        theme: plan.theme,
        art_direction: plan.art_direction,
        quest_summary: {
          quest_id: quest.quest_id,
          title: quest.title,
          giver_npc_id: quest.giver_npc_id,
          objective_label: quest.objectives[0]?.label ?? ''
        },
        style_targets: plan.style_targets,
        applied_targets: result.applied,
        explanations: { direction: directionNote, quest: questNote, styles: styleNote },
        ...(rewardItem && rewardIconPath
          ? { reward_item: { item_id: rewardItem.item_id, label: rewardItem.label, icon_path: rewardIconPath } }
          : {})
      }
      appendThemeWorkLog(logEntry)
      statusText = '적용 완료 — 변경된 에셋을 로컬 게임으로 동기화하는 중…'
      render()
      try {
        await fetch('/__sync-styled-assets', { method: 'POST' })
      } catch {
        // 동기화가 실패해도 서버 쪽 적용은 완료된 상태다. 에디터 재진입 시 다시 시도한다.
      }
      applied = true
      statusText = `적용 완료 — 퀘스트 저장 및 FLUX ${result.applied.length}개 대상 반영. 잠시 후 에디터 화면으로 이동합니다…`
      beforeAfter.view.className = 'flex flex-col gap-3'
      beforeAfter.refresh()
      window.setTimeout(() => {
        window.location.href = '/editor.html'
      }, 2000)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
      statusText = '적용에 실패했습니다.'
    } finally {
      isApplying = false
      render()
    }
  }

  const noteBlock = (text: string, className = ''): HTMLElement => {
    const box = el(
      'div',
      `rounded-xl border border-[#d9a85c]/25 bg-[#211d14] p-3 text-[12px] leading-relaxed text-[#e6d3a1] ${className}`
    )
    box.append(
      el('div', 'mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-[#b59458]', 'Qwen 설명'),
      el('div', 'whitespace-pre-wrap', text)
    )
    return box
  }

  const renderResult = (): void => {
    currentStageLabel.textContent = `현재 단계 ${stage + 1}/4 · ${steps[stage].label}`
    resultContent.replaceChildren()
    actionRow.replaceChildren()
    error.hidden = errorText.length === 0
    status.textContent = statusText
    startButton.disabled = isGenerating || isApplying || promptInput.value.trim().length === 0
    startButton.textContent = stage === 0 && !direction ? '✨ 테마 방향 생성' : '↻ 현재 단계 새로 만들기'

    stepItems.forEach(({ item, number, text }, index) => {
      const active = index === stage
      const completed = index < stage || (index === 3 && applied)
      item.className = `flex items-center gap-2 rounded-xl border px-3 py-2 ${active ? 'border-[#d9a85c]/65 bg-[#d9a85c]/10 step-pulse' : completed ? 'border-emerald-400/35 bg-emerald-400/[0.06]' : 'border-white/5 bg-white/[0.025]'}`
      number.className = `flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[12px] ${active ? 'bg-[#d9a85c] text-[#241608]' : completed ? 'bg-emerald-500/70 text-[#102015]' : 'bg-[#2c2c30] text-[#9d9d9d]'}`
      text.firstElementChild?.classList.toggle('text-[#f1dfb5]', active || completed)
    })

    if (stage === 0 && direction) {
      const card = el('div', 'flex flex-col gap-3')
      card.append(el('div', 'text-[13px] font-semibold text-[#e8d5a5]', direction.theme))
      if (directionNote) card.append(noteBlock(directionNote))
      card.append(jsonBlock(direction.art_direction))
      resultContent.append(card)
    } else if (stage === 1 && !quest && questCandidates.length > 0) {
      const grid = el('div', 'grid gap-3 md:grid-cols-2')
      questCandidates.forEach((candidate, index) => {
        const box = el('article', 'flex flex-col gap-2 rounded-xl border border-[#d9a85c]/20 bg-[#121214] p-3')
        box.append(
          el('div', 'text-[11px] font-semibold uppercase tracking-[0.14em] text-[#b59458]', `후보 ${index + 1}`),
          el('div', 'whitespace-pre-wrap rounded-lg bg-white/[0.03] p-2.5 text-[12px] leading-relaxed text-[#e3d5ae]', questReadableText(candidate.quest))
        )
        if (candidate.note) box.append(noteBlock(candidate.note))
        const detail = el('details', 'text-[11px]')
        detail.append(el('summary', 'cursor-pointer text-[#cbb27b]', 'JSON 보기'), jsonBlock(candidate.quest, 'mt-2'))
        box.append(detail)
        const pick = el('button', PRIMARY_BUTTON, '이 퀘스트 선택') as HTMLButtonElement
        pick.type = 'button'
        pick.disabled = isGenerating || isApplying
        pick.addEventListener('click', () => void selectQuestCandidate(index))
        box.append(pick)
        grid.append(box)
      })
      resultContent.append(grid)
    } else if (stage === 1 && quest) {
      const card = el('div', 'flex flex-col gap-3')
      card.append(
        el('div', 'whitespace-pre-wrap rounded-xl border border-[#d9a85c]/25 bg-white/[0.03] p-3 text-[13px] leading-relaxed text-[#e3d5ae]', questReadableText(quest))
      )
      if (questNote) card.append(noteBlock(questNote))
      if (rewardItem) {
        const itemBox = el('div', 'rounded-xl border border-[#d9a85c]/20 bg-[#121214] p-3')
        itemBox.append(
          el('div', 'mb-1 text-[12px] font-semibold text-[#e8d5a5]', `🗡 신규 보상 아이템 · ${rewardItem.label} (${rewardItem.item_id})`),
          el('div', 'whitespace-pre-wrap text-[11px] leading-relaxed text-[#9d9d9d]', `기준 아이콘: ${rewardItem.base_icon_ref}\n아이콘 프롬프트: ${rewardItem.icon_prompt}`)
        )
        if (rewardItemNote) itemBox.append(noteBlock(rewardItemNote, 'mt-2'))
        card.append(itemBox)
      }
      card.append(jsonBlock(quest))
      resultContent.append(card)
    } else if (stage === 2 && styleDraft) {
      const card = el('div', 'grid gap-3 md:grid-cols-2')
      if (styleNote) card.append(noteBlock(styleNote, 'md:col-span-2'))
      styleDraft.style_targets.forEach((target) => {
        const item = el('article', 'rounded-xl border border-[#d9a85c]/20 bg-[#121214] p-3')
        item.append(el('div', 'mb-2 text-[12px] font-semibold text-[#e8d5a5]', target.target_ref), el('div', 'mb-2 text-[11px] leading-relaxed text-[#c9c9c9]', target.prompt), makeTag(`강도 ${target.alpha}`))
        card.append(item)
      })
      resultContent.append(card)
    } else if (stage === 3) {
      const plan = buildPlan()
      if (plan) {
        const issues = catalog ? createUnifiedThemeValidationIssues(plan, profile, catalog, selectedEntity()) : []
        const dryRun = quest ? dryRunQuestApply(quest, profile, { selectedEntityId: selectedEntity()?.id }) : undefined
        const validation = el('div', `rounded-xl border p-3 text-[12px] leading-relaxed ${issues.length === 0 && dryRun?.ok ? 'border-emerald-400/30 bg-emerald-400/[0.06] text-emerald-200' : 'border-red-400/35 bg-red-400/[0.06] text-red-200'}`)
        validation.textContent = issues.length === 0 && dryRun?.ok ? '✓ 테마 JSON과 퀘스트 드라이런 검증을 통과했습니다.' : `검증 문제 ${issues.length + (dryRun?.jsonIssues.length ?? 0)}건`
        resultContent.append(validation)
        const stageNotes = [
          ['테마 방향', directionNote],
          ['퀘스트', questNote],
          ['스타일 대상', styleNote]
        ].filter(([, note]) => note.length > 0)
        if (stageNotes.length > 0) {
          resultContent.append(
            noteBlock(stageNotes.map(([label, note]) => `· ${label} — ${note}`).join('\n'))
          )
        }
        resultContent.append(jsonBlock(plan))
      }
    } else {
      resultContent.append(el('div', 'flex min-h-[320px] items-center justify-center rounded-xl border border-dashed border-white/10 text-center text-[12px] leading-relaxed text-[#77777b]', '왼쪽 프롬프트에 테마를 입력하고\n테마 방향 생성을 시작하세요.'))
    }

    if (stage === 1 && !quest && questCandidates.length > 0) {
      const regenerate = el('button', SECONDARY_BUTTON, '후보 새로 만들기') as HTMLButtonElement
      regenerate.type = 'button'
      regenerate.disabled = isGenerating || isApplying
      regenerate.addEventListener('click', regenerateStage)
      actionRow.append(regenerate)
    } else if (stage < 3 && ((stage === 0 && direction) || (stage === 1 && quest) || (stage === 2 && styleDraft))) {
      const accept = el('button', PRIMARY_BUTTON, stage === 0 ? '이 결과 사용 → 퀘스트 생성' : stage === 1 ? '이 결과 사용 → 스타일 대상 생성' : '이 결과 사용 → 최종 검토') as HTMLButtonElement
      accept.type = 'button'
      accept.disabled = isGenerating || isApplying
      accept.addEventListener('click', acceptStage)
      const regenerate = el('button', SECONDARY_BUTTON, '새로 만들기') as HTMLButtonElement
      regenerate.type = 'button'
      regenerate.disabled = isGenerating || isApplying
      regenerate.addEventListener('click', regenerateStage)
      const save = el('button', SECONDARY_BUTTON, '결과 JSON 저장') as HTMLButtonElement
      save.type = 'button'
      save.disabled = isGenerating || isApplying
      save.addEventListener('click', downloadStageResult)
      actionRow.append(accept, regenerate, save)
      if (stage === 1 && questCandidates.length > 1) {
        const back = el('button', SECONDARY_BUTTON, '다른 후보 선택') as HTMLButtonElement
        back.type = 'button'
        back.disabled = isGenerating || isApplying
        back.addEventListener('click', () => {
          quest = undefined
          questNote = ''
          rewardItem = undefined
          rewardItemNote = ''
          styleDraft = undefined
          styleNote = ''
          statusText = '두 퀘스트 후보 중 하나를 선택하세요.'
          render()
        })
        actionRow.append(back)
      }
    } else if (stage === 3 && buildPlan()) {
      const apply = el('button', PRIMARY_BUTTON, applied ? '✓ 적용 완료' : '게임에 최종 적용') as HTMLButtonElement
      apply.type = 'button'
      apply.disabled = isGenerating || isApplying || applied
      apply.addEventListener('click', () => void applyPlan())
      const exportButton = el('button', SECONDARY_BUTTON, 'JSON 내보내기') as HTMLButtonElement
      exportButton.type = 'button'
      exportButton.disabled = isApplying
      exportButton.addEventListener('click', exportPlan)
      const reset = el('button', SECONDARY_BUTTON, '처음부터 다시') as HTMLButtonElement
      reset.type = 'button'
      reset.disabled = isApplying
      reset.addEventListener('click', resetWorkflow)
      actionRow.append(apply, exportButton, reset)
    }
  }

  const render = (): void => {
    renderResult()
  }

  const renderWorkLog = (): void => {
    const entries = loadThemeWorkLog()
    logList.replaceChildren()
    if (entries.length === 0) {
      logList.append(
        el(
          'div',
          'whitespace-pre-wrap rounded-xl border border-dashed border-white/10 p-6 text-center text-[12px] leading-relaxed text-[#77777b]',
          '아직 적용 기록이 없습니다.\n게임에 최종 적용하면 기록이 남습니다.'
        )
      )
      return
    }
    entries.forEach((entry) => {
      const card = el('article', 'rounded-xl border border-[#d9a85c]/20 bg-[#121214] p-3')
      const top = el('div', 'mb-1 flex flex-wrap items-center justify-between gap-2')
      top.append(
        el('div', 'text-[13px] font-semibold text-[#e8d5a5]', entry.theme || '(제목 없음)'),
        el('div', 'text-[10px] text-[#8f8f92]', new Date(entry.created_at).toLocaleString('ko-KR'))
      )
      const meta = el(
        'div',
        'mb-2 whitespace-pre-wrap text-[11px] leading-relaxed text-[#9d9d9d]',
        `프롬프트: ${entry.user_prompt}\n퀘스트: ${entry.quest_summary.title} (기버 ${entry.quest_summary.giver_npc_id})\nFLUX 적용 ${entry.applied_targets.length}개: ${entry.applied_targets.join(', ')}`
      )
      const detail = el('details', 'text-[11px] text-[#9d9d9d]')
      detail.append(el('summary', 'cursor-pointer text-[#cbb27b]', '상세 JSON 보기'), jsonBlock(entry, 'mt-2'))
      const actions = el('div', 'mt-2 flex gap-1.5')
      const saveButton = el(
        'button',
        'rounded-lg border border-[#d9a85c]/30 bg-[#1c1c1e] px-2.5 py-1.5 text-[10px] text-[#cbb27b] transition hover:border-[#d9a85c]',
        'JSON 저장'
      ) as HTMLButtonElement
      saveButton.type = 'button'
      saveButton.addEventListener('click', () => downloadJson(`theme-log-${entry.id}.json`, entry))
      actions.append(saveButton)
      card.append(top, meta, detail, actions)
      logList.append(card)
    })
  }

  workLogButton.addEventListener('click', () => {
    renderWorkLog()
    logOverlay.style.display = 'flex'
  })
  logCloseButton.addEventListener('click', () => {
    logOverlay.style.display = 'none'
  })
  logClearButton.addEventListener('click', () => {
    clearThemeWorkLog()
    renderWorkLog()
  })
  logOverlay.addEventListener('click', (event) => {
    if (event.target === logOverlay) {
      logOverlay.style.display = 'none'
    }
  })

  startButton.addEventListener('click', () => {
    if (stage === 0 && !direction) void runDirection()
    else regenerateStage()
  })
  promptInput.addEventListener('input', render)
  giverSelect.addEventListener('change', () => {
    if (stage > 0) {
      statusText = '기버가 바뀌었습니다. 퀘스트 단계를 새로 만들어 주세요.'
    }
    render()
  })

  void loadStyleCatalog()
    .then((loadedCatalog) => {
      catalog = loadedCatalog
      catalogStatus.textContent = `FLUX 대상 준비 완료 · 정적 에셋 ${catalog.assets.length}개 · 추출 오브젝트 ${catalog.objects.length}개`
      render()
    })
    .catch(() => {
      catalogStatus.textContent = 'FLUX 카탈로그를 불러오지 못했습니다. 서버 연결을 확인하세요.'
    })

  render()
}
