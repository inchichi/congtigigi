import { openProjectDirectory } from './openProjectDirectory'
import { loadGame, type GameFile, type LoadedGame, type LoadedGameMap } from './loadGame'
import { analyzeGame, type GameAnalysis } from './analyzeGame'
import { extractTmxObjects, type TmxObject } from './tmxObjects'
import { readLocalStorage, writeLocalStorage } from './safeStorage'
import { ANTHROPIC_MODEL } from './anthropicGenerate'
import type { GameEntity, GenerationResult } from './gameAdapter'

// 하드코딩 어댑터가 엔티티를 못 찾은 미지의 게임을, LLM 분석이 찾은 editable 그룹으로 채운다.
const buildEntitiesFromAnalysis = (
  files: GameFile[],
  analysis: GameAnalysis
): LoadedGameMap[] => {
  const editableKindByGroup = new Map<string, string>()
  for (const entityGroup of analysis.entity_groups) {
    if (entityGroup.editable) {
      editableKindByGroup.set(entityGroup.group, entityGroup.kind)
    }
  }

  return files
    .filter((file) => file.name.endsWith('.tmx'))
    .map((file) => {
      const id = file.name.replace(/\.tmx$/u, '')
      // 한 맵이 깨졌다고 분석 기반 트리 재구성을 통째로 죽이지 않는다(loadGame과 동일한 격리).
      let objects: TmxObject[] = []
      try {
        objects = extractTmxObjects(file.text)
      } catch {
        // 파싱 실패 맵 → 엔티티 0개
      }
      const entities = objects
        .filter(
          (object) =>
            editableKindByGroup.has(object.group) && object.name.length > 0
        )
        .map((object) => ({
          id: `${object.group}-${object.id}`,
          name: object.name,
          kind: editableKindByGroup.get(object.group) ?? 'entity',
          mapId: id
        }))
      return { id, name: id, file: file.path, entities }
    })
}

type CreateEditorAppInput = {
  mountElement: HTMLElement
  initialFiles: GameFile[]
  gamePreviewUrl: string
}

const API_KEY_STORAGE_KEY = 'my-sample-rpg:anthropic-api-key'

const KIND_ICON: Record<string, string> = {
  npc: '👤',
  enemy: '👹',
  chest: '📦',
  loot: '💰'
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

const PRIMARY_BUTTON =
  'rounded-lg px-3.5 py-2 bg-indigo-500 text-white text-sm font-medium transition hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed'
const GHOST_BUTTON =
  'rounded-lg px-3 py-1.5 bg-white/5 text-zinc-300 text-sm transition hover:bg-white/10 hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed'
const ENTITY_BASE =
  'text-left rounded-md px-2 py-1.5 text-[0.8rem] text-zinc-400 transition hover:bg-white/5 hover:text-zinc-100'
const ENTITY_ACTIVE =
  'text-left rounded-md px-2 py-1.5 text-[0.8rem] bg-indigo-500/15 text-indigo-200 transition'

export const createEditorApp = ({
  mountElement,
  initialFiles,
  gamePreviewUrl
}: CreateEditorAppInput): void => {
  let game: LoadedGame = loadGame(initialFiles)
  let currentFiles: GameFile[] = initialFiles
  let apiKey = readLocalStorage(API_KEY_STORAGE_KEY) ?? ''
  let selectedEntity: GameEntity | undefined
  let currentResult: GenerationResult | undefined
  let currentAnalysis: GameAnalysis | undefined
  let isGenerating = false
  let isAnalyzing = false
  let entityButtons: Array<{ entity: GameEntity; node: HTMLButtonElement }> = []
  // 세션 내 생성 결과 누적(최신 우선, 최대 10개). 데모에서 여러 생성을 비교·재선택하려는 용도.
  const HISTORY_LIMIT = 10
  let history: Array<{ n: number; result: GenerationResult }> = []
  let historyCounter = 0

  // ---------- shell ----------
  const root = el('div', 'h-screen w-screen flex flex-col bg-zinc-950 text-zinc-100 overflow-hidden')

  const header = el('header', 'h-12 shrink-0 flex items-center justify-between px-4 border-b border-white/10 bg-zinc-900/60')
  const brand = el('div', 'flex items-center gap-2')
  brand.append(
    el('span', 'w-2.5 h-2.5 rounded-full bg-indigo-500'),
    el('span', 'text-sm font-semibold tracking-tight', 'Scenario Editor'),
    el('span', 'text-xs text-zinc-600', '·')
  )
  const gameLabel = el('span', 'text-xs text-zinc-400', game.adapter.name)
  brand.append(gameLabel)
  // 데모에서 어떤 LLM을 쓰는지 한눈에 보이게 모델 배지를 둔다.
  brand.append(
    el(
      'span',
      'text-[0.65rem] rounded-full px-2 py-0.5 bg-indigo-500/10 text-indigo-300 border border-indigo-500/20',
      `Claude · ${ANTHROPIC_MODEL}`
    )
  )
  const connection = el('div', 'flex items-center gap-2 text-xs text-zinc-400')
  const connectionDot = el('span', 'w-2 h-2 rounded-full bg-zinc-600')
  const connectionLabel = el('span', '', '게임 로딩...')
  connection.append(connectionDot, connectionLabel)
  header.append(brand, connection)

  const body = el('div', 'flex-1 grid grid-cols-[240px_1fr_minmax(360px,38%)] min-h-0')

  // ---------- left: project tree ----------
  const tree = el('aside', 'border-r border-white/10 flex flex-col min-h-0')
  const treeHeader = el('div', 'p-3 border-b border-white/10 flex flex-col gap-2')
  const openButton = el('button', 'rounded-lg px-3 py-2 bg-white/5 text-sm text-zinc-200 text-left transition hover:bg-white/10', '📂 게임 폴더 열기') as HTMLButtonElement
  openButton.type = 'button'
  const analyzeButton = el('button', 'rounded-lg px-3 py-2 bg-indigo-500/10 text-sm text-indigo-200 text-left transition hover:bg-indigo-500/20 disabled:opacity-50', '🔍 LLM 게임 분석') as HTMLButtonElement
  analyzeButton.type = 'button'
  const resetButton = el('button', 'rounded-lg px-3 py-1.5 bg-white/5 text-xs text-zinc-400 text-left transition hover:bg-white/10 hover:text-zinc-200', '🏠 내 게임으로 복귀') as HTMLButtonElement
  resetButton.type = 'button'
  treeHeader.append(
    el('div', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', '프로젝트'),
    openButton,
    analyzeButton,
    resetButton
  )
  const treeList = el('div', 'flex-1 overflow-auto p-3 flex flex-col gap-3')
  tree.append(treeHeader, treeList)

  // ---------- center: generation ----------
  const center = el('main', 'overflow-auto p-5 flex flex-col gap-4')
  const targetLine = el('div', 'text-sm text-zinc-400')
  const analysisPanel = el('div', 'rounded-lg border border-indigo-500/20 bg-indigo-500/5 p-3 flex flex-col gap-1')
  analysisPanel.hidden = true
  const supportNote = el('div', 'rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200')

  const apiKeyField = el('label', 'flex flex-col gap-1.5')
  apiKeyField.append(el('span', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', 'Anthropic (Claude) API 키'))
  const apiKeyInput = el('input', 'w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/30') as HTMLInputElement
  apiKeyInput.type = 'password'
  apiKeyInput.placeholder = 'sk-ant-...'
  apiKeyInput.autocomplete = 'off'
  apiKeyInput.value = apiKey
  apiKeyField.append(apiKeyInput)

  const promptField = el('label', 'flex flex-col gap-1.5')
  promptField.append(el('span', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', '자연어 프롬프트  ·  ⌘/Ctrl+Enter로 생성'))
  const promptInput = el('textarea', 'w-full min-h-[96px] rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-zinc-100 outline-none transition resize-y focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/30') as HTMLTextAreaElement
  promptInput.placeholder = '예: 대장장이가 새로 만든 검을 자랑하는 대화'
  promptField.append(promptInput)

  const actions = el('div', 'flex items-center gap-2')
  const generateButton = el('button', PRIMARY_BUTTON, '생성') as HTMLButtonElement
  generateButton.type = 'button'
  const applyButton = el('button', GHOST_BUTTON, '게임에 적용') as HTMLButtonElement
  applyButton.type = 'button'
  const copyButton = el('button', GHOST_BUTTON, '⧉ 복사') as HTMLButtonElement
  copyButton.type = 'button'
  const exportButton = el('button', GHOST_BUTTON, '↓ 내보내기') as HTMLButtonElement
  exportButton.type = 'button'
  actions.append(generateButton, applyButton, copyButton, exportButton)

  const status = el('div', 'text-sm text-zinc-400 min-h-[1.25rem]')
  const validationLine = el('div', 'text-xs')
  validationLine.hidden = true

  const resultWrap = el('div', 'flex flex-col gap-1.5')
  resultWrap.append(el('span', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', '생성 결과'))
  const result = el('pre', 'm-0 max-h-[40vh] overflow-auto rounded-lg border border-white/10 bg-black/40 p-3 text-[0.72rem] leading-relaxed text-zinc-300 whitespace-pre-wrap break-words')
  resultWrap.append(result)

  const historyWrap = el('div', 'flex flex-col gap-1.5')
  historyWrap.hidden = true
  const historyHeader = el('div', 'flex items-center justify-between')
  historyHeader.append(
    el('span', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', '생성 히스토리')
  )
  const clearHistoryButton = el('button', 'text-[0.7rem] text-zinc-500 transition hover:text-zinc-300', '비우기') as HTMLButtonElement
  clearHistoryButton.type = 'button'
  historyHeader.append(clearHistoryButton)
  const historyList = el('div', 'flex flex-col gap-1')
  historyWrap.append(historyHeader, historyList)

  center.append(targetLine, analysisPanel, supportNote, apiKeyField, promptField, actions, status, validationLine, resultWrap, historyWrap)

  // ---------- right: live game preview ----------
  const preview = el('section', 'border-l border-white/10 flex flex-col min-w-0')
  const previewBar = el('div', 'h-9 shrink-0 flex items-center justify-between px-3 border-b border-white/10 bg-zinc-900/40')
  previewBar.append(el('span', 'text-xs text-zinc-400', '🎮 라이브 게임 (실제 게임 실행 중)'))
  const previewActions = el('div', 'flex items-center gap-3')
  const popoutButton = el('button', 'text-xs text-zinc-400 transition hover:text-zinc-100', '↗ 새 창') as HTMLButtonElement
  popoutButton.type = 'button'
  const reloadButton = el('button', 'text-xs text-zinc-400 transition hover:text-zinc-100', '↻ 새로고침') as HTMLButtonElement
  reloadButton.type = 'button'
  previewActions.append(popoutButton, reloadButton)
  previewBar.append(previewActions)
  const iframe = el('iframe', 'flex-1 w-full border-0 bg-black') as HTMLIFrameElement
  iframe.src = gamePreviewUrl
  iframe.title = '게임 프리뷰'
  iframe.addEventListener('load', () => {
    connectionDot.className = 'w-2 h-2 rounded-full bg-emerald-400'
    connectionLabel.textContent = '게임 연결됨'
  })
  preview.append(previewBar, iframe)

  body.append(tree, center, preview)
  root.append(header, body)
  mountElement.append(root)

  // ---------- behavior ----------
  const setStatus = (message: string): void => {
    status.textContent = message
  }

  // 파싱 실패한 맵이 있으면 상태 메시지 끝에 붙일 경고(없으면 빈 문자열). loadGame이 throw 대신
  // game.parseErrors로 모아주므로, 에디터가 통째로 안 뜨는 일 없이 실패를 사용자에게 알린다.
  const parseErrorNote = (): string =>
    game.parseErrors.length > 0
      ? ` · ⚠️ 파싱 실패 맵 ${game.parseErrors.length}개: ${game.parseErrors.join(', ')}`
      : ''

  const renderAnalysis = (): void => {
    if (!currentAnalysis) {
      analysisPanel.hidden = true
      return
    }

    analysisPanel.hidden = false
    const analysis = currentAnalysis
    analysisPanel.replaceChildren(
      el('div', 'text-[0.7rem] uppercase tracking-wider text-indigo-300 font-medium', '🔍 LLM 게임 분석'),
      el('div', 'text-sm text-zinc-100 font-medium', `${analysis.game_name} · ${analysis.engine}`),
      el('div', 'text-xs text-zinc-400', `콘텐츠 모델: ${analysis.content_model}`),
      el('div', 'text-xs text-zinc-400', `적용 전략: ${analysis.apply_strategy}`),
      ...analysis.entity_groups.map((entityGroup) =>
        el(
          'div',
          'text-xs text-zinc-500',
          `• ${entityGroup.group} → ${entityGroup.kind}${entityGroup.editable ? ' (편집 가능)' : ''}`
        )
      )
    )
  }

  const runAnalyze = async (): Promise<void> => {
    if (isAnalyzing) {
      return
    }

    if (apiKey.trim().length === 0) {
      setStatus('분석하려면 먼저 Claude(Anthropic) API 키를 입력하세요.')
      return
    }

    isAnalyzing = true
    analyzeButton.disabled = true
    analyzeButton.textContent = '분석 중...'
    setStatus('LLM이 게임을 분석 중...')

    const filesAtStart = currentFiles
    try {
      const analysis = await analyzeGame({ apiKey: apiKey.trim(), files: filesAtStart })
      // 분석 중 다른 프로젝트를 열었으면 이 결과는 버린다(레이스 방지).
      if (currentFiles !== filesAtStart) {
        return
      }
      currentAnalysis = analysis
      // 하드코딩 어댑터가 엔티티를 못 찾았으면(미지의 게임), 분석 결과로 트리를 채운다.
      const totalEntities = game.maps.reduce((sum, map) => sum + map.entities.length, 0)
      if (totalEntities === 0) {
        game = { ...game, maps: buildEntitiesFromAnalysis(filesAtStart, analysis) }
        // 트리를 새 엔티티로 갈아끼우므로, 이전 대상의 생성 결과·히스토리는 모두 무효 처리한다.
        selectedEntity = undefined
        currentResult = undefined
        history = []
        historyCounter = 0
        renderTree()
      }
      renderAnalysis()
      render()
      setStatus(`분석 완료: ${analysis.game_name} (${analysis.engine})`)
    } catch (error) {
      setStatus(`분석 실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isAnalyzing = false
      analyzeButton.disabled = false
      analyzeButton.textContent = '🔍 LLM 게임 분석'
    }
  }

  const renderTree = (): void => {
    entityButtons = []
    const groups: HTMLElement[] = []

    for (const map of game.maps) {
      if (map.entities.length === 0) {
        continue
      }

      const group = el('div', 'flex flex-col gap-1')
      group.append(el('div', 'text-xs text-zinc-300 font-medium px-1', `🗺 ${map.name}`))

      for (const entity of map.entities) {
        const node = el('button', ENTITY_BASE, `${KIND_ICON[entity.kind] ?? '•'} ${entity.name}`) as HTMLButtonElement
        node.type = 'button'
        node.addEventListener('click', () => {
          selectedEntity = entity
          // 대상을 바꾸면 이전 생성 결과는 무효 — 새로 생성하게 한다.
          currentResult = undefined
          render()
        })
        entityButtons.push({ entity, node })
        group.append(node)
      }

      groups.push(group)
    }

    if (groups.length === 0) {
      groups.push(el('div', 'text-xs text-zinc-500 leading-relaxed', '로드된 엔티티가 없습니다. "게임 폴더 열기"로 프로젝트를 여세요.'))
    }

    treeList.replaceChildren(...groups)
  }

  const renderHistory = (): void => {
    if (history.length === 0) {
      historyWrap.hidden = true
      return
    }

    historyWrap.hidden = false
    historyList.replaceChildren(
      ...history.map((entry) => {
        const active = entry.result === currentResult
        const node = el(
          'button',
          active
            ? 'text-left rounded-md px-2.5 py-1.5 text-xs bg-indigo-500/15 text-indigo-200 transition'
            : 'text-left rounded-md px-2.5 py-1.5 text-xs text-zinc-400 transition hover:bg-white/5 hover:text-zinc-100'
        ) as HTMLButtonElement
        node.type = 'button'
        const mark = entry.result.issues.length === 0 ? '✅' : '⚠️'
        // 라벨은 LLM/열린 파일에서 온 임의 값이라 textContent로만 넣는다(주입/깨짐 방지).
        node.textContent = `#${entry.n} ${mark} ${entry.result.label}`
        node.addEventListener('click', () => {
          currentResult = entry.result
          render()
        })
        return node
      })
    )
  }

  function render(): void {
    gameLabel.textContent = game.adapter.name

    if (game.adapter.supportsApply) {
      supportNote.hidden = true
    } else {
      supportNote.hidden = false
      supportNote.textContent = `${game.adapter.name}: 생성은 되지만 라이브 적용은 아직 지원되지 않습니다 (Stage 3). 결과는 미리보기로 확인하세요.`
    }

    // 엔티티 이름/맵은 열린 TMX에서 온 임의 값이므로 textContent로만 넣는다(주입/깨짐 방지).
    if (selectedEntity) {
      targetLine.replaceChildren(
        document.createTextNode('대상: '),
        el('span', 'text-indigo-300 font-medium', selectedEntity.name),
        document.createTextNode(' '),
        el('span', 'text-zinc-500', `(${selectedEntity.kind} · ${selectedEntity.mapId})`)
      )
    } else {
      targetLine.replaceChildren(
        el('span', 'text-zinc-500', '왼쪽에서 엔티티를 선택하면 그 대상으로 생성합니다.')
      )
    }

    for (const { entity, node } of entityButtons) {
      node.className = entity.id === selectedEntity?.id ? ENTITY_ACTIVE : ENTITY_BASE
    }

    if (!currentResult) {
      validationLine.hidden = true
    } else if (currentResult.issues.length === 0) {
      validationLine.hidden = false
      validationLine.className = 'text-xs text-emerald-300'
      validationLine.replaceChildren(
        document.createTextNode('✅ Validator (생성과 분리된 자동 검증): 통과')
      )
    } else {
      validationLine.hidden = false
      validationLine.className = 'text-xs text-amber-300 flex flex-col gap-0.5'
      // 이슈 문자열은 Validator가 만든 값이지만 안전하게 textContent(el)로만 넣는다.
      validationLine.replaceChildren(
        el('div', 'font-medium', `⚠️ Validator (생성과 분리된 자동 검증): ${currentResult.issues.length}건`),
        ...currentResult.issues.map((issue) => el('div', 'pl-3 text-amber-300/80', `• ${issue}`))
      )
    }

    generateButton.textContent = isGenerating ? '생성 중...' : '생성'
    generateButton.disabled =
      isGenerating || apiKey.trim().length === 0 || promptInput.value.trim().length === 0
    applyButton.disabled =
      isGenerating || !currentResult?.apply || (currentResult?.issues.length ?? 0) > 0
    copyButton.disabled = !currentResult || isGenerating
    exportButton.disabled = !currentResult || isGenerating
    result.textContent = currentResult ? currentResult.preview : '생성 결과가 여기에 표시됩니다.'
    renderHistory()
  }

  const runGenerate = async (): Promise<void> => {
    if (isGenerating) {
      return
    }

    if (apiKey.trim().length === 0) {
      setStatus('먼저 Claude(Anthropic) API 키를 입력하세요.')
      return
    }

    if (promptInput.value.trim().length === 0) {
      setStatus('생성할 내용을 자연어 프롬프트에 입력하세요.')
      return
    }

    isGenerating = true
    setStatus(`${game.adapter.name}로 생성 중...`)
    render()

    try {
      currentResult = await game.adapter.generate({
        apiKey: apiKey.trim(),
        userPrompt: promptInput.value,
        entity: selectedEntity,
        profile: game.profile,
        gameContext: currentAnalysis
          ? `${currentAnalysis.game_name} (${currentAnalysis.engine}). 콘텐츠 모델: ${currentAnalysis.content_model}`
          : undefined
      })
      historyCounter += 1
      history = [{ n: historyCounter, result: currentResult }, ...history].slice(0, HISTORY_LIMIT)
      setStatus(`생성 완료: ${currentResult.label}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setStatus(`생성 실패: ${message}`)
    } finally {
      isGenerating = false
      render()
    }
  }

  const runApply = (): void => {
    if (!currentResult?.apply) {
      return
    }

    // apply()는 localStorage 저장을 동반해 실패할 수 있다. 조용히 죽지 않고 상태로 알린다.
    try {
      currentResult.apply()
      setStatus('게임에 적용됨 — 오른쪽 라이브 프리뷰에 즉시 반영됩니다.')
    } catch (error) {
      setStatus(`적용 실패: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const runCopy = async (): Promise<void> => {
    if (!currentResult) {
      return
    }

    // clipboard API는 비보안 컨텍스트·권한 거부에서 없거나 reject될 수 있어 방어한다.
    try {
      await navigator.clipboard.writeText(currentResult.preview)
      setStatus('생성 결과를 클립보드에 복사했습니다.')
    } catch {
      setStatus('클립보드 복사에 실패했습니다(브라우저 권한/보안 컨텍스트 확인).')
    }
  }

  const runClearHistory = (): void => {
    history = []
    historyCounter = 0
    renderHistory()
    setStatus('생성 히스토리를 비웠습니다.')
  }

  const runExport = (): void => {
    if (!currentResult) {
      return
    }

    const fileName = `${currentResult.label || 'generated'}.json`
    const blob = new Blob([currentResult.preview], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = fileName
    link.click()
    URL.revokeObjectURL(url)
    setStatus(`내보냄: ${fileName}`)
  }

  const runOpenProject = async (): Promise<void> => {
    try {
      const files = await openProjectDirectory()
      const loaded = loadGame(files)

      if (loaded.maps.length === 0) {
        setStatus('선택한 폴더에서 .tmx 맵을 찾지 못했습니다.')
        return
      }

      game = loaded
      currentFiles = files
      selectedEntity = undefined
      currentResult = undefined
      currentAnalysis = undefined
      history = []
      historyCounter = 0
      renderTree()
      renderAnalysis()
      render()
      const entityCount = game.maps.reduce((sum, map) => sum + map.entities.length, 0)
      setStatus(
        `프로젝트 로드: ${game.adapter.name} · 맵 ${game.maps.length}개 · 엔티티 ${entityCount}개${parseErrorNote()}`
      )
      // 토큰이 있으면 LLM이 이 게임을 자동 분석한다(네 아이디어: 열면 LLM이 이해).
      if (apiKey.trim().length > 0) {
        void runAnalyze()
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      setStatus(error instanceof Error ? error.message : String(error))
    }
  }

  const runReset = (): void => {
    game = loadGame(initialFiles)
    currentFiles = initialFiles
    selectedEntity = undefined
    currentResult = undefined
    currentAnalysis = undefined
    history = []
    historyCounter = 0
    renderTree()
    renderAnalysis()
    render()
    setStatus(`내 게임으로 복귀했습니다.${parseErrorNote()}`)
  }

  apiKeyInput.addEventListener('input', () => {
    apiKey = apiKeyInput.value
    // 저장이 막혀도(프라이빗 모드 등) 입력·생성 흐름은 끊기지 않게 한다. 키는 메모리에 유지된다.
    const persisted = writeLocalStorage(API_KEY_STORAGE_KEY, apiKey)
    render()
    if (!persisted && apiKey.length > 0) {
      setStatus('API 키를 저장하지 못했습니다(브라우저 저장소 차단). 이번 세션에만 사용됩니다.')
    }
  })
  resetButton.addEventListener('click', runReset)
  generateButton.addEventListener('click', () => {
    void runGenerate()
  })
  // 프롬프트가 비면 생성 버튼도 비활성(눌러보고 실패하는 대신). 전체 re-render 없이 버튼만 갱신.
  promptInput.addEventListener('input', () => {
    generateButton.disabled =
      isGenerating || apiKey.trim().length === 0 || promptInput.value.trim().length === 0
  })
  // ⌘/Ctrl+Enter로 빠르게 생성(데모 흐름용). runGenerate가 자체 가드(키·프롬프트·생성중)를 가진다.
  promptInput.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      void runGenerate()
    }
  })
  applyButton.addEventListener('click', runApply)
  copyButton.addEventListener('click', () => {
    void runCopy()
  })
  clearHistoryButton.addEventListener('click', runClearHistory)
  exportButton.addEventListener('click', runExport)
  openButton.addEventListener('click', () => {
    void runOpenProject()
  })
  analyzeButton.addEventListener('click', () => {
    void runAnalyze()
  })
  popoutButton.addEventListener('click', () => {
    window.open(gamePreviewUrl, 'game-window', 'width=1280,height=720')
  })
  reloadButton.addEventListener('click', () => {
    iframe.src = gamePreviewUrl
  })

  renderTree()
  renderAnalysis()
  render()
  if (game.parseErrors.length > 0) {
    setStatus(`기본 맵 일부를 읽지 못했습니다${parseErrorNote()}`)
  }
  // 데모 흐름: 키가 없으면 키 입력에, 있으면 바로 프롬프트에 포커스.
  ;(apiKey.trim().length > 0 ? promptInput : apiKeyInput).focus()
}
