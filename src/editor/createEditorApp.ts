import { generateEventJsonDraftWithOpenAi } from './openaiEventJsonGenerator'
import { createHolidayDialogueEventSpecFromGeneratedEventJson } from './eventCodeGenerator'
import { savePendingEvent } from './pendingEvents'
import { openProjectDirectory } from './openProjectDirectory'
import { loadGame, type GameFile, type LoadedGame } from './loadGame'
import type { GeneratedEventJson } from './eventJsonSchema'
import type { GameEntity } from './gameAdapter'

type CreateEditorAppInput = {
  mountElement: HTMLElement
  initialFiles: GameFile[]
  gamePreviewUrl: string
}

const API_KEY_STORAGE_KEY = 'my-sample-rpg:openai-api-key'

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
  let apiKey = window.localStorage.getItem(API_KEY_STORAGE_KEY) ?? ''
  let selectedEntity: GameEntity | undefined
  let currentDraft: GeneratedEventJson | undefined
  let isGenerating = false
  let entityButtons: Array<{ entity: GameEntity; node: HTMLButtonElement }> = []

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
  treeHeader.append(
    el('div', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', '프로젝트'),
    openButton
  )
  const treeList = el('div', 'flex-1 overflow-auto p-3 flex flex-col gap-3')
  tree.append(treeHeader, treeList)

  // ---------- center: generation ----------
  const center = el('main', 'overflow-auto p-5 flex flex-col gap-4')
  const targetLine = el('div', 'text-sm text-zinc-400')
  const supportNote = el('div', 'rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200')

  const apiKeyField = el('label', 'flex flex-col gap-1.5')
  apiKeyField.append(el('span', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', 'OpenAI API 키'))
  const apiKeyInput = el('input', 'w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-zinc-100 outline-none transition focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/30') as HTMLInputElement
  apiKeyInput.type = 'password'
  apiKeyInput.placeholder = 'sk-...'
  apiKeyInput.autocomplete = 'off'
  apiKeyInput.value = apiKey
  apiKeyField.append(apiKeyInput)

  const promptField = el('label', 'flex flex-col gap-1.5')
  promptField.append(el('span', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', '자연어 프롬프트'))
  const promptInput = el('textarea', 'w-full min-h-[96px] rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-zinc-100 outline-none transition resize-y focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/30') as HTMLTextAreaElement
  promptInput.placeholder = '예: 대장장이가 새로 만든 검을 자랑하는 대화'
  promptField.append(promptInput)

  const actions = el('div', 'flex items-center gap-2')
  const generateButton = el('button', PRIMARY_BUTTON, '생성') as HTMLButtonElement
  generateButton.type = 'button'
  const applyButton = el('button', GHOST_BUTTON, '게임에 적용') as HTMLButtonElement
  applyButton.type = 'button'
  actions.append(generateButton, applyButton)

  const status = el('div', 'text-sm text-zinc-400 min-h-[1.25rem]')

  const resultWrap = el('div', 'flex flex-col gap-1.5')
  resultWrap.append(el('span', 'text-[0.7rem] uppercase tracking-wider text-zinc-500 font-medium', '생성 결과 (JSON)'))
  const result = el('pre', 'm-0 max-h-[40vh] overflow-auto rounded-lg border border-white/10 bg-black/40 p-3 text-[0.72rem] leading-relaxed text-zinc-300 whitespace-pre-wrap break-words')
  resultWrap.append(result)

  center.append(targetLine, supportNote, apiKeyField, promptField, actions, status, resultWrap)

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

  function render(): void {
    const canGenerate = Boolean(game.profile)

    gameLabel.textContent = game.adapter.name

    if (canGenerate) {
      supportNote.hidden = true
    } else {
      supportNote.hidden = false
      supportNote.textContent = `${game.adapter.name}: 엔티티 보기 전용입니다. 생성·적용 어댑터는 다음 단계(Stage 2)에서 추가됩니다.`
    }

    targetLine.innerHTML = selectedEntity
      ? `대상: <span class="text-indigo-300 font-medium">${selectedEntity.name}</span> <span class="text-zinc-500">(${selectedEntity.kind} · ${selectedEntity.mapId})</span>`
      : '<span class="text-zinc-500">왼쪽에서 엔티티를 선택하면 그 대상으로 생성합니다.</span>'

    for (const { entity, node } of entityButtons) {
      node.className = entity.id === selectedEntity?.id ? ENTITY_ACTIVE : ENTITY_BASE
    }

    generateButton.textContent = isGenerating ? '생성 중...' : '생성'
    generateButton.disabled = isGenerating || apiKey.trim().length === 0 || !canGenerate
    applyButton.disabled = !currentDraft || isGenerating || !game.adapter.supportsApply
    result.textContent = currentDraft
      ? JSON.stringify(currentDraft, null, 2)
      : '생성 결과가 여기에 표시됩니다.'
  }

  const runGenerate = async (): Promise<void> => {
    const profile = game.profile

    if (isGenerating || !profile) {
      return
    }

    if (apiKey.trim().length === 0) {
      setStatus('먼저 OpenAI API 키를 입력하세요.')
      return
    }

    isGenerating = true
    setStatus('OpenAI로 생성 중...')
    render()

    try {
      const targetHint = selectedEntity
        ? ` 이 이벤트의 대상은 반드시 NPC id="${selectedEntity.id}"(${selectedEntity.name}, map=${selectedEntity.mapId})로 한다.`
        : ''
      currentDraft = await generateEventJsonDraftWithOpenAi({
        apiKey: apiKey.trim(),
        userPrompt: `${promptInput.value}${targetHint}`,
        profile
      })
      setStatus(`생성 완료: ${currentDraft.event_name}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setStatus(`생성 실패: ${message}`)
    } finally {
      isGenerating = false
      render()
    }
  }

  const runApply = (): void => {
    if (!currentDraft || !game.adapter.supportsApply) {
      return
    }

    const spec = createHolidayDialogueEventSpecFromGeneratedEventJson(currentDraft)

    if (!spec) {
      setStatus('적용 실패: 대사 라인이 비어 있습니다.')
      return
    }

    savePendingEvent(spec)
    setStatus(`게임에 적용됨: ${spec.npc.display_name} — 오른쪽 프리뷰에 즉시 반영됩니다.`)
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
      selectedEntity = undefined
      currentDraft = undefined
      renderTree()
      render()
      const entityCount = game.maps.reduce((sum, map) => sum + map.entities.length, 0)
      setStatus(`프로젝트 로드: ${game.adapter.name} · 맵 ${game.maps.length}개 · 엔티티 ${entityCount}개`)
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        return
      }
      setStatus(error instanceof Error ? error.message : String(error))
    }
  }

  apiKeyInput.addEventListener('input', () => {
    apiKey = apiKeyInput.value
    window.localStorage.setItem(API_KEY_STORAGE_KEY, apiKey)
    render()
  })
  generateButton.addEventListener('click', () => {
    void runGenerate()
  })
  applyButton.addEventListener('click', runApply)
  openButton.addEventListener('click', () => {
    void runOpenProject()
  })
  popoutButton.addEventListener('click', () => {
    window.open(gamePreviewUrl, 'game-window', 'width=1280,height=720')
  })
  reloadButton.addEventListener('click', () => {
    iframe.src = gamePreviewUrl
  })

  renderTree()
  render()
}
