// 헤더 '📦 배치' 버튼 + 팔레트 모달 + 떠 있는 모드 배너.
// 이미 있는 에셋을 마우스로 맵에 배치한다. 세 종류:
//   · 타일: 현재 맵 타일셋의 타일(배경에서 잘라 표시) 하나를 골라 칸에 놓기
//   · 오브젝트: 자동 추출된 누끼(/api/style/extracted-objects)를 골라 놓기
//   · NPC: 캐릭터 외형 + 이름 + 대사를 정해 놓기(정적 배치와 달리 게임의 기능 엔티티로 스폰 —
//          이동을 막고, 대사가 있으면 Enter 상호작용 시 말한다)
// '배치 시작'을 누르면 모달을 닫고(게임을 가리지 않게) 게임에 배치 모드를 켠다. 실제 클릭→배치는
// 게임(createPixiTiledMapView)이 처리하고 맵별 localStorage(placementStore/npcStore)에 영구 저장한다.
// '전체 지우기'는 에디터가 직접 localStorage를 비워 storage 이벤트로 게임이 즉시 다시 그린다.

import { clearPlacementsForMap, type PlacementTemplate } from './placementStore'
import { clearNpcsForMap, type NpcWireTemplate } from './npcStore'
import {
  createHoverPreview,
  buildTileSlicePreview,
  buildImagePreview
} from './createHoverPreview'

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

const LABEL = 'text-[11px] font-semibold uppercase tracking-wider text-zinc-400'
const CARD = 'rounded-xl border border-white/10 bg-white/[0.03] p-4 flex flex-col gap-2'
const PRIMARY_BUTTON =
  'rounded-lg px-4 py-2 bg-indigo-500 text-white text-sm font-medium shadow-sm shadow-indigo-500/30 transition hover:bg-indigo-400 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none'
const DANGER_BUTTON =
  'rounded-lg px-4 py-2 bg-rose-500/90 text-white text-sm font-medium transition hover:bg-rose-500 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed'
const SUBTLE_BUTTON =
  'rounded-lg px-3 py-1.5 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed'
const MODE_TAB = 'rounded-md px-2.5 py-1 text-xs text-zinc-400 border border-transparent transition hover:text-zinc-200'
const MODE_TAB_ACTIVE = 'rounded-md px-2.5 py-1 text-xs bg-indigo-500/15 text-indigo-200 border border-indigo-500/30'

const STYLE_SERVICE_BASE = '/api/style'

// 현재 맵 타일 팔레트에 필요한 정보(createEditorApp이 채운다).
export type TilePaletteContext = {
  tilesetSource: string
  tilesetImageUrl: string
  columns: number
  tileWidth: number
  tileHeight: number
}

// NPC 외형 한 종류 — 캐릭터 타일셋(tiny-dungeon-16)의 타일 id + 표시 이름.
export type NpcAppearanceOption = {
  appearanceType: string
  tileId: number
  label: string
}

// NPC 탭에 필요한 정보(createEditorApp이 채운다 — 캐릭터 타일셋 이미지/격자 + 외형 목록).
export type NpcPaletteContext = {
  tilesetImageUrl: string
  columns: number
  tileWidth: number
  tileHeight: number
  appearances: NpcAppearanceOption[]
}

type ExtractedObject = { key: string; label: string }

export interface PlacementPalette {
  button: HTMLButtonElement
  backdrop: HTMLDivElement
  banner: HTMLDivElement
  // 프로젝트(게임)를 새로 열거나 리셋했을 때 캐시된 그리드/선택을 버리고 새 컨텍스트로 다시 만든다.
  invalidate: () => void
}

export const createPlacementPalette = (options: {
  getTilePaletteContext: () => TilePaletteContext | undefined
  getNpcPaletteContext?: () => NpcPaletteContext | undefined
  getCurrentMapId: () => string | undefined
  // iframe(게임)으로 postMessage.
  sendToGame: (message: unknown) => void
}): PlacementPalette => {
  const button = el(
    'button',
    'rounded-lg px-2.5 py-1 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100',
    '📦 배치'
  ) as HTMLButtonElement
  button.type = 'button'
  button.title = '에셋을 마우스로 맵에 배치'

  const backdrop = el('div', 'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4')
  backdrop.style.display = 'none'
  const panel = el('div', 'w-full max-w-2xl max-h-[92vh] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900 p-5 flex flex-col gap-4 shadow-2xl')

  const top = el('div', 'flex items-center justify-between')
  top.append(el('span', 'text-sm font-semibold tracking-tight', '📦 에셋 배치'))
  const close = el('button', 'text-zinc-500 text-sm transition hover:text-zinc-200', '✕')
  close.type = 'button'
  top.append(close)

  const note = el(
    'div',
    'text-[11px] text-sky-300/80 leading-relaxed',
    '항목을 고르고 "배치 시작"을 누르면 모달이 닫히고, 게임 화면을 좌클릭한 칸에 놓입니다. 놓인 항목은 우클릭으로 바로 삭제할 수 있습니다. NPC는 이동을 막고, 대사가 있으면 다가가 Enter로 말을 겁니다. 맵별로 영구 저장돼 새로고침해도 유지됩니다.'
  )

  // 탭
  const tabRow = el('div', 'flex items-center gap-1')
  const tileTab = el('button', MODE_TAB_ACTIVE, '타일') as HTMLButtonElement
  tileTab.type = 'button'
  const objectTab = el('button', MODE_TAB, '오브젝트') as HTMLButtonElement
  objectTab.type = 'button'
  const npcTab = el('button', MODE_TAB, 'NPC') as HTMLButtonElement
  npcTab.type = 'button'
  tabRow.append(tileTab, objectTab, npcTab)

  // 타일 그리드(타일셋 배경에서 잘라 표시) — 성능상 dataURL 대신 background-position 사용.
  const tileCard = el('div', CARD)
  tileCard.append(el('div', LABEL, '타일 (현재 맵 타일셋)'))
  const tileGrid = el('div', 'flex flex-wrap gap-0.5 max-h-56 overflow-y-auto bg-black/30 p-1 rounded-lg')
  tileCard.append(tileGrid)

  // 오브젝트 그리드(추출 누끼)
  const objectCard = el('div', CARD)
  objectCard.append(el('div', LABEL, '오브젝트 (자동 추출 누끼)'))
  const objectGrid = el('div', 'grid grid-cols-4 gap-1.5 max-h-56 overflow-y-auto')
  objectCard.append(objectGrid)
  objectCard.style.display = 'none'

  // NPC 카드(외형 그리드 + 이름 + 대사). 외형은 캐릭터 타일셋에서 잘라 확대 표시.
  const npcCard = el('div', CARD)
  npcCard.append(el('div', LABEL, 'NPC 외형 (캐릭터)'))
  const npcGrid = el('div', 'flex flex-wrap gap-1.5 max-h-40 overflow-y-auto bg-black/30 p-1.5 rounded-lg')
  npcCard.append(npcGrid)
  npcCard.append(el('div', `${LABEL} mt-1`, '이름 (머리 위 표시 · 선택)'))
  const npcNameInput = el(
    'input',
    'rounded-lg bg-black/30 border border-white/10 px-3 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-indigo-400/60'
  ) as HTMLInputElement
  npcNameInput.type = 'text'
  npcNameInput.placeholder = '예: 마을 촌장'
  npcCard.append(npcNameInput)
  npcCard.append(el('div', `${LABEL} mt-1`, '대사 (한 줄에 한 마디 · 비우면 말 없는 NPC · 여러 줄이면 순환)'))
  const npcDialogueInput = el(
    'textarea',
    'rounded-lg bg-black/30 border border-white/10 px-3 py-1.5 text-sm text-zinc-100 placeholder:text-zinc-600 resize-y min-h-[60px] focus:outline-none focus:border-indigo-400/60'
  ) as HTMLTextAreaElement
  npcDialogueInput.placeholder = '안녕하세요, 여행자님.\n오늘도 좋은 하루 되세요.'
  npcCard.append(npcDialogueInput)
  npcCard.style.display = 'none'

  const selectedInfo = el('div', 'text-xs text-zinc-400 min-h-4', '선택된 항목이 없습니다.')

  const actions = el('div', 'flex items-center gap-2 flex-wrap')
  const placeButton = el('button', PRIMARY_BUTTON, '배치 시작') as HTMLButtonElement
  placeButton.type = 'button'
  placeButton.disabled = true
  const eraseButton = el('button', SUBTLE_BUTTON, '지우기 모드') as HTMLButtonElement
  eraseButton.type = 'button'
  const clearButton = el('button', DANGER_BUTTON, '이 맵 전체 지우기') as HTMLButtonElement
  clearButton.type = 'button'
  actions.append(placeButton, eraseButton, clearButton)

  const status = el('span', 'text-xs text-zinc-500 min-h-4', '')

  panel.append(top, note, tabRow, tileCard, objectCard, npcCard, selectedInfo, actions, status)
  backdrop.append(panel)

  // 떠 있는 모드 배너(모달이 닫혀도 모드 상태 표시 + 중지).
  const banner = el(
    'div',
    'fixed top-3 left-1/2 -translate-x-1/2 z-[60] flex items-center gap-3 rounded-full border border-indigo-400/40 bg-zinc-900/95 px-4 py-1.5 text-xs text-zinc-100 shadow-xl'
  )
  banner.style.display = 'none'
  const bannerLabel = el('span', '')
  const bannerStop = el('button', 'rounded-full bg-rose-500/90 px-2.5 py-0.5 text-white transition hover:bg-rose-500', '중지') as HTMLButtonElement
  bannerStop.type = 'button'
  banner.append(bannerLabel, bannerStop)

  // ── 상태 ──
  let activeTab: 'tile' | 'object' | 'npc' = 'tile'
  let selected: PlacementTemplate | NpcWireTemplate | null = null
  let mode: 'off' | 'place' | 'erase' = 'off'
  let tileGridBuilt = false
  let objectsLoaded = false
  let npcGridBuilt = false
  let npcSelectedAppearance: NpcAppearanceOption | null = null

  // 항목에 마우스를 올리면 확대 미리보기를 띄운다(작은 썸네일을 크게 확인).
  const hoverPreview = createHoverPreview()

  const setStatus = (message: string): void => {
    status.textContent = message
  }

  const sendMode = (next: 'off' | 'place' | 'erase'): void => {
    mode = next
    options.sendToGame({ type: 'editor:placement-mode', mode: next })
    if (next === 'off') {
      banner.style.display = 'none'
      button.textContent = '📦 배치'
    } else {
      banner.style.display = 'flex'
      bannerLabel.textContent =
        next === 'place'
          ? `🖱 배치 중: ${selected?.label ?? ''} — 좌클릭 배치 · 우클릭 삭제`
          : '🧽 지우기 중 — 클릭(또는 우클릭)으로 삭제'
      button.textContent = next === 'place' ? '📦 배치중' : '📦 지우는중'
    }
  }

  const selectTemplate = (
    template: PlacementTemplate | NpcWireTemplate
  ): void => {
    selected = template
    selectedInfo.textContent =
      template.kind === 'npc'
        ? `선택됨: NPC — ${template.label ?? template.appearanceType}`
        : `선택됨: ${template.label ?? (template.kind === 'tile' ? `타일 #${template.tileId}` : '오브젝트')}`
    placeButton.disabled = false
    // 게임에 미리 알려둔다(배치 시작 시 바로 쓰도록).
    options.sendToGame({ type: 'editor:placement-template', template })
  }

  // 모든 그리드의 선택 표시(테두리/링)를 지운다.
  const clearGridSelectionRings = (): void => {
    for (const grid of [tileGrid, objectGrid, npcGrid]) {
      for (const cell of grid.children) {
        ;(cell as HTMLElement).classList.remove('!border-indigo-400', 'ring-2', 'ring-indigo-400/60')
      }
    }
  }

  // 선택 상태 초기화 — 탭을 바꾸면 이전 탭에서 고른 항목이 그대로 배치되는 혼동을 막는다.
  const clearSelection = (): void => {
    selected = null
    npcSelectedAppearance = null
    placeButton.disabled = true
    selectedInfo.textContent = '선택된 항목이 없습니다.'
    clearGridSelectionRings()
  }

  // 외형/이름/대사가 바뀔 때마다 NPC 템플릿을 다시 만들어 게임에 보낸다(외형이 골라진 경우만).
  const updateNpcTemplate = (): void => {
    if (!npcSelectedAppearance) {
      return
    }
    const name = npcNameInput.value.trim()
    const dialogueLines = npcDialogueInput.value
      .split(/\r?\n/u)
      .map((line) => line.trim())
      .filter((line) => line.length > 0)
    selectTemplate({
      kind: 'npc',
      appearanceType: npcSelectedAppearance.appearanceType,
      name: name.length > 0 ? name : undefined,
      dialogueLines: dialogueLines.length > 0 ? dialogueLines : undefined,
      label: name.length > 0 ? name : npcSelectedAppearance.label
    })
  }

  const setTab = (tab: 'tile' | 'object' | 'npc'): void => {
    const isTabChange = tab !== activeTab
    activeTab = tab
    tileTab.className = tab === 'tile' ? MODE_TAB_ACTIVE : MODE_TAB
    objectTab.className = tab === 'object' ? MODE_TAB_ACTIVE : MODE_TAB
    npcTab.className = tab === 'npc' ? MODE_TAB_ACTIVE : MODE_TAB
    tileCard.style.display = tab === 'tile' ? '' : 'none'
    objectCard.style.display = tab === 'object' ? '' : 'none'
    npcCard.style.display = tab === 'npc' ? '' : 'none'
    // 전체 지우기 버튼은 보고 있는 탭이 다루는 것(배치 vs NPC)을 지운다.
    clearButton.textContent = tab === 'npc' ? '이 맵 NPC 전체 지우기' : '이 맵 배치 전체 지우기'
    // 탭을 실제로 바꿀 때만 선택 초기화(같은 탭 재적용=모달 열기 시엔 선택 유지).
    if (isTabChange) {
      clearSelection()
    }
    if (tab === 'tile') {
      buildTileGrid()
    } else if (tab === 'object') {
      void loadObjects()
    } else {
      buildNpcGrid()
    }
  }

  const buildNpcGrid = (): void => {
    if (npcGridBuilt) {
      return
    }
    const ctx = options.getNpcPaletteContext?.()
    npcGrid.replaceChildren()
    if (!ctx) {
      npcGrid.append(
        el(
          'div',
          'text-xs text-zinc-500 p-2',
          '캐릭터 타일셋(tiny-dungeon-16)을 찾지 못했습니다. (my-sample-rpg 맵에서만 사용 가능)'
        )
      )
      return
    }
    if (ctx.appearances.length === 0) {
      npcGrid.append(el('div', 'text-xs text-zinc-500 p-2', '사용할 수 있는 외형이 없습니다.'))
      return
    }
    // 16px 타일은 작으니 확대해 표시(배경 이미지를 비례 확대).
    const thumb = 40
    const scale = thumb / ctx.tileWidth
    const sheetWidth = ctx.columns * ctx.tileWidth * scale
    for (const appearance of ctx.appearances) {
      const col = appearance.tileId % ctx.columns
      const row = Math.floor(appearance.tileId / ctx.columns)
      const cell = el(
        'button',
        'shrink-0 rounded-md border border-transparent hover:border-indigo-400/60 [image-rendering:pixelated]'
      ) as HTMLButtonElement
      cell.type = 'button'
      cell.style.width = `${thumb}px`
      cell.style.height = `${thumb}px`
      cell.style.backgroundImage = `url(${ctx.tilesetImageUrl})`
      cell.style.backgroundRepeat = 'no-repeat'
      cell.style.backgroundSize = `${sheetWidth}px auto`
      cell.style.backgroundPosition = `-${col * ctx.tileWidth * scale}px -${row * ctx.tileHeight * scale}px`
      cell.title = appearance.label
      hoverPreview.bind(cell, () =>
        buildTileSlicePreview({
          imageUrl: ctx.tilesetImageUrl,
          columns: ctx.columns,
          tileWidth: ctx.tileWidth,
          tileHeight: ctx.tileHeight,
          tileId: appearance.tileId,
          targetSize: 128
        })
      )
      cell.addEventListener('click', () => {
        for (const other of npcGrid.children) {
          ;(other as HTMLElement).classList.remove('!border-indigo-400', 'ring-2', 'ring-indigo-400/60')
        }
        cell.classList.add('!border-indigo-400', 'ring-2', 'ring-indigo-400/60')
        npcSelectedAppearance = appearance
        updateNpcTemplate()
      })
      npcGrid.append(cell)
    }
    npcGridBuilt = true
  }

  const buildTileGrid = (): void => {
    if (tileGridBuilt) {
      return
    }
    const ctx = options.getTilePaletteContext()
    tileGrid.replaceChildren()
    if (!ctx) {
      tileGrid.append(el('div', 'text-xs text-zinc-500 p-2', '현재 맵의 타일셋을 찾지 못했습니다.'))
      return
    }
    // 타일 수는 이미지 높이로 계산. 이미지 로드 후 셀 생성.
    const probe = new Image()
    probe.onload = () => {
      const rows = Math.max(1, Math.floor(probe.naturalHeight / ctx.tileHeight))
      const count = rows * ctx.columns
      for (let tileId = 0; tileId < count; tileId += 1) {
        const cell = el('button', 'shrink-0 rounded-sm border border-transparent hover:border-indigo-400/60 [image-rendering:pixelated]') as HTMLButtonElement
        cell.type = 'button'
        cell.style.width = `${ctx.tileWidth}px`
        cell.style.height = `${ctx.tileHeight}px`
        cell.style.backgroundImage = `url(${ctx.tilesetImageUrl})`
        cell.style.backgroundRepeat = 'no-repeat'
        cell.style.backgroundPosition = `-${(tileId % ctx.columns) * ctx.tileWidth}px -${Math.floor(tileId / ctx.columns) * ctx.tileHeight}px`
        cell.title = `타일 #${tileId}`
        hoverPreview.bind(cell, () =>
          buildTileSlicePreview({
            imageUrl: ctx.tilesetImageUrl,
            columns: ctx.columns,
            tileWidth: ctx.tileWidth,
            tileHeight: ctx.tileHeight,
            tileId,
            targetSize: 128
          })
        )
        cell.addEventListener('click', () => {
          for (const other of tileGrid.children) {
            (other as HTMLElement).classList.remove('!border-indigo-400', 'ring-2', 'ring-indigo-400/60')
          }
          cell.classList.add('!border-indigo-400', 'ring-2', 'ring-indigo-400/60')
          selectTemplate({ kind: 'tile', tilesetSource: ctx.tilesetSource, tileId, label: `타일 #${tileId}` })
        })
        tileGrid.append(cell)
      }
      tileGridBuilt = true
    }
    probe.onerror = () => {
      tileGrid.append(el('div', 'text-xs text-rose-300 p-2', '타일셋 이미지를 불러오지 못했습니다.'))
    }
    probe.src = ctx.tilesetImageUrl
  }

  const loadObjects = async (): Promise<void> => {
    if (objectsLoaded) {
      return
    }
    objectGrid.replaceChildren()
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/extracted-objects`)
      if (!response.ok) {
        throw new Error(String(response.status))
      }
      const objects = ((await response.json()) as { objects: ExtractedObject[] }).objects
      if (objects.length === 0) {
        objectGrid.append(el('div', 'col-span-full text-xs text-zinc-500', '추출된 오브젝트가 없습니다 — 게임에서 맵을 열면 자동 추출됩니다.'))
      }
      const bust = Date.now()
      for (const object of objects) {
        const card = el('button', 'relative flex flex-col items-center gap-0.5 rounded-lg bg-black/40 p-1 border border-transparent transition hover:bg-white/[0.08]') as HTMLButtonElement
        card.type = 'button'
        const thumb = el('img', 'h-12 w-full object-contain [image-rendering:pixelated]') as HTMLImageElement
        thumb.src = `${STYLE_SERVICE_BASE}/extracted-objects/${object.key}.png?t=${bust}`
        thumb.loading = 'lazy'
        card.append(thumb, el('span', 'w-full truncate text-center text-[10px] text-zinc-400', object.label))
        hoverPreview.bind(card, () =>
          buildImagePreview(`${STYLE_SERVICE_BASE}/extracted-objects/${object.key}.png?t=${bust}`, {
            maxSize: 224
          })
        )
        card.addEventListener('click', () => {
          for (const other of objectGrid.children) {
            (other as HTMLElement).classList.remove('ring-2', 'ring-indigo-400/60')
          }
          card.classList.add('ring-2', 'ring-indigo-400/60')
          selectTemplate({
            kind: 'object',
            imageUrl: `${STYLE_SERVICE_BASE}/extracted-objects/${object.key}.png`,
            label: object.label
          })
        })
        objectGrid.append(card)
      }
      objectsLoaded = true
    } catch {
      objectGrid.append(el('div', 'col-span-full text-xs text-zinc-500', '스타일 서비스에 연결할 수 없습니다 (오브젝트 목록).'))
    }
  }

  // ── 이벤트 ──
  tileTab.addEventListener('click', () => setTab('tile'))
  objectTab.addEventListener('click', () => setTab('object'))
  npcTab.addEventListener('click', () => setTab('npc'))
  // 이름/대사가 바뀌면 (외형이 골라진 경우) 템플릿을 갱신해 다시 보낸다.
  npcNameInput.addEventListener('input', updateNpcTemplate)
  npcDialogueInput.addEventListener('input', updateNpcTemplate)

  placeButton.addEventListener('click', () => {
    if (!selected) {
      return
    }
    sendMode('place')
    closeModal() // 게임을 클릭할 수 있게 모달을 닫는다(배너로 모드 표시).
  })
  eraseButton.addEventListener('click', () => {
    sendMode('erase')
    closeModal()
  })
  bannerStop.addEventListener('click', () => sendMode('off'))

  clearButton.addEventListener('click', () => {
    const mapId = options.getCurrentMapId()
    if (!mapId) {
      setStatus('현재 맵을 알 수 없습니다.')
      return
    }
    try {
      if (activeTab === 'npc') {
        clearNpcsForMap(mapId)
        setStatus(`'${mapId}' 맵의 NPC를 모두 지웠습니다.`)
      } else {
        clearPlacementsForMap(mapId)
        setStatus(`'${mapId}' 맵의 배치를 모두 지웠습니다.`)
      }
    } catch (error) {
      setStatus(`지우기 실패: ${error instanceof Error ? error.message : String(error)}`)
    }
  })

  const openModal = (): void => {
    backdrop.style.display = 'flex'
    setStatus('')
    setTab(activeTab)
  }
  const closeModal = (): void => {
    backdrop.style.display = 'none'
  }
  button.addEventListener('click', () => {
    if (backdrop.style.display === 'none') {
      openModal()
    } else {
      closeModal()
    }
  })
  close.addEventListener('click', closeModal)
  backdrop.addEventListener('click', (event) => {
    if (event.target === backdrop) {
      closeModal()
    }
  })
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      if (backdrop.style.display !== 'none') {
        closeModal()
      } else if (mode !== 'off') {
        sendMode('off') // 배치/지우기 모드 중 Esc로 중지.
      }
    }
  })

  // 프로젝트 교체/리셋 시 — 캐시된 그리드(이전 게임의 타일셋·외형)와 선택을 버리고, 모달이
  // 열려 있으면 새 컨텍스트로 즉시 다시 만든다. 닫혀 있으면 다음에 열 때 setTab이 새로 빌드한다.
  const invalidate = (): void => {
    tileGridBuilt = false
    objectsLoaded = false
    npcGridBuilt = false
    tileGrid.replaceChildren()
    objectGrid.replaceChildren()
    npcGrid.replaceChildren()
    clearSelection()
    if (backdrop.style.display !== 'none') {
      setTab(activeTab)
    }
  }

  return { button, backdrop, banner, invalidate }
}
