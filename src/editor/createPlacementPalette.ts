// 헤더 '📦 배치' 버튼 + 팔레트 모달 + 떠 있는 모드 배너.
// 이미 있는 에셋을 마우스로 맵에 배치한다. 두 종류:
//   · 타일: 현재 맵 타일셋의 타일(배경에서 잘라 표시) 하나를 골라 칸에 놓기
//   · 오브젝트: 자동 추출된 누끼(/api/style/extracted-objects)를 골라 놓기
// '배치 시작'을 누르면 모달을 닫고(게임을 가리지 않게) 게임에 배치 모드를 켠다. 실제 클릭→배치는
// 게임(createPixiTiledMapView)이 처리하고 맵별 localStorage(placementStore)에 영구 저장한다.
// '전체 지우기'는 에디터가 직접 localStorage를 비워 storage 이벤트로 게임이 즉시 다시 그린다.

import { clearPlacementsForMap, type PlacementTemplate } from './placementStore'

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

type ExtractedObject = { key: string; label: string }

export interface PlacementPalette {
  button: HTMLButtonElement
  backdrop: HTMLDivElement
  banner: HTMLDivElement
}

export const createPlacementPalette = (options: {
  getTilePaletteContext: () => TilePaletteContext | undefined
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
    '항목을 고르고 "배치 시작"을 누르면 모달이 닫히고, 게임 화면을 좌클릭한 칸에 놓입니다. 놓인 항목은 우클릭으로 바로 삭제할 수 있습니다. 맵별로 영구 저장돼 새로고침해도 유지됩니다.'
  )

  // 탭
  const tabRow = el('div', 'flex items-center gap-1')
  const tileTab = el('button', MODE_TAB_ACTIVE, '타일') as HTMLButtonElement
  tileTab.type = 'button'
  const objectTab = el('button', MODE_TAB, '오브젝트') as HTMLButtonElement
  objectTab.type = 'button'
  tabRow.append(tileTab, objectTab)

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

  panel.append(top, note, tabRow, tileCard, objectCard, selectedInfo, actions, status)
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
  let activeTab: 'tile' | 'object' = 'tile'
  let selected: PlacementTemplate | null = null
  let mode: 'off' | 'place' | 'erase' = 'off'
  let tileGridBuilt = false
  let objectsLoaded = false

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

  const selectTemplate = (template: PlacementTemplate): void => {
    selected = template
    selectedInfo.textContent = `선택됨: ${template.label ?? (template.kind === 'tile' ? `타일 #${template.tileId}` : '오브젝트')}`
    placeButton.disabled = false
    // 게임에 미리 알려둔다(배치 시작 시 바로 쓰도록).
    options.sendToGame({ type: 'editor:placement-template', template })
  }

  const setTab = (tab: 'tile' | 'object'): void => {
    activeTab = tab
    tileTab.className = tab === 'tile' ? MODE_TAB_ACTIVE : MODE_TAB
    objectTab.className = tab === 'object' ? MODE_TAB_ACTIVE : MODE_TAB
    tileCard.style.display = tab === 'tile' ? '' : 'none'
    objectCard.style.display = tab === 'object' ? '' : 'none'
    if (tab === 'tile') {
      buildTileGrid()
    } else {
      void loadObjects()
    }
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
      clearPlacementsForMap(mapId)
      setStatus(`'${mapId}' 맵의 배치를 모두 지웠습니다.`)
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

  return { button, backdrop, banner }
}
