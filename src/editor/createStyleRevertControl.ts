// 헤더 '↩ 스타일 되돌리기' 버튼 + 전체/선택 되돌리기 모달.
// 스타일이 적용된 에셋(서비스 originals/가 추적)을 전체 또는 골라서 최초 원본으로 복원한다.
// 복원은 디스크의 src/games/my-sample-rpg/assets 파일을 원본 바이트로 되써, Vite가 게임을 자동 리로드한다.
// 스타일 변환 모달의 인라인 '↩ 원본으로 되돌리기'(작업 중 한 대상용)와 같은 백엔드를 공유한다.
// 선택 되돌리기 목록은 스타일 변환 모달의 그리드와 같은 썸네일 컴포넌트(다중 선택)를 쓴다.

import { createThumbnailPicker } from './createThumbnailPicker'

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

const STYLE_SERVICE_BASE = '/api/style'
const SERVICE_GUIDE = '스타일 서비스에 연결할 수 없습니다 — style-service 폴더에서 python server.py를 실행하세요.'

type StyledAsset = { path: string; size: number }

export interface StyleRevertControl {
  button: HTMLButtonElement
  backdrop: HTMLDivElement
  // 스타일 적용 상태가 바뀐 뒤(적용/되돌리기) 호출 — 버튼 활성/비활성 카운트를 갱신한다.
  refresh: () => void
}

export const createStyleRevertControl = (): StyleRevertControl => {
  const button = el('button', 'rounded-lg px-2.5 py-1 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed', '↩ 스타일 되돌리기') as HTMLButtonElement
  button.type = 'button'
  button.disabled = true
  button.title = '스타일이 적용된 항목이 없습니다'

  const backdrop = el('div', 'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4')
  backdrop.style.display = 'none'
  const panel = el('div', 'w-full max-w-xl max-h-[90vh] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900 p-5 flex flex-col gap-4 shadow-2xl')

  const top = el('div', 'flex items-center justify-between')
  top.append(el('span', 'text-sm font-semibold tracking-tight', '↩ 스타일 되돌리기'))
  const close = el('button', 'text-zinc-500 text-sm transition hover:text-zinc-200', '✕')
  close.type = 'button'
  top.append(close)

  const summary = el('div', 'text-xs text-zinc-400', '')

  // 전체 되돌리기 — 인라인 2-클릭 확인(실수 방지). 첫 클릭에 확인 상태로 바뀐다.
  const allCard = el('div', CARD)
  allCard.append(el('div', LABEL, '전체 되돌리기'))
  const allHint = el('div', 'text-xs text-zinc-500', '스타일이 적용된 모든 항목을 원본으로 되돌립니다.')
  const allButton = el('button', DANGER_BUTTON, '전체 되돌리기') as HTMLButtonElement
  allButton.type = 'button'
  const allRow = el('div', 'flex items-center gap-2')
  allRow.append(allButton)
  allCard.append(allHint, allRow)

  // 선택 되돌리기 — 썸네일 그리드(다중 선택)로 이미지를 보고 고른다.
  const selectCard = el('div', CARD)
  const selectHead = el('div', 'flex items-center justify-between gap-2')
  selectHead.append(el('div', LABEL, '선택 되돌리기'))
  const selectAllToggle = el('button', 'text-[11px] text-zinc-400 transition hover:text-zinc-200', '모두 선택') as HTMLButtonElement
  selectAllToggle.type = 'button'
  selectHead.append(selectAllToggle)
  const picker = createThumbnailPicker({
    multiSelect: true,
    onChange: () => syncSelectButton()
  })
  const selectButton = el('button', PRIMARY_BUTTON, '선택 항목 되돌리기') as HTMLButtonElement
  selectButton.type = 'button'
  selectButton.disabled = true
  selectCard.append(selectHead, picker.node, selectButton)

  const status = el('span', 'text-xs text-zinc-500 min-h-4', '')

  panel.append(top, summary, allCard, selectCard, status)
  backdrop.append(panel)

  const setStatus = (message: string): void => {
    status.textContent = message
  }

  let assets: StyledAsset[] = []
  let isBusy = false
  // 전체 되돌리기 확인 대기 상태(첫 클릭 후). 닫거나 목록이 바뀌면 해제된다.
  let allConfirmArmed = false
  // 썸네일 캐시버스트 — 같은 경로 파일이 되돌리기로 바뀌어도 새 이미지를 받게.
  let assetCacheBust = 0

  // 헤더 버튼 활성/비활성용 가벼운 카운트 갱신 — 목록 fetch는 모달 열 때만.
  const refresh = (): void => {
    void (async () => {
      try {
        const response = await fetch(`${STYLE_SERVICE_BASE}/styled-assets`)
        if (!response.ok) {
          throw new Error(String(response.status))
        }
        const data = (await response.json()) as { assets: StyledAsset[] }
        const count = data.assets.length
        button.disabled = count === 0
        button.title = count === 0 ? '스타일이 적용된 항목이 없습니다' : `스타일 적용 항목 ${count}개 되돌리기`
      } catch {
        // 서비스 미기동: 알 수 없으니 비활성으로 둔다(클릭해도 안내만 뜬다).
        button.disabled = true
        button.title = SERVICE_GUIDE
      }
    })()
  }

  const assetName = (path: string): string => path.replace(/^src\/games\/my-sample-rpg\/assets\//, '')

  const syncSelectButton = (): void => {
    const count = picker.getSelected().length
    selectButton.disabled = isBusy || count === 0
    selectButton.textContent = count > 0 ? `선택 항목 되돌리기 (${count})` : '선택 항목 되돌리기'
    selectAllToggle.textContent = assets.length > 0 && count === assets.length ? '모두 해제' : '모두 선택'
  }

  const resetAllConfirm = (): void => {
    allConfirmArmed = false
    allButton.textContent = '전체 되돌리기'
    allButton.className = DANGER_BUTTON
  }

  const renderList = (): void => {
    summary.textContent = `스타일이 적용된 항목 ${assets.length}개`
    allButton.disabled = isBusy || assets.length === 0
    // Vite가 src/games/my-sample-rpg/assets를 그대로 서빙한다. 현재(스타일 적용된) 파일을 캐시버스트로 받는다.
    picker.setItems(
      assets.map((asset) => ({
        id: asset.path,
        label: assetName(asset.path),
        thumbUrl: `/${asset.path}?t=${assetCacheBust}`
      })),
      '스타일이 적용된 항목이 없습니다.'
    )
    syncSelectButton()
  }

  const loadList = async (): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/styled-assets`)
      if (!response.ok) {
        throw new Error(String(response.status))
      }
      const data = (await response.json()) as { assets: StyledAsset[] }
      assets = data.assets
      assetCacheBust += 1
      resetAllConfirm()
      // setItems가 사라진 항목의 선택을 자동으로 정리한다.
      renderList()
      if (assets.length === 0) {
        setStatus('')
      }
    } catch {
      assets = []
      renderList()
      setStatus(SERVICE_GUIDE)
    }
  }

  const revertPaths = async (paths: string[]): Promise<void> => {
    if (isBusy || paths.length === 0) {
      return
    }
    isBusy = true
    renderList()
    setStatus('원본으로 복원 중...')
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/revert-assets`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paths })
      })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      const result = (await response.json()) as { reverted: string[]; failed: Array<{ path: string }> }
      try {
        // 서버에서 복원된 원본을 로컬 게임 파일에도 반영한다(원격 GPU 구성용 dev 엔드포인트).
        await fetch('/__sync-styled-assets?force=1', { method: 'POST' })
      } catch {
        // 미러 동기화 실패가 되돌리기 결과 표시를 막지는 않는다.
      }
      isBusy = false
      await loadList()
      refresh()
      const failedNote = result.failed.length > 0 ? ` (실패 ${result.failed.length}개)` : ''
      setStatus(`${result.reverted.length}개 항목을 원본으로 되돌렸습니다 — 게임 화면이 자동 새로고침됩니다.${failedNote}`)
    } catch (error) {
      isBusy = false
      renderList()
      const failedToFetch = error instanceof TypeError
      setStatus(failedToFetch ? SERVICE_GUIDE : `되돌리기 실패: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  allButton.addEventListener('click', () => {
    if (isBusy || assets.length === 0) {
      return
    }
    if (!allConfirmArmed) {
      // 첫 클릭: 확인 상태로 전환(실수 방지).
      allConfirmArmed = true
      allButton.textContent = `⚠ ${assets.length}개 모두 되돌리기 확정`
      allButton.className = DANGER_BUTTON + ' ring-2 ring-rose-400/60'
      setStatus('전체 되돌리기를 다시 누르면 실행됩니다.')
      return
    }
    void revertPaths(assets.map((asset) => asset.path))
  })

  selectButton.addEventListener('click', () => {
    void revertPaths(picker.getSelected())
  })

  selectAllToggle.addEventListener('click', () => {
    // setSelected가 onChange→syncSelectButton을 발생시켜 버튼/토글 라벨이 갱신된다.
    if (picker.getSelected().length === assets.length) {
      picker.setSelected([])
    } else {
      picker.setSelected(assets.map((asset) => asset.path))
    }
  })

  const openModal = (): void => {
    backdrop.style.display = 'flex'
    setStatus('')
    resetAllConfirm()
    void loadList()
  }
  const closeModal = (): void => {
    backdrop.style.display = 'none'
    resetAllConfirm()
  }
  button.addEventListener('click', () => {
    if (!button.disabled) {
      openModal()
    }
  })
  close.addEventListener('click', closeModal)
  backdrop.addEventListener('click', (event) => {
    if (event.target === backdrop) {
      closeModal()
    }
  })
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && backdrop.style.display !== 'none') {
      closeModal()
    }
  })

  refresh()
  return { button, backdrop, refresh }
}
