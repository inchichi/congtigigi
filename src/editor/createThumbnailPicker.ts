// 썸네일 그리드 선택 컴포넌트 — 이미지를 직접 보고 고르는 UI.
// 스타일 변환 모달의 '게임 에셋'·'추출 오브젝트' 탭과 '스타일 되돌리기'의 선택 목록이
// 모두 이걸 쓴다(단일/다중 선택만 다름). 선택의 시각 상태(테두리·✓ 배지)는 이 컴포넌트가,
// 선택에 따른 동작(이미지 로드·되돌리기 등)은 onChange를 받는 호출부가 담당한다.

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

const CARD_BASE =
  'relative flex flex-col items-center gap-0.5 rounded-lg bg-black/40 p-1 border border-transparent transition hover:bg-white/[0.08]'
const CARD_SELECTED =
  'relative flex flex-col items-center gap-0.5 rounded-lg bg-indigo-500/15 p-1 border border-indigo-500/60 ring-2 ring-indigo-500/60 transition'

export type ThumbnailItem = {
  id: string
  label: string
  thumbUrl: string
}

export type ThumbnailPickerOptions = {
  // true면 여러 개를 토글 선택(되돌리기), false(기본)면 하나만 선택(게임 에셋/추출 오브젝트).
  multiSelect?: boolean
  onChange?: (selectedIds: string[]) => void
}

export type ThumbnailPicker = {
  node: HTMLDivElement
  // 목록을 다시 그린다. 여전히 존재하는 id의 선택 상태는 유지한다. items가 비면 emptyMessage 표시.
  setItems: (items: ThumbnailItem[], emptyMessage: string) => void
  getSelected: () => string[]
  // 선택을 지정한다(사용자 동작과 동일하게 onChange를 발생 — '모두 선택/해제'용).
  setSelected: (ids: string[]) => void
  // 선택만 비운다(onChange 없음 — 탭/모드 전환 리셋용).
  clearSelection: () => void
}

export const createThumbnailPicker = (
  options: ThumbnailPickerOptions = {}
): ThumbnailPicker => {
  const multi = options.multiSelect === true
  const node = el('div', 'grid grid-cols-4 gap-1.5 max-h-56 overflow-y-auto')

  let items: ThumbnailItem[] = []
  const selected = new Set<string>()
  const cardById = new Map<string, HTMLButtonElement>()

  const applyCardStyle = (id: string, card: HTMLButtonElement): void => {
    card.className = selected.has(id) ? CARD_SELECTED : CARD_BASE
    const badge = card.querySelector('[data-badge]') as HTMLElement | null
    if (badge) {
      badge.style.display = selected.has(id) ? 'flex' : 'none'
    }
  }

  const refreshStyles = (): void => {
    for (const [id, card] of cardById) {
      applyCardStyle(id, card)
    }
  }

  const render = (emptyMessage: string): void => {
    node.replaceChildren()
    cardById.clear()
    if (items.length === 0) {
      node.append(
        el('div', 'col-span-full text-xs text-zinc-500 leading-relaxed', emptyMessage)
      )
      return
    }
    for (const item of items) {
      const card = el('button', CARD_BASE) as HTMLButtonElement
      card.type = 'button'
      card.title = item.label
      const thumb = el('img', 'h-12 w-full object-contain [image-rendering:pixelated]')
      thumb.src = item.thumbUrl
      thumb.loading = 'lazy'
      const caption = el('span', 'w-full truncate text-center text-[10px] text-zinc-400', item.label)
      card.append(thumb, caption)
      if (multi) {
        // 다중 선택은 테두리만으로는 모호해 모서리 ✓ 배지를 함께 둔다.
        const badge = el(
          'span',
          'absolute top-0.5 right-0.5 w-4 h-4 items-center justify-center rounded-full bg-indigo-500 text-white text-[10px] leading-none',
          '✓'
        )
        badge.setAttribute('data-badge', '')
        badge.style.display = 'none'
        card.append(badge)
      }
      card.addEventListener('click', () => {
        if (multi) {
          if (selected.has(item.id)) {
            selected.delete(item.id)
          } else {
            selected.add(item.id)
          }
        } else {
          // 이미 유일하게 선택된 항목 재클릭이면 onChange를 생략한다 — 이전 <select> 시맨틱과
          // 동일. (재발생 시 호출부가 콘텐츠를 다시 로드해 완료된 변환 결과가 폐기되는 회귀 방지.)
          if (selected.has(item.id) && selected.size === 1) {
            return
          }
          selected.clear()
          selected.add(item.id)
        }
        refreshStyles()
        options.onChange?.([...selected])
      })
      cardById.set(item.id, card)
      node.append(card)
    }
    refreshStyles()
  }

  return {
    node,
    setItems: (next, emptyMessage) => {
      items = next
      // 사라진 항목은 선택에서 제거(되돌리기로 목록이 줄어드는 경우).
      const present = new Set(next.map((item) => item.id))
      for (const id of [...selected]) {
        if (!present.has(id)) {
          selected.delete(id)
        }
      }
      render(emptyMessage)
    },
    getSelected: () => [...selected],
    setSelected: (ids) => {
      selected.clear()
      for (const id of ids) {
        selected.add(id)
      }
      refreshStyles()
      options.onChange?.([...selected])
    },
    clearSelection: () => {
      selected.clear()
      refreshStyles()
    }
  }
}
