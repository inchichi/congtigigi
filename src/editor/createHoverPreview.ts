// 마우스 호버 시 떠 있는 "확대 미리보기" 박스를 보여주는 공용 컨트롤러.
// 배치 팔레트(타일/NPC/오브젝트)와 좌측 '현재 맵 에셋' 트리가 함께 쓴다.
//   · bind(target, build): target에 마우스를 올리면 build()가 만든 콘텐츠(이미지/슬라이스/캔버스)를
//     커서 옆 떠 있는 박스에 보여주고, 벗어나면 숨긴다. build()는 동기/비동기 모두 허용하며,
//     undefined를 반환하면 미리보기를 띄우지 않는다(예: 미리보기 이미지를 못 구한 엔티티).
//   · 결과 콘텐츠는 target별로 캐시한다(같은 항목 재호버 시 다시 만들지 않음).
// 박스 자체는 document.body에 한 번만 붙이고 pointer-events:none이라 마우스를 가로채지 않는다.

export interface HoverPreview {
  bind: (target: HTMLElement, build: () => HoverPreviewContent) => void
  destroy: () => void
}

type HoverPreviewContent = HTMLElement | undefined | Promise<HTMLElement | undefined>

// target 화면 위치 기준으로 박스를 오른쪽(공간 없으면 왼쪽)에 세로 중앙 정렬해 띄운다.
const placeBox = (box: HTMLElement, rect: DOMRect): void => {
  const gap = 12
  const margin = 8
  const boxWidth = box.offsetWidth
  const boxHeight = box.offsetHeight
  let left = rect.right + gap
  if (left + boxWidth > window.innerWidth - margin) {
    left = rect.left - gap - boxWidth
  }
  if (left < margin) {
    left = Math.max(margin, Math.min(window.innerWidth - boxWidth - margin, rect.left))
  }
  let top = rect.top + rect.height / 2 - boxHeight / 2
  top = Math.max(margin, Math.min(top, window.innerHeight - boxHeight - margin))
  box.style.left = `${Math.round(left)}px`
  box.style.top = `${Math.round(top)}px`
}

export const createHoverPreview = (): HoverPreview => {
  const box = document.createElement('div')
  box.className = 'hover-preview'
  box.style.display = 'none'
  document.body.append(box)

  // 현재 호버 중인 target. 비동기 build가 끝났을 때 여전히 같은 target 위인지 확인하는 데 쓴다.
  let activeTarget: HTMLElement | null = null

  const hide = (): void => {
    activeTarget = null
    box.style.display = 'none'
    box.replaceChildren()
  }

  const show = (content: HTMLElement, target: HTMLElement): void => {
    box.replaceChildren(content)
    box.style.display = 'block'
    // 콘텐츠를 넣은 뒤 크기가 정해지므로 그 다음에 위치를 잡는다.
    placeBox(box, target.getBoundingClientRect())
  }

  const bind = (target: HTMLElement, build: () => HoverPreviewContent): void => {
    let cached: HTMLElement | undefined
    let cacheBuilt = false
    // 비동기 build 도중 호버가 바뀌면 늦게 도착한 결과를 무시하기 위한 토큰.
    let token = 0

    target.addEventListener('mouseenter', () => {
      activeTarget = target
      const myToken = ++token
      if (cacheBuilt) {
        if (cached) {
          show(cached, target)
        }
        return
      }
      const built = build()
      if (built instanceof Promise) {
        void built.then((content) => {
          // 빌드 중 다른 곳으로 옮겨갔거나 떠났으면 버린다.
          if (myToken !== token || activeTarget !== target) {
            return
          }
          cached = content ?? undefined
          cacheBuilt = true
          if (cached) {
            show(cached, target)
          }
        })
      } else {
        cached = built ?? undefined
        cacheBuilt = true
        if (cached) {
          show(cached, target)
        }
      }
    })

    target.addEventListener('mouseleave', () => {
      token += 1
      if (activeTarget === target) {
        hide()
      }
    })
  }

  const destroy = (): void => {
    box.remove()
  }

  return { bind, destroy }
}

// ── 콘텐츠 빌더 ──

// 타일셋 한 칸을 확대해 보여주는 div(배경 슬라이스). 이미지 로드 불필요 — 즉시 반환.
export const buildTileSlicePreview = (opts: {
  imageUrl: string
  columns: number
  tileWidth: number
  tileHeight: number
  tileId: number
  targetSize?: number
}): HTMLElement => {
  const target = opts.targetSize ?? 128
  const scale = Math.max(1, Math.round(target / Math.max(opts.tileWidth, opts.tileHeight)))
  const col = opts.tileId % opts.columns
  const row = Math.floor(opts.tileId / opts.columns)
  const node = document.createElement('div')
  node.style.width = `${opts.tileWidth * scale}px`
  node.style.height = `${opts.tileHeight * scale}px`
  node.style.backgroundImage = `url(${opts.imageUrl})`
  node.style.backgroundRepeat = 'no-repeat'
  node.style.backgroundSize = `${opts.columns * opts.tileWidth * scale}px auto`
  node.style.backgroundPosition = `-${col * opts.tileWidth * scale}px -${row * opts.tileHeight * scale}px`
  node.style.imageRendering = 'pixelated'
  return node
}

// 통짜 이미지(추출 오브젝트 누끼 PNG·몬스터 스프라이트 시트)를 비율 유지로 보여주는 img.
export const buildImagePreview = (url: string, opts?: { maxSize?: number }): HTMLElement => {
  const max = opts?.maxSize ?? 192
  const img = document.createElement('img')
  img.src = url
  img.loading = 'lazy'
  img.style.display = 'block'
  img.style.maxWidth = `${max}px`
  img.style.maxHeight = `${max}px`
  img.style.imageRendering = 'pixelated'
  return img
}

// 여러 타일 셀(나무·분수 등 군집 오브젝트)을 경계 사각형에 맞춰 캔버스로 조립해 보여준다.
// 이미지를 로드해야 하므로 Promise를 반환한다(로드 실패/빈 셀이면 undefined → 미리보기 없음).
const imageCache = new Map<string, Promise<HTMLImageElement | undefined>>()

const loadImage = (url: string): Promise<HTMLImageElement | undefined> => {
  const existing = imageCache.get(url)
  if (existing) {
    return existing
  }
  const promise = new Promise<HTMLImageElement | undefined>((resolve) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => resolve(undefined)
    image.src = url
  })
  imageCache.set(url, promise)
  return promise
}

export const buildCellsCanvasPreview = async (opts: {
  imageUrl: string
  columns: number
  tileWidth: number
  tileHeight: number
  // col/row = 맵 타일 좌표(배치 위치), tileId = 타일셋 로컬 인덱스(어느 타일을 그릴지).
  cells: ReadonlyArray<{ col: number; row: number; tileId: number }>
  targetSize?: number
}): Promise<HTMLElement | undefined> => {
  if (opts.cells.length === 0) {
    return undefined
  }
  const image = await loadImage(opts.imageUrl)
  if (!image) {
    return undefined
  }
  const cols = opts.cells.map((cell) => cell.col)
  const rows = opts.cells.map((cell) => cell.row)
  const minCol = Math.min(...cols)
  const maxCol = Math.max(...cols)
  const minRow = Math.min(...rows)
  const maxRow = Math.max(...rows)
  const widthTiles = maxCol - minCol + 1
  const heightTiles = maxRow - minRow + 1
  const canvas = document.createElement('canvas')
  canvas.width = widthTiles * opts.tileWidth
  canvas.height = heightTiles * opts.tileHeight
  const context = canvas.getContext('2d')
  if (!context) {
    return undefined
  }
  context.imageSmoothingEnabled = false
  for (const cell of opts.cells) {
    // 소스는 tileId로 타일셋에서 찾고(맵 좌표가 아니다), 목적지는 군집의 경계 사각형 기준 상대 위치.
    const srcCol = cell.tileId % opts.columns
    const srcRow = Math.floor(cell.tileId / opts.columns)
    context.drawImage(
      image,
      srcCol * opts.tileWidth,
      srcRow * opts.tileHeight,
      opts.tileWidth,
      opts.tileHeight,
      (cell.col - minCol) * opts.tileWidth,
      (cell.row - minRow) * opts.tileHeight,
      opts.tileWidth,
      opts.tileHeight
    )
  }
  // 작은 오브젝트는 확대, 큰 건물은 축소해 미리보기 박스 크기를 일정하게 유지한다.
  const target = opts.targetSize ?? 176
  const natural = Math.max(canvas.width, canvas.height)
  const scale =
    natural >= target ? target / natural : Math.min(8, Math.max(1, Math.floor(target / natural)))
  canvas.style.width = `${Math.round(canvas.width * scale)}px`
  canvas.style.height = `${Math.round(canvas.height * scale)}px`
  canvas.style.imageRendering = 'pixelated'
  canvas.style.display = 'block'
  return canvas
}
