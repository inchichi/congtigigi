const STYLE_SERVICE_BASE = '/api/style'

type StyledObject = {
  key: string
  label: string
  tilesetPath: string
  beforeUrl: string
  afterUrl: string
}

export interface BeforeAfterView {
  view: HTMLElement
  refresh: () => void
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

const assetLabel = (path: string): string => {
  const relative = path.replace(/^src\/games\/my-sample-rpg\/assets\//, '')
  return relative.replace(/\.[^.]+$/, '')
}

const sanitizeFileToken = (value: string): string =>
  value.replace(/[^a-zA-Z0-9가-힣_-]+/g, '-').replace(/^-+|-+$/g, '') || 'object'

const downloadBlob = (blob: Blob, fileName: string): void => {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.click()
  URL.revokeObjectURL(url)
}

const downloadImageFile = async (url: string, fileName: string): Promise<void> => {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) {
    throw new Error(`이미지를 내려받지 못했습니다 (HTTP ${response.status})`)
  }
  downloadBlob(await response.blob(), fileName)
}

const loadImage = (url: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('비교 이미지를 불러오지 못했습니다.'))
    image.src = url
  })

// 픽셀 에셋은 원본 크기가 작아 그대로 합성하면 라벨조차 안 보이므로,
// 한 변이 최소 96px이 되도록 정수 배율로 확대해 비교 이미지를 만든다.
const downloadComparisonImage = async (object: StyledObject, fileName: string): Promise<void> => {
  const [before, after] = await Promise.all([loadImage(object.beforeUrl), loadImage(object.afterUrl)])
  const scale = Math.max(1, Math.ceil(96 / Math.max(before.naturalWidth, after.naturalWidth, 1)))
  const gap = 12
  const labelHeight = 16
  const canvas = document.createElement('canvas')
  canvas.width = (before.naturalWidth + after.naturalWidth) * scale + gap
  canvas.height = Math.max(before.naturalHeight, after.naturalHeight) * scale + labelHeight
  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('캔버스를 사용할 수 없습니다.')
  }
  context.imageSmoothingEnabled = false
  context.font = '600 11px sans-serif'
  context.fillStyle = '#d9a85c'
  context.fillText('BEFORE', 0, 11)
  context.fillText('AFTER', before.naturalWidth * scale + gap, 11)
  context.drawImage(before, 0, labelHeight, before.naturalWidth * scale, before.naturalHeight * scale)
  context.drawImage(after, before.naturalWidth * scale + gap, labelHeight, after.naturalWidth * scale, after.naturalHeight * scale)
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
  if (!blob) {
    throw new Error('비교 이미지 생성에 실패했습니다.')
  }
  downloadBlob(blob, fileName)
}

const createPreview = (label: string, url: string): HTMLElement => {
  const frame = el(
    'div',
    'min-w-0 rounded-xl border border-[#d9a85c]/18 bg-[#171717] p-2'
  )
  const title = el('div', 'mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#9d9d9d]', label)
  const image = el('img', 'block h-[132px] w-full object-contain [image-rendering:pixelated]') as HTMLImageElement
  image.alt = label
  image.loading = 'lazy'
  image.src = url
  frame.append(title, image)
  return frame
}

export const createBeforeAfterView = (): BeforeAfterView => {
  const view = el('div', 'flex flex-col gap-3')
  view.hidden = true

  const heading = el('div', 'flex items-center justify-between gap-2')
  heading.append(
    el('div', 'text-[15px] text-[#e8d5a5]', '비포 / 애프터'),
    el('div', 'text-[11px] text-[#9d9d9d]', '변경된 오브젝트만 표시')
  )
  const status = el('div', 'text-[11px] text-[#9d9d9d]', '변경 내역을 불러오는 중…')
  const grid = el('div', 'grid grid-cols-1 gap-3 xl:grid-cols-2')
  view.append(heading, status, grid)

  let refreshToken = 0

  const render = (objects: StyledObject[]): void => {
    grid.replaceChildren()
    status.textContent = objects.length > 0
      ? `변경된 오브젝트 ${objects.length}개`
      : '아직 변경된 오브젝트가 없습니다.'
    if (objects.length === 0) {
      grid.append(
        el(
          'div',
          'col-span-full flex min-h-[220px] items-center justify-center rounded-xl border border-dashed border-[#d9a85c]/20 text-center text-[12px] leading-relaxed text-[#777777]',
          'FLUX로 오브젝트 스타일을 적용하면\n이곳에서 원본과 결과를 비교할 수 있습니다.'
        )
      )
      return
    }

    for (const object of objects) {
      const card = el('article', 'rounded-2xl border border-[#d9a85c]/24 bg-[#242426] p-3 shadow-[0_0_12px_rgba(0,0,0,0.18)]')
      const title = el('div', 'mb-2 flex items-center justify-between gap-2')
      title.append(
        el('div', 'truncate text-[13px] font-semibold text-[#e8d5a5]', object.label || object.key),
        el('div', 'shrink-0 rounded-full bg-[#d9a85c]/10 px-2 py-1 text-[10px] text-[#d9a85c]', object.key)
      )
      const comparison = el('div', 'grid grid-cols-2 gap-2')
      comparison.append(
        createPreview('BEFORE · 원본', object.beforeUrl),
        createPreview('AFTER · 적용 결과', object.afterUrl)
      )
      const downloads = el('div', 'mt-2 flex flex-wrap gap-1.5')
      const downloadButton = (label: string, action: () => Promise<void>): HTMLButtonElement => {
        const button = el(
          'button',
          'rounded-lg border border-[#d9a85c]/30 bg-[#1c1c1e] px-2.5 py-1.5 text-[10px] text-[#cbb27b] transition hover:border-[#d9a85c] hover:bg-[#2b2b2f] disabled:cursor-not-allowed disabled:opacity-45',
          label
        ) as HTMLButtonElement
        button.type = 'button'
        button.addEventListener('click', () => {
          button.disabled = true
          void action()
            .catch((caught) => {
              status.textContent = caught instanceof Error ? caught.message : String(caught)
            })
            .finally(() => {
              button.disabled = false
            })
        })
        return button
      }
      const fileToken = sanitizeFileToken(object.key)
      downloads.append(
        downloadButton('원본 저장', () => downloadImageFile(object.beforeUrl, `${fileToken}-before.png`)),
        downloadButton('결과 저장', () => downloadImageFile(object.afterUrl, `${fileToken}-after.png`)),
        downloadButton('비교 저장', () => downloadComparisonImage(object, `${fileToken}-compare.png`))
      )
      const path = el('div', 'mt-2 truncate text-[10px] text-[#777777]', assetLabel(object.tilesetPath))
      card.append(title, comparison, downloads, path)
      grid.append(card)
    }
  }

  const refresh = (): void => {
    const token = ++refreshToken
    status.textContent = '변경 내역을 불러오는 중…'
    void fetch(`${STYLE_SERVICE_BASE}/styled-objects`, { cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        return (await response.json()) as { objects?: StyledObject[] }
      })
      .then((data) => {
        if (token === refreshToken) {
          render(data.objects ?? [])
        }
      })
      .catch(() => {
        if (token !== refreshToken) {
          return
        }
        grid.replaceChildren()
        status.textContent = '스타일 서비스에 연결하지 못했습니다.'
        grid.append(
          el(
            'div',
            'col-span-full rounded-xl border border-dashed border-[#e06c6c]/35 p-6 text-center text-[12px] leading-relaxed text-[#e06c6c]',
            'FLUX 서비스 연결을 확인한 뒤 다시 시도하세요.'
          )
        )
      })
  }

  refresh()
  return { view, refresh }
}
