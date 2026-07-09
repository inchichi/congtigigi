// @ts-nocheck
import { createThumbnailPicker } from './createThumbnailPicker'

const el = <K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string
): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag)
  node.className = className
  if (text !== undefined) node.textContent = text
  return node
}

const LABEL = 'text-[11px] font-semibold uppercase tracking-wider text-zinc-400'
const CARD = 'rounded-xl border border-white/10 bg-white/[0.03] p-4 flex flex-col gap-2'
const PRIMARY_BUTTON =
  'rounded-lg px-4 py-2 bg-indigo-500 text-white text-sm font-medium shadow-sm shadow-indigo-500/30 transition hover:bg-indigo-400 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none'
const GHOST_BUTTON =
  'rounded-lg px-3.5 py-2 bg-white/[0.04] text-zinc-300 text-sm border border-white/10 transition hover:bg-white/[0.08] hover:text-zinc-100 hover:border-white/20 disabled:opacity-40 disabled:cursor-not-allowed'
const MODE_TAB =
  'rounded-md px-2.5 py-1 text-xs text-zinc-400 border border-transparent transition hover:text-zinc-200'
const MODE_TAB_ACTIVE =
  'rounded-md px-2.5 py-1 text-xs bg-indigo-500/15 text-indigo-200 border border-indigo-500/30 transition'

const STYLE_SERVICE_BASE = '/api/style'
const SERVICE_GUIDE =
  '편집 서비스에 연결할 수 없습니다 — style-service 폴더에서 python server.py를 실행하세요.'

const fileStem = (name: string): string => name.replace(/\.[^.]+$/, '')
const safeName = (name: string): string => name.replace(/[^\w가-힣]+/gu, '_')

const base64ToBlob = (base64: string): Blob => {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return new Blob([bytes], { type: 'image/png' })
}

type ContentMode = 'file' | 'asset' | 'object' | 'extracted'

export type StyleTransferMapObject = {
  label: string
  tilesetImagePath: string
  tileWidth: number
  tileHeight: number
  columns: number
  cells: Array<{ col: number; row: number; tileId: number }>
  sharedOutsideCells: number
  bannerText?: string
}

export type StyleTransferAssetTarget = {
  path: string
  label: string
  note?: string
  monsterKey?: string
}

type ApplyContext =
  | { kind: 'asset'; path: string }
  | { kind: 'extracted-object'; key: string; path: string }

type TileStyleContext = {
  tilesetImagePath: string
  tilesetImageUrl: string
  columns: number
  tileWidth: number
  tileHeight: number
}

export interface StyleTransferModal {
  openButton: HTMLButtonElement
  backdrop: HTMLDivElement
  openForMapObject: (target: StyleTransferMapObject) => void
  openForAsset: (target: StyleTransferAssetTarget) => void
}

export type CreateStyleTransferModalInput = {
  onAssetChanged?: () => void
  getTileStyleTarget?: () => TileStyleContext | undefined
}

const createImagePicker = (): {
  input: HTMLInputElement
  thumb: HTMLImageElement
  getFile: () => File | undefined
  setFile: (file: File | undefined) => void
  onChange: (listener: () => void) => void
} => {
  const input = el(
    'input',
    'text-xs text-zinc-400 file:mr-2 file:rounded-lg file:border-0 file:bg-white/[0.08] file:px-3 file:py-1.5 file:text-xs file:text-zinc-200 file:cursor-pointer'
  ) as HTMLInputElement
  input.type = 'file'
  input.accept = 'image/png,image/jpeg,image/webp'
  const thumb = el('img', 'max-h-36 w-full rounded-lg object-contain bg-black/40')
  thumb.style.display = 'none'

  let file: File | undefined
  let listener: (() => void) | undefined
  const setFile = (next: File | undefined): void => {
    file = next
    if (thumb.src) URL.revokeObjectURL(thumb.src)
    if (file) {
      thumb.src = URL.createObjectURL(file)
      thumb.style.display = 'block'
    } else {
      thumb.removeAttribute('src')
      thumb.style.display = 'none'
    }
    listener?.()
  }
  input.addEventListener('change', () => setFile(input.files?.[0] ?? undefined))

  return {
    input,
    thumb,
    getFile: () => file,
    setFile,
    onChange: (next) => {
      listener = next
    }
  }
}

const promptSuffix = (alpha: number): string => {
  if (alpha <= 0.33) return 'Subtle edit. Keep the original composition and identity mostly intact.'
  if (alpha <= 0.66) return 'Balanced edit. Preserve the subject while clearly applying the requested style.'
  return 'Strong edit. Transform the image noticeably while keeping it coherent.'
}

export const createStyleTransferModal = (
  input: CreateStyleTransferModalInput = {}
): StyleTransferModal => {
  const { onAssetChanged } = input
  const openButton = el(
    'button',
    'rounded-lg px-2.5 py-1 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100',
    '🎨 편집'
  ) as HTMLButtonElement
  openButton.type = 'button'

  const backdrop = el('div', 'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4')
  backdrop.style.display = 'none'
  const panel = el('div', 'w-full max-w-2xl max-h-[92vh] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900 p-5 flex flex-col gap-4 shadow-2xl')

  const top = el('div', 'flex items-center justify-between')
  top.append(el('span', 'text-sm font-semibold tracking-tight', '🎨 이미지 편집 (FLUX.1 Kontext)'))
  const close = el('button', 'text-zinc-500 text-sm transition hover:text-zinc-200', '✕') as HTMLButtonElement
  close.type = 'button'
  top.append(close)

  const modeTabs = el('div', 'flex items-center gap-1')
  const fileTab = el('button', MODE_TAB_ACTIVE, '파일') as HTMLButtonElement
  const assetTab = el('button', MODE_TAB, '게임 에셋') as HTMLButtonElement
  const extractedTab = el('button', MODE_TAB, '추출 오브젝트') as HTMLButtonElement
  fileTab.type = assetTab.type = extractedTab.type = 'button'
  modeTabs.append(fileTab, assetTab, extractedTab)

  const contentCard = el('div', CARD)
  const contentHead = el('div', 'flex items-center justify-between gap-2')
  contentHead.append(el('div', LABEL, '콘텐츠'), modeTabs)
  const contentPicker = createImagePicker()
  const assetPicker = createThumbnailPicker({
    multiSelect: true,
    onChange: (ids) => {
      selectedAssetPaths = ids
      updateBatchToolbar()
      syncButtons()
    }
  })
  const extractedPicker = createThumbnailPicker({
    multiSelect: true,
    onChange: (ids) => {
      selectedExtractedKeys = ids
      updateBatchToolbar()
      syncButtons()
    }
  })
  const objectBanner = el('div', 'flex items-center justify-between gap-2 rounded-lg bg-indigo-500/10 border border-indigo-500/25 px-3 py-2')
  objectBanner.style.display = 'none'
  const objectLabel = el('span', 'text-xs text-indigo-200 truncate')
  const objectClear = el('button', 'shrink-0 text-xs text-zinc-400 transition hover:text-zinc-200', '해제') as HTMLButtonElement
  objectClear.type = 'button'
  objectBanner.append(objectLabel, objectClear)
  const objectPreview = el('img', 'max-h-36 w-full rounded-lg object-contain bg-black/40 [image-rendering:pixelated]')
  objectPreview.style.display = 'none'
  const assetNote = el('div', 'text-xs text-amber-300/90 leading-relaxed')
  assetNote.style.display = 'none'
  contentCard.append(contentHead, objectBanner, assetNote, contentPicker.input, assetPicker.node, extractedPicker.node, contentPicker.thumb, objectPreview)

  const promptCard = el('div', CARD)
  const promptInput = document.createElement('textarea')
  promptInput.className =
    'min-h-28 w-full rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-500'
  promptInput.rows = 5
  promptInput.placeholder =
    '예: 따뜻한 수채화 느낌, 부드러운 파스텔 톤, 손그림 질감, 배경은 유지하고 캐릭터만 더 감성적으로'
  promptCard.append(
    el('div', LABEL, '편집 프롬프트'),
    el('div', 'text-xs text-zinc-500 leading-relaxed', '스타일 이미지는 사용하지 않습니다. 원하는 결과를 텍스트로 적어주세요.'),
    promptInput
  )

  const alphaWrap = el('div', CARD)
  const alphaHead = el('div', 'flex items-center justify-between')
  const alphaValue = el('span', 'text-xs text-zinc-300', '1.00')
  alphaHead.append(el('div', LABEL, '편집 강도 (alpha)'), alphaValue)
  const alphaSlider = el('input', 'w-full accent-indigo-500') as HTMLInputElement
  alphaSlider.type = 'range'
  alphaSlider.min = '0'
  alphaSlider.max = '1'
  alphaSlider.step = '0.05'
  alphaSlider.value = '1'
  alphaSlider.addEventListener('input', () => {
    alphaValue.textContent = Number(alphaSlider.value).toFixed(2)
  })
  alphaWrap.append(alphaHead, alphaSlider)

  const actions = el('div', 'flex flex-wrap items-center gap-2')
  const runButton = el('button', PRIMARY_BUTTON, '변환') as HTMLButtonElement
  const applyButton = el('button', GHOST_BUTTON, '게임에 적용') as HTMLButtonElement
  const revertButton = el('button', GHOST_BUTTON, '원본으로 되돌리기') as HTMLButtonElement
  const saveButton = el('button', GHOST_BUTTON, 'PNG 저장') as HTMLButtonElement
  const saveOrigButton = el('button', GHOST_BUTTON, '원본 PNG 저장') as HTMLButtonElement
  const batchButton = el('button', PRIMARY_BUTTON, '일괄 변환') as HTMLButtonElement
  const status = el('span', 'text-xs text-zinc-500 min-h-4', '')
  for (const button of [runButton, applyButton, revertButton, saveButton, saveOrigButton, batchButton]) {
    button.type = 'button'
  }
  applyButton.disabled = revertButton.disabled = saveButton.disabled = saveOrigButton.disabled = batchButton.disabled = true
  batchButton.style.display = 'none'
  actions.append(runButton, batchButton, applyButton, revertButton, saveOrigButton, saveButton, status)

  const progressWrap = el('div', 'flex items-center gap-2')
  const progressTrack = el('div', 'h-2 flex-1 overflow-hidden rounded-full bg-white/10')
  const progressBar = el('div', 'h-full w-0 rounded-full bg-amber-400')
  const progressLabel = el('span', 'w-9 shrink-0 text-right text-xs tabular-nums text-amber-300', '0%')
  progressBar.style.transition = 'width 0.15s ease-out'
  progressTrack.append(progressBar)
  progressWrap.append(progressTrack, progressLabel)
  progressWrap.style.display = 'none'

  const resultWrap = el('div', CARD)
  const resultImage = el('img', 'max-h-72 w-full rounded-lg object-contain bg-black/40')
  resultImage.style.display = 'none'
  resultWrap.append(el('div', LABEL, '결과 미리보기'), resultImage)

  panel.append(top, contentCard, promptCard, alphaWrap, actions, progressWrap, resultWrap)
  backdrop.append(panel)

  let mode: ContentMode = 'file'
  let mapObject: StyleTransferMapObject | undefined
  let assetPath: string | undefined
  let assetMonsterKey: string | undefined
  let selectedAssetPaths: string[] = []
  let selectedExtractedKeys: string[] = []
  let isRunning = false
  let isApplying = false
  let isReverting = false
  let isBatchRunning = false
  let runSeq = 0
  let progressTimer: number | undefined
  let progressPct = 0
  let progressBusy = false
  let resultBlob: Blob | undefined
  let resultName = 'edited.png'
  let applyTarget: { kind: 'asset' | 'patched-tileset' | 'extracted-object'; path?: string; key?: string; blob: Blob } | undefined
  let revertPath: string | undefined

  const setStatus = (message: string): void => {
    status.textContent = message
  }

  const updateProgress = (pct: number): void => {
    progressPct = pct
    progressBar.style.width = `${pct}%`
    progressLabel.textContent = `${Math.round(pct)}%`
  }
  const startProgress = (): void => {
    if (progressTimer !== undefined) window.clearInterval(progressTimer)
    updateProgress(0)
    progressWrap.style.display = 'flex'
    progressTimer = window.setInterval(() => updateProgress(progressPct + (90 - progressPct) * 0.08), 120)
  }
  const finishProgress = (): void => {
    if (progressTimer !== undefined) {
      window.clearInterval(progressTimer)
      progressTimer = undefined
    }
    updateProgress(100)
    window.setTimeout(() => {
      progressWrap.style.display = 'none'
      updateProgress(0)
    }, 350)
  }

  const clearResult = (): void => {
    resultBlob = undefined
    applyTarget = undefined
    if (resultImage.src) URL.revokeObjectURL(resultImage.src)
    resultImage.removeAttribute('src')
    resultImage.style.display = 'none'
  }

  const loadImage = (src: string): Promise<HTMLImageElement> =>
    new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error('이미지를 읽을 수 없습니다.'))
      img.src = src
    })

  const composeOriginalObject = async (obj: StyleTransferMapObject): Promise<Blob> => {
    const img = await loadImage(`/${obj.tilesetImagePath}`)
    const cols = obj.cells.map((c) => c.col)
    const rows = obj.cells.map((c) => c.row)
    const minCol = Math.min(...cols)
    const minRow = Math.min(...rows)
    const w = (Math.max(...cols) - minCol + 1) * obj.tileWidth
    const h = (Math.max(...rows) - minRow + 1) * obj.tileHeight
    const canvas = document.createElement('canvas')
    canvas.width = w
    canvas.height = h
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('canvas 컨텍스트를 만들 수 없습니다.')
    ctx.imageSmoothingEnabled = false
    for (const cell of obj.cells) {
      const sx = (cell.tileId % obj.columns) * obj.tileWidth
      const sy = Math.floor(cell.tileId / obj.columns) * obj.tileHeight
      const dx = (cell.col - minCol) * obj.tileWidth
      const dy = (cell.row - minRow) * obj.tileHeight
      ctx.drawImage(img, sx, sy, obj.tileWidth, obj.tileHeight, dx, dy, obj.tileWidth, obj.tileHeight)
    }
    return await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG 생성 실패'))), 'image/png')
    )
  }

  const refreshRevertState = async (): Promise<void> => {
    const path = mapObject?.tilesetImagePath ?? (mode === 'asset' && selectedAssetPaths.length === 1 ? selectedAssetPaths[0] : undefined)
    if (!path) {
      revertPath = undefined
      revertButton.disabled = true
      return
    }
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/asset-status?path=${encodeURIComponent(path)}`)
      if (!response.ok) throw new Error(String(response.status))
      const info = (await response.json()) as { hasOriginal?: boolean }
      revertPath = info.hasOriginal ? path : undefined
      revertButton.disabled = revertPath === undefined || isReverting || isRunning || isApplying || isBatchRunning
    } catch {
      revertPath = undefined
      revertButton.disabled = true
    }
  }

  const currentSelection = (): string[] =>
    mode === 'asset' ? selectedAssetPaths : mode === 'extracted' ? selectedExtractedKeys : []

  const updateBatchToolbar = (): void => {
    const grid = mode === 'asset' || mode === 'extracted'
    batchButton.style.display = grid ? 'inline-flex' : 'none'
    batchButton.disabled = isRunning || isApplying || isReverting || isBatchRunning || currentSelection().length < 2 || promptInput.value.trim().length === 0
  }

  const syncButtons = (): void => {
    const busy = isRunning || isApplying || isReverting || isBatchRunning
    if (busy && !progressBusy) startProgress()
    else if (!busy && progressBusy) finishProgress()
    progressBusy = busy

    const hasContent =
      mapObject !== undefined ||
      contentPicker.getFile() !== undefined ||
      (mode === 'asset' && selectedAssetPaths.length === 1) ||
      (mode === 'extracted' && selectedExtractedKeys.length === 1)
    const hasPrompt = promptInput.value.trim().length > 0

    runButton.disabled = busy || !hasContent || !hasPrompt
    applyButton.disabled = busy || !resultBlob || !applyTarget
    saveButton.disabled = !resultBlob || busy
    saveOrigButton.disabled = !hasContent || busy
    revertButton.disabled = busy || revertPath === undefined
    batchButton.disabled = busy || currentSelection().length < 2 || !hasPrompt
  }

  const setMode = (next: ContentMode): void => {
    mode = next
    fileTab.className = mode === 'file' ? MODE_TAB_ACTIVE : MODE_TAB
    assetTab.className = mode === 'asset' ? MODE_TAB_ACTIVE : MODE_TAB
    extractedTab.className = mode === 'extracted' ? MODE_TAB_ACTIVE : MODE_TAB
    contentPicker.input.style.display = mode === 'file' ? 'block' : 'none'
    assetPicker.node.style.display = mode === 'asset' ? 'grid' : 'none'
    extractedPicker.node.style.display = mode === 'extracted' ? 'grid' : 'none'
    objectBanner.style.display = mode === 'object' ? 'flex' : 'none'
    objectPreview.style.display = mode === 'object' ? 'block' : 'none'
    contentPicker.thumb.style.display = mode === 'file' && contentPicker.getFile() ? 'block' : 'none'
    assetNote.style.display = mode === 'asset' && assetNote.textContent ? 'block' : 'none'
    clearResult()
    void refreshRevertState()
    updateBatchToolbar()
    syncButtons()
  }

  const loadAssetList = async (): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/assets`)
      if (!response.ok) throw new Error(String(response.status))
      const data = (await response.json()) as { assets: Array<{ path: string; size: number }> }
      assetPicker.setItems(
        data.assets.map((asset) => ({ id: asset.path, label: asset.path.split('/').pop() ?? asset.path, thumbUrl: `/${asset.path}` })),
        '게임 에셋 PNG가 없습니다.'
      )
      selectedAssetPaths = assetPicker.getSelected()
    } catch {
      assetPicker.setItems([], '게임 에셋 목록을 불러오지 못했습니다.')
      setStatus(SERVICE_GUIDE)
    }
    updateBatchToolbar()
    syncButtons()
  }

  const loadExtractedList = async (): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/extracted-objects`)
      if (!response.ok) throw new Error(String(response.status))
      const data = (await response.json()) as {
        objects: Array<{ id: string; key: string; label: string; tilesetPath: string; sharedOutsideCells: number }>
      }
      extractedPicker.setItems(
        data.objects.map((obj) => ({ id: obj.key, label: obj.label, thumbUrl: `${STYLE_SERVICE_BASE}/extracted-objects/${encodeURIComponent(obj.key)}.png` })),
        '추출 오브젝트가 없습니다.'
      )
      selectedExtractedKeys = extractedPicker.getSelected()
    } catch {
      extractedPicker.setItems([], '추출 오브젝트 목록을 불러오지 못했습니다.')
      setStatus(SERVICE_GUIDE)
    }
    updateBatchToolbar()
    syncButtons()
  }

  const openObjectPreview = async (target: StyleTransferMapObject): Promise<void> => {
    try {
      const blob = await composeOriginalObject(target)
      if (objectPreview.src) URL.revokeObjectURL(objectPreview.src)
      objectPreview.src = URL.createObjectURL(blob)
      objectPreview.style.display = 'block'
    } catch {
      objectPreview.style.display = 'none'
    }
  }

  const setObjectTarget = (target: StyleTransferMapObject): void => {
    mode = 'object'
    mapObject = target
    objectLabel.textContent = target.bannerText ?? `맵 오브젝트: ${target.label} · ${target.cells.length}개 타일`
    objectBanner.style.display = 'flex'
    objectPreview.style.display = 'block'
    void openObjectPreview(target)
    clearResult()
    updateBatchToolbar()
    void refreshRevertState()
    syncButtons()
  }

  const runObjectTransfer = async (target: StyleTransferMapObject, prompt: string, alpha: number, seq: number): Promise<void> => {
    const form = new FormData()
    form.append('prompt', promptSuffix(alpha) + ' ' + prompt)
    form.append('tileset_path', target.tilesetImagePath)
    form.append('tile_width', String(target.tileWidth))
    form.append('tile_height', String(target.tileHeight))
    form.append('columns', String(target.columns))
    form.append('cells', JSON.stringify(target.cells))
    form.append('alpha', String(alpha))
    const response = await fetch(`${STYLE_SERVICE_BASE}/stylize-object`, { method: 'POST', body: form })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
    }
    const data = (await response.json()) as { object_png: string; tileset_png: string }
    if (seq !== runSeq) return
    resultBlob = base64ToBlob(data.object_png)
    resultName = `${safeName(target.label)}_edited.png`
    applyTarget = {
      kind: 'patched-tileset',
      path: target.tilesetImagePath,
      blob: base64ToBlob(data.tileset_png)
    }
    setStatus('완료: 오브젝트와 타일셋이 편집되었습니다.')
  }

  const runImageTransfer = async (
    contentFile: File,
    prompt: string,
    alpha: number,
    applyContext: ApplyContext | undefined,
    seq: number
  ): Promise<void> => {
    const form = new FormData()
    form.append('content', contentFile)
    form.append('prompt', promptSuffix(alpha) + ' ' + prompt)
    form.append('alpha', String(alpha))
    let endpoint = `${STYLE_SERVICE_BASE}/style-transfer`
    if (assetMonsterKey && assetPath) {
      endpoint = `${STYLE_SERVICE_BASE}/stylize-monster`
      form.append('sheet_path', assetPath)
      form.append('monster_key', assetMonsterKey)
    }
    const response = await fetch(endpoint, { method: 'POST', body: form })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
    }
    const blob = await response.blob()
    if (seq !== runSeq) return
    resultBlob = blob
    const promptSlug = safeName(prompt.slice(0, 24) || 'prompt')
    resultName = `${fileStem(contentFile.name)}_edited_${promptSlug}.png`
    applyTarget = applyContext ? { ...applyContext, blob } : undefined
    setStatus(applyTarget ? '완료: 결과를 게임에 적용할 수 있습니다.' : `완료: ${resultName}`)
  }

  const resolveContentFile = async (): Promise<File | undefined> => {
    if (mode === 'file') return contentPicker.getFile()
    if (mode === 'asset' && selectedAssetPaths.length === 1) {
      const path = selectedAssetPaths[0]
      const response = await fetch(`/${path}`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      return new File([await response.blob()], path.split('/').pop() ?? 'asset.png', { type: 'image/png' })
    }
    if (mode === 'extracted' && selectedExtractedKeys.length === 1) {
      const key = selectedExtractedKeys[0]
      const response = await fetch(`${STYLE_SERVICE_BASE}/extracted-objects/${encodeURIComponent(key)}.png`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      return new File([await response.blob()], `${key}.png`, { type: 'image/png' })
    }
    return undefined
  }

  const runTransfer = async (): Promise<void> => {
    const prompt = promptInput.value.trim()
    const alpha = Number(alphaSlider.value)
    if (!prompt || isRunning) return
    const target = mapObject
    const seq = ++runSeq
    isRunning = true
    syncButtons()
    setStatus('변환 중...')
    try {
      if (target) {
        await runObjectTransfer(target, prompt, alpha, seq)
      } else {
        const contentFile = await resolveContentFile()
        if (!contentFile) throw new Error('콘텐츠 이미지를 찾을 수 없습니다.')
        const applyContext: ApplyContext | undefined =
          mode === 'asset' && selectedAssetPaths.length === 1
            ? ({ kind: 'asset', path: selectedAssetPaths[0] as string } as ApplyContext)
            : mode === 'extracted' && selectedExtractedKeys.length === 1
              ? ({ kind: 'extracted-object', key: selectedExtractedKeys[0], path: `${selectedExtractedKeys[0]}.png` } as ApplyContext)
              : undefined
        await runImageTransfer(contentFile, prompt, alpha, applyContext, seq)
      }
      if (resultBlob) {
        if (resultImage.src) URL.revokeObjectURL(resultImage.src)
        resultImage.src = URL.createObjectURL(resultBlob)
        resultImage.style.display = 'block'
      }
      onAssetChanged?.()
    } catch (error) {
      setStatus(error instanceof TypeError ? SERVICE_GUIDE : `실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isRunning = false
      syncButtons()
    }
  }

  const runBatch = async (): Promise<void> => {
    const prompt = promptInput.value.trim()
    const alpha = Number(alphaSlider.value)
    const selection = currentSelection()
    if (!prompt || selection.length < 2 || isBatchRunning) return
    const targets =
      mode === 'asset'
        ? selection.map((path) => ({ kind: 'asset', path }))
        : selection.map((key) => ({ kind: 'object', key }))
    isBatchRunning = true
    syncButtons()
    setStatus(`일괄 변환 중... (${selection.length}개)`)
    try {
      const form = new FormData()
      form.append('prompt', promptSuffix(alpha) + ' ' + prompt)
      form.append('alpha', String(alpha))
      form.append('targets', JSON.stringify(targets))
      const response = await fetch(`${STYLE_SERVICE_BASE}/batch-apply`, { method: 'POST', body: form })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      const result = (await response.json()) as { applied: string[]; failed: unknown[] }
      onAssetChanged?.()
      setStatus(`일괄 적용 완료: ${result.applied.length}개`)
    } catch (error) {
      setStatus(error instanceof TypeError ? SERVICE_GUIDE : `실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isBatchRunning = false
      syncButtons()
    }
  }

  const runApply = async (): Promise<void> => {
    if (!resultBlob || !applyTarget || isApplying) return
    isApplying = true
    syncButtons()
    try {
      const form = new FormData()
      form.append('file', new File([resultBlob], 'patched.png', { type: 'image/png' }))
      let endpoint = `${STYLE_SERVICE_BASE}/apply-asset`
      if (applyTarget.kind === 'extracted-object') {
        endpoint = `${STYLE_SERVICE_BASE}/apply-object`
        form.append('object_key', applyTarget.key ?? '')
      } else {
        form.append('path', applyTarget.path ?? '')
      }
      const response = await fetch(endpoint, { method: 'POST', body: form })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      revertPath = applyTarget.path
      onAssetChanged?.()
      setStatus('게임에 적용되었습니다.')
    } catch (error) {
      setStatus(error instanceof TypeError ? SERVICE_GUIDE : `실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isApplying = false
      syncButtons()
    }
  }

  const runRevert = async (): Promise<void> => {
    if (!revertPath || isReverting) return
    isReverting = true
    syncButtons()
    try {
      const form = new FormData()
      form.append('path', revertPath)
      const response = await fetch(`${STYLE_SERVICE_BASE}/revert-asset`, { method: 'POST', body: form })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      revertPath = undefined
      onAssetChanged?.()
      setStatus('원본으로 되돌렸습니다.')
    } catch (error) {
      setStatus(error instanceof TypeError ? SERVICE_GUIDE : `실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isReverting = false
      syncButtons()
    }
  }

  const runSaveOriginal = async (): Promise<void> => {
    try {
      if (mapObject) {
        const blob = await composeOriginalObject(mapObject)
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `${safeName(mapObject.label)}_original.png`
        link.click()
        URL.revokeObjectURL(url)
        setStatus('원본 PNG를 저장했습니다.')
        return
      }
      const contentFile = await resolveContentFile()
      if (!contentFile) {
        setStatus('저장할 원본이 없습니다.')
        return
      }
      const url = URL.createObjectURL(contentFile)
      const link = document.createElement('a')
      link.href = url
      link.download = contentFile.name
      link.click()
      URL.revokeObjectURL(url)
      setStatus('원본 PNG를 저장했습니다.')
    } catch (error) {
      setStatus(`실패: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const checkHealth = async (): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/health`)
      if (!response.ok) throw new Error(String(response.status))
      const info = (await response.json()) as { status?: string; service?: string; endpoint?: string; loaded?: boolean }
      if (info.status === 'ok') {
        if (info.loaded) {
          setStatus(`편집 서비스 연결됨 (${info.service ?? 'FLUX.1 Kontext'})`)
        } else {
          setStatus(`편집 서비스는 연결됨 (${info.service ?? 'FLUX.1 Kontext'}) - 모델 로딩 중`)
        }
      } else {
        setStatus('편집 서비스는 실행 중이지만 설정이 부족합니다.')
      }
    } catch {
      setStatus(SERVICE_GUIDE)
    }
  }

  const openModal = (): void => {
    backdrop.style.display = 'flex'
    void checkHealth()
  }

  fileTab.addEventListener('click', () => setMode('file'))
  assetTab.addEventListener('click', () => {
    setMode('asset')
    void loadAssetList()
  })
  extractedTab.addEventListener('click', () => {
    setMode('extracted')
    void loadExtractedList()
  })
  promptInput.addEventListener('input', syncButtons)
  alphaSlider.addEventListener('input', syncButtons)
  contentPicker.onChange(() => {
    clearResult()
    syncButtons()
  })
  objectClear.addEventListener('click', () => {
    mapObject = undefined
    objectBanner.style.display = 'none'
    objectPreview.style.display = 'none'
    setMode('file')
  })
  runButton.addEventListener('click', () => void runTransfer())
  batchButton.addEventListener('click', () => void runBatch())
  applyButton.addEventListener('click', () => void runApply())
  revertButton.addEventListener('click', () => void runRevert())
  saveOrigButton.addEventListener('click', () => void runSaveOriginal())
  saveButton.addEventListener('click', () => {
    if (!resultBlob) return
    const url = URL.createObjectURL(resultBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = resultName
    link.click()
    URL.revokeObjectURL(url)
    setStatus(`저장됨: ${resultName}`)
  })
  close.addEventListener('click', () => {
    backdrop.style.display = 'none'
  })
  backdrop.addEventListener('click', (event) => {
    if (event.target === backdrop) backdrop.style.display = 'none'
  })
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && backdrop.style.display !== 'none') backdrop.style.display = 'none'
  })

  const setObjectMode = (target: StyleTransferMapObject): void => {
    mapObject = target
    objectLabel.textContent = target.bannerText ?? `맵 오브젝트: ${target.label} · ${target.cells.length}개 타일`
    objectBanner.style.display = 'flex'
    void openObjectPreview(target)
    clearResult()
    setMode('object')
  }

  const openForMapObject = (target: StyleTransferMapObject): void => {
    setObjectMode(target)
    openModal()
  }

  const openForAsset = (target: StyleTransferAssetTarget): void => {
    mode = 'asset'
    assetPath = target.path
    assetMonsterKey = target.monsterKey
    assetNote.textContent = target.note ?? ''
    assetNote.style.display = target.note ? 'block' : 'none'
    contentPicker.setFile(undefined)
    setMode('asset')
    void loadAssetList().then(() => {
      selectedAssetPaths = [target.path]
      assetPicker.setSelected([target.path])
      syncButtons()
    })
    openModal()
  }

  void checkHealth()
  syncButtons()
  updateBatchToolbar()

  return { openButton, backdrop, openForMapObject, openForAsset }
}
