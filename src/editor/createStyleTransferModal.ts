// AdaIN 스타일 트랜스퍼 모달 (헤더 🎨 버튼).
// 콘텐츠는 세 가지로 고를 수 있다 — ① 파일 업로드 ② 게임 에셋(src/games/my-sample-rpg/assets의 PNG)
// ③ 맵 오브젝트(엔티티 트리에서 클릭한 타일 군집 — openForMapObject로 진입).
// 변환은 로컬 Python 서비스(style-service, /api/style 프록시)가 수행하고, ②③은 결과를
// '게임에 적용'으로 src/games/my-sample-rpg/assets에 백업 후 덮어써 게임(iframe)이 Vite 리로드로 즉시 반영된다.
// fetch가 비동기라 추론 중에도 에디터 UI는 멈추지 않는다.
// 모달 패턴(인라인 display 토글·백드롭 클릭·Escape 닫기)은 createEditorApp의 설정 모달과 동일.

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

const LABEL =
  'text-[11px] font-semibold uppercase tracking-wider text-zinc-400'
const CARD =
  'rounded-xl border border-white/10 bg-white/[0.03] p-4 flex flex-col gap-2'
const PRIMARY_BUTTON =
  'rounded-lg px-4 py-2 bg-indigo-500 text-white text-sm font-medium shadow-sm shadow-indigo-500/30 transition hover:bg-indigo-400 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none'
const GHOST_BUTTON =
  'rounded-lg px-3.5 py-2 bg-white/[0.04] text-zinc-300 text-sm border border-white/10 transition hover:bg-white/[0.08] hover:text-zinc-100 hover:border-white/20 disabled:opacity-40 disabled:cursor-not-allowed'
const MODE_TAB =
  'rounded-md px-2.5 py-1 text-xs text-zinc-400 border border-transparent transition hover:text-zinc-200'
const MODE_TAB_ACTIVE =
  'rounded-md px-2.5 py-1 text-xs bg-indigo-500/15 text-indigo-200 border border-indigo-500/30 transition'
const FIELD_SELECT =
  'w-full rounded-lg border border-white/15 bg-black/40 px-2 py-1.5 text-xs text-zinc-100 outline-none'

const STYLE_SERVICE_BASE = '/api/style'
const SERVICE_GUIDE =
  '스타일 서비스에 연결할 수 없습니다 — style-service 폴더에서 python server.py를 실행하세요.'

const fileStem = (name: string): string => name.replace(/\.[^.]+$/, '')

const base64ToBlob = (base64: string): Blob => {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Blob([bytes], { type: 'image/png' })
}

// 엔티티 트리에서 클릭한 타일 군집을 부분 변환하는 데 필요한 모든 것 — createEditorApp이 만든다.
export type StyleTransferMapObject = {
  label: string
  tilesetImagePath: string
  tileWidth: number
  tileHeight: number
  columns: number
  cells: Array<{ col: number; row: number; tileId: number }>
  // 이 오브젝트의 타일을 영역 밖에서도 쓰는 맵 칸 수 — 0이 아니면 그 칸들도 함께 바뀐다(경고 표시).
  sharedOutsideCells: number
  // 배너에 표시할 맞춤 문구(NPC 등). 없으면 기본 '맵 오브젝트: … N타일 …' 문구를 만든다.
  bannerText?: string
}

// 통짜 파일(스프라이트 시트 등)을 대상으로 모달을 여는 입력 — 몬스터 시트처럼 타일 개념이
// 없는 에셋용. 기존 '게임 에셋' 모드를 재사용해 파일 전체를 변환·적용한다.
export type StyleTransferAssetTarget = {
  path: string
  label: string
  // 안내 문구(예: '몬스터 시트 전체가 변환됩니다 …'). 콘텐츠 카드 아래에 표시.
  note?: string
  // 몬스터 시트면 종류('pig'|'slime'). 있으면 배경 보존 전용 변환(/stylize-monster)을 쓴다 —
  // 게임의 색 기반 프레임 슬라이싱이 깨지지 않게 캐릭터(전경)만 스타일하고 배경은 원본 유지.
  monsterKey?: string
}

type ContentMode = 'file' | 'asset' | 'object' | 'extracted' | 'tile'

// '타일' 탭: 현재 맵 타일셋에서 타일 하나를 골라 스타일을 입힌다(타일셋 패치 = 그 타일을
// 쓰는 모든 곳이 함께 바뀜). createEditorApp이 현재 맵 타일셋 정보를 공급한다.
export type TileStyleContext = {
  tilesetImagePath: string // 에셋 상대 경로(서비스 API용, 예: src/games/.../town-32.png)
  tilesetImageUrl: string // 썸네일 슬라이스용 URL
  columns: number
  tileWidth: number
  tileHeight: number
}

// 자동 추출된 오브젝트(서비스 GET /extracted-objects의 사이드카 메타).
type ExtractedObject = {
  id: string
  key: string
  label: string
  tilesetPath: string
  sharedOutsideCells: number
}

// '게임에 적용'의 대상 — 모드에 따라 적용 방식이 다르다.
type ApplyTarget =
  | { kind: 'asset'; path: string; blob: Blob } // 결과 PNG로 에셋 자체를 덮어쓰기
  | { kind: 'patched-tileset'; path: string; blob: Blob } // 맵 오브젝트 모드: 서비스가 만든 패치 타일셋
  | { kind: 'extracted-object'; key: string; path: string; blob: Blob } // 추출 모드: 서비스가 역패치

export interface StyleTransferModal {
  openButton: HTMLButtonElement
  backdrop: HTMLDivElement
  openForMapObject: (target: StyleTransferMapObject) => void
  openForAsset: (target: StyleTransferAssetTarget) => void
}

export type CreateStyleTransferModalInput = {
  // 적용/되돌리기로 에셋의 스타일 상태가 바뀐 뒤 호출 — 헤더 '스타일 되돌리기' 카운트 갱신용.
  onAssetChanged?: () => void
  // '타일' 탭용 — 현재 맵의 타일셋 정보(없으면 타일 탭이 비활성 안내).
  getTileStyleTarget?: () => TileStyleContext | undefined
}

// 이미지 한 장 선택용 파일 입력 + 썸네일. 새 파일을 고르면 이전 미리보기 URL을 해제한다.
const createImagePicker = (): {
  input: HTMLInputElement
  thumb: HTMLImageElement
  getFile: () => File | undefined
  setFile: (file: File | undefined) => void
  onChange: (listener: () => void) => void
} => {
  const input = el('input', 'text-xs text-zinc-400 file:mr-2 file:rounded-lg file:border-0 file:bg-white/[0.08] file:px-3 file:py-1.5 file:text-xs file:text-zinc-200 file:cursor-pointer')
  input.type = 'file'
  input.accept = 'image/png,image/jpeg,image/webp'
  const thumb = el('img', 'max-h-36 w-full rounded-lg object-contain bg-black/40')
  thumb.style.display = 'none'

  let file: File | undefined
  let listener: (() => void) | undefined
  const setFile = (next: File | undefined): void => {
    file = next
    if (thumb.src) {
      URL.revokeObjectURL(thumb.src)
    }
    if (file) {
      thumb.src = URL.createObjectURL(file)
      thumb.style.display = 'block'
    } else {
      thumb.removeAttribute('src')
      thumb.style.display = 'none'
    }
    listener?.()
  }
  input.addEventListener('change', () => {
    setFile(input.files?.[0] ?? undefined)
  })

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

export const createStyleTransferModal = (
  input: CreateStyleTransferModalInput = {}
): StyleTransferModal => {
  const { onAssetChanged, getTileStyleTarget } = input
  const openButton = el('button', 'rounded-lg px-2.5 py-1 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100', '🎨 스타일 변환')
  openButton.type = 'button'

  const backdrop = el('div', 'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4')
  // 설정 모달과 동일: hidden 속성 대신 인라인 display — `flex` 클래스가 [hidden]을 덮는 사고 방지.
  backdrop.style.display = 'none'
  const panel = el('div', 'w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900 p-5 flex flex-col gap-4 shadow-2xl')

  const top = el('div', 'flex items-center justify-between')
  top.append(el('span', 'text-sm font-semibold tracking-tight', '🎨 스타일 변환 (AdaIN)'))
  const close = el('button', 'text-zinc-500 text-sm transition hover:text-zinc-200', '✕')
  close.type = 'button'
  top.append(close)

  // ---------- 콘텐츠 카드: 모드 탭(파일/게임 에셋) + 맵 오브젝트 배너 ----------
  let contentMode: ContentMode = 'file'
  let mapObject: StyleTransferMapObject | undefined
  let assetPath: string | undefined
  // 현재 asset 대상이 몬스터 시트면 종류('pig'|'slime') — 배경 보존 변환 경로 선택용.
  let assetMonsterKey: string | undefined
  let extractedObject: ExtractedObject | undefined
  // 추출 그리드의 id(=key)로 메타를 되찾기 위한 맵. loadExtractedList가 채운다.
  const extractedById = new Map<string, ExtractedObject>()
  // 다중 선택(일괄 변환) 상태. 그리드별 전체 id 목록과 현재 선택.
  let selectedAssetPaths: string[] = []
  let selectedExtractedKeys: string[] = []
  let assetAllIds: string[] = []
  let extractedAllIds: string[] = []
  let isBatchRunning = false
  const currentGridAllIds = (): string[] =>
    contentMode === 'asset' ? assetAllIds : contentMode === 'extracted' ? extractedAllIds : []

  const contentCard = el('div', CARD)
  const contentHead = el('div', 'flex items-center justify-between gap-2')
  const modeTabs = el('div', 'flex items-center gap-1')
  const fileTab = el('button', MODE_TAB_ACTIVE, '파일')
  fileTab.type = 'button'
  const assetTab = el('button', MODE_TAB, '게임 에셋')
  assetTab.type = 'button'
  const tileTab = el('button', MODE_TAB, '타일')
  tileTab.type = 'button'
  tileTab.title = '현재 맵 타일셋의 타일 하나를 골라 스타일 적용(그 타일을 쓰는 모든 곳이 함께 바뀜)'
  const extractedTab = el('button', MODE_TAB, '추출 오브젝트')
  extractedTab.type = 'button'
  extractedTab.title = '맵 인식 시 자동으로 누끼 추출된 오브젝트들'
  modeTabs.append(fileTab, assetTab, tileTab, extractedTab)
  contentHead.append(el('div', LABEL, '콘텐츠 이미지'), modeTabs)

  const contentPicker = createImagePicker()
  // 게임 에셋: src/games/my-sample-rpg/assets PNG들을 썸네일 그리드로 본다(다중 선택 — 1개면 미리보기, 2개+면 일괄).
  const assetPicker = createThumbnailPicker({
    multiSelect: true,
    onChange: (ids) => {
      selectedAssetPaths = ids
      // 정확히 1개일 때만 미리보기로 로드(단일 변환 흐름). 그 외엔 미리보기 비움.
      loadAssetByPath(ids.length === 1 ? ids[0] : undefined)
      updateBatchToolbar()
      syncButtons()
    }
  })
  assetPicker.node.style.display = 'none'
  // 맵 오브젝트 모드 배너 — 트리에서 클릭해 들어온 대상 표시. '해제'로 파일 모드로 돌아간다.
  const objectBanner = el('div', 'flex items-center justify-between gap-2 rounded-lg bg-indigo-500/10 border border-indigo-500/25 px-3 py-2')
  objectBanner.style.display = 'none'
  const objectLabel = el('span', 'text-xs text-indigo-200 truncate')
  const objectClear = el('button', 'shrink-0 text-xs text-zinc-400 transition hover:text-zinc-200', '해제')
  objectClear.type = 'button'
  objectBanner.append(objectLabel, objectClear)
  const objectPreview = el('img', 'max-h-36 w-full rounded-lg object-contain bg-black/40 [image-rendering:pixelated]')
  objectPreview.style.display = 'none'
  // 추출 오브젝트 썸네일 그리드 — 자동 누끼 PNG들(다중 선택 — 1개면 미리보기, 2개+면 일괄).
  const extractedPicker = createThumbnailPicker({
    multiSelect: true,
    onChange: (ids) => {
      selectedExtractedKeys = ids
      if (ids.length === 1) {
        const meta = extractedById.get(ids[0])
        if (meta) {
          selectExtractedObject(meta)
        }
      } else {
        // 0개 또는 2개+: 단일 미리보기 상태 해제.
        extractedObject = undefined
        contentPicker.setFile(undefined)
        clearResult()
      }
      updateBatchToolbar()
      syncButtons()
    }
  })
  extractedPicker.node.style.display = 'none'
  // 타일 모드: 현재 맵 타일셋을 CSS background-position으로 슬라이스한 타일 그리드.
  const tileGridNode = el('div', 'flex flex-wrap gap-0.5 max-h-44 overflow-y-auto bg-black/30 p-1 rounded-lg')
  tileGridNode.style.display = 'none'
  let tileGridBuilt = false
  // 그리드(게임 에셋/추출) 다중 선택 툴바 — 모두 선택/해제 + 선택 개수. 그리드 모드에만 표시.
  const batchToolbar = el('div', 'flex items-center justify-between gap-2')
  batchToolbar.style.display = 'none'
  const batchSelectAll = el('button', 'text-[11px] text-zinc-400 transition hover:text-zinc-200', '모두 선택') as HTMLButtonElement
  batchSelectAll.type = 'button'
  const batchCount = el('span', 'text-[11px] text-zinc-500', '')
  batchToolbar.append(batchSelectAll, batchCount)
  // openForAsset(몬스터 시트 등)이 표시하는 안내 문구 — 그 외 모드에선 숨김.
  const assetNote = el('div', 'text-xs text-amber-300/90 leading-relaxed')
  assetNote.style.display = 'none'
  contentCard.append(contentHead, objectBanner, assetNote, batchToolbar, contentPicker.input, assetPicker.node, extractedPicker.node, tileGridNode, contentPicker.thumb, objectPreview)
  const setAssetNote = (text: string): void => {
    assetNote.textContent = text
    assetNote.style.display = text ? 'block' : 'none'
  }

  const currentGridPicker = () =>
    contentMode === 'asset' ? assetPicker : contentMode === 'extracted' ? extractedPicker : undefined
  const currentGridCount = (): number => currentGridAllIds().length
  const currentSelection = () =>
    contentMode === 'asset' ? selectedAssetPaths : contentMode === 'extracted' ? selectedExtractedKeys : []
  const updateBatchToolbar = (): void => {
    const grid = contentMode === 'asset' || contentMode === 'extracted'
    batchToolbar.style.display = grid ? 'flex' : 'none'
    if (!grid) {
      return
    }
    const selected = currentSelection().length
    batchCount.textContent = selected > 0 ? `${selected}개 선택됨` : '여러 개를 선택하면 한 번에 변환됩니다'
    batchSelectAll.textContent = selected > 0 && selected === currentGridCount() ? '모두 해제' : '모두 선택'
  }
  batchSelectAll.addEventListener('click', () => {
    const picker = currentGridPicker()
    if (!picker) {
      return
    }
    picker.setSelected(picker.getSelected().length === currentGridCount() ? [] : currentGridAllIds())
  })

  const stylePicker = createImagePicker()
  const styleCard = el('div', CARD)
  styleCard.append(el('div', LABEL, '스타일 이미지'), stylePicker.input, stylePicker.thumb)

  const pickers = el('div', 'grid grid-cols-1 sm:grid-cols-2 gap-3')
  pickers.append(contentCard, styleCard)

  // ---------- 스타일 강도(alpha) + 변환 크기 ----------
  const alphaWrap = el('div', CARD)
  const alphaHead = el('div', 'flex items-center justify-between')
  const alphaValue = el('span', 'text-xs text-zinc-300', '1.00')
  alphaHead.append(el('div', LABEL, '스타일 강도 (alpha)'), alphaValue)
  const alphaSlider = el('input', 'w-full accent-indigo-500')
  alphaSlider.type = 'range'
  alphaSlider.min = '0'
  alphaSlider.max = '1'
  alphaSlider.step = '0.05'
  alphaSlider.value = '1'
  alphaSlider.addEventListener('input', () => {
    alphaValue.textContent = Number(alphaSlider.value).toFixed(2)
  })
  const sizeRow = el('div', 'flex flex-wrap items-center gap-2')
  const sizeSelect = el('select', FIELD_SELECT.replace('w-full', ''))
  for (const [value, label] of [['512', '512px (기본)'], ['256', '256px'], ['0', '원본 크기 유지']]) {
    const option = el('option', '', label)
    option.value = value
    sizeSelect.append(option)
  }
  // 경계 침식: 투명 경계에 스타일 색이 번져 보일 때 알파를 살짝 깎는 옵션(기본 0 = 없음).
  const erodeSelect = el('select', FIELD_SELECT.replace('w-full', ''))
  for (const [value, label] of [['0', '0px (없음)'], ['1', '1px'], ['2', '2px'], ['3', '3px']]) {
    const option = el('option', '', label)
    option.value = value
    erodeSelect.append(option)
  }
  erodeSelect.title = '투명 경계의 색 번짐을 줄입니다'
  sizeRow.append(
    el('span', 'text-xs text-zinc-500', '변환 크기'),
    sizeSelect,
    el('span', 'text-xs text-zinc-500', '경계 침식'),
    erodeSelect
  )
  alphaWrap.append(alphaHead, alphaSlider, sizeRow)

  // ---------- 실행/저장/적용 ----------
  const actions = el('div', 'flex flex-wrap items-center gap-2')
  const runButton = el('button', PRIMARY_BUTTON, '변환')
  runButton.type = 'button'
  runButton.disabled = true
  const applyButton = el('button', GHOST_BUTTON, '🕹 게임에 적용')
  applyButton.type = 'button'
  applyButton.disabled = true
  const revertButton = el('button', GHOST_BUTTON, '↩ 원본으로 되돌리기')
  revertButton.type = 'button'
  revertButton.disabled = true
  revertButton.title = '스타일이 적용된 에셋을 최초 원본으로 복원합니다'
  const saveButton = el('button', GHOST_BUTTON, 'PNG 저장')
  saveButton.type = 'button'
  saveButton.disabled = true
  // 변환 전 원본(자동 조립된 타일/오브젝트·추출·에셋)을 PNG로 저장.
  const saveOrigButton = el('button', GHOST_BUTTON, '원본 PNG 저장')
  saveOrigButton.type = 'button'
  saveOrigButton.disabled = true
  // 일괄 변환·적용 — 그리드에서 2개 이상 선택했을 때만 보인다. 미리보기 없이 서버에서
  // 한 번에 변환·적용한다(클라이언트 루프는 첫 적용의 Vite 리로드로 끊기므로 서버 일괄 처리).
  const batchButton = el('button', PRIMARY_BUTTON, '🎨 일괄 변환·적용')
  batchButton.type = 'button'
  batchButton.disabled = true
  batchButton.style.display = 'none'
  const status = el('span', 'text-xs text-zinc-500 min-h-4', '')
  actions.append(runButton, batchButton, applyButton, revertButton, saveOrigButton, saveButton, status)

  // 변환/적용 진행 표시 (0~100%). AdaIN 변환은 단일 백엔드 호출이라 실제 진행률 스트림이 없어,
  // 추정 진행도로 보여준다 — 시작하면 90%까지 감속하며 차오르고, 응답이 오면 100%로 마감한다.
  const progressWrap = el('div', 'flex items-center gap-2')
  const progressTrack = el('div', 'h-2 flex-1 overflow-hidden rounded-full bg-white/10')
  const progressBar = el('div', 'h-full w-0 rounded-full bg-amber-400')
  progressBar.style.transition = 'width 0.15s ease-out'
  progressTrack.append(progressBar)
  const progressLabel = el('span', 'w-9 shrink-0 text-right text-xs tabular-nums text-amber-300', '0%')
  progressWrap.append(progressTrack, progressLabel)
  progressWrap.style.display = 'none'
  let progressTimer: number | undefined
  let progressPct = 0
  let progressWasBusy = false
  const setProgressPct = (pct: number): void => {
    progressPct = pct
    progressBar.style.width = `${pct}%`
    progressLabel.textContent = `${Math.round(pct)}%`
  }
  const startProgress = (): void => {
    if (progressTimer !== undefined) {
      window.clearInterval(progressTimer)
    }
    setProgressPct(0)
    progressWrap.style.display = 'flex'
    // 90%까지 감속하며 접근 — 실제 완료 전엔 100%에 닿지 않게 한다.
    progressTimer = window.setInterval(() => {
      setProgressPct(progressPct + (90 - progressPct) * 0.08)
    }, 120)
  }
  const finishProgress = (): void => {
    if (progressTimer !== undefined) {
      window.clearInterval(progressTimer)
      progressTimer = undefined
    }
    setProgressPct(100)
    window.setTimeout(() => {
      progressWrap.style.display = 'none'
      setProgressPct(0)
    }, 350)
  }

  const resultWrap = el('div', CARD)
  const resultImage = el('img', 'max-h-72 w-full rounded-lg object-contain bg-black/40')
  resultWrap.append(el('div', LABEL, '결과 미리보기'), resultImage)
  resultImage.style.display = 'none'

  panel.append(top, pickers, alphaWrap, actions, progressWrap, resultWrap)
  backdrop.append(panel)

  const setStatus = (message: string): void => {
    status.textContent = message
  }

  let isRunning = false
  let isApplying = false
  let isReverting = false
  let resultBlob: Blob | undefined
  let resultName = 'stylized.png'
  // '게임에 적용'이 쓸 대상 — kind에 따라 적용 엔드포인트가 다르다.
  let applyTarget: ApplyTarget | undefined
  // '원본으로 되돌리기' 대상 — 현재 대상 에셋이 스타일 적용 상태일 때만 채워진다.
  let revertPath: string | undefined
  // 요청 세대 토큰: 변환이 도는 동안 콘텐츠/모드가 바뀌면 증가시켜, 완료된 옛 요청이
  // 이미 클리어된 결과·적용 대상을 되살리지 못하게 한다.
  let runSeq = 0

  // 변환 전 원본을 PNG로 저장할 수 있는 상태인지 — 타일/오브젝트(mapObject), 추출 단일 선택,
  // 에셋 단일 선택, 또는 업로드 파일이 있을 때.
  const hasOriginal = (): boolean =>
    mapObject !== undefined ||
    (contentMode === 'asset' && selectedAssetPaths.length === 1) ||
    (contentMode === 'extracted' && selectedExtractedKeys.length === 1) ||
    contentPicker.getFile() !== undefined

  const syncButtons = (): void => {
    const busy = isRunning || isApplying || isReverting || isBatchRunning
    // 변환·일괄·적용·복원 시작/완료 엣지에서 0~100% 진행 바를 시작/마감한다
    // (모든 run 함수가 플래그 토글 직후 syncButtons를 호출하므로 엣지가 정확히 잡힌다).
    if (busy && !progressWasBusy) {
      startProgress()
    } else if (!busy && progressWasBusy) {
      finishProgress()
    }
    progressWasBusy = busy
    // 오브젝트·타일 모드는 mapObject(타일 셀)가 콘텐츠, 그 외는 업로드/선택한 파일이 콘텐츠.
    const hasContent =
      mapObject !== undefined || contentPicker.getFile() !== undefined
    const hasStyle = stylePicker.getFile() !== undefined
    // 그리드에서 2개 이상 선택하면 일괄 버튼을, 1개 이하면 단일 변환 버튼을 쓴다.
    const batchCountSel = currentSelection().length
    const batchVisible = (contentMode === 'asset' || contentMode === 'extracted') && batchCountSel >= 2
    batchButton.style.display = batchVisible ? 'inline-flex' : 'none'
    batchButton.textContent = `🎨 일괄 변환·적용 (${batchCountSel})`
    batchButton.disabled = busy || !batchVisible || !hasStyle
    runButton.disabled = busy || !hasContent || !hasStyle
    saveButton.disabled = isRunning || !resultBlob
    saveOrigButton.disabled = busy || !hasOriginal()
    applyButton.disabled = busy || !applyTarget
    revertButton.disabled = busy || !revertPath
  }

  // 현재 모드의 "게임 에셋 경로" — 적용/되돌리기의 대상.
  const currentTargetPath = (): string | undefined => {
    if (contentMode === 'asset') {
      return assetPath
    }
    if (contentMode === 'object' || contentMode === 'tile') {
      return mapObject?.tilesetImagePath
    }
    if (contentMode === 'extracted') {
      return extractedObject?.tilesetPath
    }
    return undefined
  }

  // 대상 에셋이 스타일 적용 상태인지 서비스에 물어 되돌리기 버튼을 갱신한다.
  // 응답이 올 때까지 대상이 바뀌었으면 결과를 버린다(레이스 가드).
  const refreshRevertState = async (): Promise<void> => {
    const path = currentTargetPath()
    if (!path) {
      revertPath = undefined
      syncButtons()
      return
    }
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/asset-status?path=${encodeURIComponent(path)}`)
      if (!response.ok) {
        throw new Error(String(response.status))
      }
      const info = (await response.json()) as { styled?: boolean }
      if (path !== currentTargetPath()) {
        return
      }
      revertPath = info.styled === true ? path : undefined
    } catch {
      revertPath = undefined
    }
    syncButtons()
  }

  // 결과 무효화는 한 곳으로 — blob·적용 대상·미리보기·상태 메시지를 같이 지워야
  // "이전 결과가 새 대상의 결과처럼 보이는" 혼동이 없다.
  const clearResult = (): void => {
    runSeq += 1
    resultBlob = undefined
    applyTarget = undefined
    if (resultImage.src) {
      URL.revokeObjectURL(resultImage.src)
      resultImage.removeAttribute('src')
    }
    resultImage.style.display = 'none'
    setStatus('')
  }

  contentPicker.onChange(() => {
    clearResult()
    syncButtons()
  })
  stylePicker.onChange(syncButtons)

  const setContentMode = (mode: ContentMode): void => {
    const modeChanged = mode !== contentMode
    // 모드가 실제로 바뀔 때만 콘텐츠 상태를 리셋한다 — 탭 전환 후 이전 모드의 파일/선택이
    // 입력으로 남아 UI와 실제 입력이 어긋나는 사고 방지.
    if (modeChanged) {
      contentPicker.setFile(undefined)
      assetPath = undefined
      assetMonsterKey = undefined
      assetPicker.clearSelection()
      selectedAssetPaths = []
      extractedObject = undefined
      extractedPicker.clearSelection()
      selectedExtractedKeys = []
    }
    contentMode = mode
    // object는 openForMapObject가 mapObject를 세팅·유지한다. tile은 그리드 클릭으로 세팅하므로
    // 모드가 "실제로 바뀔 때만" 비워서, 타일 탭 재클릭이 선택을 지우지 않게 한다.
    if (modeChanged && mode !== 'object') {
      mapObject = undefined
    }
    fileTab.className = mode === 'file' ? MODE_TAB_ACTIVE : MODE_TAB
    assetTab.className = mode === 'asset' ? MODE_TAB_ACTIVE : MODE_TAB
    tileTab.className = mode === 'tile' ? MODE_TAB_ACTIVE : MODE_TAB
    extractedTab.className = mode === 'extracted' ? MODE_TAB_ACTIVE : MODE_TAB
    modeTabs.style.display = mode === 'object' ? 'none' : 'flex'
    objectBanner.style.display = mode === 'object' ? 'flex' : 'none'
    objectPreview.style.display = 'none'
    contentPicker.input.style.display = mode === 'file' ? 'block' : 'none'
    assetPicker.node.style.display = mode === 'asset' ? 'grid' : 'none'
    extractedPicker.node.style.display = mode === 'extracted' ? 'grid' : 'none'
    tileGridNode.style.display = mode === 'tile' ? 'flex' : 'none'
    contentPicker.thumb.style.display =
      mode !== 'object' && mode !== 'tile' && contentPicker.getFile() ? 'block' : 'none'
    // 안내 문구는 openForAsset이 모드 설정 직후 다시 채운다(여기선 일단 비운다).
    setAssetNote('')
    if (mode === 'tile') {
      buildTileGrid()
    }
    clearResult()
    updateBatchToolbar()
    void refreshRevertState()
    syncButtons()
  }

  fileTab.addEventListener('click', () => {
    setContentMode('file')
  })
  tileTab.addEventListener('click', () => {
    setContentMode('tile')
  })

  // ---------- 타일 모드: 타일셋 슬라이스 그리드에서 타일 하나를 골라 1셀 오브젝트로 변환 ----------
  const buildTileGrid = (): void => {
    if (tileGridBuilt) {
      return
    }
    const ctx = getTileStyleTarget?.()
    tileGridNode.replaceChildren()
    if (!ctx) {
      tileGridNode.append(el('div', 'text-xs text-zinc-500 p-2', '현재 맵의 타일셋을 찾지 못했습니다.'))
      return
    }
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
          for (const other of tileGridNode.children) {
            (other as HTMLElement).classList.remove('ring-2', 'ring-indigo-400/70')
          }
          cell.classList.add('ring-2', 'ring-indigo-400/70')
          selectTile(ctx, tileId)
        })
        tileGridNode.append(cell)
      }
      tileGridBuilt = true
    }
    probe.onerror = () => {
      tileGridNode.append(el('div', 'text-xs text-rose-300 p-2', '타일셋 이미지를 불러오지 못했습니다.'))
    }
    probe.src = ctx.tilesetImageUrl
  }

  // 고른 타일을 1셀 mapObject로 만들어 기존 오브젝트 변환·적용(타일셋 패치) 경로를 그대로 탄다.
  const selectTile = (ctx: TileStyleContext, tileId: number): void => {
    mapObject = {
      label: `타일 #${tileId}`,
      tilesetImagePath: ctx.tilesetImagePath,
      tileWidth: ctx.tileWidth,
      tileHeight: ctx.tileHeight,
      columns: ctx.columns,
      cells: [{ col: 0, row: 0, tileId }],
      sharedOutsideCells: 0
    }
    // 미리보기: 고른 타일을 잘라 보여준다(objectPreview 재사용).
    const canvas = document.createElement('canvas')
    canvas.width = ctx.tileWidth
    canvas.height = ctx.tileHeight
    const context = canvas.getContext('2d')
    const probe = new Image()
    probe.onload = () => {
      if (!context) {
        return
      }
      context.imageSmoothingEnabled = false
      context.drawImage(
        probe,
        (tileId % ctx.columns) * ctx.tileWidth,
        Math.floor(tileId / ctx.columns) * ctx.tileHeight,
        ctx.tileWidth,
        ctx.tileHeight,
        0,
        0,
        ctx.tileWidth,
        ctx.tileHeight
      )
      objectPreview.src = canvas.toDataURL()
      objectPreview.style.display = 'block'
    }
    probe.src = ctx.tilesetImageUrl
    clearResult()
    void refreshRevertState()
    syncButtons()
  }

  // ---------- 게임 에셋 모드: 서비스의 PNG 목록을 썸네일 그리드로 ----------
  // 썸네일 캐시버스트 — 적용/되돌리기로 같은 경로 파일이 바뀌어도 새 이미지를 받게.
  let assetCacheBust = 0
  const loadAssetList = async (): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/assets`)
      if (!response.ok) {
        throw new Error(String(response.status))
      }
      const data = (await response.json()) as { assets: Array<{ path: string }> }
      assetCacheBust += 1
      assetAllIds = data.assets.map((asset) => asset.path)
      // Vite dev 서버가 소스 파일을 같은 경로로 서빙하므로 썸네일은 /{path}로 바로 받는다.
      assetPicker.setItems(
        data.assets.map((asset) => ({
          id: asset.path,
          label: asset.path.replace(/^src\/games\/my-sample-rpg\/assets\//, ''),
          thumbUrl: `/${asset.path}?t=${assetCacheBust}`
        })),
        '에셋이 없습니다.'
      )
      selectedAssetPaths = assetPicker.getSelected()
      updateBatchToolbar()
    } catch {
      assetAllIds = []
      assetPicker.setItems([], '에셋 목록을 불러오지 못했습니다.')
      setStatus(SERVICE_GUIDE)
    }
  }
  // 그리드에서 에셋을 고르면 현재 파일로 로드한다(단일 선택). 그리드 선택은 일반 에셋이므로
  // 몬스터 키를 비운다 — openForAsset이 몬스터로 열 땐 이 호출 뒤에 다시 키를 설정한다.
  const loadAssetByPath = (path: string | undefined): void => {
    assetPath = path
    assetMonsterKey = undefined
    if (!path) {
      contentPicker.setFile(undefined)
      return
    }
    void (async () => {
      try {
        const response = await fetch(`/${path}?t=${assetCacheBust}`)
        if (!response.ok) {
          throw new Error(String(response.status))
        }
        const blob = await response.blob()
        const name = path.split('/').pop() ?? 'asset.png'
        contentPicker.setFile(new File([blob], name, { type: 'image/png' }))
        contentPicker.thumb.style.display = 'block'
        void refreshRevertState()
      } catch {
        contentPicker.setFile(undefined)
        setStatus('에셋 이미지를 불러오지 못했습니다.')
      }
    })()
  }
  assetTab.addEventListener('click', () => {
    setContentMode('asset')
    // 적용/되돌리기로 내용이 바뀔 수 있어 탭을 열 때마다 새로 받는다(목록 자체는 정적).
    void loadAssetList()
  })
  objectClear.addEventListener('click', () => {
    setContentMode('file')
  })

  // ---------- 추출 오브젝트 모드: 자동 누끼 추출본 썸네일 그리드 ----------
  const selectExtractedObject = (meta: ExtractedObject): void => {
    void (async () => {
      // 클릭 시점에 세대를 올려서 캡처한다 — 빠른 연속 클릭 시 늦게 도착한 이전 응답이
      // 마지막 선택을 덮거나, 모드 전환으로 지운 선택을 부활시키는 것을 막는다.
      runSeq += 1
      const seq = runSeq
      try {
        const response = await fetch(`${STYLE_SERVICE_BASE}/extracted-objects/${encodeURIComponent(meta.key)}.png`)
        if (!response.ok) {
          throw new Error(String(response.status))
        }
        const blob = await response.blob()
        if (seq !== runSeq) {
          return
        }
        extractedObject = meta
        contentPicker.setFile(new File([blob], `${meta.key}.png`, { type: 'image/png' }))
        contentPicker.thumb.style.display = 'block'
        setStatus(
          meta.sharedOutsideCells > 0
            ? `${meta.label} 선택됨 · ⚠ 같은 타일을 쓰는 영역 밖 ${meta.sharedOutsideCells}칸도 함께 바뀝니다`
            : `${meta.label} 선택됨`
        )
        void refreshRevertState()
      } catch {
        if (seq === runSeq) {
          setStatus('추출 오브젝트 이미지를 불러오지 못했습니다.')
        }
      }
    })()
  }

  const loadExtractedList = async (): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/extracted-objects`)
      if (!response.ok) {
        throw new Error(String(response.status))
      }
      const data = (await response.json()) as { objects: ExtractedObject[] }
      extractedById.clear()
      for (const meta of data.objects) {
        extractedById.set(meta.key, meta)
      }
      extractedAllIds = data.objects.map((meta) => meta.key)
      extractedPicker.setItems(
        data.objects.map((meta) => ({
          id: meta.key,
          label: meta.label,
          thumbUrl: `${STYLE_SERVICE_BASE}/extracted-objects/${encodeURIComponent(meta.key)}.png`
        })),
        '추출된 오브젝트가 없습니다 — 게임에서 맵을 열면 자동으로 추출됩니다.'
      )
      selectedExtractedKeys = extractedPicker.getSelected()
      updateBatchToolbar()
    } catch {
      extractedAllIds = []
      extractedPicker.setItems([], '추출 오브젝트 목록을 불러오지 못했습니다.')
      setStatus(SERVICE_GUIDE)
    }
  }
  extractedTab.addEventListener('click', () => {
    setContentMode('extracted')
    // 맵 인식 때마다 새 오브젝트가 추가될 수 있어 탭을 열 때마다 새로 받는다.
    void loadExtractedList()
  })

  // ---------- 변환 ----------
  // 대상(target/applyPath)은 호출 시점에 캡처하고, 완료 시 세대(seq)가 그대로일 때만 결과를
  // 반영한다 — 변환 중 모드·선택이 바뀌어 엉뚱한 에셋에 적용되거나 stale 결과가 살아나는 것 방지.
  const runObjectTransfer = async (
    target: StyleTransferMapObject,
    styleFile: File,
    seq: number
  ): Promise<void> => {
    const form = new FormData()
    form.append('style', styleFile)
    form.append('alpha', alphaSlider.value)
    form.append('tileset_path', target.tilesetImagePath)
    form.append('tile_width', String(target.tileWidth))
    form.append('tile_height', String(target.tileHeight))
    form.append('columns', String(target.columns))
    form.append('cells', JSON.stringify(target.cells))
    form.append('alpha_erode', erodeSelect.value)
    const response = await fetch(`${STYLE_SERVICE_BASE}/stylize-object`, {
      method: 'POST',
      body: form
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
    }
    const data = (await response.json()) as { object_png: string; tileset_png: string }
    if (seq !== runSeq) {
      return
    }
    resultBlob = base64ToBlob(data.object_png)
    resultName = `${target.label.replace(/[^\w가-힣]+/gu, '_')}_stylized.png`
    applyTarget = {
      kind: 'patched-tileset',
      path: target.tilesetImagePath,
      blob: base64ToBlob(data.tileset_png)
    }
    setStatus('완료 — 미리보기를 확인하고 🕹 게임에 적용을 누르세요. (같은 타일을 쓰는 동일 오브젝트가 함께 바뀝니다)')
  }

  // 적용 컨텍스트(대상 종류·경로·키)는 변환 시작 시점에 캡처되어 그대로 들어온다.
  type ApplyContext =
    | { kind: 'asset'; path: string }
    | { kind: 'extracted-object'; key: string; path: string }

  const runImageTransfer = async (
    contentFile: File,
    styleFile: File,
    applyContext: ApplyContext | undefined,
    seq: number
  ): Promise<void> => {
    const form = new FormData()
    form.append('style', styleFile)
    form.append('alpha', alphaSlider.value)
    form.append('alpha_erode', erodeSelect.value)
    let endpoint = `${STYLE_SERVICE_BASE}/style-transfer`
    if (assetMonsterKey && assetPath) {
      // 몬스터 시트: 배경 보존 전용 변환 — 캐릭터(전경)만 스타일하고 배경은 원본 유지해
      // 게임의 색 기반 프레임 슬라이싱이 깨지지 않게 한다. 시트는 서버가 경로로 읽는다.
      endpoint = `${STYLE_SERVICE_BASE}/stylize-monster`
      form.append('sheet_path', assetPath)
      form.append('monster_key', assetMonsterKey)
    } else {
      form.append('content', contentFile)
      form.append('content_size', sizeSelect.value)
      form.append('style_size', '512')
      // 게임 에셋을 덮어쓸 때는(asset 모드) 원본 크기를 정확히 보존한다(파일 업로드 미리보기 제외).
      if (contentMode === 'asset') {
        form.append('preserve_size', '1')
      }
    }
    const response = await fetch(endpoint, {
      method: 'POST',
      body: form
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
    }
    const blob = await response.blob()
    if (seq !== runSeq) {
      return
    }
    resultBlob = blob
    resultName = `${fileStem(contentFile.name)}_stylized_${fileStem(styleFile.name)}.png`
    applyTarget = applyContext !== undefined ? { ...applyContext, blob } : undefined
    if (!applyTarget) {
      setStatus(`완료: ${resultName}`)
    } else if (applyTarget.kind === 'extracted-object' && extractedObject && extractedObject.sharedOutsideCells > 0) {
      setStatus(`완료: ${resultName} — 🕹 게임에 적용 시 영역 밖 ${extractedObject.sharedOutsideCells}칸도 함께 바뀝니다.`)
    } else {
      setStatus(`완료: ${resultName} — 🕹 게임에 적용으로 에셋을 덮어쓸 수 있습니다.`)
    }
  }

  const runTransfer = async (): Promise<void> => {
    const styleFile = stylePicker.getFile()
    const contentFile = contentPicker.getFile()
    // object·tile 모드는 mapObject(타일 셀)가 콘텐츠다 — 둘 다 /stylize-object 경로로 처리한다.
    const target = mapObject
    if (isRunning || !styleFile) {
      return
    }
    if (target === undefined && !contentFile) {
      return
    }
    // 적용 대상·세대를 await 이전(동기 구간)에 캡처한다.
    const applyContext: ApplyContext | undefined =
      contentMode === 'asset' && assetPath !== undefined
        ? { kind: 'asset', path: assetPath }
        : contentMode === 'extracted' && extractedObject !== undefined
          ? { kind: 'extracted-object', key: extractedObject.key, path: extractedObject.tilesetPath }
          : undefined
    const seq = runSeq
    isRunning = true
    syncButtons()
    setStatus('변환 중... (CPU에서 수 초 걸릴 수 있습니다)')
    try {
      if (target !== undefined) {
        await runObjectTransfer(target, styleFile, seq)
      } else {
        await runImageTransfer(contentFile as File, styleFile, applyContext, seq)
      }
      if (seq === runSeq && resultBlob) {
        if (resultImage.src) {
          URL.revokeObjectURL(resultImage.src)
        }
        resultImage.src = URL.createObjectURL(resultBlob)
        resultImage.style.display = 'block'
      }
    } catch (error) {
      const failedToFetch = error instanceof TypeError
      setStatus(failedToFetch ? SERVICE_GUIDE : `실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isRunning = false
      syncButtons()
    }
  }
  runButton.addEventListener('click', () => {
    void runTransfer()
  })

  // 일괄 변환·적용: 선택한 그리드 항목들을 한 스타일로 서버에서 한 번에 변환·적용한다.
  // 미리보기는 없고, 적용 후 Vite가 게임/에디터를 새로고침해 결과가 반영된다.
  const runBatch = async (): Promise<void> => {
    const styleFile = stylePicker.getFile()
    const selection = currentSelection()
    if (isBatchRunning || !styleFile || selection.length < 2) {
      return
    }
    const targets =
      contentMode === 'asset'
        ? selection.map((path) => ({ kind: 'asset', path }))
        : selection.map((key) => ({ kind: 'object', key }))
    isBatchRunning = true
    syncButtons()
    setStatus(`일괄 변환·적용 중... (${selection.length}개, 잠시 걸릴 수 있습니다)`)
    try {
      const form = new FormData()
      form.append('style', styleFile)
      form.append('alpha', alphaSlider.value)
      form.append('alpha_erode', erodeSelect.value)
      form.append('targets', JSON.stringify(targets))
      const response = await fetch(`${STYLE_SERVICE_BASE}/batch-apply`, {
        method: 'POST',
        body: form
      })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      const result = (await response.json()) as { applied: string[]; failed: unknown[] }
      onAssetChanged?.()
      const failNote = result.failed.length > 0 ? ` (실패 ${result.failed.length}개)` : ''
      setStatus(`일괄 적용 완료: ${result.applied.length}개${failNote} — 게임 화면이 자동 새로고침됩니다.`)
    } catch (error) {
      const failedToFetch = error instanceof TypeError
      setStatus(failedToFetch ? SERVICE_GUIDE : `일괄 적용 실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isBatchRunning = false
      syncButtons()
    }
  }
  batchButton.addEventListener('click', () => {
    void runBatch()
  })

  // ---------- 게임에 적용: src/games/my-sample-rpg/assets의 원본을 백업 후 덮어쓰기 → Vite가 게임을 자동 리로드 ----------
  const runApply = async (): Promise<void> => {
    if (isApplying || !applyTarget) {
      return
    }
    isApplying = true
    syncButtons()
    setStatus('게임 에셋에 적용 중...')
    try {
      const form = new FormData()
      form.append('file', new File([applyTarget.blob], 'patched.png', { type: 'image/png' }))
      let endpoint = `${STYLE_SERVICE_BASE}/apply-asset`
      if (applyTarget.kind === 'extracted-object') {
        // 추출 오브젝트: 결과 PNG를 서비스가 셀 정보로 타일셋에 역패치한다.
        endpoint = `${STYLE_SERVICE_BASE}/apply-object`
        form.append('object_key', applyTarget.key)
      } else {
        form.append('path', applyTarget.path)
      }
      const response = await fetch(endpoint, {
        method: 'POST',
        body: form
      })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      // 적용됐으니 이 에셋은 되돌릴 수 있다(최초 원본은 서비스가 originals/에 시드).
      revertPath = applyTarget?.path
      onAssetChanged?.()
      setStatus('게임에 적용됨 — 게임 화면이 자동 새로고침됩니다. (원본은 style-service/backups에 백업)')
    } catch (error) {
      const failedToFetch = error instanceof TypeError
      setStatus(failedToFetch ? SERVICE_GUIDE : `적용 실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isApplying = false
      syncButtons()
    }
  }
  applyButton.addEventListener('click', () => {
    void runApply()
  })

  // 원본으로 되돌리기: 서비스의 originals/에 시드된 최초 원본으로 복원한다.
  // 여러 번 적용했어도 한 번에 최초 원본으로 돌아간다. 파일이 바뀌므로 Vite가 게임을
  // 자동 새로고침해 화면도 함께 갱신된다.
  const runRevert = async (): Promise<void> => {
    if (isReverting || !revertPath) {
      return
    }
    isReverting = true
    syncButtons()
    setStatus('원본으로 복원 중...')
    try {
      const form = new FormData()
      form.append('path', revertPath)
      const response = await fetch(`${STYLE_SERVICE_BASE}/revert-asset`, {
        method: 'POST',
        body: form
      })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      revertPath = undefined
      onAssetChanged?.()
      setStatus('원본으로 복원됨 — 게임 화면이 자동 새로고침됩니다.')
    } catch (error) {
      const failedToFetch = error instanceof TypeError
      setStatus(failedToFetch ? SERVICE_GUIDE : `복원 실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isReverting = false
      syncButtons()
    }
  }
  revertButton.addEventListener('click', () => {
    void runRevert()
  })

  saveButton.addEventListener('click', () => {
    if (!resultBlob) {
      return
    }
    const url = URL.createObjectURL(resultBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = resultName
    link.click()
    URL.revokeObjectURL(url)
    setStatus(`저장됨: ${resultName}`)
  })

  // ── 변환 전 원본 PNG 저장 ──
  const triggerDownload = (blob: Blob, name: string): void => {
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = name
    link.click()
    URL.revokeObjectURL(url)
  }
  const safeName = (s: string): string => s.replace(/[^\w가-힣]+/gu, '_')
  const loadImage = (src: string): Promise<HTMLImageElement> =>
    new Promise((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error('타일셋 이미지를 불러오지 못했습니다.'))
      img.src = src
    })
  // 타일/오브젝트(mapObject)를 타일셋에서 셀 좌표로 조립해 원본 누끼 PNG를 만든다(백엔드 compose_object_canvas와 동일 로직).
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
    if (!ctx) {
      throw new Error('canvas 컨텍스트를 만들지 못했습니다.')
    }
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
  const runSaveOriginal = async (): Promise<void> => {
    try {
      if (mapObject) {
        const blob = await composeOriginalObject(mapObject)
        const name = `${safeName(mapObject.label)}_original.png`
        triggerDownload(blob, name)
        setStatus(`원본 저장됨: ${name}`)
        return
      }
      if (contentMode === 'extracted' && selectedExtractedKeys.length === 1) {
        const key = selectedExtractedKeys[0]
        const res = await fetch(`/api/style/extracted-objects/${encodeURIComponent(key)}.png`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        triggerDownload(await res.blob(), `${safeName(key)}_original.png`)
        setStatus('원본 저장됨')
        return
      }
      if (contentMode === 'asset' && selectedAssetPaths.length === 1) {
        const path = selectedAssetPaths[0]
        const res = await fetch(`/${path}`)
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`)
        }
        const base = (path.split('/').pop() ?? 'asset.png').replace(/\.png$/i, '')
        triggerDownload(await res.blob(), `${base}_original.png`)
        setStatus('원본 저장됨')
        return
      }
      const file = contentPicker.getFile()
      if (file) {
        triggerDownload(file, file.name)
        setStatus(`원본 저장됨: ${file.name}`)
        return
      }
      setStatus('저장할 원본이 없습니다 — 타일/오브젝트나 에셋을 먼저 선택하세요.')
    } catch (error) {
      setStatus(`원본 저장 실패: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  saveOrigButton.addEventListener('click', () => {
    void runSaveOriginal()
  })

  // 모달을 열 때마다 서비스 생존 확인 — 미기동이면 실행 안내를 먼저 보여준다.
  // isRunning 검사는 await 이후(setStatus 직전)에 한다 — 응답 대기 중 변환이 시작됐다면
  // '변환 중...' 상태 메시지를 헬스 결과로 덮어쓰지 않는다.
  const checkHealth = async (): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/health`)
      if (!response.ok) {
        throw new Error(String(response.status))
      }
      const info = (await response.json()) as {
        status?: string
        device?: string
        missing_weights?: string[]
      }
      if (isRunning) {
        return
      }
      if (info.status === 'ok') {
        setStatus(`스타일 서비스 연결됨 (${info.device ?? '?'})`)
      } else {
        setStatus(`서비스는 떠 있지만 모델 가중치가 없습니다: ${(info.missing_weights ?? []).join(', ')}`)
      }
    } catch {
      if (!isRunning) {
        setStatus(SERVICE_GUIDE)
      }
    }
  }

  const openModal = (): void => {
    backdrop.style.display = 'flex'
    void checkHealth()
  }
  const closeModal = (): void => {
    backdrop.style.display = 'none'
  }
  openButton.addEventListener('click', openModal)
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

  const openForMapObject = (target: StyleTransferMapObject): void => {
    setContentMode('object')
    mapObject = target
    objectLabel.textContent =
      target.bannerText ??
      `맵 오브젝트: ${target.label} · ${target.cells.length}타일` +
        (target.sharedOutsideCells > 0
          ? ` · ⚠ 같은 타일을 쓰는 영역 밖 ${target.sharedOutsideCells}칸도 함께 바뀝니다`
          : '')
    // 오브젝트 모드의 콘텐츠 미리보기: 타일셋 원본을 받아 그대로 보여주긴 크니, 생략하고
    // 결과 미리보기로 확인하게 한다. (타일 조립은 서비스가 수행)
    // setContentMode 시점에는 mapObject가 아직 이전 값이라, 대상 확정 후 다시 조회한다.
    void refreshRevertState()
    syncButtons()
    openModal()
  }

  // 통짜 파일(몬스터 시트 등) 대상으로 열기 — '게임 에셋' 모드를 재사용하되 그리드 대신
  // 주어진 path를 직접 로드한다. 시트는 프레임 좌표 정렬을 위해 원본 크기 유지(0)로 변환한다.
  const openForAsset = (target: StyleTransferAssetTarget): void => {
    setContentMode('asset')
    assetPath = target.path
    // 변환은 512로 빠르게 하고, preserve_size(asset 모드)가 결과를 원본 시트 크기로 정확히
    // 되돌린다 — CPU에서 큰 시트를 원본 해상도로 추론하면 매우 느리고 메모리 부담이 크다.
    sizeSelect.value = '512'
    loadAssetByPath(target.path)
    // loadAssetByPath가 몬스터 키를 비우므로 그 뒤에 설정한다(몬스터면 배경 보존 경로 사용).
    assetMonsterKey = target.monsterKey
    if (target.note) {
      setAssetNote(target.note)
    }
    syncButtons()
    openModal()
  }

  return { openButton, backdrop, openForMapObject, openForAsset }
}
