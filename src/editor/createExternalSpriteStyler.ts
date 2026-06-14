// 헤더 '🎮 외부 게임 스타일' 버튼 + 모달.
// 외부 게임(예: Love2D 'Legend of Lua')의 스프라이트 PNG에 AdaIN 스타일을 직접 입힌다.
// my-sample-rpg 에셋과는 별개로, 스타일 서비스의 /ext/* 엔드포인트(external_assets 게이트,
// originals-ext/·backups-ext/)를 쓴다. 외부 폴더는 Vite 감시 밖이라 적용해도 자동 리로드가
// 없다 — 게임을 다시 실행해야 반영된다(스프라이트는 시작 시 1회 로드).
//
// 구성: 스프라이트 썸네일 그리드(다중 선택) + 스타일 이미지 업로드 + alpha/erode →
//       적용(/ext/apply·/ext/batch-apply), 그리고 적용된 항목을 골라/전체 되돌리기(/ext/revert).

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
const SUBTLE_BUTTON =
  'rounded-lg px-3 py-1.5 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed'

const STYLE_SERVICE_BASE = '/api/style'
const SERVICE_GUIDE = '스타일 서비스에 연결할 수 없습니다 — style-service 폴더에서 python server.py를 실행하세요.'

type ExtProject = { id: string; name: string }
type ExtAsset = { path: string; size: number }

export interface ExternalSpriteStyler {
  button: HTMLButtonElement
  backdrop: HTMLDivElement
}

export const createExternalSpriteStyler = (): ExternalSpriteStyler => {
  const button = el(
    'button',
    'rounded-lg px-2.5 py-1 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100',
    '🎮 외부 게임 스타일'
  ) as HTMLButtonElement
  button.type = 'button'
  button.title = '외부 게임(Love2D 등) 스프라이트에 스타일 적용'

  const backdrop = el('div', 'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4')
  backdrop.style.display = 'none'
  const panel = el('div', 'w-full max-w-2xl max-h-[92vh] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900 p-5 flex flex-col gap-4 shadow-2xl')

  const top = el('div', 'flex items-center justify-between')
  const title = el('span', 'text-sm font-semibold tracking-tight', '🎮 외부 게임 스프라이트 스타일')
  const close = el('button', 'text-zinc-500 text-sm transition hover:text-zinc-200', '✕')
  close.type = 'button'
  top.append(title, close)

  const projectLine = el('div', 'text-xs text-zinc-400', '')
  const note = el(
    'div',
    'text-[11px] text-amber-300/80 leading-relaxed',
    '외부 게임 폴더는 Vite 감시 밖이라 적용해도 에디터가 자동 새로고침되지 않습니다 — 게임을 다시 실행하면 반영됩니다.'
  )

  // ── 스프라이트 선택 ──
  const spriteCard = el('div', CARD)
  const spriteHead = el('div', 'flex items-center justify-between gap-2')
  spriteHead.append(el('div', LABEL, '① 스프라이트 선택 (다중 가능)'))
  const spriteSelectAll = el('button', 'text-[11px] text-zinc-400 transition hover:text-zinc-200', '모두 선택') as HTMLButtonElement
  spriteSelectAll.type = 'button'
  spriteHead.append(spriteSelectAll)
  const spritePicker = createThumbnailPicker({ multiSelect: true, onChange: () => syncApplyButton() })
  spriteCard.append(spriteHead, spritePicker.node)

  // ── 스타일 이미지 ──
  const styleCard = el('div', CARD)
  styleCard.append(el('div', LABEL, '② 스타일 이미지'))
  const styleRow = el('div', 'flex items-center gap-3')
  const styleInput = el('input', 'hidden') as HTMLInputElement
  styleInput.type = 'file'
  styleInput.accept = 'image/png,image/jpeg,image/webp'
  const stylePickButton = el('button', SUBTLE_BUTTON, '이미지 선택...') as HTMLButtonElement
  stylePickButton.type = 'button'
  const stylePreview = el('img', 'h-12 w-12 rounded-md object-cover border border-white/10 [image-rendering:auto]') as HTMLImageElement
  stylePreview.style.display = 'none'
  const styleName = el('span', 'text-xs text-zinc-400 truncate', '선택된 이미지가 없습니다')
  styleRow.append(stylePickButton, stylePreview, styleName)
  styleCard.append(styleRow, styleInput)

  // ── 옵션 ──
  const optCard = el('div', CARD)
  optCard.append(el('div', LABEL, '③ 옵션'))
  const alphaRow = el('div', 'flex items-center gap-3')
  alphaRow.append(el('span', 'text-xs text-zinc-400 w-28', '스타일 강도(alpha)'))
  const alphaInput = el('input', 'flex-1') as HTMLInputElement
  alphaInput.type = 'range'
  alphaInput.min = '0'
  alphaInput.max = '1'
  alphaInput.step = '0.05'
  alphaInput.value = '1'
  const alphaValue = el('span', 'text-xs text-zinc-300 w-10 text-right', '1.00')
  alphaInput.addEventListener('input', () => {
    alphaValue.textContent = Number(alphaInput.value).toFixed(2)
  })
  alphaRow.append(alphaInput, alphaValue)
  const erodeRow = el('div', 'flex items-center gap-3')
  erodeRow.append(el('span', 'text-xs text-zinc-400 w-28', '경계 침식(erode)'))
  const erodeSelect = el('select', 'rounded-md bg-zinc-800 border border-white/10 text-xs text-zinc-200 px-2 py-1') as HTMLSelectElement
  for (const px of [0, 1, 2, 3]) {
    const opt = el('option', '', px === 0 ? '없음' : `${px}px`) as HTMLOptionElement
    opt.value = String(px)
    erodeSelect.append(opt)
  }
  erodeRow.append(erodeSelect)
  optCard.append(alphaRow, erodeRow)

  const applyButton = el('button', PRIMARY_BUTTON, '스타일 적용') as HTMLButtonElement
  applyButton.type = 'button'
  applyButton.disabled = true

  // ── 되돌리기 ──
  const revertCard = el('div', CARD)
  const revertHead = el('div', 'flex items-center justify-between gap-2')
  revertHead.append(el('div', LABEL, '되돌리기 (적용된 항목)'))
  const revertSelectAll = el('button', 'text-[11px] text-zinc-400 transition hover:text-zinc-200', '모두 선택') as HTMLButtonElement
  revertSelectAll.type = 'button'
  revertHead.append(revertSelectAll)
  const revertPicker = createThumbnailPicker({ multiSelect: true, onChange: () => syncRevertButtons() })
  const revertRow = el('div', 'flex items-center gap-2')
  const revertSelectedButton = el('button', SUBTLE_BUTTON, '선택 되돌리기') as HTMLButtonElement
  revertSelectedButton.type = 'button'
  revertSelectedButton.disabled = true
  const revertAllButton = el('button', DANGER_BUTTON, '전체 되돌리기') as HTMLButtonElement
  revertAllButton.type = 'button'
  revertRow.append(revertSelectedButton, revertAllButton)
  revertCard.append(revertHead, revertPicker.node, revertRow)

  const status = el('span', 'text-xs text-zinc-400 min-h-4 leading-relaxed', '')

  panel.append(top, projectLine, note, spriteCard, styleCard, optCard, applyButton, revertCard, status)
  backdrop.append(panel)

  // ── 상태 ──
  let project: ExtProject | null = null
  let sprites: ExtAsset[] = []
  let styled: ExtAsset[] = []
  let styleFile: File | null = null
  let isBusy = false
  let cacheBust = 0
  let allRevertConfirmArmed = false

  const setStatus = (message: string): void => {
    status.textContent = message
  }

  const shortName = (path: string): string => path.replace(/^sprites\//, '')

  const thumbUrl = (path: string): string =>
    `${STYLE_SERVICE_BASE}/ext/asset?project=${encodeURIComponent(project!.id)}&path=${encodeURIComponent(path)}&t=${cacheBust}`

  const syncApplyButton = (): void => {
    const count = spritePicker.getSelected().length
    applyButton.disabled = isBusy || count === 0 || styleFile === null
    applyButton.textContent = count > 1 ? `스타일 적용 (${count}개)` : '스타일 적용'
    spriteSelectAll.textContent = sprites.length > 0 && count === sprites.length ? '모두 해제' : '모두 선택'
  }

  const syncRevertButtons = (): void => {
    const count = revertPicker.getSelected().length
    revertSelectedButton.disabled = isBusy || count === 0
    revertSelectedButton.textContent = count > 0 ? `선택 되돌리기 (${count})` : '선택 되돌리기'
    revertAllButton.disabled = isBusy || styled.length === 0
    revertSelectAll.textContent = styled.length > 0 && count === styled.length ? '모두 해제' : '모두 선택'
  }

  const resetAllRevertConfirm = (): void => {
    allRevertConfirmArmed = false
    revertAllButton.textContent = '전체 되돌리기'
    revertAllButton.className = DANGER_BUTTON
  }

  const renderSprites = (): void => {
    spritePicker.setItems(
      sprites.map((asset) => ({ id: asset.path, label: shortName(asset.path), thumbUrl: thumbUrl(asset.path) })),
      '스프라이트가 없습니다.'
    )
    syncApplyButton()
  }

  const renderStyled = (): void => {
    revertPicker.setItems(
      styled.map((asset) => ({ id: asset.path, label: shortName(asset.path), thumbUrl: thumbUrl(asset.path) })),
      '아직 스타일이 적용된 스프라이트가 없습니다.'
    )
    resetAllRevertConfirm()
    syncRevertButtons()
  }

  const loadStyled = async (): Promise<void> => {
    if (!project) {
      return
    }
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/ext/styled?project=${encodeURIComponent(project.id)}`)
      if (!response.ok) {
        throw new Error(String(response.status))
      }
      styled = ((await response.json()) as { assets: ExtAsset[] }).assets
    } catch {
      styled = []
    }
    renderStyled()
  }

  const load = async (): Promise<void> => {
    setStatus('외부 게임 목록을 불러오는 중...')
    try {
      const projectsResponse = await fetch(`${STYLE_SERVICE_BASE}/ext/projects`)
      if (!projectsResponse.ok) {
        throw new Error(String(projectsResponse.status))
      }
      const projects = ((await projectsResponse.json()) as { projects: ExtProject[] }).projects
      if (projects.length === 0) {
        project = null
        projectLine.textContent = '등록된 외부 게임이 없습니다 (config.json의 externalProjects 확인).'
        sprites = []
        styled = []
        renderSprites()
        renderStyled()
        setStatus('')
        return
      }
      project = projects[0]
      projectLine.textContent = `대상: ${project.name}`

      cacheBust += 1
      const assetsResponse = await fetch(`${STYLE_SERVICE_BASE}/ext/assets?project=${encodeURIComponent(project.id)}`)
      sprites = assetsResponse.ok ? ((await assetsResponse.json()) as { assets: ExtAsset[] }).assets : []
      renderSprites()
      await loadStyled()
      setStatus(sprites.length > 0 ? '' : '스프라이트를 찾지 못했습니다.')
    } catch {
      project = null
      sprites = []
      styled = []
      renderSprites()
      renderStyled()
      setStatus(SERVICE_GUIDE)
    }
  }

  const buildForm = (): FormData => {
    const form = new FormData()
    form.set('project', project!.id)
    form.set('alpha', alphaInput.value)
    form.set('alpha_erode', erodeSelect.value)
    form.set('style', styleFile!)
    return form
  }

  const runApply = async (): Promise<void> => {
    if (isBusy || !project || !styleFile) {
      return
    }
    const paths = spritePicker.getSelected()
    if (paths.length === 0) {
      return
    }
    isBusy = true
    syncApplyButton()
    syncRevertButtons()
    setStatus(`스타일 변환 중... (CPU 추론, 장당 수 초 — ${paths.length}장)`)
    try {
      let appliedCount: number
      let failed: Array<{ path?: string; error?: string }> = []
      if (paths.length === 1) {
        const form = buildForm()
        form.set('path', paths[0])
        const response = await fetch(`${STYLE_SERVICE_BASE}/ext/apply`, { method: 'POST', body: form })
        if (!response.ok) {
          throw new Error(`${response.status}: ${(await response.text()).slice(0, 200)}`)
        }
        appliedCount = 1
      } else {
        const form = buildForm()
        form.set('paths', JSON.stringify(paths))
        const response = await fetch(`${STYLE_SERVICE_BASE}/ext/batch-apply`, { method: 'POST', body: form })
        if (!response.ok) {
          throw new Error(`${response.status}: ${(await response.text()).slice(0, 200)}`)
        }
        const result = (await response.json()) as { applied: string[]; failed: Array<{ path?: string; error?: string }> }
        appliedCount = result.applied.length
        failed = result.failed
      }
      isBusy = false
      cacheBust += 1
      renderSprites()
      await loadStyled()
      const failNote = failed.length > 0 ? ` · 실패 ${failed.length}개` : ''
      setStatus(`✅ ${appliedCount}개 스프라이트에 스타일을 적용했습니다${failNote} — 게임을 다시 실행하면 반영됩니다.`)
    } catch (error) {
      isBusy = false
      renderSprites()
      syncRevertButtons()
      const failedToFetch = error instanceof TypeError
      setStatus(failedToFetch ? SERVICE_GUIDE : `적용 실패: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const runRevert = async (paths: string[]): Promise<void> => {
    if (isBusy || !project || paths.length === 0) {
      return
    }
    isBusy = true
    syncApplyButton()
    syncRevertButtons()
    setStatus('원본으로 복원 중...')
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/ext/revert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: project.id, paths })
      })
      if (!response.ok) {
        throw new Error(`${response.status}: ${(await response.text()).slice(0, 200)}`)
      }
      const result = (await response.json()) as { reverted: string[]; failed: Array<{ path: string }> }
      isBusy = false
      cacheBust += 1
      renderSprites()
      await loadStyled()
      const failNote = result.failed.length > 0 ? ` (실패 ${result.failed.length}개)` : ''
      setStatus(`↩ ${result.reverted.length}개를 원본으로 되돌렸습니다${failNote} — 게임을 다시 실행하면 반영됩니다.`)
    } catch (error) {
      isBusy = false
      renderSprites()
      syncRevertButtons()
      const failedToFetch = error instanceof TypeError
      setStatus(failedToFetch ? SERVICE_GUIDE : `되돌리기 실패: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  // ── 이벤트 ──
  stylePickButton.addEventListener('click', () => styleInput.click())
  styleInput.addEventListener('change', () => {
    const file = styleInput.files?.[0] ?? null
    styleFile = file
    if (file) {
      styleName.textContent = file.name
      stylePreview.src = URL.createObjectURL(file)
      stylePreview.style.display = ''
    } else {
      styleName.textContent = '선택된 이미지가 없습니다'
      stylePreview.style.display = 'none'
    }
    syncApplyButton()
  })

  spriteSelectAll.addEventListener('click', () => {
    if (spritePicker.getSelected().length === sprites.length) {
      spritePicker.setSelected([])
    } else {
      spritePicker.setSelected(sprites.map((asset) => asset.path))
    }
  })

  revertSelectAll.addEventListener('click', () => {
    if (revertPicker.getSelected().length === styled.length) {
      revertPicker.setSelected([])
    } else {
      revertPicker.setSelected(styled.map((asset) => asset.path))
    }
  })

  applyButton.addEventListener('click', () => void runApply())
  revertSelectedButton.addEventListener('click', () => void runRevert(revertPicker.getSelected()))
  revertAllButton.addEventListener('click', () => {
    if (isBusy || styled.length === 0) {
      return
    }
    if (!allRevertConfirmArmed) {
      allRevertConfirmArmed = true
      revertAllButton.textContent = `⚠ ${styled.length}개 모두 되돌리기 확정`
      revertAllButton.className = DANGER_BUTTON + ' ring-2 ring-rose-400/60'
      setStatus('전체 되돌리기를 다시 누르면 실행됩니다.')
      return
    }
    void runRevert(styled.map((asset) => asset.path))
  })

  const openModal = (): void => {
    backdrop.style.display = 'flex'
    setStatus('')
    resetAllRevertConfirm()
    void load()
  }
  const closeModal = (): void => {
    backdrop.style.display = 'none'
    resetAllRevertConfirm()
  }
  button.addEventListener('click', openModal)
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

  return { button, backdrop }
}
