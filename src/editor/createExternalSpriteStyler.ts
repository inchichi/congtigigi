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
const SUBTLE_BUTTON =
  'rounded-lg px-3 py-1.5 text-sm bg-white/[0.04] border border-white/10 text-zinc-300 transition hover:bg-white/[0.08] hover:text-zinc-100 disabled:opacity-40 disabled:cursor-not-allowed'

const STYLE_SERVICE_BASE = '/api/style'
const SERVICE_GUIDE =
  '편집 서비스에 연결할 수 없습니다 — style-service 폴더에서 python server.py를 실행하세요.'

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
    '외부 편집'
  ) as HTMLButtonElement
  button.type = 'button'

  const backdrop = el('div', 'fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4')
  backdrop.style.display = 'none'
  const panel = el('div', 'w-full max-w-2xl max-h-[92vh] overflow-y-auto rounded-2xl border border-white/10 bg-zinc-900 p-5 flex flex-col gap-4 shadow-2xl')

  const top = el('div', 'flex items-center justify-between')
  top.append(el('span', 'text-sm font-semibold tracking-tight', '외부 게임 편집'))
  const close = el('button', 'text-zinc-500 text-sm transition hover:text-zinc-200', '✕') as HTMLButtonElement
  close.type = 'button'
  top.append(close)

  const projectLine = el('div', 'text-xs text-zinc-400', '')
  const note = el(
    'div',
    'text-[11px] text-amber-300/80 leading-relaxed',
    '외부 게임의 PNG에 프롬프트 기반 편집을 적용합니다. 스타일 이미지는 사용하지 않습니다.'
  )

  const projectCard = el('div', CARD)
  const projectSelect = el('select', 'rounded-lg border border-white/15 bg-black/40 px-2 py-1.5 text-xs text-zinc-100 outline-none') as HTMLSelectElement
  projectCard.append(el('div', LABEL, '프로젝트'), projectSelect)

  const assetCard = el('div', CARD)
  const assetHead = el('div', 'flex items-center justify-between gap-2')
  assetHead.append(el('div', LABEL, '스프라이트 선택'))
  const assetSelectAll = el('button', 'text-[11px] text-zinc-400 transition hover:text-zinc-200', '모두 선택') as HTMLButtonElement
  assetSelectAll.type = 'button'
  assetHead.append(assetSelectAll)
  const assetPicker = createThumbnailPicker({ multiSelect: true })
  assetCard.append(assetHead, assetPicker.node)

  const promptCard = el('div', CARD)
  const promptInput = document.createElement('textarea')
  promptInput.className =
    'min-h-28 w-full rounded-lg border border-white/15 bg-black/40 px-3 py-2 text-sm text-zinc-100 outline-none placeholder:text-zinc-500'
  promptInput.rows = 5
  promptInput.placeholder =
    '예: 셀 셰이딩 만화풍, 강한 외곽선, 채도가 높은 색감, 원본 구도는 유지'
  promptCard.append(el('div', LABEL, '편집 프롬프트'), promptInput)

  const alphaCard = el('div', CARD)
  const alphaHead = el('div', 'flex items-center justify-between')
  const alphaValue = el('span', 'text-xs text-zinc-300', '1.00')
  alphaHead.append(el('div', LABEL, '편집 강도 (alpha)'), alphaValue)
  const alphaInput = el('input', 'w-full accent-indigo-500') as HTMLInputElement
  alphaInput.type = 'range'
  alphaInput.min = '0'
  alphaInput.max = '1'
  alphaInput.step = '0.05'
  alphaInput.value = '1'
  alphaInput.addEventListener('input', () => {
    alphaValue.textContent = Number(alphaInput.value).toFixed(2)
  })
  alphaCard.append(alphaHead, alphaInput)

  const buttons = el('div', 'flex flex-wrap items-center gap-2')
  const runButton = el('button', PRIMARY_BUTTON, '변환') as HTMLButtonElement
  const batchButton = el('button', PRIMARY_BUTTON, '일괄 변환') as HTMLButtonElement
  const revertButton = el('button', SUBTLE_BUTTON, '되돌리기') as HTMLButtonElement
  const status = el('span', 'text-xs text-zinc-500 min-h-4', '')
  for (const btn of [runButton, batchButton, revertButton]) btn.type = 'button'
  batchButton.style.display = 'none'
  buttons.append(runButton, batchButton, revertButton, status)

  const resultWrap = el('div', CARD)
  const resultImage = el('img', 'max-h-72 w-full rounded-lg object-contain bg-black/40')
  resultImage.style.display = 'none'
  resultWrap.append(el('div', LABEL, '결과 미리보기'), resultImage)

  panel.append(top, projectLine, note, projectCard, assetCard, promptCard, alphaCard, buttons, resultWrap)
  backdrop.append(panel)

  let projects: ExtProject[] = []
  let assets: ExtAsset[] = []
  let selectedProject: ExtProject | undefined
  let selectedAssetPaths: string[] = []
  let isBusy = false
  let currentBackup: string | undefined

  const setStatus = (message: string): void => {
    status.textContent = message
  }

  const syncButtons = (): void => {
    const hasPrompt = promptInput.value.trim().length > 0
    const busy = isBusy
    runButton.disabled = busy || !selectedProject || selectedAssetPaths.length !== 1 || !hasPrompt
    batchButton.disabled = busy || !selectedProject || selectedAssetPaths.length < 2 || !hasPrompt
    revertButton.disabled = busy || !currentBackup
  }

  const loadProjects = async (): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/ext/projects`)
      if (!response.ok) throw new Error(String(response.status))
      const data = (await response.json()) as { projects: ExtProject[] }
      projects = data.projects
      projectSelect.replaceChildren()
      for (const project of projects) {
        const option = el('option', '', project.name) as HTMLOptionElement
        option.value = project.id
        projectSelect.append(option)
      }
      selectedProject = projects[0]
      projectLine.textContent = selectedProject ? `프로젝트: ${selectedProject.name}` : ''
      if (selectedProject) await loadAssets(selectedProject.id)
    } catch {
      setStatus(SERVICE_GUIDE)
    }
    syncButtons()
  }

  const loadAssets = async (projectId: string): Promise<void> => {
    try {
      const response = await fetch(`${STYLE_SERVICE_BASE}/ext/assets?project=${encodeURIComponent(projectId)}`)
      if (!response.ok) throw new Error(String(response.status))
      const data = (await response.json()) as { assets: ExtAsset[] }
      assets = data.assets
      assetPicker.setItems(
        assets.map((asset) => ({ id: asset.path, label: asset.path.split('/').pop() ?? asset.path, thumbUrl: `${STYLE_SERVICE_BASE}/ext/asset?project=${encodeURIComponent(projectId)}&path=${encodeURIComponent(asset.path)}` })),
        '외부 PNG가 없습니다.'
      )
      selectedAssetPaths = assetPicker.getSelected()
      batchButton.style.display = assets.length > 1 ? 'inline-flex' : 'none'
    } catch {
      assets = []
      assetPicker.setItems([], '외부 에셋 목록을 불러오지 못했습니다.')
      setStatus(SERVICE_GUIDE)
    }
    syncButtons()
  }

  const applySingle = async (): Promise<void> => {
    if (!selectedProject || selectedAssetPaths.length !== 1 || isBusy) return
    isBusy = true
    syncButtons()
    const path = selectedAssetPaths[0]
    try {
      const form = new FormData()
      form.append('prompt', promptInput.value.trim())
      form.append('project', selectedProject.id)
      form.append('path', path)
      form.append('alpha', alphaInput.value)
      const response = await fetch(`${STYLE_SERVICE_BASE}/ext/apply`, { method: 'POST', body: form })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      const data = (await response.json()) as { ok?: boolean; backup?: string }
      currentBackup = data.backup
      setStatus('적용 완료')
    } catch (error) {
      setStatus(error instanceof TypeError ? SERVICE_GUIDE : `실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isBusy = false
      syncButtons()
    }
  }

  const applyBatch = async (): Promise<void> => {
    if (!selectedProject || selectedAssetPaths.length < 2 || isBusy) return
    isBusy = true
    syncButtons()
    try {
      const form = new FormData()
      form.append('prompt', promptInput.value.trim())
      form.append('project', selectedProject.id)
      form.append('paths', JSON.stringify(selectedAssetPaths))
      form.append('alpha', alphaInput.value)
      const response = await fetch(`${STYLE_SERVICE_BASE}/ext/batch-apply`, { method: 'POST', body: form })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      const result = (await response.json()) as { applied: string[]; failed: unknown[] }
      setStatus(`일괄 적용 완료: ${result.applied.length}개`)
    } catch (error) {
      setStatus(error instanceof TypeError ? SERVICE_GUIDE : `실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isBusy = false
      syncButtons()
    }
  }

  const applyRevert = async (): Promise<void> => {
    if (!selectedProject || !currentBackup || isBusy) return
    isBusy = true
    syncButtons()
    try {
      const payload = { project: selectedProject.id, paths: selectedAssetPaths }
      const response = await fetch(`${STYLE_SERVICE_BASE}/ext/revert`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      if (!response.ok) {
        const detail = await response.text()
        throw new Error(`서비스 오류 (${response.status}): ${detail.slice(0, 200)}`)
      }
      currentBackup = undefined
      setStatus('원본으로 되돌렸습니다.')
    } catch (error) {
      setStatus(error instanceof TypeError ? SERVICE_GUIDE : `실패: ${error instanceof Error ? error.message : String(error)}`)
    } finally {
      isBusy = false
      syncButtons()
    }
  }

  const openModal = (): void => {
    backdrop.style.display = 'flex'
  }

  button.addEventListener('click', () => {
    openModal()
    void loadProjects()
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

  projectSelect.addEventListener('change', () => {
    selectedProject = projects.find((project) => project.id === projectSelect.value)
    projectLine.textContent = selectedProject ? `프로젝트: ${selectedProject.name}` : ''
    if (selectedProject) void loadAssets(selectedProject.id)
  })
  promptInput.addEventListener('input', syncButtons)
  alphaInput.addEventListener('input', syncButtons)
  runButton.addEventListener('click', () => void applySingle())
  batchButton.addEventListener('click', () => void applyBatch())
  revertButton.addEventListener('click', () => void applyRevert())
  assetSelectAll.addEventListener('click', () => {
    assetPicker.setSelected(assetPicker.getSelected().length === assets.length ? [] : assets.map((asset) => asset.path))
  })

  syncButtons()
  return { button, backdrop }
}
