import { createBeforeAfterView } from './createBeforeAfterView'
import { loadThemeWorkLog } from './themeWorkLog'

// 변환 결과 요약 페이지(/editor.html?workspace=summary).
// 적용 이력 JSON 전체와 스타일 전이 전/후 타일 비교·다운로드를 한 화면에 모은다.

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

const NAV_LINK =
  'rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-[12px] text-[#c9c9c9] hover:border-[#d9a85c]/60 hover:text-[#f1dfb5]'
const SMALL_BUTTON =
  'rounded-lg border border-[#d9a85c]/30 bg-[#1c1c1e] px-2.5 py-1.5 text-[10px] text-[#cbb27b] hover:border-[#d9a85c]'

const downloadJson = (fileName: string, payload: unknown): void => {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.click()
  URL.revokeObjectURL(url)
}

export const createSummaryPage = ({ mountElement }: { mountElement: HTMLElement }): void => {
  document.title = '변환 결과 요약'
  const entries = loadThemeWorkLog()

  const shell = el('div', 'settings-game-font min-h-screen bg-[#171719] p-5 text-[#d4d4d4] lg:p-10')
  const header = el('div', 'mb-5 flex flex-wrap items-center justify-between gap-3')
  header.append(el('h1', 'text-[22px] font-bold text-[#f1dfb5]', '📊 변환 결과 요약'))
  const nav = el('div', 'flex gap-2')
  const themeLink = el('a', NAV_LINK, '테마 작업실') as HTMLAnchorElement
  themeLink.href = '/editor.html?workspace=theme'
  const editorLink = el('a', NAV_LINK, '에디터') as HTMLAnchorElement
  editorLink.href = '/editor.html'
  nav.append(themeLink, editorLink)
  header.append(nav)
  shell.append(header)

  const logSection = el('section', 'mb-6 rounded-2xl border border-[#d9a85c]/25 bg-[#19191b] p-4')
  const logHeader = el('div', 'mb-3 flex flex-wrap items-center justify-between gap-2')
  logHeader.append(el('h2', 'text-[15px] font-semibold text-[#e8d5a5]', `적용 이력 JSON · ${entries.length}건`))
  if (entries.length > 0) {
    const allButton = el('button', SMALL_BUTTON, '전체 JSON 다운로드') as HTMLButtonElement
    allButton.type = 'button'
    allButton.addEventListener('click', () => downloadJson('theme-work-log-all.json', entries))
    logHeader.append(allButton)
  }
  logSection.append(logHeader)
  if (entries.length === 0) {
    logSection.append(
      el('div', 'rounded-xl border border-dashed border-white/10 p-6 text-center text-[12px] text-[#77777b]', '아직 적용 이력이 없습니다. 테마 작업실에서 게임에 최종 적용하면 이곳에 쌓입니다.')
    )
  }
  entries.forEach((entry) => {
    const card = el('article', 'mb-3 rounded-xl border border-[#d9a85c]/20 bg-[#121214] p-3')
    const top = el('div', 'mb-1 flex flex-wrap items-center justify-between gap-2')
    top.append(
      el('div', 'text-[13px] font-semibold text-[#e8d5a5]', entry.theme || '(제목 없음)'),
      el('div', 'text-[10px] text-[#8f8f92]', new Date(entry.created_at).toLocaleString('ko-KR'))
    )
    const meta = el(
      'div',
      'mb-2 whitespace-pre-wrap text-[11px] leading-relaxed text-[#9d9d9d]',
      `퀘스트: ${entry.quest_summary.title} (기버 ${entry.quest_summary.giver_npc_id})\nFLUX 적용 ${entry.applied_targets.length}개${entry.reward_item ? `\n보상 아이템: ${entry.reward_item.label} (${entry.reward_item.item_id})` : ''}`
    )
    const detail = el('details', 'text-[11px] text-[#9d9d9d]')
    const pre = el('pre', 'mt-2 max-h-[320px] overflow-auto whitespace-pre-wrap break-words rounded-xl border border-[#d9a85c]/15 bg-[#0f0f11] p-3 text-[11px] leading-[1.6] text-[#d4d4d4]')
    pre.textContent = JSON.stringify(entry, null, 2)
    detail.append(el('summary', 'cursor-pointer text-[#cbb27b]', '전체 JSON 보기'), pre)
    const actions = el('div', 'mt-2 flex gap-1.5')
    const saveButton = el('button', SMALL_BUTTON, 'JSON 다운로드') as HTMLButtonElement
    saveButton.type = 'button'
    saveButton.addEventListener('click', () => downloadJson(`theme-log-${entry.id}.json`, entry))
    actions.append(saveButton)
    card.append(top, meta, detail, actions)
    logSection.append(card)
  })
  shell.append(logSection)

  const baSection = el('section', 'rounded-2xl border border-[#d9a85c]/25 bg-[#19191b] p-4')
  baSection.append(el('h2', 'mb-3 text-[15px] font-semibold text-[#e8d5a5]', '스타일 전이 전/후 타일 비교'))
  const beforeAfter = createBeforeAfterView()
  beforeAfter.view.hidden = false
  beforeAfter.view.className = 'flex flex-col gap-3'
  beforeAfter.refresh()
  baSection.append(beforeAfter.view)
  shell.append(baSection)

  mountElement.append(shell)
}
