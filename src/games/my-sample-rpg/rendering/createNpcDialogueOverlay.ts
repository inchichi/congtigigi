// 비주얼노벨 스타일 NPC 대화 오버레이.
// 화면 왼쪽에 큰 초상화, 그 위 중앙에 이름 명패, 하단에 금테 대사창을 띄운다.
// 대사는 한 줄씩 보여주고 클릭/Space/Enter 로 다음 줄로 넘어가며, 마지막 줄에서 더 넘기면
// 닫히고 onComplete 가 호출된다. DOM + 인라인 스타일로 자급자족하며 게임 styles.css 에 의존하지 않는다.

const STYLE_ELEMENT_ID = 'npc-dialogue-overlay-style'

// blink 애니메이션 등 인라인 style 로 표현하기 번거로운 것만 한 번 주입한다.
const ensureStyleInjected = () => {
  if (document.getElementById(STYLE_ELEMENT_ID)) {
    return
  }
  const style = document.createElement('style')
  style.id = STYLE_ELEMENT_ID
  style.textContent = `
@keyframes npc-dialogue-advance-blink {
  0%, 100% { opacity: 0.25; transform: translateY(0); }
  50% { opacity: 1; transform: translateY(3px); }
}
.npc-dialogue-overlay__advance {
  animation: npc-dialogue-advance-blink 1.1s ease-in-out infinite;
}
.npc-dialogue-overlay__portrait {
  transition: opacity 220ms ease, transform 220ms ease;
}
`
  document.head.append(style)
}

type ShowNpcDialogueInput = {
  portraitUrl: string
  name: string
  lines: string[]
  onComplete?: () => void
}

export type NpcDialogueOverlay = {
  show: (input: ShowNpcDialogueInput) => void
  hide: () => void
  isOpen: () => boolean
  destroy: () => void
}

type CreateNpcDialogueOverlayInput = {
  mountElement: HTMLElement
}

// 게임 HUD/상점과 통일: 양피지 크림 패널 + 얇은 갈색 테두리 + 진갈색 글자(NeoDunggeunmo).
const PARCHMENT = '#fff9ee'
const PARCHMENT_DEEP = '#f7ecd4'
const BORDER_BROWN = 'rgba(111, 89, 58, 0.5)'
const BORDER_BROWN_SOFT = 'rgba(111, 89, 58, 0.3)'
const TEXT_DARK = '#2e2313'
const LABEL_BROWN = '#6b5534'
const BOX_BG = 'rgba(255, 249, 238, 0.96)'

export const createNpcDialogueOverlay = ({
  mountElement
}: CreateNpcDialogueOverlayInput): NpcDialogueOverlay => {
  ensureStyleInjected()

  const overlayRoot = document.createElement('div')
  const portrait = document.createElement('img')
  const nameBanner = document.createElement('div')
  const dialogueBox = document.createElement('div')
  const dialogueText = document.createElement('div')
  const advanceIndicator = document.createElement('div')

  overlayRoot.className = 'npc-dialogue-overlay'
  overlayRoot.setAttribute('aria-hidden', 'true')
  Object.assign(overlayRoot.style, {
    position: 'absolute',
    inset: '0',
    zIndex: '70',
    display: 'none',
    pointerEvents: 'none',
    // 하단을 살짝 어둡게 깔아 초상화·대화창을 바닥에 앉힌다(과하지 않게).
    background:
      'linear-gradient(to top, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.12) 24%, rgba(0,0,0,0) 42%)',
    fontFamily: "'NeoDunggeunmo', 'Jersey 25', monospace",
    userSelect: 'none'
  } as CSSStyleDeclaration)

  portrait.className = 'npc-dialogue-overlay__portrait'
  portrait.alt = ''
  portrait.draggable = false
  Object.assign(portrait.style, {
    position: 'absolute',
    left: '1.5%',
    bottom: '0',
    height: '94%',
    maxWidth: '46%',
    objectFit: 'contain',
    objectPosition: 'left bottom',
    filter: 'drop-shadow(0 12px 26px rgba(0,0,0,0.6))',
    pointerEvents: 'none'
  } as CSSStyleDeclaration)

  nameBanner.className = 'npc-dialogue-overlay__name'
  Object.assign(nameBanner.style, {
    position: 'absolute',
    left: '50%',
    transform: 'translateX(-50%)',
    bottom: 'calc(4% + 1px)', // 대사창 상단 테두리 위에 걸치게
    padding: '7px 30px',
    color: TEXT_DARK,
    fontWeight: '700',
    fontSize: 'clamp(15px, 2.4vh, 26px)',
    letterSpacing: '0.04em',
    background: `linear-gradient(180deg, ${PARCHMENT} 0%, ${PARCHMENT_DEEP} 100%)`,
    border: `1px solid ${BORDER_BROWN}`,
    borderRadius: '6px',
    boxShadow:
      `0 0 0 1px ${BORDER_BROWN_SOFT}, 0 4px 10px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.75)`,
    whiteSpace: 'nowrap',
    pointerEvents: 'none'
  } as CSSStyleDeclaration)

  dialogueBox.className = 'npc-dialogue-overlay__box'
  Object.assign(dialogueBox.style, {
    position: 'absolute',
    left: '4%',
    right: '4%',
    bottom: '4%',
    minHeight: '20%',
    boxSizing: 'border-box',
    padding: 'clamp(18px, 3.2vh, 34px) clamp(26px, 4vw, 52px)',
    background: BOX_BG,
    border: `1px solid ${BORDER_BROWN}`,
    borderRadius: '8px',
    boxShadow:
      `0 0 0 1px ${BORDER_BROWN_SOFT}, inset 0 1px 0 rgba(255,255,255,0.7), ` +
      '0 14px 30px rgba(0,0,0,0.4)',
    color: TEXT_DARK,
    cursor: 'pointer',
    pointerEvents: 'auto'
  } as CSSStyleDeclaration)

  dialogueText.className = 'npc-dialogue-overlay__text'
  Object.assign(dialogueText.style, {
    fontSize: 'clamp(15px, 2.6vh, 28px)',
    lineHeight: '1.6',
    whiteSpace: 'pre-wrap',
    color: TEXT_DARK,
    // 초상화와 겹치지 않도록 왼쪽 여백을 살짝 둔다.
    paddingLeft: 'clamp(0px, 6vw, 90px)'
  } as CSSStyleDeclaration)

  advanceIndicator.className = 'npc-dialogue-overlay__advance'
  advanceIndicator.textContent = '▼'
  Object.assign(advanceIndicator.style, {
    position: 'absolute',
    right: 'clamp(16px, 2vw, 28px)',
    bottom: 'clamp(10px, 1.4vh, 18px)',
    color: LABEL_BROWN,
    fontSize: 'clamp(13px, 1.8vh, 20px)',
    pointerEvents: 'none'
  } as CSSStyleDeclaration)

  dialogueBox.append(dialogueText, advanceIndicator)
  overlayRoot.append(portrait, nameBanner, dialogueBox)
  mountElement.append(overlayRoot)

  let lines: string[] = []
  let lineIndex = 0
  let open = false
  let onComplete: (() => void) | undefined

  const renderCurrentLine = () => {
    dialogueText.textContent = lines[lineIndex] ?? ''
    const isLast = lineIndex >= lines.length - 1
    advanceIndicator.textContent = isLast ? '✕' : '▼'
  }

  const hide = () => {
    if (!open) {
      return
    }
    open = false
    overlayRoot.style.display = 'none'
    overlayRoot.style.pointerEvents = 'none'
    overlayRoot.setAttribute('aria-hidden', 'true')
    window.removeEventListener('keydown', handleKeyDown, true)
    lines = []
    lineIndex = 0
  }

  // 다음 줄로. 마지막 줄에서 더 넘기면 닫고 onComplete 호출.
  const advance = () => {
    if (!open) {
      return
    }
    if (lineIndex < lines.length - 1) {
      lineIndex += 1
      renderCurrentLine()
      return
    }
    const completion = onComplete
    hide()
    completion?.()
  }

  const handleKeyDown = (event: KeyboardEvent) => {
    if (!open) {
      return
    }
    if (
      event.code === 'Space' ||
      event.code === 'Enter' ||
      event.code === 'NumpadEnter'
    ) {
      // 게임 입력(점프/상호작용 등)과 겹치지 않도록 이 오버레이가 키를 소비한다.
      event.preventDefault()
      event.stopImmediatePropagation()
      advance()
      return
    }
    if (event.code === 'Escape') {
      event.preventDefault()
      event.stopImmediatePropagation()
      const completion = onComplete
      hide()
      completion?.()
    }
  }

  dialogueBox.addEventListener('click', (event) => {
    event.preventDefault()
    event.stopPropagation()
    advance()
  })

  const show = (input: ShowNpcDialogueInput) => {
    const validLines = input.lines.filter((line) => line.trim().length > 0)
    if (validLines.length === 0) {
      return
    }
    lines = validLines
    lineIndex = 0
    onComplete = input.onComplete
    if (input.portraitUrl) {
      portrait.src = input.portraitUrl
      portrait.style.display = ''
    } else {
      portrait.removeAttribute('src')
      portrait.style.display = 'none'
    }
    nameBanner.textContent = input.name
    nameBanner.style.display = input.name.trim().length > 0 ? '' : 'none'
    renderCurrentLine()

    open = true
    overlayRoot.style.display = 'block'
    overlayRoot.style.pointerEvents = 'auto'
    overlayRoot.setAttribute('aria-hidden', 'false')
    window.addEventListener('keydown', handleKeyDown, true)
  }

  return {
    show,
    hide,
    isOpen: () => open,
    destroy: () => {
      window.removeEventListener('keydown', handleKeyDown, true)
      overlayRoot.remove()
    }
  }
}
