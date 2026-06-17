import type { QuestObjectiveDefinition } from '../questLog'
import { getPlayerEquipmentItemDefinitionById } from '../playerEquipment'
import { renderItemIconById } from './createBlacksmithShopOverlay'

// 퀘스트 목표의 "대상"(몬스터/아이템)을 파란 밑줄 링크로 만들고, 클릭하면 그 대상의 이미지를
// 보여주는 팝업을 띄운다. 유저가 무엇을 잡고/얻어야 하는지 시각적으로 알 수 있게 한다.
// 퀘스트 UI가 DOM이라 전부 DOM으로 구현한다(트래커·B창 공용).

const MONSTER_SLIME_SHEET_URL = new URL(
  '../assets/monsters/몬스터-말캉이.png',
  import.meta.url
).href
const MONSTER_PIG_SHEET_URL = new URL(
  '../assets/monsters/monster-pig-sheet.png',
  import.meta.url
).href

// 몬스터 스프라이트 시트의 idle 밴드(상/하)와 프레임 수 — 첫 idle 프레임만 잘라 팝업에 보여준다.
// (시트 프레임은 동적 검출이라 정확 크롭이 어려워 첫 프레임 베스트에포트.)
type MonsterPopupSprite = {
  sheetUrl: string
  idleTop: number
  idleBottom: number
  idleFrameCount: number
  label: string
}
const MONSTER_POPUP_SPRITES: Record<string, MonsterPopupSprite> = {
  monster_slime: {
    sheetUrl: MONSTER_SLIME_SHEET_URL,
    idleTop: 45,
    idleBottom: 140,
    idleFrameCount: 4,
    label: '말캉이'
  },
  monster_pig: {
    sheetUrl: MONSTER_PIG_SHEET_URL,
    idleTop: 40,
    idleBottom: 145,
    idleFrameCount: 4,
    label: '돼지'
  }
}

const POTION_LABELS: Record<string, string> = {
  'health-potion': '체력 회복 포션',
  'mana-potion': '마나 회복 포션'
}

const resolveItemLabel = (itemId: string): string =>
  getPlayerEquipmentItemDefinitionById(itemId)?.label ??
  POTION_LABELS[itemId] ??
  itemId

export type QuestTargetDescriptor =
  | { kind: 'monster'; label: string; appearanceType: string }
  | { kind: 'item'; label: string; itemId: string }

// 목표 → 클릭 가능한 대상(이미지 있는 것). shop/scene/talk는 이미지 대상이 없어 undefined.
export const describeQuestObjectiveTarget = (
  objective: QuestObjectiveDefinition
): QuestTargetDescriptor | undefined => {
  if (objective.type === 'monster-defeat' && objective.target.appearanceType) {
    const appearanceType = objective.target.appearanceType
    return {
      kind: 'monster',
      appearanceType,
      label: MONSTER_POPUP_SPRITES[appearanceType]?.label ?? appearanceType
    }
  }
  if (
    (objective.type === 'item-use' || objective.type === 'item-acquire') &&
    objective.target.itemId
  ) {
    return {
      kind: 'item',
      itemId: objective.target.itemId,
      label: resolveItemLabel(objective.target.itemId)
    }
  }
  return undefined
}

let activePopup: HTMLElement | undefined

const closeQuestTargetPopup = (): void => {
  activePopup?.remove()
  activePopup = undefined
  document.removeEventListener('keydown', handlePopupKeyDown)
}

const handlePopupKeyDown = (event: KeyboardEvent): void => {
  if (event.key === 'Escape') {
    event.stopPropagation()
    closeQuestTargetPopup()
  }
}

const renderMonsterIdleFrame = (
  box: HTMLElement,
  appearanceType: string
): void => {
  const sprite = MONSTER_POPUP_SPRITES[appearanceType]
  if (!sprite) {
    box.textContent = '?'
    return
  }
  const maxWidth = 180
  const maxHeight = 150
  const image = new Image()
  image.src = sprite.sheetUrl
  image.addEventListener('load', () => {
    const frameWidth = image.naturalWidth / sprite.idleFrameCount
    const bandHeight = sprite.idleBottom - sprite.idleTop
    if (frameWidth <= 0 || bandHeight <= 0) {
      return
    }
    const scale = Math.min(maxWidth / frameWidth, maxHeight / bandHeight)
    box.style.width = `${Math.round(frameWidth * scale)}px`
    box.style.height = `${Math.round(bandHeight * scale)}px`
    box.style.backgroundImage = `url(${sprite.sheetUrl})`
    box.style.backgroundRepeat = 'no-repeat'
    box.style.backgroundPosition = `0px -${Math.round(sprite.idleTop * scale)}px`
    box.style.backgroundSize = `${Math.round(image.naturalWidth * scale)}px ${Math.round(image.naturalHeight * scale)}px`
    box.style.imageRendering = 'pixelated'
  })
}

const showQuestTargetImagePopup = (descriptor: QuestTargetDescriptor): void => {
  closeQuestTargetPopup()

  const backdrop = document.createElement('div')
  backdrop.style.cssText =
    'position:fixed;inset:0;z-index:9000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.55);'

  const card = document.createElement('div')
  card.style.cssText =
    'display:flex;flex-direction:column;align-items:center;gap:10px;padding:18px 22px;border-radius:14px;background:#1f1b16;border:2px solid #d9a85c;box-shadow:0 8px 30px rgba(0,0,0,0.5);min-width:160px;'

  const imageBox = document.createElement('div')
  imageBox.style.cssText =
    'display:flex;align-items:center;justify-content:center;min-width:64px;min-height:64px;'
  if (descriptor.kind === 'item') {
    renderItemIconById(imageBox, descriptor.itemId, 6)
  } else {
    renderMonsterIdleFrame(imageBox, descriptor.appearanceType)
  }

  const caption = document.createElement('div')
  caption.textContent = descriptor.label
  caption.style.cssText = 'color:#f3d88b;font-size:14px;font-weight:600;'

  card.append(imageBox, caption)
  backdrop.append(card)
  backdrop.addEventListener('click', closeQuestTargetPopup)
  card.addEventListener('click', (event) => event.stopPropagation())

  document.body.append(backdrop)
  document.addEventListener('keydown', handlePopupKeyDown)
  activePopup = backdrop
}

// 목표 대상 이름의 파란 밑줄 클릭 링크. 이미지 대상이 없으면(상점/이동/대화) undefined.
export const createQuestTargetLink = (
  objective: QuestObjectiveDefinition
): HTMLButtonElement | undefined => {
  const descriptor = describeQuestObjectiveTarget(objective)
  if (!descriptor) {
    return undefined
  }
  const link = document.createElement('button')
  link.type = 'button'
  link.textContent = descriptor.label
  link.style.cssText =
    'background:none;border:none;padding:0;margin:0;color:#5db3ff;text-decoration:underline;cursor:pointer;font:inherit;'
  link.addEventListener('click', (event) => {
    event.preventDefault()
    event.stopPropagation()
    showQuestTargetImagePopup(descriptor)
  })
  return link
}
