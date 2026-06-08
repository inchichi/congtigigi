import type { GameStructureProfile } from './gameStructureProfile'
import type { GeneratedEventJson } from './eventJsonSchema'

type EventJsonGeneratorResult = {
  eventJson: GeneratedEventJson
  reasoning: string[]
}

export const generateMockEventJsonDraft = (
  userPrompt: string,
  profile: GameStructureProfile
): EventJsonGeneratorResult => {
  const normalizedPrompt = userPrompt.toLowerCase()
  const isChristmasEvent =
    normalizedPrompt.includes('크리스마스') ||
    normalizedPrompt.includes('christmas') ||
    normalizedPrompt.includes('xmas')
  const isHalloweenEvent =
    normalizedPrompt.includes('할로윈') || normalizedPrompt.includes('halloween')
  const targetMapId = pickKnownMapId(profile, normalizedPrompt)
  const targetNpc = pickKnownNpc(profile, normalizedPrompt, {
    isChristmasEvent,
    isHalloweenEvent
  })
  const targetRewardItemId = pickKnownItemId(profile, normalizedPrompt)
  const eventId = isChristmasEvent
    ? 'christmas_event'
    : isHalloweenEvent
      ? 'halloween_event'
      : createSlugFromText(userPrompt, 'custom_event')
  const eventName = isChristmasEvent
    ? 'christmas_event'
    : isHalloweenEvent
      ? 'halloween_event'
      : eventId
  const dialogueLines = createDialogueLines({
    isChristmasEvent,
    isHalloweenEvent,
    npcName: targetNpc.name,
    prompt: userPrompt
  })
  const requiresNewAsset = determineRequiresNewAsset({
    prompt: normalizedPrompt,
    profile,
    mapId: targetMapId,
    npcId: targetNpc.id,
    itemId: targetRewardItemId
  })

  return {
    eventJson: {
      event_id: eventId,
      event_name: eventName,
      description: createDescription({
        isChristmasEvent,
        isHalloweenEvent,
        npcName: targetNpc.name
      }),
      trigger: {
        type: 'talk',
        target: targetNpc.id,
        condition: 'player_interacts_with_npc'
      },
      location: {
        map_id: targetMapId,
        x: targetNpc.x,
        y: targetNpc.y
      },
      npc: {
        id: targetNpc.id,
        name: targetNpc.name,
        dialogue_id: `${targetNpc.id}_dialogue`
      },
      dialogue: dialogueLines,
      reward: {
        item_id: targetRewardItemId,
        amount: isHalloweenEvent ? 3 : 1
      },
      duration: {
        start: new Date().toISOString(),
        end: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString()
      },
      requires_new_asset: requiresNewAsset.requires_new_asset,
      requires_new_asset_reasons: requiresNewAsset.requires_new_asset_reasons
    },
    reasoning: requiresNewAsset.reasoning
  }
}

const pickKnownMapId = (profile: GameStructureProfile, prompt: string): string => {
  if (prompt.includes('사냥터') || prompt.includes('hunting')) {
    return 'hunting-ground'
  }

  if (prompt.includes('동굴') || prompt.includes('cave')) {
    return 'cave'
  }

  return profile.maps.find((map) => map.id === 'town')?.id ?? profile.maps[0].id
}

const pickKnownNpc = (
  profile: GameStructureProfile,
  prompt: string,
  {
    isChristmasEvent,
    isHalloweenEvent
  }: {
    isChristmasEvent: boolean
    isHalloweenEvent: boolean
  }
): { id: string; name: string; x: number; y: number } => {
  const santa = createNpcPlacement('santa', '산타', 544, 516)
  const wizard = createNpcPlacement('wizard', '마법사', 832, 543)
  const blacksmith = createNpcPlacement('blacksmith', '대장장이', 400, 516)
  const potionMerchant = createNpcPlacement('potion_merchant', '물약상인', 256, 516)

  if (isChristmasEvent) {
    return santa
  }

  if (isHalloweenEvent) {
    return santa
  }

  if (prompt.includes('마법사') || prompt.includes('wizard')) {
    return wizard
  }

  if (prompt.includes('대장장이') || prompt.includes('blacksmith')) {
    return blacksmith
  }

  if (prompt.includes('물약') || prompt.includes('potion')) {
    return potionMerchant
  }

  const candidate = profile.npcs.find((npc) => npc.id === 'wizard')

  return candidate
    ? wizard
    : profile.npcs.length > 0
      ? createNpcPlacement(profile.npcs[0].id, profile.npcs[0].name, 400, 516)
      : santa
}

const pickKnownItemId = (profile: GameStructureProfile, prompt: string): string => {
  if (prompt.includes('마나') || prompt.includes('mana')) {
    return profile.items.find((item) => item.id === 'mana-potion')?.id ?? profile.items[0].id
  }

  if (prompt.includes('검') || prompt.includes('sword')) {
    return profile.items.find((item) => item.id === 'basic-sword')?.id ?? profile.items[0].id
  }

  return profile.items.find((item) => item.id === 'health-potion')?.id ?? profile.items[0].id
}

const createDialogueLines = ({
  isChristmasEvent,
  isHalloweenEvent,
  npcName,
  prompt
}: {
  isChristmasEvent: boolean
  isHalloweenEvent: boolean
  npcName: string
  prompt: string
}): Array<{ speaker: string; text: string }> => {
  if (isChristmasEvent) {
    return [
      { speaker: npcName, text: '메리 크리스마스!' },
      { speaker: npcName, text: '오늘은 특별한 선물을 받을 수 있는 날이란다.' }
    ]
  }

  if (isHalloweenEvent) {
    return [
      { speaker: npcName, text: '할로윈이야!' },
      { speaker: npcName, text: '오늘은 사탕 대신 특별한 보상을 줄게.' }
    ]
  }

  return [
    {
      speaker: npcName,
      text: prompt.trim().length > 0 ? prompt.trim() : '새로운 이벤트가 시작되었다.'
    }
  ]
}

const createDescription = ({
  isChristmasEvent,
  isHalloweenEvent,
  npcName
}: {
  isChristmasEvent: boolean
  isHalloweenEvent: boolean
  npcName: string
}): string => {
  if (isChristmasEvent) {
    return `${npcName}가 등장하는 크리스마스 이벤트다.`
  }

  if (isHalloweenEvent) {
    return `${npcName}가 등장하는 할로윈 이벤트다.`
  }

  return `${npcName}가 등장하는 사용자 요청 기반 이벤트다.`
}

const determineRequiresNewAsset = ({
  prompt,
  profile,
  mapId,
  npcId,
  itemId
}: {
  prompt: string
  profile: GameStructureProfile
  mapId: string
  npcId: string
  itemId: string
}): {
  requires_new_asset: boolean
  requires_new_asset_reasons: string[]
  reasoning: string[]
} => {
  const reasons: string[] = []
  const reasoning: string[] = []

  if (!profile.maps.some((map) => map.id === mapId)) {
    reasons.push(`맵 ID "${mapId}"는 현재 프로필에 없다.`)
  }

  if (!profile.npcs.some((npc) => npc.id === npcId)) {
    reasons.push(`NPC ID "${npcId}"는 현재 프로필에 없다.`)
  }

  if (!profile.items.some((item) => item.id === itemId)) {
    reasons.push(`아이템 ID "${itemId}"는 현재 프로필에 없다.`)
  }

  if (prompt.includes('새 자산') || prompt.includes('새 npc') || prompt.includes('새 아이템')) {
    reasons.push('사용자가 새 자산 생성을 요구했다.')
  }

  if (reasons.length > 0) {
    reasoning.push('기존 게임 프로필에 없는 항목이 있어 새 자산 필요 플래그를 켰다.')
  } else {
    reasoning.push('현재 게임 프로필 안의 자산만으로 이벤트를 구성했다.')
  }

  return {
    requires_new_asset: reasons.length > 0,
    requires_new_asset_reasons: reasons,
    reasoning
  }
}

const createSlugFromText = (text: string, fallback: string): string => {
  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9가-힣]+/gu, '_')
    .replace(/^_+|_+$/gu, '')

  return slug.length > 0 ? slug : fallback
}

const createNpcPlacement = (
  id: string,
  name: string,
  x: number,
  y: number
): {
  id: string
  name: string
  x: number
  y: number
} => ({
  id,
  name,
  x,
  y
})
