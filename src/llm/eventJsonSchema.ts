import type { GameStructureProfile } from './gameStructureProfile'

export type GeneratedEventDialogueLine = {
  speaker: string
  text: string
}

export type GeneratedEventJson = {
  event_id: string
  event_name: string
  description: string
  trigger: {
    type: string
    target: string
    condition: string
  }
  location: {
    map_id: string
    x: number
    y: number
  }
  npc: {
    id: string
    name: string
    dialogue_id: string
  }
  dialogue: GeneratedEventDialogueLine[]
  reward: {
    item_id: string
    amount: number
  }
  duration: {
    start: string
    end: string
  }
  requires_new_asset?: boolean
  requires_new_asset_reasons?: string[]
}

export type GeneratedEventJsonValidationIssue = {
  path: string
  message: string
}

export const createGeneratedEventJsonValidationIssues = (
  eventJson: GeneratedEventJson,
  profile: GameStructureProfile
): GeneratedEventJsonValidationIssue[] => {
  const issues: GeneratedEventJsonValidationIssue[] = []

  if (!isSnakeCase(eventJson.event_id)) {
    issues.push({
      path: 'event_id',
      message: 'event_id는 snake_case여야 한다.'
    })
  }

  if (!isSnakeCase(eventJson.event_name)) {
    issues.push({
      path: 'event_name',
      message: 'event_name은 snake_case여야 한다.'
    })
  }

  if (eventJson.description.trim().length === 0) {
    issues.push({
      path: 'description',
      message: 'description은 비어 있을 수 없다.'
    })
  }

  if (eventJson.trigger.type.trim().length === 0) {
    issues.push({
      path: 'trigger.type',
      message: 'trigger.type는 비어 있을 수 없다.'
    })
  }

  if (eventJson.trigger.target.trim().length === 0) {
    issues.push({
      path: 'trigger.target',
      message: 'trigger.target는 비어 있을 수 없다.'
    })
  }

  if (eventJson.trigger.condition.trim().length === 0) {
    issues.push({
      path: 'trigger.condition',
      message: 'trigger.condition는 비어 있을 수 없다.'
    })
  }

  if (!profile.maps.some((map) => map.id === eventJson.location.map_id)) {
    pushOrRequireNewAssetIssue({
      issues,
      requiresNewAsset: eventJson.requires_new_asset === true,
      path: 'location.map_id',
      message: `존재하는 맵 ID가 아니다: ${eventJson.location.map_id}`
    })
  }

  if (!Number.isFinite(eventJson.location.x)) {
    issues.push({
      path: 'location.x',
      message: 'location.x는 유효한 숫자여야 한다.'
    })
  }

  if (!Number.isFinite(eventJson.location.y)) {
    issues.push({
      path: 'location.y',
      message: 'location.y는 유효한 숫자여야 한다.'
    })
  }

  if (!profile.npcs.some((npc) => npc.id === eventJson.npc.id)) {
    pushOrRequireNewAssetIssue({
      issues,
      requiresNewAsset: eventJson.requires_new_asset === true,
      path: 'npc.id',
      message: `존재하는 NPC ID가 아니다: ${eventJson.npc.id}`
    })
  }

  if (eventJson.npc.name.trim().length === 0) {
    issues.push({
      path: 'npc.name',
      message: 'npc.name은 비어 있을 수 없다.'
    })
  }

  if (eventJson.npc.dialogue_id.trim().length === 0) {
    issues.push({
      path: 'npc.dialogue_id',
      message: 'npc.dialogue_id는 비어 있을 수 없다.'
    })
  }

  if (eventJson.dialogue.length === 0) {
    issues.push({
      path: 'dialogue',
      message: 'dialogue는 최소 1개 이상의 줄이 필요하다.'
    })
  }

  for (const [index, line] of eventJson.dialogue.entries()) {
    if (line.speaker.trim().length === 0) {
      issues.push({
        path: `dialogue[${index}].speaker`,
        message: 'speaker는 비어 있을 수 없다.'
      })
    }

    if (line.text.trim().length === 0) {
      issues.push({
        path: `dialogue[${index}].text`,
        message: 'text는 비어 있을 수 없다.'
      })
    }
  }

  if (!profile.items.some((item) => item.id === eventJson.reward.item_id)) {
    pushOrRequireNewAssetIssue({
      issues,
      requiresNewAsset: eventJson.requires_new_asset === true,
      path: 'reward.item_id',
      message: `존재하는 아이템 ID가 아니다: ${eventJson.reward.item_id}`
    })
  }

  if (!Number.isFinite(eventJson.reward.amount) || eventJson.reward.amount <= 0) {
    issues.push({
      path: 'reward.amount',
      message: 'reward.amount는 1 이상의 숫자여야 한다.'
    })
  }

  if (eventJson.duration.start.trim().length === 0) {
    issues.push({
      path: 'duration.start',
      message: 'duration.start는 비어 있을 수 없다.'
    })
  }

  if (eventJson.duration.end.trim().length === 0) {
    issues.push({
      path: 'duration.end',
      message: 'duration.end는 비어 있을 수 없다.'
    })
  }

  if (eventJson.requires_new_asset === true) {
    const reasons = eventJson.requires_new_asset_reasons ?? []

    if (reasons.length === 0) {
      issues.push({
        path: 'requires_new_asset_reasons',
        message: 'requires_new_asset가 true이면 이유를 최소 1개 이상 적어야 한다.'
      })
    }
  }

  return dedupeValidationIssues(issues)
}

export const isGeneratedEventJsonValid = (
  eventJson: GeneratedEventJson,
  profile: GameStructureProfile
): boolean => createGeneratedEventJsonValidationIssues(eventJson, profile).length === 0

const isSnakeCase = (value: string): boolean => /^[a-z0-9]+(?:_[a-z0-9]+)*$/u.test(value)

const pushOrRequireNewAssetIssue = ({
  issues,
  requiresNewAsset,
  path,
  message
}: {
  issues: GeneratedEventJsonValidationIssue[]
  requiresNewAsset: boolean
  path: string
  message: string
}): void => {
  if (!requiresNewAsset) {
    issues.push({ path, message })
    return
  }

  issues.push({
    path,
    message: `${message} (새 자산 필요로 표시됨)`
  })
}

const dedupeValidationIssues = (
  issues: GeneratedEventJsonValidationIssue[]
): GeneratedEventJsonValidationIssue[] => {
  const seen = new Set<string>()

  return issues.filter((issue) => {
    const key = `${issue.path}:${issue.message}`

    if (seen.has(key)) {
      return false
    }

    seen.add(key)
    return true
  })
}
