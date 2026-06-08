import {
  HOLIDAY_DIALOGUE_EVENT_TYPE,
  TALK_EVENT_TRIGGER_TYPE,
  type HolidayDialogueEventSpec
} from '../game/eventGeneration'
import type { GameStructureProfile } from './gameStructureProfile'
import type { GeneratedEventJson } from './eventJsonSchema'

export type GeneratedEventCodePreview = {
  code: string
  warnings: string[]
}

export const generateEventCodePreview = (
  eventJson: GeneratedEventJson,
  profile: GameStructureProfile
): GeneratedEventCodePreview => {
  const warnings = createEventCodeWarnings(eventJson, profile)
  const code = [
    "import { registerDynamicEventDefinition } from '../events/DynamicEventManager'",
    '',
    'registerDynamicEventDefinition({',
    `  event_json: ${JSON.stringify(eventJson, null, 2)
      .split('\n')
      .map((line) => `  ${line}`)
      .join('\n')},`,
    '  generated_code: "runtime-injected-event",',
    `  created_at: ${Date.now()}`,
    '})'
  ].join('\n')

  return {
    code,
    warnings
  }
}

export const createHolidayDialogueEventSpecFromGeneratedEventJson = (
  eventJson: GeneratedEventJson
): HolidayDialogueEventSpec | undefined => {
  const dialogueLines = eventJson.dialogue
    .map((line) => line.text.trim())
    .filter((line) => line.length > 0)

  if (dialogueLines.length === 0) {
    return undefined
  }

  return {
    event_name: eventJson.event_name,
    event_type: HOLIDAY_DIALOGUE_EVENT_TYPE,
    npc: {
      id: eventJson.npc.id,
      display_name: eventJson.npc.name,
      appearance_type: 'character_villager_brown_tunic'
    },
    trigger: {
      type: TALK_EVENT_TRIGGER_TYPE,
      target_scene: eventJson.location.map_id
    },
    dialogue: {
      opening_lines: dialogueLines,
      active_lines: [],
      completion_lines: []
    },
    reward: {
      type: 'item',
      id: eventJson.reward.item_id,
      count: eventJson.reward.amount
    },
    duration: 7
  }
}

const createEventCodeWarnings = (
  eventJson: GeneratedEventJson,
  profile: GameStructureProfile
): string[] => {
  const warnings: string[] = []

  if (!profile.maps.some((map) => map.id === eventJson.location.map_id)) {
    warnings.push(`알 수 없는 맵 ID: ${eventJson.location.map_id}`)
  }

  if (!profile.npcs.some((npc) => npc.id === eventJson.npc.id)) {
    warnings.push(`알 수 없는 NPC ID: ${eventJson.npc.id}`)
  }

  if (!profile.items.some((item) => item.id === eventJson.reward.item_id)) {
    warnings.push(`알 수 없는 아이템 ID: ${eventJson.reward.item_id}`)
  }

  if (eventJson.requires_new_asset === true) {
    warnings.push('새 자산 필요 플래그가 설정되어 있으므로 자동 적용 전에 검토가 필요하다.')
  }

  return warnings
}
