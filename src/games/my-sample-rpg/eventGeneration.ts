export const HOLIDAY_DIALOGUE_EVENT_TYPE = 'holiday_dialogue'
export const TALK_EVENT_TRIGGER_TYPE = 'talk'
export const SCENE_ENTER_EVENT_TRIGGER_TYPE = 'scene_enter'
export const QUEST_START_EVENT_TRIGGER_TYPE = 'quest_start'
export const QUEST_COMPLETE_EVENT_TRIGGER_TYPE = 'quest_complete'
export const HOLIDAY_DIALOGUE_CONTROLLER_SCRIPT_ID = 'reply-with-message'

export type EventTriggerType =
  | typeof TALK_EVENT_TRIGGER_TYPE
  | typeof SCENE_ENTER_EVENT_TRIGGER_TYPE
  | typeof QUEST_START_EVENT_TRIGGER_TYPE
  | typeof QUEST_COMPLETE_EVENT_TRIGGER_TYPE

export type EventReward =
  | {
      type: 'none'
    }
  | {
      type: 'item'
      id: string
      count: number
    }
  | {
      type: 'gold'
      count: number
    }
  | {
      type: 'experience'
      count: number
    }

export type HolidayDialogueEventSpec = {
  event_name: string
  event_type: typeof HOLIDAY_DIALOGUE_EVENT_TYPE
  npc: {
    id: string
    display_name: string
    appearance_type: string
  }
  trigger: {
    type: EventTriggerType
    target_scene: string
  }
  dialogue: {
    opening_lines: string[]
    active_lines: string[]
    completion_lines: string[]
  }
  reward: EventReward
  duration: number
}

export type TiledNpcEventProperties = {
  displayText: string
  'controller.scriptId': string
  'controller.dialogueLines': string[]
  'controller.messageDurationSeconds': number
}

export type TiledNpcEventObject = {
  name: string
  type: string
  properties: TiledNpcEventProperties
}

export const createHolidayDialogueEventValidationErrors = (
  event: HolidayDialogueEventSpec
): string[] => {
  const errors: string[] = []

  if (event.event_name.trim().length === 0) {
    errors.push('event_name must not be empty')
  }

  if (!/^[a-z0-9_]+$/u.test(event.event_name)) {
    errors.push('event_name must use lowercase letters, numbers, and underscores only')
  }

  if (event.event_type !== HOLIDAY_DIALOGUE_EVENT_TYPE) {
    errors.push(`event_type must be "${HOLIDAY_DIALOGUE_EVENT_TYPE}"`)
  }

  if (event.npc.id.trim().length === 0) {
    errors.push('npc.id must not be empty')
  }

  if (event.npc.display_name.trim().length === 0) {
    errors.push('npc.display_name must not be empty')
  }

  if (event.npc.appearance_type.trim().length === 0) {
    errors.push('npc.appearance_type must not be empty')
  }

  if (
    event.trigger.type !== TALK_EVENT_TRIGGER_TYPE &&
    event.trigger.type !== SCENE_ENTER_EVENT_TRIGGER_TYPE &&
    event.trigger.type !== QUEST_START_EVENT_TRIGGER_TYPE &&
    event.trigger.type !== QUEST_COMPLETE_EVENT_TRIGGER_TYPE
  ) {
    errors.push('trigger.type must be a supported trigger type')
  }

  if (event.trigger.target_scene.trim().length === 0) {
    errors.push('trigger.target_scene must not be empty')
  }

  if (event.dialogue.opening_lines.length === 0) {
    errors.push('dialogue.opening_lines must contain at least one line')
  }

  if (!event.dialogue.opening_lines.every(isNonEmptyString)) {
    errors.push('dialogue.opening_lines must contain non-empty strings')
  }

  if (!event.dialogue.active_lines.every(isNonEmptyString)) {
    errors.push('dialogue.active_lines must contain non-empty strings')
  }

  if (!event.dialogue.completion_lines.every(isNonEmptyString)) {
    errors.push('dialogue.completion_lines must contain non-empty strings')
  }

  if (!Number.isFinite(event.duration) || event.duration <= 0) {
    errors.push('duration must be a positive finite number')
  }

  validateReward(event.reward, errors)

  return errors
}

export const isHolidayDialogueEventValid = (
  event: HolidayDialogueEventSpec
): boolean => createHolidayDialogueEventValidationErrors(event).length === 0

export const createTiledNpcEventObject = (
  event: HolidayDialogueEventSpec
): TiledNpcEventObject => ({
  name: event.npc.id,
  type: event.npc.appearance_type,
  properties: {
    displayText: event.npc.display_name,
    'controller.scriptId': HOLIDAY_DIALOGUE_CONTROLLER_SCRIPT_ID,
    'controller.dialogueLines': [...event.dialogue.opening_lines],
    'controller.messageDurationSeconds': event.duration
  }
})

const validateReward = (reward: EventReward, errors: string[]) => {
  switch (reward.type) {
    case 'none':
      return
    case 'item':
      if (reward.id.trim().length === 0) {
        errors.push('reward.id must not be empty when reward.type is "item"')
      }

      if (!Number.isFinite(reward.count) || reward.count <= 0) {
        errors.push('reward.count must be a positive finite number when reward.type is "item"')
      }
      return
    case 'gold':
    case 'experience':
      if (!Number.isFinite(reward.count) || reward.count <= 0) {
        errors.push(`reward.count must be a positive finite number when reward.type is "${reward.type}"`)
      }
      return
  }
}

const isNonEmptyString = (value: string): boolean => value.trim().length > 0
