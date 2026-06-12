export type PlayerSkillSlotAssignment = {
  skillId: string
}

export type PlayerSkillSlots = {
  slots: Array<PlayerSkillSlotAssignment | undefined>
}

export const PLAYER_SKILL_SLOT_COUNT = 4
export const PLAYER_SKILL_SLOT_HOTKEY_LABELS = ['Q', 'W', 'E', 'R'] as const
export const PLAYER_SKILL_SLOT_HOTKEY_CODES = [
  'KeyQ',
  'KeyW',
  'KeyE',
  'KeyR'
] as const
export const PLAYER_SKILL_DRAG_MIME_TYPE = 'application/x-player-skill-id'

type CreateInitialPlayerSkillSlotsInput = {
  slotCount?: number
  initialSkillIds?: readonly (string | undefined)[]
}

type SetPlayerSkillSlotAssignmentInput = {
  skillSlots: PlayerSkillSlots
  skillSlotIndex: number
  skillId: string
}

type ClearPlayerSkillSlotAssignmentInput = {
  skillSlots: PlayerSkillSlots
  skillSlotIndex: number
}

export const createInitialPlayerSkillSlots = ({
  slotCount = PLAYER_SKILL_SLOT_COUNT,
  initialSkillIds = []
}: CreateInitialPlayerSkillSlotsInput = {}): PlayerSkillSlots => ({
  slots: Array.from({ length: slotCount }, (_, slotIndex) => {
    const skillId = initialSkillIds[slotIndex]

    return skillId ? { skillId } : undefined
  })
})

export const setPlayerSkillSlotAssignment = ({
  skillSlots,
  skillSlotIndex,
  skillId
}: SetPlayerSkillSlotAssignmentInput): PlayerSkillSlots => {
  assertPlayerSkillSlotIndex(skillSlots, skillSlotIndex)

  if (skillId.length === 0) {
    throw new Error('Invalid skill id')
  }

  const slots = skillSlots.slots.map((slot, index) =>
    index !== skillSlotIndex && slot?.skillId === skillId ? undefined : slot
  )

  slots[skillSlotIndex] = {
    skillId
  }

  return {
    slots
  }
}

export const clearPlayerSkillSlotAssignment = ({
  skillSlots,
  skillSlotIndex
}: ClearPlayerSkillSlotAssignmentInput): PlayerSkillSlots => {
  assertPlayerSkillSlotIndex(skillSlots, skillSlotIndex)

  const slots = [...skillSlots.slots]

  slots[skillSlotIndex] = undefined

  return {
    slots
  }
}

export const findPlayerSkillSlotIndexBySkillId = (
  skillSlots: PlayerSkillSlots,
  skillId: string
): number | undefined => {
  const skillSlotIndex = skillSlots.slots.findIndex(
    (slot) => slot?.skillId === skillId
  )

  return skillSlotIndex < 0 ? undefined : skillSlotIndex
}

export const getPlayerSkillSlotAssignment = (
  skillSlots: PlayerSkillSlots,
  skillSlotIndex: number
): PlayerSkillSlotAssignment | undefined => skillSlots.slots[skillSlotIndex]

export const getPlayerSkillSlotIndexFromCode = (
  code: string
): number | undefined => {
  const skillSlotIndex = PLAYER_SKILL_SLOT_HOTKEY_CODES.findIndex(
    (hotkeyCode) => hotkeyCode === code
  )

  return skillSlotIndex < 0 ? undefined : skillSlotIndex
}

const assertPlayerSkillSlotIndex = (
  skillSlots: PlayerSkillSlots,
  skillSlotIndex: number
) => {
  if (skillSlotIndex < 0 || skillSlotIndex >= skillSlots.slots.length) {
    throw new Error(`Invalid skill slot index ${skillSlotIndex}`)
  }
}
