import { describe, expect, it } from 'vitest'

import {
  createInitialPlayerSkillSlots,
  clearPlayerSkillSlotAssignment,
  findPlayerSkillSlotIndexBySkillId,
  getPlayerSkillSlotIndexFromCode,
  setPlayerSkillSlotAssignment
} from './playerSkillSlots'

describe('playerSkillSlots', () => {
  it('creates empty skill slots and maps QWER to slot indexes', () => {
    expect(createInitialPlayerSkillSlots()).toEqual({
      slots: [undefined, undefined, undefined, undefined]
    })

    expect(getPlayerSkillSlotIndexFromCode('KeyQ')).toBe(0)
    expect(getPlayerSkillSlotIndexFromCode('KeyW')).toBe(1)
    expect(getPlayerSkillSlotIndexFromCode('KeyE')).toBe(2)
    expect(getPlayerSkillSlotIndexFromCode('KeyR')).toBe(3)
    expect(getPlayerSkillSlotIndexFromCode('KeyA')).toBeUndefined()
  })

  it('moves a skill id to the requested slot and clears duplicates', () => {
    const initialSlots = createInitialPlayerSkillSlots({
      initialSkillIds: ['smash']
    })

    expect(findPlayerSkillSlotIndexBySkillId(initialSlots, 'smash')).toBe(0)

    const nextSlots = setPlayerSkillSlotAssignment({
      skillSlots: initialSlots,
      skillSlotIndex: 1,
      skillId: 'smash'
    })

    expect(nextSlots).toEqual({
      slots: [undefined, { skillId: 'smash' }, undefined, undefined]
    })

    expect(
      clearPlayerSkillSlotAssignment({
        skillSlots: nextSlots,
        skillSlotIndex: 1
      })
    ).toEqual({
      slots: [undefined, undefined, undefined, undefined]
    })
  })
})
