import { describe, expect, it } from 'vitest'

import {
  PLAYER_PROTECT_SKILL_ID,
  getPlayerSkillDisplayInfoById,
  getPlayerSkillDamageById,
  getPlayerSkillManaCostById,
  getPlayerProtectSkillDurationByLevel,
  isPlayerSkillUnlockedInProfile
} from './playerSkills'
import { createInitialPlayerProfile } from './playerProfile'

describe('playerSkills', () => {
  it('looks up display info for the smash and protect skills', () => {
    expect(getPlayerSkillDisplayInfoById('smash')).toMatchObject({
      label: '스매시',
      description: '직선으로 뻗는 검 잔상 스킬'
    })
    expect(getPlayerSkillDisplayInfoById('smash').iconUrl).toContain(
      'Slash_skill.png'
    )

    expect(getPlayerSkillDisplayInfoById(PLAYER_PROTECT_SKILL_ID)).toMatchObject(
      {
        label: '방어 자세',
        description: '레벨이 오를수록 더 오래 유지되는 방어 스킬'
      }
    )
    expect(
      getPlayerSkillDisplayInfoById(PLAYER_PROTECT_SKILL_ID).iconUrl
    ).toContain('Protect_skill.png')

    expect(getPlayerSkillDisplayInfoById('future-skill')).toEqual({
      label: 'future-skill',
      description: '',
      iconUrl: undefined
    })
  })

  it('treats the starting skills as locked at level 0 and unlocks them at level 1', () => {
    const profile = createInitialPlayerProfile()

    expect(isPlayerSkillUnlockedInProfile(profile, 'smash')).toBe(false)
    expect(isPlayerSkillUnlockedInProfile(profile, PLAYER_PROTECT_SKILL_ID)).toBe(
      false
    )

    expect(
      isPlayerSkillUnlockedInProfile(
        {
          ...profile,
          skills: profile.skills.map((skill, index) =>
            index < 2 ? { ...skill, level: 1 } : skill
          )
        },
        'smash'
      )
    ).toBe(true)
  })

  it('adds more smash damage as the skill level rises', () => {
    const profile = createInitialPlayerProfile()

    expect(getPlayerSkillDamageById(profile, 'smash')).toBe(10)
    expect(
      getPlayerSkillDamageById(
        {
          ...profile,
          skills: profile.skills.map((skill, index) =>
            index === 0
              ? {
                  ...skill,
                  level: 2
                }
              : skill
          )
        },
        'smash'
      )
    ).toBe(14)

    expect(
      getPlayerSkillDamageById(
        {
          ...profile,
          skills: profile.skills.map((skill, index) =>
            index === 0
              ? {
                  ...skill,
                  level: 3
                }
              : skill
          )
        },
        'smash'
      )
    ).toBe(18)
  })

  it('returns the slash mana cost by skill level', () => {
    const profile = createInitialPlayerProfile()

    expect(getPlayerSkillManaCostById(profile, 'smash')).toBe(4)
    expect(
      getPlayerSkillManaCostById(
        {
          ...profile,
          skills: profile.skills.map((skill, index) =>
            index === 0
              ? {
                  ...skill,
                  level: 3
                }
              : skill
          )
        },
        'smash'
      )
    ).toBe(6)

    expect(
      getPlayerSkillManaCostById(
        {
          ...profile,
          skills: profile.skills.map((skill, index) =>
            index === 0
              ? {
                  ...skill,
                  level: 5
                }
              : skill
          )
        },
        'smash'
      )
    ).toBe(8)
  })

  it('increases protect duration as the skill level rises', () => {
    expect(getPlayerProtectSkillDurationByLevel(1)).toBe(1800)
    expect(getPlayerProtectSkillDurationByLevel(2)).toBe(2200)
    expect(getPlayerProtectSkillDurationByLevel(3)).toBe(2600)
    expect(getPlayerProtectSkillDurationByLevel(4)).toBe(3000)
    expect(getPlayerProtectSkillDurationByLevel(5)).toBe(3400)
  })
})
