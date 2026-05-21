import { describe, expect, it } from 'vitest'

import { createInitialPlayerProfile } from './playerProfile'
import {
  getPlayerSkillLevelLabel,
  grantPlayerLevelUpRewards,
  spendPlayerSkillPoint,
  spendPlayerStatPoint
} from './playerProgression'

describe('grantPlayerLevelUpRewards', () => {
  it('adds level-up rewards for the requested levels', () => {
    expect(grantPlayerLevelUpRewards(createInitialPlayerProfile())).toEqual({
      ...createInitialPlayerProfile(),
      level: 2,
      statPoints: 3,
      skillPoints: 1,
      hp: {
        current: 28,
        max: 28
      },
      mp: {
        current: 14,
        max: 14
      }
    })
  })

  it('does not advance beyond level 100', () => {
    const profile = {
      ...grantPlayerLevelUpRewards(createInitialPlayerProfile(), 99),
      experience: {
        current: 0
      }
    }

    expect(grantPlayerLevelUpRewards(profile, 5)).toEqual(profile)
  })
})

describe('spendPlayerStatPoint', () => {
  it('spends one point and increases the selected stat', () => {
    const profile = grantPlayerLevelUpRewards(createInitialPlayerProfile())

    expect(spendPlayerStatPoint(profile, 'strength')).toEqual({
      ...profile,
      statPoints: 2,
      stats: {
        strength: profile.stats.strength + 1,
        agility: profile.stats.agility,
        intelligence: profile.stats.intelligence,
        luck: profile.stats.luck
      }
    })
  })

  it('returns undefined when no stat points remain', () => {
    expect(
      spendPlayerStatPoint(createInitialPlayerProfile(), 'strength')
    ).toBeUndefined()
  })

  it('increases max mana when intelligence goes up', () => {
    const profile = grantPlayerLevelUpRewards(createInitialPlayerProfile())

    expect(spendPlayerStatPoint(profile, 'intelligence')).toEqual({
      ...profile,
      statPoints: 2,
      stats: {
        strength: profile.stats.strength,
        agility: profile.stats.agility,
        intelligence: profile.stats.intelligence + 1,
        luck: profile.stats.luck
      },
      mp: {
        current: profile.mp.current + 2,
        max: profile.mp.max + 2
      }
    })
  })
})

describe('spendPlayerSkillPoint', () => {
  it('spends one point and increases the selected skill level', () => {
    const profile = grantPlayerLevelUpRewards(createInitialPlayerProfile())

    expect(spendPlayerSkillPoint(profile, 0)).toEqual({
      ...profile,
      skillPoints: 0,
      skills: [
        {
          ...profile.skills[0],
          level: 2
        },
        profile.skills[1],
        profile.skills[2],
        profile.skills[3]
      ]
    })
  })

  it('returns undefined when the skill is already maxed', () => {
    const profile = grantPlayerLevelUpRewards(createInitialPlayerProfile())
    const maxedProfile = {
      ...profile,
      skills: profile.skills.map((skill, index) =>
        index === 0
          ? {
              ...skill,
              level: skill.maxLevel
            }
          : skill
      )
    }

    expect(spendPlayerSkillPoint(maxedProfile, 0)).toBeUndefined()
  })
})

describe('getPlayerSkillLevelLabel', () => {
  it('shows MAX when a skill reaches its limit', () => {
    expect(
      getPlayerSkillLevelLabel({
        hotkey: '1',
        label: '베기',
        description: '재빠른 근거리 공격',
        level: 5,
        maxLevel: 5
      })
    ).toBe('MAX')
  })
})
