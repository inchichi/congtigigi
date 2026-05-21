import { describe, expect, it } from 'vitest'

import { createInitialPlayerProfile } from './playerProfile'
import {
  getPlayerMaxManaBySkillUserLevel,
  getPlayerSkillLevelLabel,
  getPlayerSkillPointCost,
  getPlayerSkillUserLevel,
  grantPlayerLevelUpRewards,
  grantPlayerSkillPoints,
  spendPlayerSkillPoint,
  spendPlayerStatPoint
} from './playerProgression'

describe('grantPlayerLevelUpRewards', () => {
  it('adds level-up rewards for the requested levels', () => {
    expect(grantPlayerLevelUpRewards(createInitialPlayerProfile())).toEqual({
      ...createInitialPlayerProfile(),
      level: 2,
      statPoints: 3,
      hp: {
        current: 28,
        max: 28
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
  it('spends the required points and increases the selected skill level', () => {
    const profile = grantPlayerSkillPoints(createInitialPlayerProfile(), 2)

    expect(spendPlayerSkillPoint(profile, 0)).toEqual({
      ...profile,
      availableSkillPoints: 0,
      skills: [
        {
          ...profile.skills[0],
          level: 1
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

describe('getPlayerSkillPointCost', () => {
  it('increases the required points as the skill levels up', () => {
    const profile = createInitialPlayerProfile()

    expect(getPlayerSkillPointCost(profile.skills[0])).toBe(2)
    expect(
      getPlayerSkillPointCost({
        ...profile.skills[0],
        level: 2
      })
    ).toBe(3)
    expect(
      getPlayerSkillPointCost({
        ...profile.skills[0],
        level: 3
      })
    ).toBe(4)
    expect(
      getPlayerSkillPointCost({
        ...profile.skills[0],
        level: 4
      })
    ).toBe(4)
    expect(
      getPlayerSkillPointCost({
        ...profile.skills[0],
        level: 5
      })
    ).toBe(0)
  })
})

describe('grantPlayerSkillPoints', () => {
  it('adds skill points from monster rewards and raises max mana by skill user level', () => {
    const profile = createInitialPlayerProfile()

    expect(grantPlayerSkillPoints(profile, 3)).toEqual({
      ...profile,
      availableSkillPoints: 3,
      totalSkillPointsEarned: 3,
      mp: {
        current: 24,
        max: 24
      }
    })
  })
})

describe('getPlayerSkillUserLevel', () => {
  it('maps total skill points to the user level without spending penalties', () => {
    expect(getPlayerSkillUserLevel(0)).toBe(1)
    expect(getPlayerSkillUserLevel(1)).toBe(2)
    expect(getPlayerSkillUserLevel(3)).toBe(4)
  })
})

describe('getPlayerMaxManaBySkillUserLevel', () => {
  it('follows the level-based mana table', () => {
    expect(getPlayerMaxManaBySkillUserLevel(1)).toBe(12)
    expect(getPlayerMaxManaBySkillUserLevel(2)).toBe(16)
    expect(getPlayerMaxManaBySkillUserLevel(3)).toBe(20)
    expect(getPlayerMaxManaBySkillUserLevel(4)).toBe(24)
    expect(getPlayerMaxManaBySkillUserLevel(5)).toBe(28)
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
