import { describe, expect, it } from 'vitest'

import { createInitialPlayerProfile } from './playerProfile'
import { grantPlayerLevelUpRewards } from './playerProgression'
import {
  grantPlayerExperience,
  getPlayerExperienceToNextLevel
} from './playerExperience'

describe('getPlayerExperienceToNextLevel', () => {
  it('uses a level-based growth curve and stops at level 100', () => {
    expect(getPlayerExperienceToNextLevel(1)).toBe(140)
    expect(getPlayerExperienceToNextLevel(5)).toBe(260)
    expect(getPlayerExperienceToNextLevel(100)).toBe(0)
  })
})

describe('grantPlayerExperience', () => {
  it('stores experience within the current level until enough is earned', () => {
    const profile = createInitialPlayerProfile()

    expect(grantPlayerExperience(profile, 55)).toEqual({
      nextProfile: {
        ...profile,
        experience: {
          current: 55
        }
      },
      levelsGained: 0,
      gainedExperience: 55
    })
  })

  it('levels up and carries overflow experience forward', () => {
    const profile = createInitialPlayerProfile()

    expect(grantPlayerExperience(profile, 150)).toEqual({
      nextProfile: {
        ...profile,
        level: 2,
        statPoints: 3,
        availableSkillPoints: 0,
        totalSkillPointsEarned: 0,
        hp: {
          current: 28,
          max: 28
        },
        experience: {
          current: 10
        }
      },
      levelsGained: 1,
      gainedExperience: 150
    })
  })

  it('does not go beyond level 100', () => {
    const profile = {
      ...grantPlayerLevelUpRewards(createInitialPlayerProfile(), 98),
      experience: {
        current: 90
      }
    }

    expect(grantPlayerExperience(profile, 5000)).toEqual({
      nextProfile: {
        ...profile,
        level: 100,
        statPoints: profile.statPoints + 3,
        availableSkillPoints: profile.availableSkillPoints,
        totalSkillPointsEarned: profile.totalSkillPointsEarned,
        hp: {
          current: profile.hp.max + 4,
          max: profile.hp.max + 4
        },
        experience: {
          current: 0
        }
      },
      levelsGained: 1,
      gainedExperience: 5000
    })
  })
})
