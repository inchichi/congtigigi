import { grantPlayerLevelUpRewards } from './playerProgression'
import {
  PLAYER_MAX_LEVEL,
  type PlayerProfile
} from './playerProfile'

export const PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP = 140
export const PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL = 30

export type GrantPlayerExperienceResult = {
  nextProfile: PlayerProfile
  levelsGained: number
  gainedExperience: number
}

export const getPlayerExperienceToNextLevel = (level: number): number => {
  const normalizedLevel = Math.max(1, Math.floor(level))

  if (normalizedLevel >= PLAYER_MAX_LEVEL) {
    return 0
  }

  return (
    PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP +
    (normalizedLevel - 1) * PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL
  )
}

export const grantPlayerExperience = (
  profile: PlayerProfile,
  experience: number
): GrantPlayerExperienceResult => {
  const gainedExperience = Math.max(0, Math.floor(experience))

  if (gainedExperience === 0) {
    if (profile.level >= PLAYER_MAX_LEVEL) {
      return {
        nextProfile: {
          ...profile,
          experience: {
            current: 0
          }
        },
        levelsGained: 0,
        gainedExperience: 0
      }
    }

    return {
      nextProfile: profile,
      levelsGained: 0,
      gainedExperience: 0
    }
  }

  if (profile.level >= PLAYER_MAX_LEVEL) {
    return {
      nextProfile: {
        ...profile,
        experience: {
          current: 0
        }
      },
      levelsGained: 0,
      gainedExperience
    }
  }

  let nextProfile = profile
  let remainingExperience = profile.experience.current + gainedExperience
  let levelsGained = 0

  while (nextProfile.level < PLAYER_MAX_LEVEL) {
    const requiredExperience = getPlayerExperienceToNextLevel(nextProfile.level)

    if (remainingExperience < requiredExperience) {
      break
    }

    remainingExperience -= requiredExperience
    nextProfile = grantPlayerLevelUpRewards(nextProfile)
    levelsGained += 1
  }

  if (nextProfile.level >= PLAYER_MAX_LEVEL) {
    remainingExperience = 0
  }

  return {
    nextProfile: {
      ...nextProfile,
      experience: {
        current: remainingExperience
      }
    },
    levelsGained,
    gainedExperience
  }
}
