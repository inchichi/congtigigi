import {
  PLAYER_MAX_LEVEL,
  type PlayerProfile,
  type PlayerSkillSlot,
  type PlayerStatId
} from './playerProfile'
import {
  PLAYER_BASE_INTELLIGENCE_STAT,
  PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT
} from './playerStatEffects'

export const PLAYER_LEVEL_UP_STAT_POINTS = 3
export const PLAYER_LEVEL_UP_HP_BONUS = 4
const PLAYER_SKILL_USER_LEVEL_BASE_MANA_BY_LEVEL: Record<number, number> = {
  1: 12,
  2: 16,
  3: 20,
  4: 24,
  5: 28
}

export const grantPlayerLevelUpRewards = (
  profile: PlayerProfile,
  levels = 1
): PlayerProfile => {
  const nextLevels = Math.max(0, Math.floor(levels))
  const appliedLevels = Math.min(
    nextLevels,
    Math.max(0, PLAYER_MAX_LEVEL - profile.level)
  )

  if (appliedLevels === 0) {
    return profile
  }

  const nextHpMax = profile.hp.max + appliedLevels * PLAYER_LEVEL_UP_HP_BONUS

  return {
    ...profile,
    level: profile.level + appliedLevels,
    statPoints: profile.statPoints + appliedLevels * PLAYER_LEVEL_UP_STAT_POINTS,
    hp: {
      current: nextHpMax,
      max: nextHpMax
    }
  }
}

export const grantPlayerSkillPoints = (
  profile: PlayerProfile,
  gainedSkillPoints: number
): PlayerProfile => {
  const normalizedSkillPoints = Math.max(0, Math.floor(gainedSkillPoints))

  if (normalizedSkillPoints === 0) {
    return profile
  }

  const nextProfile = {
    ...profile,
    availableSkillPoints:
      profile.availableSkillPoints + normalizedSkillPoints,
    totalSkillPointsEarned:
      profile.totalSkillPointsEarned + normalizedSkillPoints
  }

  return syncPlayerManaFromSkillProgress(nextProfile)
}

export const spendPlayerStatPoint = (
  profile: PlayerProfile,
  statId: PlayerStatId
): PlayerProfile | undefined => {
  if (profile.statPoints <= 0) {
    return undefined
  }

  const nextProfile = {
    ...profile,
    stats: {
      ...profile.stats,
      [statId]: profile.stats[statId] + 1
    }
  }

  return {
    ...nextProfile,
    statPoints: profile.statPoints - 1,
    mp:
      statId === 'intelligence'
        ? syncPlayerManaFromSkillProgress(nextProfile).mp
        : profile.mp
  }
}

export const spendPlayerSkillPoint = (
  profile: PlayerProfile,
  skillIndex: number
): PlayerProfile | undefined => {
  const skill = profile.skills[skillIndex]

  if (!skill) {
    return undefined
  }

  const skillPointCost = getPlayerSkillPointCost(skill)

  if (skillPointCost <= 0 || profile.availableSkillPoints < skillPointCost) {
    return undefined
  }

  const nextSkills = profile.skills.map((currentSkill, currentIndex) =>
    currentIndex === skillIndex
      ? {
          ...currentSkill,
          level: currentSkill.level + 1
        }
      : currentSkill
  )

  return {
    ...profile,
    availableSkillPoints: profile.availableSkillPoints - skillPointCost,
    skills: nextSkills
  }
}

export const getPlayerSkillPointCost = (
  skill: PlayerSkillSlot
): number => {
  if (skill.level >= skill.maxLevel) {
    return 0
  }

  if (skill.level <= 1) {
    return 2
  }

  if (skill.level === 2) {
    return 3
  }

  return 4
}

export const getPlayerSkillLevelLabel = (skill: PlayerSkillSlot): string =>
  skill.level >= skill.maxLevel ? 'MAX' : `${skill.level} / ${skill.maxLevel}`

export const getPlayerSkillUserLevel = (
  totalSkillPointsEarned: number
): number => Math.max(1, Math.floor(totalSkillPointsEarned) + 1)

export const getPlayerMaxManaBySkillUserLevel = (skillUserLevel: number): number => {
  const normalizedLevel = Math.max(1, Math.floor(skillUserLevel))

  return (
    PLAYER_SKILL_USER_LEVEL_BASE_MANA_BY_LEVEL[normalizedLevel] ??
    PLAYER_SKILL_USER_LEVEL_BASE_MANA_BY_LEVEL[5] +
      (normalizedLevel - 5) * 4
  )
}

export const getPlayerMaxManaForProfile = (
  profile: Pick<PlayerProfile, 'stats' | 'totalSkillPointsEarned'>
): number =>
  getPlayerMaxManaBySkillUserLevel(
    getPlayerSkillUserLevel(profile.totalSkillPointsEarned)
  ) +
  Math.max(0, profile.stats.intelligence - PLAYER_BASE_INTELLIGENCE_STAT) *
    PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT

const syncPlayerManaFromSkillProgress = (
  profile: PlayerProfile
): PlayerProfile => {
  const nextMpMax = getPlayerMaxManaForProfile(profile)

  if (nextMpMax === profile.mp.max) {
    return profile
  }

  const nextMpCurrent = Math.min(
    nextMpMax,
    profile.mp.current + Math.max(0, nextMpMax - profile.mp.max)
  )

  return {
    ...profile,
    mp: {
      current: nextMpCurrent,
      max: nextMpMax
    }
  }
}
