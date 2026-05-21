import {
  PLAYER_MAX_LEVEL,
  type PlayerProfile,
  type PlayerSkillSlot,
  type PlayerStatId
} from './playerProfile'
import { PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT } from './playerStatEffects'

export const PLAYER_LEVEL_UP_STAT_POINTS = 3
export const PLAYER_LEVEL_UP_SKILL_POINTS = 1
export const PLAYER_LEVEL_UP_HP_BONUS = 4
export const PLAYER_LEVEL_UP_MP_BONUS = 2

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
  const nextMpMax = profile.mp.max + appliedLevels * PLAYER_LEVEL_UP_MP_BONUS

  return {
    ...profile,
    level: profile.level + appliedLevels,
    statPoints: profile.statPoints + appliedLevels * PLAYER_LEVEL_UP_STAT_POINTS,
    skillPoints: profile.skillPoints + appliedLevels * PLAYER_LEVEL_UP_SKILL_POINTS,
    hp: {
      current: nextHpMax,
      max: nextHpMax
    },
    mp: {
      current: nextMpMax,
      max: nextMpMax
    }
  }
}

export const spendPlayerStatPoint = (
  profile: PlayerProfile,
  statId: PlayerStatId
): PlayerProfile | undefined => {
  if (profile.statPoints <= 0) {
    return undefined
  }

  return {
    ...profile,
    statPoints: profile.statPoints - 1,
    stats: {
      ...profile.stats,
      [statId]: profile.stats[statId] + 1
    },
    mp:
      statId === 'intelligence'
        ? {
            current: Math.min(
              profile.mp.max + PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT,
              profile.mp.current + PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT
            ),
            max: profile.mp.max + PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT
          }
        : profile.mp
  }
}

export const spendPlayerSkillPoint = (
  profile: PlayerProfile,
  skillIndex: number
): PlayerProfile | undefined => {
  const skill = profile.skills[skillIndex]

  if (
    profile.skillPoints <= 0 ||
    !skill ||
    skill.level >= skill.maxLevel
  ) {
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
    skillPoints: profile.skillPoints - 1,
    skills: nextSkills
  }
}

export const getPlayerSkillLevelLabel = (skill: PlayerSkillSlot): string =>
  skill.level >= skill.maxLevel ? 'MAX' : `${skill.level} / ${skill.maxLevel}`
