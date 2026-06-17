// playerProgression의 레벨업/스탯/스킬 보상 규칙을 Lua로 실행하는 래퍼(PlayerProfile 객체 ↔ callJson).
// 원본의 export 함수 시그니처를 그대로 미러링한다. spend* 의 undefined 반환은 Lua의 json_null(→null)로
// 표현되며 TS undefined로 정규화한다. 상수 소유는 TS — 호출 시 인자로 Lua에 전달한다(공식만 Lua).

import playerProgressionSource from '../assets/lua/player-progression.lua?raw'

import {
  PLAYER_MAX_LEVEL,
  type PlayerProfile,
  type PlayerSkillSlot,
  type PlayerStatId
} from '../playerProfile'
// 원본 playerProgression.ts가 소유한 공개 상수(단일 출처: TS). Lua엔 인자로 전달한다.
import {
  PLAYER_LEVEL_UP_HP_BONUS,
  PLAYER_LEVEL_UP_STAT_POINTS
} from '../playerProgression'
import {
  PLAYER_BASE_INTELLIGENCE_STAT,
  PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT
} from '../playerStatEffects'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type PlayerProgressionLua = {
  grantPlayerLevelUpRewards: (
    profile: PlayerProfile,
    levels?: number
  ) => PlayerProfile
  grantPlayerSkillPoints: (
    profile: PlayerProfile,
    gainedSkillPoints: number
  ) => PlayerProfile
  spendPlayerStatPoint: (
    profile: PlayerProfile,
    statId: PlayerStatId
  ) => PlayerProfile | undefined
  spendPlayerSkillPoint: (
    profile: PlayerProfile,
    skillIndex: number
  ) => PlayerProfile | undefined
  getPlayerSkillPointCost: (skill: PlayerSkillSlot) => number
  getPlayerSkillLevelLabel: (skill: PlayerSkillSlot) => string
  getPlayerSkillUserLevel: (totalSkillPointsEarned: number) => number
  getPlayerMaxManaBySkillUserLevel: (skillUserLevel: number) => number
  getPlayerMaxManaForProfile: (
    profile: Pick<PlayerProfile, 'stats' | 'totalSkillPointsEarned'>
  ) => number
  close: () => void
}

export const createPlayerProgressionLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerProgressionLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerProgressionSource, '@player-progression.lua')

  return {
    grantPlayerLevelUpRewards: (
      profile: PlayerProfile,
      levels = 1
    ): PlayerProfile =>
      host.callJson<PlayerProfile>(
        'progression_grant_level_up_rewards',
        profile,
        levels,
        PLAYER_MAX_LEVEL,
        PLAYER_LEVEL_UP_STAT_POINTS,
        PLAYER_LEVEL_UP_HP_BONUS
      ),
    grantPlayerSkillPoints: (
      profile: PlayerProfile,
      gainedSkillPoints: number
    ): PlayerProfile =>
      host.callJson<PlayerProfile>(
        'progression_grant_skill_points',
        profile,
        gainedSkillPoints,
        PLAYER_BASE_INTELLIGENCE_STAT,
        PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT
      ),
    spendPlayerStatPoint: (
      profile: PlayerProfile,
      statId: PlayerStatId
    ): PlayerProfile | undefined => {
      const result = host.callJson<PlayerProfile | null>(
        'progression_spend_stat_point',
        profile,
        statId,
        PLAYER_BASE_INTELLIGENCE_STAT,
        PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT
      )

      return result === null ? undefined : result
    },
    spendPlayerSkillPoint: (
      profile: PlayerProfile,
      skillIndex: number
    ): PlayerProfile | undefined => {
      const result = host.callJson<PlayerProfile | null>(
        'progression_spend_skill_point',
        profile,
        skillIndex
      )

      return result === null ? undefined : result
    },
    getPlayerSkillPointCost: (skill: PlayerSkillSlot): number =>
      host.callJson<number>('progression_skill_point_cost', skill),
    getPlayerSkillLevelLabel: (skill: PlayerSkillSlot): string =>
      host.callJson<string>('progression_skill_level_label', skill),
    getPlayerSkillUserLevel: (totalSkillPointsEarned: number): number =>
      host.callJson<number>(
        'progression_skill_user_level',
        totalSkillPointsEarned
      ),
    getPlayerMaxManaBySkillUserLevel: (skillUserLevel: number): number =>
      host.callJson<number>(
        'progression_max_mana_by_skill_user_level',
        skillUserLevel
      ),
    getPlayerMaxManaForProfile: (
      profile: Pick<PlayerProfile, 'stats' | 'totalSkillPointsEarned'>
    ): number =>
      host.callJson<number>(
        'progression_max_mana_for_profile',
        profile,
        PLAYER_BASE_INTELLIGENCE_STAT,
        PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
