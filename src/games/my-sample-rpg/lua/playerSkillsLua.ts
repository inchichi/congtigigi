// playerSkills.ts 의 순수 로직 6개 함수를 Lua로 실행하는 래퍼.
// Vite 에셋 URL이 필요한 getPlayerSkillDisplayInfoById 는 제외한다.
// 모든 함수가 구조적 인자를 받으므로 callJson을 사용한다.
// getPlayerSkillLevelById 반환값 null → undefined 정규화.

import playerSkillsLuaSource from '../assets/lua/player-skills.lua?raw'

import type { PlayerProfile } from '../playerProfile'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

type ProfileArg = Pick<PlayerProfile, 'skills'>

export type PlayerSkillsLua = {
  isPlayerSkillUnlockedInProfile: (profile: ProfileArg, skillId: string) => boolean
  getPlayerSkillLevelById: (profile: ProfileArg, skillId: string) => number | undefined
  getPlayerSkillDamageById: (profile: ProfileArg, skillId: string) => number
  getPlayerSkillDamageBonusById: (profile: ProfileArg, skillId: string) => number
  getPlayerSkillManaCostById: (profile: ProfileArg, skillId: string) => number
  getPlayerProtectSkillDurationByLevel: (skillLevel: number) => number
  close: () => void
}

export const createPlayerSkillsLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerSkillsLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerSkillsLuaSource, '@player-skills.lua')

  return {
    isPlayerSkillUnlockedInProfile: (profile: ProfileArg, skillId: string): boolean =>
      host.callJson<boolean>('player_skills_is_unlocked', profile, skillId),

    getPlayerSkillLevelById: (profile: ProfileArg, skillId: string): number | undefined => {
      const result = host.callJson<number | null>('player_skills_get_level', profile, skillId)
      return result === null ? undefined : result
    },

    getPlayerSkillDamageById: (profile: ProfileArg, skillId: string): number =>
      host.callJson<number>('player_skills_damage', profile, skillId),

    getPlayerSkillDamageBonusById: (profile: ProfileArg, skillId: string): number =>
      host.callJson<number>('player_skills_damage_bonus', profile, skillId),

    getPlayerSkillManaCostById: (profile: ProfileArg, skillId: string): number =>
      host.callJson<number>('player_skills_mana_cost', profile, skillId),

    getPlayerProtectSkillDurationByLevel: (skillLevel: number): number =>
      host.callJson<number>('player_skills_protect_duration', skillLevel),

    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
