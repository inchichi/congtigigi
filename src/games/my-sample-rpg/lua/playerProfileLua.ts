// playerProfile의 순수 헬퍼를 Lua로 실행하는 래퍼(객체/문자열 입력 → callJson 마샬링).
// string|undefined 반환 함수(getPlayerJobPrimaryStatId/Label)는 Lua가 json_null을 돌려주고,
// 여기서 null→undefined로 정규화한다. createInitialPlayerProfile은 데이터 구성이라 TS에 남긴다.

import playerProfileSource from '../assets/lua/player-profile.lua?raw'

import type { PlayerProfile, PlayerStatId } from '../playerProfile'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type PlayerProfileLua = {
  getPlayerJobDisplayName: (
    profile: Pick<PlayerProfile, 'job' | 'level'>
  ) => string
  isPlayerJobPromotionAvailable: (
    profile: Pick<PlayerProfile, 'level'>
  ) => boolean
  isPlayerAtMaxLevel: (profile: Pick<PlayerProfile, 'level'>) => boolean
  getPlayerJobPrimaryStatId: (job: string) => PlayerStatId | undefined
  getPlayerJobPrimaryStatLabel: (job: string) => string | undefined
  close: () => void
}

export const createPlayerProfileLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerProfileLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerProfileSource, '@player-profile.lua')

  return {
    getPlayerJobDisplayName: (
      profile: Pick<PlayerProfile, 'job' | 'level'>
    ): string =>
      host.callJson<string>('player_profile_job_display_name', profile),
    isPlayerJobPromotionAvailable: (
      profile: Pick<PlayerProfile, 'level'>
    ): boolean =>
      host.callJson<boolean>('player_profile_job_promotion_available', profile),
    isPlayerAtMaxLevel: (profile: Pick<PlayerProfile, 'level'>): boolean =>
      host.callJson<boolean>('player_profile_at_max_level', profile),
    getPlayerJobPrimaryStatId: (job: string): PlayerStatId | undefined => {
      const result = host.callJson<PlayerStatId | null>(
        'player_profile_job_primary_stat_id',
        job
      )

      return result === null ? undefined : result
    },
    getPlayerJobPrimaryStatLabel: (job: string): string | undefined => {
      const result = host.callJson<string | null>(
        'player_profile_job_primary_stat_label',
        job
      )

      return result === null ? undefined : result
    },
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
