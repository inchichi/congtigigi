// 플레이어 경험치 규칙의 Lua 구현 래퍼 — playerExperience.ts(TS)와 동일한 API를 제공하되 실제
// 계산은 player-experience.lua(Lua VM)에서 한다. 게임 부팅 시 1회 await로 만들고, 이후 동기 호출.
// 상수는 TS가 보유해 인자로 넘긴다.
// getPlayerExperienceToNextLevel 은 순수 산술(원시값 마샬링, callNumber).
// grantPlayerExperience 는 PlayerProfile 객체 in/out(callJson)이며 레벨업 보상을
// player-progression.lua 의 progression_grant_level_up_rewards 전역에 위임한다. 따라서 자체 호스트를
// 만들 때(input.host 미지정)는 player-profile → player-stat-effects → player-progression →
// player-experience 순으로 .lua 를 로드한다. 공유 호스트(facade)에선 의존성이 이미 로드돼 있다고 가정.

import playerProfileSource from '../assets/lua/player-profile.lua?raw'
import playerStatEffectsSource from '../assets/lua/player-stat-effects.lua?raw'
import playerProgressionSource from '../assets/lua/player-progression.lua?raw'
import playerExperienceLuaSource from '../assets/lua/player-experience.lua?raw'

import { PLAYER_MAX_LEVEL, type PlayerProfile } from '../playerProfile'
import {
  PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP,
  PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL,
  type GrantPlayerExperienceResult
} from '../playerExperience'
import {
  PLAYER_LEVEL_UP_HP_BONUS,
  PLAYER_LEVEL_UP_STAT_POINTS
} from '../playerProgression'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type PlayerExperienceLua = {
  getPlayerExperienceToNextLevel: (level: number) => number
  grantPlayerExperience: (
    profile: PlayerProfile,
    experience: number
  ) => GrantPlayerExperienceResult
  close: () => void
}

export const PLAYER_EXPERIENCE_LUA_SOURCE = playerExperienceLuaSource

export const createPlayerExperienceLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerExperienceLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))

  if (!input.host) {
    host.runModule(playerProfileSource, '@player-profile.lua')
    host.runModule(playerStatEffectsSource, '@player-stat-effects.lua')
    host.runModule(playerProgressionSource, '@player-progression.lua')
  }
  host.runModule(playerExperienceLuaSource, '@player-experience.lua')

  return {
    getPlayerExperienceToNextLevel: (level: number): number =>
      host.callNumber(
        'player_experience_to_next_level',
        level,
        PLAYER_MAX_LEVEL,
        PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP,
        PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL
      ),
    grantPlayerExperience: (
      profile: PlayerProfile,
      experience: number
    ): GrantPlayerExperienceResult =>
      host.callJson<GrantPlayerExperienceResult>(
        'player_experience_grant',
        profile,
        experience,
        PLAYER_MAX_LEVEL,
        PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP,
        PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL,
        PLAYER_LEVEL_UP_STAT_POINTS,
        PLAYER_LEVEL_UP_HP_BONUS
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
