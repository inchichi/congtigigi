// 플레이어 경험치 규칙의 Lua 구현 래퍼 — playerExperience.ts(TS)의 순수 leaf 함수와 동일한
// API를 제공하되 실제 계산은 player-experience.lua(Lua VM)에서 한다. 게임 부팅 시 1회 await로
// 만들고, 이후 동기 호출(원시값 마샬링이라 가볍다). 상수는 TS가 보유해 인자로 넘긴다.

import playerExperienceLuaSource from '../assets/lua/player-experience.lua?raw'

import { PLAYER_MAX_LEVEL } from '../playerProfile'
import {
  PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP,
  PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL
} from '../playerExperience'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput
} from './luaLogicHost'

export type PlayerExperienceLua = {
  getPlayerExperienceToNextLevel: (level: number) => number
  close: () => void
}

export const PLAYER_EXPERIENCE_LUA_SOURCE = playerExperienceLuaSource

export const createPlayerExperienceLua = async (
  input: CreateLuaLogicHostInput = {}
): Promise<PlayerExperienceLua> => {
  const host = await createLuaLogicHost(input)
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
    close: (): void => {
      host.close()
    }
  }
}
