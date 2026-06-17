// 몬스터 보상 규칙의 Lua 구현 래퍼 — monsterRewards.ts(TS)와 동일한 API를 제공하되
// 실제 계산은 monster-rewards.lua(Lua VM)에서 한다. 게임 부팅 시 1회 await로 만들고,
// 이후 동기 호출(원시값 마샬링이라 가볍다). "게임 로직을 Lua로" 전환의 첫 모듈.

import monsterRewardsLuaSource from '../assets/lua/monster-rewards.lua?raw'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput
} from './luaLogicHost'

export type MonsterRewardsLua = {
  getMonsterGoldDropAmount: (monsterLevel: number) => number
  getMonsterExperienceDropAmount: (monsterLevel: number) => number
  getMonsterSkillPointDropAmount: (monsterLevel: number) => number
  close: () => void
}

export const MONSTER_REWARDS_LUA_SOURCE = monsterRewardsLuaSource

export const createMonsterRewardsLua = async (
  input: CreateLuaLogicHostInput = {}
): Promise<MonsterRewardsLua> => {
  const host = await createLuaLogicHost(input)
  host.runModule(monsterRewardsLuaSource, '@monster-rewards.lua')

  return {
    getMonsterGoldDropAmount: (monsterLevel: number): number =>
      host.callNumber('monster_rewards_gold', monsterLevel),
    getMonsterExperienceDropAmount: (monsterLevel: number): number =>
      host.callNumber('monster_rewards_experience', monsterLevel),
    getMonsterSkillPointDropAmount: (monsterLevel: number): number =>
      host.callNumber('monster_rewards_skill_point', monsterLevel),
    close: (): void => {
      host.close()
    }
  }
}
