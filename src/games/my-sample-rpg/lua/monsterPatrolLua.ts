// monsterPatrol의 순수 초기화(createMonsterPatrolState)를 Lua로 실행하는 래퍼.
// character 객체 입력 → MonsterPatrolState 객체 반환이라 callJson 마샬링을 쓴다.
//
// 변환 대상은 createMonsterPatrolState뿐이다. stepMonsterPatrol은 가변 횟수의
// Math.random 추첨에 의존해 Lua에서 동등 검증이 불가능하므로 TS(원본)에 그대로 남긴다.

import monsterPatrolSource from '../assets/lua/monster-patrol.lua?raw'

import type { CharacterState } from '../characterState'
import type { MonsterPatrolState } from '../monsterPatrol'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type MonsterPatrolLua = {
  createMonsterPatrolState: (character: CharacterState) => MonsterPatrolState
  close: () => void
}

export const MONSTER_PATROL_LUA_SOURCE = monsterPatrolSource

export const createMonsterPatrolLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<MonsterPatrolLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(monsterPatrolSource, '@monster-patrol.lua')

  return {
    createMonsterPatrolState: (
      character: CharacterState
    ): MonsterPatrolState =>
      host.callJson<MonsterPatrolState>(
        'monster_patrol_create_state',
        character
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
