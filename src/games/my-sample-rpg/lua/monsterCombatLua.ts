// 몬스터 전투 상태 규칙의 Lua 구현 래퍼 — monsterCombat.ts(TS)와 동일한 API를 제공하되
// 실제 계산은 monster-combat.lua(Lua VM)에서 한다. 게임 부팅 시 1회 await로 만들고,
// 이후 동기 호출(원시값 마샬링이라 가볍다). "게임 로직을 Lua로" 전환 모듈.

import monsterCombatLuaSource from '../assets/lua/monster-combat.lua?raw'

import type { MonsterCombatState } from '../monsterCombat'
import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput
} from './luaLogicHost'

type CreateMonsterCombatStateOptions = {
  hpMultiplier?: number
  damageMultiplier?: number
}

export type MonsterCombatLua = {
  createMonsterCombatState: (
    monsterLevel?: number,
    options?: CreateMonsterCombatStateOptions
  ) => MonsterCombatState
  applyMonsterDamage: (
    state: MonsterCombatState,
    damage: number
  ) => MonsterCombatState
  isMonsterDefeated: (state: MonsterCombatState) => boolean
  close: () => void
}

export const MONSTER_COMBAT_LUA_SOURCE = monsterCombatLuaSource

export const createMonsterCombatLua = async (
  input: CreateLuaLogicHostInput = {}
): Promise<MonsterCombatLua> => {
  const host = await createLuaLogicHost(input)
  host.runModule(monsterCombatLuaSource, '@monster-combat.lua')

  return {
    createMonsterCombatState: (
      monsterLevel?: number,
      options: CreateMonsterCombatStateOptions = {}
    ): MonsterCombatState => {
      const [maxHp, currentHp, contactDamage] = host.callNumbers(
        'monster_combat_create',
        [
          monsterLevel ?? 1,
          options.hpMultiplier ?? 1,
          options.damageMultiplier ?? 1
        ],
        3
      )

      return { maxHp, currentHp, contactDamage }
    },
    applyMonsterDamage: (
      state: MonsterCombatState,
      damage: number
    ): MonsterCombatState => {
      const next = host.callNumber(
        'monster_combat_apply_damage',
        state.currentHp,
        damage
      )

      if (next === state.currentHp) {
        return state
      }

      return { ...state, currentHp: next }
    },
    isMonsterDefeated: (state: MonsterCombatState): boolean =>
      host.callNumber('monster_combat_is_defeated', state.currentHp) === 1,
    close: (): void => {
      host.close()
    }
  }
}
