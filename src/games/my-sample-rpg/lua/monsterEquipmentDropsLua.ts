// 몬스터 장비 드롭 규칙의 Lua 구현 래퍼 — monsterEquipmentDrops.ts(TS)와 동일한 API를 제공하되
// 드롭 인덱스 계산은 monster-equipment-drops.lua(Lua VM)에서 한다. 게임 부팅 시 1회 await로 만들고,
// 이후 동기 호출(원시값 마샬링이라 가볍다). 정의 배열 + 확률 상수는 TS 데이터로 남는다.
//
// RNG 호출 순서 보존: TS는 random()을 확률 게이트에 1회 부르고, 통과한 경우에만 인덱스용으로
// random()을 다시 부른다. 그래서 확률 게이트는 이 래퍼(첫 random())가 담당하고, Lua에는 인덱스
// 수학만 둔다.

import monsterEquipmentDropsLuaSource from '../assets/lua/monster-equipment-drops.lua?raw'

import {
  MONSTER_EQUIPMENT_DROP_CHANCE,
  MONSTER_EQUIPMENT_DROP_DEFINITIONS,
  type MonsterEquipmentDropDefinition
} from '../monsterEquipmentDrops'
import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput
} from './luaLogicHost'

export type MonsterEquipmentDropsLua = {
  rollMonsterEquipmentDrop: (
    random?: () => number
  ) => MonsterEquipmentDropDefinition | undefined
  close: () => void
}

export const MONSTER_EQUIPMENT_DROPS_LUA_SOURCE = monsterEquipmentDropsLuaSource

export const createMonsterEquipmentDropsLua = async (
  input: CreateLuaLogicHostInput = {}
): Promise<MonsterEquipmentDropsLua> => {
  const host = await createLuaLogicHost(input)
  host.runModule(monsterEquipmentDropsLuaSource, '@monster-equipment-drops.lua')

  return {
    rollMonsterEquipmentDrop: (
      random: () => number = Math.random
    ): MonsterEquipmentDropDefinition | undefined => {
      const chanceRoll = random()

      if (chanceRoll >= MONSTER_EQUIPMENT_DROP_CHANCE) {
        return undefined
      }

      const index = host.callNumber(
        'monster_equipment_drop_index',
        random(),
        MONSTER_EQUIPMENT_DROP_DEFINITIONS.length
      )

      return MONSTER_EQUIPMENT_DROP_DEFINITIONS[index]
    },
    close: (): void => {
      host.close()
    }
  }
}
