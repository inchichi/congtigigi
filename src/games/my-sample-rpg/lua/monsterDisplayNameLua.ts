// 몬스터 표시 이름 규칙의 Lua 구현 래퍼 — monsterDisplayName.ts(TS)와 동일한 API를 제공하되
// 실제 계산은 monster-display-name.lua(Lua VM)에서 한다. 게임 부팅 시 1회 await로 만들고,
// 이후 동기 호출(원시값 마샬링이라 가볍다). "게임 로직을 Lua로" 전환 모듈 중 하나.

import monsterDisplayNameLuaSource from '../assets/lua/monster-display-name.lua?raw'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput
} from './luaLogicHost'

export type MonsterDisplayNameLua = {
  getMonsterDisplayName: (args: { id: string; displayText?: string }) => string
  close: () => void
}

export const MONSTER_DISPLAY_NAME_LUA_SOURCE = monsterDisplayNameLuaSource

export const createMonsterDisplayNameLua = async (
  input: CreateLuaLogicHostInput = {}
): Promise<MonsterDisplayNameLua> => {
  const host = await createLuaLogicHost(input)
  host.runModule(monsterDisplayNameLuaSource, '@monster-display-name.lua')

  return {
    getMonsterDisplayName: ({
      id,
      displayText
    }: {
      id: string
      displayText?: string
    }): string =>
      host.callString('monster_display_name', id, displayText ?? ''),
    close: (): void => {
      host.close()
    }
  }
}
