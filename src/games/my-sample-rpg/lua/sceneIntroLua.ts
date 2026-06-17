// 씬 인트로 메시지 규칙의 Lua 구현 래퍼 — sceneIntro.ts(TS)와 동일한 API를 제공하되
// 실제 계산은 scene-intro.lua(Lua VM)에서 한다. 순수 문자열 마샬링(callString)이라 가볍다.
// 공유 호스트(facade) 경로면 input.host를 재사용하고 이 모듈 소스만 로드, 아니면 독립 호스트를 만든다.

import sceneIntroSource from '../assets/lua/scene-intro.lua?raw'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type SceneIntroLua = {
  getSceneIntroMessage: (sceneId: string) => string
  close: () => void
}

export const SCENE_INTRO_LUA_SOURCE = sceneIntroSource

export const createSceneIntroLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<SceneIntroLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(sceneIntroSource, '@scene-intro.lua')

  return {
    getSceneIntroMessage: (sceneId: string): string =>
      host.callString('scene_intro_message', sceneId),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
