import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import { getSceneIntroMessage } from '../sceneIntro'
import { createSceneIntroLua, type SceneIntroLua } from './sceneIntroLua'

// 실제 Lua 5.3.6 WASM VM을 node에서 로드해(디스크의 mjs/wasm 주입), Lua 구현이 TS 기준
// 구현과 모든 입력에서 동일한지 검증한다. (다른 브리지 스펙과 같은 로더 패턴.)
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createSceneIntro = async (): Promise<SceneIntroLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createSceneIntroLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

// 알려진 sceneId 모두 + 알 수 없는/엣지 케이스(빈 문자열, 대소문자, 유사 키).
const CASES: ReadonlyArray<string> = [
  'town',
  'hunting-ground',
  'cave',
  'unknown',
  '',
  'Town',
  'TOWN',
  'town ',
  ' town',
  'hunting',
  'caves',
  'dungeon',
  'hunting-ground-2',
  '티르코네일 마을'
]

describe('sceneIntroLua (real wasm bridge)', () => {
  let sceneIntro: SceneIntroLua | undefined

  afterEach(() => {
    sceneIntro?.close()
    sceneIntro = undefined
  })

  it('matches the TS getSceneIntroMessage across every case', async () => {
    sceneIntro = await createSceneIntro()

    for (const sceneId of CASES) {
      expect(sceneIntro.getSceneIntroMessage(sceneId)).toBe(
        getSceneIntroMessage(sceneId)
      )
    }
  })
})
