import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  getPlayerJobDisplayName,
  isPlayerJobPromotionAvailable,
  isPlayerAtMaxLevel,
  getPlayerJobPrimaryStatId,
  getPlayerJobPrimaryStatLabel
} from '../playerProfile'
import {
  createPlayerProfileLua,
  type PlayerProfileLua
} from './playerProfileLua'

// playerProfile의 순수 헬퍼를 Lua로 변환한 결과가 TS와 동등한지 실제 WASM으로 검증한다.
// (callJson 마샬링 + string|undefined → json_null → undefined 정규화 포함)
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const JOBS = ['초보자', '전사', '궁수', '마법사', '도적', '없는직업']
const LEVELS = [1, 9, 10, 11, 99, 100, 101]

describe('playerProfileLua (real wasm)', () => {
  let lua: PlayerProfileLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerProfile across jobs and levels', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerProfileLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // 직업별: primary stat id/label (string|undefined 경로 포함)
    for (const job of JOBS) {
      expect(lua.getPlayerJobPrimaryStatId(job)).toBe(
        getPlayerJobPrimaryStatId(job)
      )
      expect(lua.getPlayerJobPrimaryStatLabel(job)).toBe(
        getPlayerJobPrimaryStatLabel(job)
      )
    }

    // 레벨별: 전직 가능 / 최대 레벨 (경계값 9/10/11, 99/100/101 포함)
    for (const level of LEVELS) {
      expect(lua.isPlayerJobPromotionAvailable({ level })).toBe(
        isPlayerJobPromotionAvailable({ level })
      )
      expect(lua.isPlayerAtMaxLevel({ level })).toBe(
        isPlayerAtMaxLevel({ level })
      )
    }

    // 직업×레벨 조합: 표시 이름(전직 가능 접미사 분기)
    for (const job of JOBS) {
      for (const level of LEVELS) {
        expect(lua.getPlayerJobDisplayName({ job, level })).toBe(
          getPlayerJobDisplayName({ job, level })
        )
      }
    }
  })
})
