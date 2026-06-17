import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  isPlayerSkillUnlockedInProfile,
  getPlayerSkillLevelById,
  getPlayerSkillDamageById,
  getPlayerSkillDamageBonusById,
  getPlayerSkillManaCostById,
  getPlayerProtectSkillDurationByLevel
} from '../playerSkills'
import { createInitialPlayerProfile } from '../playerProfile'
import {
  createPlayerSkillsLua,
  type PlayerSkillsLua
} from './playerSkillsLua'

// 실제 Lua 5.3.6 WASM VM을 node에서 로드해(디스크의 mjs/wasm 주입), Lua 구현이 TS 기준
// 구현과 동일한지 검증한다.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const SKILL_IDS = ['smash', 'protect', 'unknown', '', 'dash']

describe('playerSkillsLua (real wasm bridge)', () => {
  let lua: PlayerSkillsLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerSkills for all 6 functions across skill IDs and level variants', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerSkillsLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // ── 기본 프로필 (level=0, maxLevel=5) ──
    const baseProfile = createInitialPlayerProfile()

    for (const skillId of SKILL_IDS) {
      expect(lua.isPlayerSkillUnlockedInProfile(baseProfile, skillId)).toBe(
        isPlayerSkillUnlockedInProfile(baseProfile, skillId)
      )
      expect(lua.getPlayerSkillLevelById(baseProfile, skillId)).toBe(
        getPlayerSkillLevelById(baseProfile, skillId)
      )
      expect(lua.getPlayerSkillDamageById(baseProfile, skillId)).toBe(
        getPlayerSkillDamageById(baseProfile, skillId)
      )
      expect(lua.getPlayerSkillDamageBonusById(baseProfile, skillId)).toBe(
        getPlayerSkillDamageBonusById(baseProfile, skillId)
      )
      expect(lua.getPlayerSkillManaCostById(baseProfile, skillId)).toBe(
        getPlayerSkillManaCostById(baseProfile, skillId)
      )
    }

    // ── 레벨별 변형 프로필 ──
    // level 1 ~ 5 (정상 범위) + 0(잠금) + 6/10(maxLevel 초과 클램프)
    const LEVELS = [0, 1, 2, 3, 4, 5, 6, 10]
    const MAX_LEVEL = 5

    for (const lvl of LEVELS) {
      const profile = {
        ...baseProfile,
        skills: baseProfile.skills.map((s, i) =>
          i === 0 || i === 1 ? { ...s, level: lvl, maxLevel: MAX_LEVEL } : s
        )
      }

      for (const skillId of ['smash', 'protect', 'unknown']) {
        expect(lua.isPlayerSkillUnlockedInProfile(profile, skillId)).toBe(
          isPlayerSkillUnlockedInProfile(profile, skillId)
        )
        expect(lua.getPlayerSkillLevelById(profile, skillId)).toBe(
          getPlayerSkillLevelById(profile, skillId)
        )
        expect(lua.getPlayerSkillDamageById(profile, skillId)).toBe(
          getPlayerSkillDamageById(profile, skillId)
        )
        expect(lua.getPlayerSkillDamageBonusById(profile, skillId)).toBe(
          getPlayerSkillDamageBonusById(profile, skillId)
        )
        expect(lua.getPlayerSkillManaCostById(profile, skillId)).toBe(
          getPlayerSkillManaCostById(profile, skillId)
        )
      }
    }

    // ── getPlayerProtectSkillDurationByLevel ──
    // 레벨 1~5(정상), 0/0.5/-1(클램핑 → 1로 정규화), 6/8/10(beyond-5 선형 외삽)
    const DURATION_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 0, 0.5, -1, 3.7]

    for (const skillLevel of DURATION_LEVELS) {
      expect(lua.getPlayerProtectSkillDurationByLevel(skillLevel)).toBe(
        getPlayerProtectSkillDurationByLevel(skillLevel)
      )
    }

    // ── maxLevel 클램핑: profile.skills[i].maxLevel = 3 → getLevel returns min(level, 3) ──
    const clampProfile = {
      ...baseProfile,
      skills: baseProfile.skills.map((s, i) =>
        i === 0 ? { ...s, level: 5, maxLevel: 3 } : s
      )
    }
    expect(lua.getPlayerSkillLevelById(clampProfile, 'smash')).toBe(
      getPlayerSkillLevelById(clampProfile, 'smash')
    )
    expect(lua.getPlayerSkillDamageById(clampProfile, 'smash')).toBe(
      getPlayerSkillDamageById(clampProfile, 'smash')
    )
    expect(lua.getPlayerSkillManaCostById(clampProfile, 'smash')).toBe(
      getPlayerSkillManaCostById(clampProfile, 'smash')
    )
  })
})
