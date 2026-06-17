import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import { createInitialPlayerProfile, type PlayerProfile } from '../playerProfile'
import {
  grantPlayerLevelUpRewards,
  grantPlayerSkillPoints,
  spendPlayerStatPoint,
  spendPlayerSkillPoint,
  getPlayerSkillPointCost,
  getPlayerSkillLevelLabel,
  getPlayerSkillUserLevel,
  getPlayerMaxManaBySkillUserLevel,
  getPlayerMaxManaForProfile
} from '../playerProgression'
import {
  createPlayerProgressionLua,
  type PlayerProgressionLua
} from './playerProgressionLua'

// playerProgression의 레벨업/스탯/스킬 보상 규칙(PlayerProfile in/out)을 callJson으로 변환한 결과가
// TS와 동등한지 실제 WASM으로 검증한다. wasm/mjs URL은 4단계 위(public/vendor/lua).
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

// 기본 프로필을 변형해 다양한 입력(레벨/스탯/스킬포인트/스킬레벨)을 만든다.
const withOverrides = (
  overrides: Partial<PlayerProfile>
): PlayerProfile => ({
  ...createInitialPlayerProfile(),
  ...overrides
})

const PROFILES: PlayerProfile[] = [
  createInitialPlayerProfile(),
  withOverrides({ level: 1, statPoints: 0 }),
  withOverrides({ level: 50, statPoints: 7 }),
  withOverrides({ level: 99 }),
  withOverrides({ level: 100 }),
  withOverrides({
    level: 30,
    statPoints: 3,
    availableSkillPoints: 5,
    totalSkillPointsEarned: 10,
    mp: { current: 5, max: 40 },
    stats: { strength: 8, agility: 6, intelligence: 7, luck: 4 }
  }),
  withOverrides({
    level: 12,
    statPoints: 1,
    availableSkillPoints: 0,
    totalSkillPointsEarned: 0,
    mp: { current: 12, max: 12 },
    stats: { strength: 5, agility: 4, intelligence: 3, luck: 2 }
  }),
  withOverrides({
    level: 75,
    statPoints: 2,
    availableSkillPoints: 9,
    totalSkillPointsEarned: 20,
    hp: { current: 100, max: 200 },
    mp: { current: 60, max: 60 },
    stats: { strength: 20, agility: 15, intelligence: 12, luck: 9 },
    skills: [
      { hotkey: '1', label: 'a', description: '', level: 0, maxLevel: 5 },
      { hotkey: '2', label: 'b', description: '', level: 1, maxLevel: 5 },
      { hotkey: '3', label: 'c', description: '', level: 2, maxLevel: 5 },
      { hotkey: '4', label: 'd', description: '', level: 5, maxLevel: 5 }
    ]
  })
]

const STAT_IDS = ['strength', 'agility', 'intelligence', 'luck'] as const

describe('playerProgressionLua (real wasm)', () => {
  let lua: PlayerProgressionLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS playerProgression across many profiles + edge cases', async () => {
    const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
      import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
      readFile(LUA_MODULE_WASM_URL)
    ])
    lua = await createPlayerProgressionLua({
      createLuaModuleFactory: async () => createLuaModule,
      createLuaModuleOptions: { wasmBinary }
    })

    // grantPlayerLevelUpRewards: 다양한 levels(기본/0/음수/소수/대량/캡 초과)
    for (const profile of PROFILES) {
      for (const levels of [undefined, 0, 1, 3, -2, 2.9, 60, 999]) {
        expect(
          lua.grantPlayerLevelUpRewards(profile, levels)
        ).toEqual(grantPlayerLevelUpRewards(profile, levels))
      }
    }

    // grantPlayerSkillPoints: 0/음수/소수/양수 — mp 동기화 포함
    for (const profile of PROFILES) {
      for (const gained of [0, -5, 1, 2.7, 3, 10]) {
        expect(
          lua.grantPlayerSkillPoints(profile, gained)
        ).toEqual(grantPlayerSkillPoints(profile, gained))
      }
    }

    // spendPlayerStatPoint: 모든 스탯 id — intelligence는 mp 동기화 경로, statPoints<=0은 undefined
    for (const profile of PROFILES) {
      for (const statId of STAT_IDS) {
        expect(
          lua.spendPlayerStatPoint(profile, statId)
        ).toEqual(spendPlayerStatPoint(profile, statId))
      }
    }

    // spendPlayerSkillPoint: 유효/무효 인덱스, 비용/포인트 부족 경계
    for (const profile of PROFILES) {
      for (const skillIndex of [-1, 0, 1, 2, 3, 4, 99]) {
        expect(
          lua.spendPlayerSkillPoint(profile, skillIndex)
        ).toEqual(spendPlayerSkillPoint(profile, skillIndex))
      }
    }

    // getPlayerSkillPointCost + getPlayerSkillLevelLabel: 레벨 0..6, maxLevel 1/5
    for (const maxLevel of [1, 5]) {
      for (let level = 0; level <= 6; level++) {
        const skill = {
          hotkey: '1',
          label: 'x',
          description: '',
          level,
          maxLevel
        }
        expect(lua.getPlayerSkillPointCost(skill)).toBe(
          getPlayerSkillPointCost(skill)
        )
        expect(lua.getPlayerSkillLevelLabel(skill)).toBe(
          getPlayerSkillLevelLabel(skill)
        )
      }
    }

    // getPlayerSkillUserLevel: 0/음수/소수/큰 값
    for (const total of [-3, 0, 0.4, 1, 4, 9.9, 25]) {
      expect(lua.getPlayerSkillUserLevel(total)).toBe(
        getPlayerSkillUserLevel(total)
      )
    }

    // getPlayerMaxManaBySkillUserLevel: 1..5 직접 맵 + >5 외삽 + 클램프(<1)
    for (const level of [-2, 0, 1, 2, 3, 4, 5, 6, 10, 20.5]) {
      expect(lua.getPlayerMaxManaBySkillUserLevel(level)).toBe(
        getPlayerMaxManaBySkillUserLevel(level)
      )
    }

    // getPlayerMaxManaForProfile: 지력 보너스 포함
    for (const profile of PROFILES) {
      expect(lua.getPlayerMaxManaForProfile(profile)).toBe(
        getPlayerMaxManaForProfile(profile)
      )
    }
  })
})
