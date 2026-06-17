import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  getMonsterGoldDropAmount as tsGold,
  getMonsterExperienceDropAmount as tsExp,
  getMonsterSkillPointDropAmount as tsSkill
} from '../monsterRewards'
import {
  createMonsterCombatState as tsCreate,
  applyMonsterDamage as tsDamage,
  isMonsterDefeated as tsDefeated
} from '../monsterCombat'
import { getMonsterDisplayName as tsDisplayName } from '../monsterDisplayName'
import { getPlayerExperienceToNextLevel as tsToNext } from '../playerExperience'
import { rollMonsterEquipmentDrop as tsRoll } from '../monsterEquipmentDrops'
import { createInitialPlayerProfile } from '../playerProfile'
import { createInitialPlayerInventory as tsCreateInv } from '../playerInventory'
import {
  getPlayerPhysicalAttackPower as tsAtk,
  getPlayerEvadeChance as tsEvade
} from '../playerStatEffects'
import {
  getPlayerJobDisplayName as tsJobName,
  getPlayerJobPrimaryStatId as tsPrimaryStat
} from '../playerProfile'
import {
  grantPlayerLevelUpRewards as tsLevelUp,
  getPlayerMaxManaForProfile as tsMaxMana
} from '../playerProgression'

import {
  initLuaGameLogic,
  closeLuaGameLogic,
  isLuaGameLogicReady,
  getMonsterGoldDropAmount,
  getMonsterExperienceDropAmount,
  getMonsterSkillPointDropAmount,
  createMonsterCombatState,
  applyMonsterDamage,
  isMonsterDefeated,
  getMonsterDisplayName,
  getPlayerExperienceToNextLevel,
  rollMonsterEquipmentDrop,
  getPlayerPhysicalAttackPower,
  getPlayerEvadeChance,
  createInitialPlayerInventory,
  getPlayerJobDisplayName,
  getPlayerJobPrimaryStatId,
  grantPlayerLevelUpRewards,
  getPlayerMaxManaForProfile
} from './luaGameLogic'

// 단일 호스트 퍼사드(게임이 실제로 쓰는 경로)가 모든 함수에서 TS와 동등한지 실제 WASM으로 검증.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const initWithRealWasm = async (): Promise<void> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  await initLuaGameLogic({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

// 고정 시퀀스 난수(드롭 동등성용) — 매 호출 새로 만들어 양쪽이 동일하게 소비하게 한다.
const makeRng = (values: number[]): (() => number) => {
  let index = 0
  return () => values[Math.min(index++, values.length - 1)]
}

const LEVELS = [-3, 0, 1, 2, 3, 5, 10, 49, 50, 51, 100, 3.7]

describe('luaGameLogic facade (real wasm)', () => {
  afterEach(() => {
    closeLuaGameLogic()
  })

  it('matches the TS implementations across all wired functions', async () => {
    await initWithRealWasm()
    expect(isLuaGameLogicReady()).toBe(true)

    for (const level of LEVELS) {
      expect(getMonsterGoldDropAmount(level)).toBe(tsGold(level))
      expect(getMonsterExperienceDropAmount(level)).toBe(tsExp(level))
      expect(getMonsterSkillPointDropAmount(level)).toBe(tsSkill(level))
      expect(getPlayerExperienceToNextLevel(level)).toBe(tsToNext(level))

      const luaState = createMonsterCombatState(level, { hpMultiplier: 2, damageMultiplier: 3 })
      const tsState = tsCreate(level, { hpMultiplier: 2, damageMultiplier: 3 })
      expect(luaState).toEqual(tsState)

      for (const damage of [0, 1, 5, 9999, 3.7]) {
        expect(applyMonsterDamage(luaState, damage)).toEqual(tsDamage(tsState, damage))
      }
      expect(isMonsterDefeated(luaState)).toBe(tsDefeated(tsState))
    }

    for (const args of [
      { id: 'slime-1' },
      { id: 'pig-2', displayText: 'kkulkkul' },
      { id: 'boss' },
      { id: '-leading' },
      { id: 'gold-dragon-3' }
    ]) {
      expect(getMonsterDisplayName(args)).toBe(tsDisplayName(args))
    }

    for (const seq of [[0.95], [0.0, 0.0], [0.5, 0.999], [0.1, 0.5], [0.89, 1.0]]) {
      expect(rollMonsterEquipmentDrop(makeRng(seq))).toEqual(tsRoll(makeRng(seq)))
    }

    // 와이어링된 플레이어 시스템 모듈도 퍼사드 경로로 TS와 동등해야 한다.
    const profile = createInitialPlayerProfile()
    expect(getPlayerPhysicalAttackPower(profile)).toBe(tsAtk(profile))
    expect(getPlayerEvadeChance(profile)).toBe(tsEvade(profile))
    expect(getPlayerMaxManaForProfile(profile)).toBe(tsMaxMana(profile))
    expect(createInitialPlayerInventory()).toEqual(tsCreateInv())
    expect(createInitialPlayerInventory({ slotCount: 6 })).toEqual(
      tsCreateInv({ slotCount: 6 })
    )
    expect(grantPlayerLevelUpRewards(profile)).toEqual(tsLevelUp(profile))
    for (const job of ['초보자', '전사', '궁수', '마법사', '도적', '없음']) {
      expect(getPlayerJobDisplayName({ job, level: 12 })).toBe(
        tsJobName({ job, level: 12 })
      )
      expect(getPlayerJobPrimaryStatId(job)).toBe(tsPrimaryStat(job))
    }
  })

  it('falls back to TS before init', () => {
    expect(isLuaGameLogicReady()).toBe(false)
    expect(getMonsterGoldDropAmount(5)).toBe(tsGold(5))
    expect(getMonsterDisplayName({ id: 'slime-1' })).toBe(tsDisplayName({ id: 'slime-1' }))
  })
})
