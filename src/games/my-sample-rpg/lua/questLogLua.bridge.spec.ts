import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import {
  createInitialQuestLog,
  startQuest,
  recordMonsterDefeatQuestProgress,
  recordItemUseQuestProgress,
  recordShopOpenQuestProgress,
  recordSceneEnterQuestProgress,
  recordTalkQuestProgress,
  completeQuest,
  abandonQuest,
  setQuestTrackerVisible,
  hideVisibleQuestTrackers,
  getQuestProgress,
  getQuestDefinition,
  getVisibleQuestTrackers,
  getQuestNpcBadgeKind,
  getQuestNpcBadgeKindForNpc,
  getNextQuestInteractionForNpc,
  formatQuestText,
  formatQuestTextLines,
  FIRST_SLIME_HUNT_QUEST_ID,
  POTION_SURVIVAL_BASICS_QUEST_ID,
  PIG_TROUBLE_QUEST_ID,
  BLACKSMITH_PREPARATION_QUEST_ID,
  CAVE_ENTRANCE_INVESTIGATION_QUEST_ID,
  WIZARD_NPC_ID,
  POTION_MERCHANT_NPC_ID,
  QUEST_DEFINITIONS
} from '../questLog'
import { createQuestLogLua, type QuestLogLua } from './questLogLua'

// 실제 Lua 5.3.6 WASM VM + QUEST_DEFINITIONS 주입으로 Lua 구현이 TS와 동등한지 검증.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createLua = async (): Promise<QuestLogLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createQuestLogLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

describe('questLogLua (real wasm bridge)', () => {
  let lua: QuestLogLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('createInitialQuestLog: all quests not-started, objectives at 0', async () => {
    lua = await createLua()
    const tsResult = createInitialQuestLog()
    const luaResult = lua.createInitialQuestLog()

    expect(luaResult).toEqual(tsResult)

    // spot-check: all quests exist and are not-started
    for (const def of QUEST_DEFINITIONS) {
      expect(luaResult.progressByQuestId[def.id].status).toBe('not-started')
      expect(luaResult.progressByQuestId[def.id].trackerVisible).toBe(false)
      for (const obj of def.objectives) {
        expect(luaResult.progressByQuestId[def.id].objectives[obj.id]).toBe(0)
      }
    }
  })

  it('getQuestProgress and getQuestDefinition match TS', async () => {
    lua = await createLua()
    const log = createInitialQuestLog()

    for (const def of QUEST_DEFINITIONS) {
      expect(lua.getQuestProgress(log, def.id)).toEqual(
        getQuestProgress(log, def.id)
      )
      expect(lua.getQuestDefinition(def.id)).toEqual(getQuestDefinition(def.id))
    }
  })

  it('startQuest and recordMonsterDefeatQuestProgress: q001 slime quest flow', async () => {
    lua = await createLua()
    let log = createInitialQuestLog()

    // startQuest: q001 is unlocked (no prerequisites)
    const tsAfterStart = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    const luaAfterStart = lua.startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    expect(luaAfterStart).toEqual(tsAfterStart)
    expect(luaAfterStart.progressByQuestId[FIRST_SLIME_HUNT_QUEST_ID].status).toBe('active')
    expect(luaAfterStart.progressByQuestId[FIRST_SLIME_HUNT_QUEST_ID].trackerVisible).toBe(true)

    // startQuest is idempotent when already active
    expect(lua.startQuest(luaAfterStart, FIRST_SLIME_HUNT_QUEST_ID)).toEqual(
      startQuest(tsAfterStart, FIRST_SLIME_HUNT_QUEST_ID)
    )

    // record monster defeats — 3 required
    log = tsAfterStart
    let luaLog = luaAfterStart
    for (let i = 1; i <= 3; i++) {
      const target = { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
      const tsNext = recordMonsterDefeatQuestProgress(log, target)
      const luaNext = lua.recordMonsterDefeatQuestProgress(luaLog, target)
      expect(luaNext).toEqual(tsNext)
      log = tsNext
      luaLog = luaNext
    }

    // after 3 kills → ready-to-turn-in
    expect(log.progressByQuestId[FIRST_SLIME_HUNT_QUEST_ID].status).toBe(
      'ready-to-turn-in'
    )
    expect(luaLog.progressByQuestId[FIRST_SLIME_HUNT_QUEST_ID].status).toBe(
      'ready-to-turn-in'
    )

    // extra defeat doesn't push count past required
    const target = { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
    expect(lua.recordMonsterDefeatQuestProgress(luaLog, target)).toEqual(
      recordMonsterDefeatQuestProgress(log, target)
    )
  })

  it('recordItemUseQuestProgress matches TS', async () => {
    lua = await createLua()
    // unlock q002: complete q001 first
    let log = createInitialQuestLog()
    log = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) {
      log = recordMonsterDefeatQuestProgress(log, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    const { nextQuestLog: logAfterQ1 } = completeQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    log = logAfterQ1
    log = startQuest(log, POTION_SURVIVAL_BASICS_QUEST_ID)

    let luaLog = lua.createInitialQuestLog()
    luaLog = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) {
      luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    const luaCompleteResult = lua.completeQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    luaLog = luaCompleteResult.nextQuestLog
    luaLog = lua.startQuest(luaLog, POTION_SURVIVAL_BASICS_QUEST_ID)

    // record item use
    const tsNext = recordItemUseQuestProgress(log, 'health-potion')
    const luaNext = lua.recordItemUseQuestProgress(luaLog, 'health-potion')
    expect(luaNext).toEqual(tsNext)
    expect(luaNext.progressByQuestId[POTION_SURVIVAL_BASICS_QUEST_ID].status).toBe(
      'ready-to-turn-in'
    )
  })

  it('recordShopOpenQuestProgress matches TS', async () => {
    lua = await createLua()
    // q004 requires q003 which requires q001+q002
    // Build state: q001, q002, q003 completed, q004 started
    let log = createInitialQuestLog()

    const slimeTarget = { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
    const pigTarget = { sceneId: 'hunting-ground', appearanceType: 'monster_pig' }

    log = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) log = recordMonsterDefeatQuestProgress(log, slimeTarget)
    log = completeQuest(log, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog
    log = startQuest(log, POTION_SURVIVAL_BASICS_QUEST_ID)
    log = recordItemUseQuestProgress(log, 'health-potion')
    log = completeQuest(log, POTION_SURVIVAL_BASICS_QUEST_ID).nextQuestLog
    log = startQuest(log, PIG_TROUBLE_QUEST_ID)
    for (let i = 0; i < 2; i++) log = recordMonsterDefeatQuestProgress(log, pigTarget)
    log = completeQuest(log, PIG_TROUBLE_QUEST_ID).nextQuestLog
    log = startQuest(log, BLACKSMITH_PREPARATION_QUEST_ID)

    let luaLog = lua.createInitialQuestLog()
    luaLog = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, slimeTarget)
    luaLog = lua.completeQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, POTION_SURVIVAL_BASICS_QUEST_ID)
    luaLog = lua.recordItemUseQuestProgress(luaLog, 'health-potion')
    luaLog = lua.completeQuest(luaLog, POTION_SURVIVAL_BASICS_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, PIG_TROUBLE_QUEST_ID)
    for (let i = 0; i < 2; i++) luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, pigTarget)
    luaLog = lua.completeQuest(luaLog, PIG_TROUBLE_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, BLACKSMITH_PREPARATION_QUEST_ID)

    const tsNext = recordShopOpenQuestProgress(log, 'blacksmith')
    const luaNext = lua.recordShopOpenQuestProgress(luaLog, 'blacksmith')
    expect(luaNext).toEqual(tsNext)
    expect(luaNext.progressByQuestId[BLACKSMITH_PREPARATION_QUEST_ID].status).toBe(
      'ready-to-turn-in'
    )
  })

  it('recordSceneEnterQuestProgress matches TS', async () => {
    lua = await createLua()
    // q005 requires q004 completed — build up to q005 started
    const slimeTarget = { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
    const pigTarget = { sceneId: 'hunting-ground', appearanceType: 'monster_pig' }

    let log = createInitialQuestLog()
    log = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) log = recordMonsterDefeatQuestProgress(log, slimeTarget)
    log = completeQuest(log, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog
    log = startQuest(log, POTION_SURVIVAL_BASICS_QUEST_ID)
    log = recordItemUseQuestProgress(log, 'health-potion')
    log = completeQuest(log, POTION_SURVIVAL_BASICS_QUEST_ID).nextQuestLog
    log = startQuest(log, PIG_TROUBLE_QUEST_ID)
    for (let i = 0; i < 2; i++) log = recordMonsterDefeatQuestProgress(log, pigTarget)
    log = completeQuest(log, PIG_TROUBLE_QUEST_ID).nextQuestLog
    log = startQuest(log, BLACKSMITH_PREPARATION_QUEST_ID)
    log = recordShopOpenQuestProgress(log, 'blacksmith')
    log = completeQuest(log, BLACKSMITH_PREPARATION_QUEST_ID).nextQuestLog
    log = startQuest(log, CAVE_ENTRANCE_INVESTIGATION_QUEST_ID)

    let luaLog = lua.createInitialQuestLog()
    luaLog = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, slimeTarget)
    luaLog = lua.completeQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, POTION_SURVIVAL_BASICS_QUEST_ID)
    luaLog = lua.recordItemUseQuestProgress(luaLog, 'health-potion')
    luaLog = lua.completeQuest(luaLog, POTION_SURVIVAL_BASICS_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, PIG_TROUBLE_QUEST_ID)
    for (let i = 0; i < 2; i++) luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, pigTarget)
    luaLog = lua.completeQuest(luaLog, PIG_TROUBLE_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, BLACKSMITH_PREPARATION_QUEST_ID)
    luaLog = lua.recordShopOpenQuestProgress(luaLog, 'blacksmith')
    luaLog = lua.completeQuest(luaLog, BLACKSMITH_PREPARATION_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, CAVE_ENTRANCE_INVESTIGATION_QUEST_ID)

    const tsNext = recordSceneEnterQuestProgress(log, 'cave')
    const luaNext = lua.recordSceneEnterQuestProgress(luaLog, 'cave')
    expect(luaNext).toEqual(tsNext)
    expect(luaNext.progressByQuestId[CAVE_ENTRANCE_INVESTIGATION_QUEST_ID].status).toBe(
      'ready-to-turn-in'
    )
  })

  it('recordTalkQuestProgress matches TS', async () => {
    lua = await createLua()
    // q007 (FINAL_SUPPLIES) requires q006 (SLIME_BOSS_SHADOW).
    // Build a shortcut: manually set q001-q006 all completed, start q007.
    // We do it via the TS chain and replicate on Lua side.
    const slimeTarget = { sceneId: 'hunting-ground', appearanceType: 'monster_slime' }
    const pigTarget = { sceneId: 'hunting-ground', appearanceType: 'monster_pig' }
    const slimeBossTarget = { sceneId: 'cave', appearanceType: 'monster_slime' }

    let log = createInitialQuestLog()
    log = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) log = recordMonsterDefeatQuestProgress(log, slimeTarget)
    log = completeQuest(log, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog
    log = startQuest(log, POTION_SURVIVAL_BASICS_QUEST_ID)
    log = recordItemUseQuestProgress(log, 'health-potion')
    log = completeQuest(log, POTION_SURVIVAL_BASICS_QUEST_ID).nextQuestLog
    log = startQuest(log, PIG_TROUBLE_QUEST_ID)
    for (let i = 0; i < 2; i++) log = recordMonsterDefeatQuestProgress(log, pigTarget)
    log = completeQuest(log, PIG_TROUBLE_QUEST_ID).nextQuestLog
    log = startQuest(log, BLACKSMITH_PREPARATION_QUEST_ID)
    log = recordShopOpenQuestProgress(log, 'blacksmith')
    log = completeQuest(log, BLACKSMITH_PREPARATION_QUEST_ID).nextQuestLog
    log = startQuest(log, CAVE_ENTRANCE_INVESTIGATION_QUEST_ID)
    log = recordSceneEnterQuestProgress(log, 'cave')
    log = completeQuest(log, CAVE_ENTRANCE_INVESTIGATION_QUEST_ID).nextQuestLog
    log = startQuest(log, 'q006-slime-boss-shadow')
    log = recordMonsterDefeatQuestProgress(log, slimeBossTarget)
    log = completeQuest(log, 'q006-slime-boss-shadow').nextQuestLog
    log = startQuest(log, 'q007-final-supplies')

    let luaLog = lua.createInitialQuestLog()
    luaLog = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, slimeTarget)
    luaLog = lua.completeQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, POTION_SURVIVAL_BASICS_QUEST_ID)
    luaLog = lua.recordItemUseQuestProgress(luaLog, 'health-potion')
    luaLog = lua.completeQuest(luaLog, POTION_SURVIVAL_BASICS_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, PIG_TROUBLE_QUEST_ID)
    for (let i = 0; i < 2; i++) luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, pigTarget)
    luaLog = lua.completeQuest(luaLog, PIG_TROUBLE_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, BLACKSMITH_PREPARATION_QUEST_ID)
    luaLog = lua.recordShopOpenQuestProgress(luaLog, 'blacksmith')
    luaLog = lua.completeQuest(luaLog, BLACKSMITH_PREPARATION_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, CAVE_ENTRANCE_INVESTIGATION_QUEST_ID)
    luaLog = lua.recordSceneEnterQuestProgress(luaLog, 'cave')
    luaLog = lua.completeQuest(luaLog, CAVE_ENTRANCE_INVESTIGATION_QUEST_ID).nextQuestLog
    luaLog = lua.startQuest(luaLog, 'q006-slime-boss-shadow')
    luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, slimeBossTarget)
    luaLog = lua.completeQuest(luaLog, 'q006-slime-boss-shadow').nextQuestLog
    luaLog = lua.startQuest(luaLog, 'q007-final-supplies')

    const tsNext = recordTalkQuestProgress(log, POTION_MERCHANT_NPC_ID)
    const luaNext = lua.recordTalkQuestProgress(luaLog, POTION_MERCHANT_NPC_ID)
    expect(luaNext).toEqual(tsNext)
    expect(luaNext.progressByQuestId['q007-final-supplies'].status).toBe(
      'ready-to-turn-in'
    )
  })

  it('completeQuest: rewards returned correctly', async () => {
    lua = await createLua()
    let log = createInitialQuestLog()
    log = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) {
      log = recordMonsterDefeatQuestProgress(log, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    let luaLog = lua.createInitialQuestLog()
    luaLog = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) {
      luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }

    const tsResult = completeQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    const luaResult = lua.completeQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)

    expect(luaResult.didComplete).toBe(tsResult.didComplete)
    expect(luaResult.goldReward).toBe(tsResult.goldReward)
    expect(luaResult.experienceReward).toBe(tsResult.experienceReward)
    expect(luaResult.itemRewards).toEqual(tsResult.itemRewards)
    expect(luaResult.nextQuestLog).toEqual(tsResult.nextQuestLog)

    // not ready-to-turn-in → didComplete false
    const fresh = lua.createInitialQuestLog()
    const noComplete = lua.completeQuest(fresh, FIRST_SLIME_HUNT_QUEST_ID)
    expect(noComplete.didComplete).toBe(false)
    expect(noComplete.goldReward).toBe(0)
    expect(noComplete.experienceReward).toBe(0)
    expect(noComplete.itemRewards).toEqual([])
  })

  it('abandonQuest resets to initial progress', async () => {
    lua = await createLua()
    let log = createInitialQuestLog()
    log = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    log = recordMonsterDefeatQuestProgress(log, {
      sceneId: 'hunting-ground',
      appearanceType: 'monster_slime'
    })

    let luaLog = lua.createInitialQuestLog()
    luaLog = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, {
      sceneId: 'hunting-ground',
      appearanceType: 'monster_slime'
    })

    expect(lua.abandonQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)).toEqual(
      abandonQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    )
  })

  it('setQuestTrackerVisible and hideVisibleQuestTrackers match TS', async () => {
    lua = await createLua()
    let log = createInitialQuestLog()
    log = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)

    let luaLog = lua.createInitialQuestLog()
    luaLog = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)

    // hide tracker
    expect(
      lua.setQuestTrackerVisible(luaLog, FIRST_SLIME_HUNT_QUEST_ID, false)
    ).toEqual(setQuestTrackerVisible(log, FIRST_SLIME_HUNT_QUEST_ID, false))

    // show tracker (already visible → same object semantics)
    expect(
      lua.setQuestTrackerVisible(luaLog, FIRST_SLIME_HUNT_QUEST_ID, true)
    ).toEqual(setQuestTrackerVisible(log, FIRST_SLIME_HUNT_QUEST_ID, true))

    // hideVisibleQuestTrackers
    expect(lua.hideVisibleQuestTrackers(luaLog)).toEqual(
      hideVisibleQuestTrackers(log)
    )
  })

  it('getVisibleQuestTrackers matches TS', async () => {
    lua = await createLua()
    let log = createInitialQuestLog()
    log = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    log = recordMonsterDefeatQuestProgress(log, {
      sceneId: 'hunting-ground',
      appearanceType: 'monster_slime'
    })

    let luaLog = lua.createInitialQuestLog()
    luaLog = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    luaLog = lua.recordMonsterDefeatQuestProgress(luaLog, {
      sceneId: 'hunting-ground',
      appearanceType: 'monster_slime'
    })

    expect(lua.getVisibleQuestTrackers(luaLog)).toEqual(
      getVisibleQuestTrackers(log)
    )

    // ready-to-turn-in tracker text (turnInTrackerText)
    let log2 = log
    for (let i = 0; i < 2; i++) {
      log2 = recordMonsterDefeatQuestProgress(log2, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    let luaLog2 = luaLog
    for (let i = 0; i < 2; i++) {
      luaLog2 = lua.recordMonsterDefeatQuestProgress(luaLog2, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    expect(lua.getVisibleQuestTrackers(luaLog2)).toEqual(
      getVisibleQuestTrackers(log2)
    )

    // empty when nothing visible
    const fresh = lua.createInitialQuestLog()
    expect(lua.getVisibleQuestTrackers(fresh)).toEqual(
      getVisibleQuestTrackers(createInitialQuestLog())
    )
  })

  it('getQuestNpcBadgeKind matches TS', async () => {
    lua = await createLua()
    const log = createInitialQuestLog()
    const luaLog = lua.createInitialQuestLog()

    // q001 not-started and unlocked → 'new'
    expect(lua.getQuestNpcBadgeKind(luaLog, FIRST_SLIME_HUNT_QUEST_ID)).toBe(
      getQuestNpcBadgeKind(log, FIRST_SLIME_HUNT_QUEST_ID)
    )

    // q002 not-started but locked (prerequisite not done) → undefined
    expect(lua.getQuestNpcBadgeKind(luaLog, POTION_SURVIVAL_BASICS_QUEST_ID)).toBe(
      getQuestNpcBadgeKind(log, POTION_SURVIVAL_BASICS_QUEST_ID)
    )
  })

  it('getQuestNpcBadgeKindForNpc matches TS', async () => {
    lua = await createLua()
    const log = createInitialQuestLog()
    const luaLog = lua.createInitialQuestLog()

    // wizard: q001 unlocked → 'new'
    expect(lua.getQuestNpcBadgeKindForNpc(luaLog, WIZARD_NPC_ID)).toBe(
      getQuestNpcBadgeKindForNpc(log, WIZARD_NPC_ID)
    )
    // potion_merchant: q002 locked → undefined
    expect(lua.getQuestNpcBadgeKindForNpc(luaLog, POTION_MERCHANT_NPC_ID)).toBe(
      getQuestNpcBadgeKindForNpc(log, POTION_MERCHANT_NPC_ID)
    )

    // after q001 completed, q002 unlocked → potion_merchant shows 'new'
    let log2 = createInitialQuestLog()
    log2 = startQuest(log2, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) {
      log2 = recordMonsterDefeatQuestProgress(log2, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    log2 = completeQuest(log2, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog

    let luaLog2 = lua.createInitialQuestLog()
    luaLog2 = lua.startQuest(luaLog2, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) {
      luaLog2 = lua.recordMonsterDefeatQuestProgress(luaLog2, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    luaLog2 = lua.completeQuest(luaLog2, FIRST_SLIME_HUNT_QUEST_ID).nextQuestLog

    expect(lua.getQuestNpcBadgeKindForNpc(luaLog2, POTION_MERCHANT_NPC_ID)).toBe(
      getQuestNpcBadgeKindForNpc(log2, POTION_MERCHANT_NPC_ID)
    )

    // q001 ready-to-turn-in → wizard shows 'finish'
    let log3 = createInitialQuestLog()
    log3 = startQuest(log3, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) {
      log3 = recordMonsterDefeatQuestProgress(log3, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    let luaLog3 = lua.createInitialQuestLog()
    luaLog3 = lua.startQuest(luaLog3, FIRST_SLIME_HUNT_QUEST_ID)
    for (let i = 0; i < 3; i++) {
      luaLog3 = lua.recordMonsterDefeatQuestProgress(luaLog3, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }

    expect(lua.getQuestNpcBadgeKindForNpc(luaLog3, WIZARD_NPC_ID)).toBe(
      getQuestNpcBadgeKindForNpc(log3, WIZARD_NPC_ID)
    )
  })

  it('getNextQuestInteractionForNpc matches TS', async () => {
    lua = await createLua()
    const log = createInitialQuestLog()
    const luaLog = lua.createInitialQuestLog()

    // wizard: q001 unlocked → start interaction
    const tsInteraction = getNextQuestInteractionForNpc(log, WIZARD_NPC_ID)
    const luaInteraction = lua.getNextQuestInteractionForNpc(luaLog, WIZARD_NPC_ID)
    expect(luaInteraction).toEqual(tsInteraction)
    expect(luaInteraction?.action).toBe('start')

    // potion_merchant: nothing unlocked → undefined
    expect(lua.getNextQuestInteractionForNpc(luaLog, POTION_MERCHANT_NPC_ID)).toBe(
      getNextQuestInteractionForNpc(log, POTION_MERCHANT_NPC_ID)
    )

    // wizard with q001 active → active interaction
    let log2 = startQuest(log, FIRST_SLIME_HUNT_QUEST_ID)
    let luaLog2 = lua.startQuest(luaLog, FIRST_SLIME_HUNT_QUEST_ID)
    expect(lua.getNextQuestInteractionForNpc(luaLog2, WIZARD_NPC_ID)).toEqual(
      getNextQuestInteractionForNpc(log2, WIZARD_NPC_ID)
    )

    // wizard with q001 ready-to-turn-in → complete interaction
    for (let i = 0; i < 3; i++) {
      log2 = recordMonsterDefeatQuestProgress(log2, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
      luaLog2 = lua.recordMonsterDefeatQuestProgress(luaLog2, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    }
    expect(lua.getNextQuestInteractionForNpc(luaLog2, WIZARD_NPC_ID)).toEqual(
      getNextQuestInteractionForNpc(log2, WIZARD_NPC_ID)
    )
  })

  it('formatQuestText and formatQuestTextLines replace {playerName}', async () => {
    lua = await createLua()
    const ctx = { playerName: '홍길동' }

    const texts = [
      '안녕하세요, {playerName}!',
      '돌아왔구나, {playerName}.',
      '{playerName}은 영웅이다.',
      '변수 없는 텍스트',
      ''
    ]

    for (const text of texts) {
      expect(lua.formatQuestText(text, ctx)).toBe(formatQuestText(text, ctx))
    }

    expect(lua.formatQuestTextLines(texts, ctx)).toEqual(
      formatQuestTextLines(texts, ctx)
    )

    // quest definition completion dialogue with {playerName}
    const completionLines = [
      '돌아왔구나, {playerName}.',
      '동굴의 어두운 기운이 사라졌다. 네가 해낸 것이다.',
      '이제 티르코네일 마을은 당분간 안전할 것이다.'
    ]
    expect(lua.formatQuestTextLines(completionLines, ctx)).toEqual(
      formatQuestTextLines(completionLines, ctx)
    )
  })

  it('setQuestTrackerVisible blocked for not-started quest', async () => {
    lua = await createLua()
    const log = createInitialQuestLog()
    const luaLog = lua.createInitialQuestLog()

    // trying to show tracker on not-started quest → no change
    expect(
      lua.setQuestTrackerVisible(luaLog, FIRST_SLIME_HUNT_QUEST_ID, true)
    ).toEqual(setQuestTrackerVisible(log, FIRST_SLIME_HUNT_QUEST_ID, true))
  })
})
