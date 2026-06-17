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
  createInitialPlayerEquipment as tsCreateEquip,
  getPlayerEquipmentItemDefinitionById as tsEquipDefById
} from '../playerEquipment'
import {
  createInitialPotionInventory as tsCreatePotionInv,
  getPotionShopItemDefinitionById as tsPotionDefById,
  getPotionShopBuyPriceById as tsPotionBuy,
  getPotionShopSellPriceById as tsPotionSell,
  buyPotionShopItem as tsBuyPotion,
  sellPotionShopItem as tsSellPotion
} from '../potionShop'
import {
  createInitialPlayerQuickslots as tsCreateQuickslots,
  getPlayerQuickslotAssignment as tsGetQuickslotAssignment
} from '../playerQuickslots'
import { createInitialPlayerSkillSlots as tsCreateSkillSlots } from '../playerSkillSlots'
import {
  isPlayerSkillUnlockedInProfile as tsIsSkillUnlocked,
  getPlayerSkillManaCostById as tsSkillManaCost
} from '../playerSkills'
import { usePlayerInventoryConsumable as tsUseConsumable } from '../playerConsumables'
import { equipPlayerInventorySlot as tsEquipInventorySlot } from '../playerLoadout'
import { getBlacksmithShopBuyPriceById as tsBlacksmithBuy } from '../blacksmithShop'
import {
  createInitialQuestLog as tsCreateQuestLog,
  recordMonsterDefeatQuestProgress as tsRecordMonsterDefeat,
  startQuest as tsStartQuest,
  FIRST_SLIME_HUNT_QUEST_ID
} from '../questLog'
import { grantPlayerExperience as tsGrantExperience } from '../playerExperience'
import {
  createInitialPlayerControlBindings as tsCreateControlBindings,
  getPlayerControlBindingDisplayText as tsControlDisplayText
} from '../playerControls'
import {
  serializePlayerSaveState as tsSerializeSave,
  parseStoredPlayerSaveState as tsParseSave
} from '../playerSaveState'
import {
  getPlayerRollDirectionVector as tsRollDirection,
  getPlayerRollVisualState as tsRollVisual
} from '../playerRoll'
import {
  getPlayerSmashSkillSegmentPlacement as tsSmashPlacement,
  getPlayerSmashSkillDirectionVector as tsSmashDirection
} from '../playerSmashSkill'
import {
  isHolidayDialogueEventValid as tsEventValid,
  createTiledNpcEventObject as tsTiledNpc
} from '../eventGeneration'
import { createHolidayDialogueEventDraftFromText as tsEventDraft } from '../eventDrafting'
import {
  createInitialPlayerCharacter as tsCreatePlayerCharacter,
  getCharacterMoveDirectionFromKey as tsMoveDirFromKey
} from '../characterState'
import { createMonsterPatrolState as tsCreatePatrol } from '../monsterPatrol'
import { getSceneIntroMessage as tsSceneIntro } from '../sceneIntro'

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
  getPlayerMaxManaForProfile,
  createInitialPlayerEquipment,
  getPlayerEquipmentItemDefinitionById,
  createInitialPotionInventory,
  getPotionShopItemDefinitionById,
  getPotionShopBuyPriceById,
  getPotionShopSellPriceById,
  buyPotionShopItem,
  sellPotionShopItem,
  createInitialPlayerQuickslots,
  getPlayerQuickslotAssignment,
  createInitialPlayerSkillSlots,
  isPlayerSkillUnlockedInProfile,
  getPlayerSkillManaCostById,
  usePlayerInventoryConsumable,
  equipPlayerInventorySlot,
  getBlacksmithShopBuyPriceById,
  createInitialQuestLog,
  recordMonsterDefeatQuestProgress,
  grantPlayerExperience,
  createInitialPlayerControlBindings,
  getPlayerControlBindingDisplayText,
  serializePlayerSaveState,
  parseStoredPlayerSaveState,
  getPlayerRollDirectionVector,
  getPlayerRollVisualState,
  getPlayerSmashSkillSegmentPlacement,
  getPlayerSmashSkillDirectionVector,
  isHolidayDialogueEventValid,
  createTiledNpcEventObject,
  createHolidayDialogueEventDraftFromText,
  createInitialPlayerCharacter,
  getCharacterMoveDirectionFromKey,
  createMonsterPatrolState,
  getSceneIntroMessage
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

    // playerEquipment 위임도 TS와 동등해야 한다(set/clear 등 상세는 per-module 스펙이 검증).
    expect(createInitialPlayerEquipment()).toEqual(tsCreateEquip())
    expect(getPlayerEquipmentItemDefinitionById('iron-sword')).toEqual(
      tsEquipDefById('iron-sword')
    )
    expect(getPlayerEquipmentItemDefinitionById('does-not-exist')).toBeUndefined()

    // playerPotionShop 위임도 TS와 동등해야 한다(정의/가격 + 구매·판매 트랜잭션, 장비 매입가 폴백 포함).
    expect(createInitialPotionInventory()).toEqual(tsCreatePotionInv())
    expect(createInitialPotionInventory({ slotCount: 4 })).toEqual(
      tsCreatePotionInv({ slotCount: 4 })
    )
    for (const id of ['health-potion', 'mana-potion', 'iron-sword', 'does-not-exist']) {
      expect(getPotionShopItemDefinitionById(id)).toEqual(tsPotionDefById(id))
      expect(getPotionShopBuyPriceById(id)).toBe(tsPotionBuy(id))
      expect(getPotionShopSellPriceById(id)).toBe(tsPotionSell(id))
    }
    const buyMerchant = tsCreatePotionInv()
    const buyPlayer = tsCreateInv()
    const buyInput = {
      playerInventory: buyPlayer,
      merchantInventory: buyMerchant,
      merchantSlotIndex: 0,
      quantity: 2
    }
    expect(buyPotionShopItem(buyInput)).toEqual(tsBuyPotion(buyInput))
    const sellPlayer = tsCreatePotionInv()
    const sellMerchant = tsCreatePotionInv()
    const sellInput = {
      playerInventory: sellPlayer,
      merchantInventory: sellMerchant,
      playerSlotIndex: 0,
      quantity: 1
    }
    expect(sellPotionShopItem(sellInput)).toEqual(tsSellPotion(sellInput))

    // ── 신규 7개 모듈 스팟 체크 ──

    // playerQuickslots
    expect(createInitialPlayerQuickslots()).toEqual(tsCreateQuickslots())
    const qs = tsCreateQuickslots()
    expect(getPlayerQuickslotAssignment(qs, 0)).toBe(
      tsGetQuickslotAssignment(qs, 0)
    )

    // playerSkillSlots
    expect(createInitialPlayerSkillSlots()).toEqual(tsCreateSkillSlots())

    // playerSkills
    expect(isPlayerSkillUnlockedInProfile(profile, 'smash')).toBe(
      tsIsSkillUnlocked(profile, 'smash')
    )
    expect(getPlayerSkillManaCostById(profile, 'smash')).toBe(
      tsSkillManaCost(profile, 'smash')
    )

    // playerConsumables — 체력 회복 포션 사용 (슬롯에 아이템이 없으면 undefined)
    const consumableInv = tsCreateInv()
    const consumableInput = { profile, inventory: consumableInv, slotIndex: 0 }
    expect(usePlayerInventoryConsumable(consumableInput)).toEqual(
      tsUseConsumable(consumableInput)
    )

    // playerLoadout — iron-sword 장착 시도 (인벤토리가 비어있으면 undefined)
    const loadoutEquip = tsCreateEquip()
    const loadoutInv = tsCreateInv()
    const loadoutInput = { equipment: loadoutEquip, inventory: loadoutInv, slotIndex: 0 }
    expect(equipPlayerInventorySlot(loadoutInput)).toEqual(
      tsEquipInventorySlot(loadoutInput)
    )

    // blacksmithShop
    expect(getBlacksmithShopBuyPriceById('iron-sword')).toBe(
      tsBlacksmithBuy('iron-sword')
    )
    expect(getBlacksmithShopBuyPriceById('does-not-exist')).toBe(
      tsBlacksmithBuy('does-not-exist')
    )

    // questLog
    expect(createInitialQuestLog()).toEqual(tsCreateQuestLog())
    const questLog = tsCreateQuestLog()
    const startedQuestLog = tsStartQuest(questLog, FIRST_SLIME_HUNT_QUEST_ID)
    expect(
      recordMonsterDefeatQuestProgress(startedQuestLog, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    ).toEqual(
      tsRecordMonsterDefeat(startedQuestLog, {
        sceneId: 'hunting-ground',
        appearanceType: 'monster_slime'
      })
    )

    // ── 신규 10개 모듈 스팟 체크 ──

    // playerExperience — grant (progression 전역에 위임)
    expect(grantPlayerExperience(profile, 500)).toEqual(
      tsGrantExperience(profile, 500)
    )
    expect(grantPlayerExperience(profile, 0)).toEqual(
      tsGrantExperience(profile, 0)
    )

    // playerControls
    expect(createInitialPlayerControlBindings()).toEqual(tsCreateControlBindings())
    for (const code of ['ArrowUp', 'KeyA', 'Digit3', 'Numpad5', 'Escape']) {
      expect(getPlayerControlBindingDisplayText(code)).toBe(
        tsControlDisplayText(code)
      )
    }

    // playerSaveState — serialize 후 parse 왕복이 TS와 동등
    const saveInput = {
      profile,
      equipment: tsCreateEquip(),
      inventory: tsCreateInv(),
      quickslots: tsCreateQuickslots(),
      skillSlots: tsCreateSkillSlots()
    }
    // 직렬화 문자열은 키 순서가 비결정적이므로 파싱해 같은 데이터인지 비교한다.
    expect(JSON.parse(serializePlayerSaveState(saveInput))).toEqual(
      JSON.parse(tsSerializeSave(saveInput))
    )
    expect(parseStoredPlayerSaveState(serializePlayerSaveState(saveInput))).toEqual(
      tsParseSave(tsSerializeSave(saveInput))
    )
    expect(parseStoredPlayerSaveState('not json')).toEqual(tsParseSave('not json'))

    // playerRoll
    for (const dir of ['up', 'down', 'left', 'right'] as const) {
      expect(getPlayerRollDirectionVector(dir)).toEqual(tsRollDirection(dir))
    }
    expect(
      getPlayerRollVisualState({ vector: { x: 1, y: 0 }, progress: 0.5 })
    ).toEqual(tsRollVisual({ vector: { x: 1, y: 0 }, progress: 0.5 }))

    // playerSmashSkill
    for (const facing of ['up', 'down', 'left', 'right'] as const) {
      expect(getPlayerSmashSkillDirectionVector(facing)).toEqual(
        tsSmashDirection(facing)
      )
    }
    expect(
      getPlayerSmashSkillSegmentPlacement({
        characterCenterX: 100,
        characterCenterY: 80,
        facing: 'right',
        segmentIndex: 1,
        progress: 0.4
      })
    ).toEqual(
      tsSmashPlacement({
        characterCenterX: 100,
        characterCenterY: 80,
        facing: 'right',
        segmentIndex: 1,
        progress: 0.4
      })
    )

    // eventGeneration / eventDrafting
    const draft = tsEventDraft('크리스마스 이벤트 만들어줘')
    expect(createHolidayDialogueEventDraftFromText('크리스마스 이벤트 만들어줘')).toEqual(
      draft
    )
    if (draft) {
      expect(isHolidayDialogueEventValid(draft)).toBe(tsEventValid(draft))
      expect(createTiledNpcEventObject(draft)).toEqual(tsTiledNpc(draft))
    }
    expect(createHolidayDialogueEventDraftFromText('관련없는 입력')).toEqual(
      tsEventDraft('관련없는 입력')
    )

    // characterState
    expect(
      createInitialPlayerCharacter({ mapWidth: 20, mapHeight: 16 })
    ).toEqual(tsCreatePlayerCharacter({ mapWidth: 20, mapHeight: 16 }))
    for (const key of ['ArrowUp', 'ArrowLeft', 'Enter', 'KeyX']) {
      expect(getCharacterMoveDirectionFromKey(key)).toBe(tsMoveDirFromKey(key))
    }

    // monsterPatrol — createMonsterPatrolState 만 변환됨(stepMonsterPatrol 은 TS 유지)
    const patrolCharacter = tsCreatePlayerCharacter({ mapWidth: 20, mapHeight: 16 })
    expect(createMonsterPatrolState(patrolCharacter)).toEqual(
      tsCreatePatrol(patrolCharacter)
    )

    // sceneIntro
    for (const sceneId of ['town', 'hunting-ground', 'cave', 'unknown']) {
      expect(getSceneIntroMessage(sceneId)).toBe(tsSceneIntro(sceneId))
    }
  })

  it('falls back to TS before init', () => {
    expect(isLuaGameLogicReady()).toBe(false)
    expect(getMonsterGoldDropAmount(5)).toBe(tsGold(5))
    expect(getMonsterDisplayName({ id: 'slime-1' })).toBe(tsDisplayName({ id: 'slime-1' }))
  })
})
