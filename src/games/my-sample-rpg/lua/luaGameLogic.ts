// 게임이 실제로 Lua 로직을 쓰도록 하는 단일 호스트 퍼사드.
// 변환된 .lua 모듈들을 하나의 Lua VM에 로드하고, 원본 TS 모듈과 "동일한 시그니처"의 함수를
// 내보낸다 → 게임의 호출부는 import 출처만 이 파일로 바꾸면 되고, 코드는 그대로다.
// initLuaGameLogic()를 부팅 시 1회 await 하기 전(또는 실패 시)에는 TS 구현으로 폴백한다(안전).
// 각 함수의 Lua↔TS 동등성은 *Lua.bridge.spec.ts + luaGameLogic.bridge.spec.ts로 검증된다.

import monsterRewardsSource from '../assets/lua/monster-rewards.lua?raw'
import monsterCombatSource from '../assets/lua/monster-combat.lua?raw'
import monsterDisplayNameSource from '../assets/lua/monster-display-name.lua?raw'
import playerExperienceSource from '../assets/lua/player-experience.lua?raw'
import monsterEquipmentDropsSource from '../assets/lua/monster-equipment-drops.lua?raw'

import {
  getMonsterGoldDropAmount as tsGetMonsterGoldDropAmount,
  getMonsterExperienceDropAmount as tsGetMonsterExperienceDropAmount,
  getMonsterSkillPointDropAmount as tsGetMonsterSkillPointDropAmount
} from '../monsterRewards'
import {
  createMonsterCombatState as tsCreateMonsterCombatState,
  applyMonsterDamage as tsApplyMonsterDamage,
  isMonsterDefeated as tsIsMonsterDefeated,
  type MonsterCombatState
} from '../monsterCombat'
import { getMonsterDisplayName as tsGetMonsterDisplayName } from '../monsterDisplayName'
import {
  getPlayerExperienceToNextLevel as tsGetPlayerExperienceToNextLevel,
  grantPlayerExperience as tsGrantPlayerExperience,
  PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP,
  PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL,
  type GrantPlayerExperienceResult
} from '../playerExperience'
import { PLAYER_MAX_LEVEL } from '../playerProfile'
import {
  rollMonsterEquipmentDrop as tsRollMonsterEquipmentDrop,
  MONSTER_EQUIPMENT_DROP_CHANCE,
  MONSTER_EQUIPMENT_DROP_DEFINITIONS,
  type MonsterEquipmentDropDefinition
} from '../monsterEquipmentDrops'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

// 플레이어 시스템 모듈 — 검증된 래퍼를 그대로 인스턴스화해 위임(마샬링 재구현/드리프트 없음).
import {
  createPlayerStatEffectsLua,
  type PlayerStatEffectsLua
} from './playerStatEffectsLua'
import {
  createPlayerInventoryLua,
  type PlayerInventoryLua
} from './playerInventoryLua'
import {
  createPlayerProfileLua,
  type PlayerProfileLua
} from './playerProfileLua'
import {
  createPlayerProgressionLua,
  type PlayerProgressionLua
} from './playerProgressionLua'
import {
  createPlayerEquipmentLua,
  type PlayerEquipmentLua
} from './playerEquipmentLua'
import {
  createPlayerPotionShopLua,
  type PlayerPotionShopLua
} from './playerPotionShopLua'
import {
  createPlayerQuickslotsLua,
  type PlayerQuickslotsLua
} from './playerQuickslotsLua'
import {
  createPlayerSkillSlotsLua,
  type PlayerSkillSlotsLua
} from './playerSkillSlotsLua'
import {
  createPlayerSkillsLua,
  type PlayerSkillsLua
} from './playerSkillsLua'
import {
  createPlayerConsumablesLua,
  type PlayerConsumablesLua
} from './playerConsumablesLua'
import {
  createPlayerLoadoutLua,
  type PlayerLoadoutLua
} from './playerLoadoutLua'
import {
  createBlacksmithShopLua,
  type BlacksmithShopLua
} from './blacksmithShopLua'
import {
  createQuestLogLua,
  type QuestLogLua
} from './questLogLua'
import {
  createPlayerExperienceLua,
  type PlayerExperienceLua
} from './playerExperienceLua'
import {
  createPlayerControlsLua,
  type PlayerControlsLua
} from './playerControlsLua'
import {
  createPlayerSaveStateLua,
  type PlayerSaveStateLua
} from './playerSaveStateLua'
import {
  createPlayerRollLua,
  type PlayerRollLua
} from './playerRollLua'
import {
  createPlayerSmashSkillLua,
  type PlayerSmashSkillLua
} from './playerSmashSkillLua'
import {
  createEventGenerationLua,
  type EventGenerationLua
} from './eventGenerationLua'
import {
  createEventDraftingLua,
  type EventDraftingLua
} from './eventDraftingLua'
import {
  createCharacterStateLua,
  type CharacterStateLua
} from './characterStateLua'
import {
  createMonsterPatrolLua,
  type MonsterPatrolLua
} from './monsterPatrolLua'
import {
  createSceneIntroLua,
  type SceneIntroLua
} from './sceneIntroLua'

// 위 모듈들의 TS 폴백 + 게임 호출부 시그니처용 타입.
import {
  getPlayerPhysicalAttackPower as tsGetPlayerPhysicalAttackPower,
  getPlayerMovementSpeedTilesPerSecond as tsGetPlayerMovementSpeedTilesPerSecond,
  getPlayerEvadeChance as tsGetPlayerEvadeChance,
  shouldPlayerEvadeDamage as tsShouldPlayerEvadeDamage
} from '../playerStatEffects'
import {
  createInitialPlayerInventory as tsCreateInitialPlayerInventory,
  setPlayerInventorySlot as tsSetPlayerInventorySlot,
  clearPlayerInventorySlot as tsClearPlayerInventorySlot,
  getPlayerInventoryFilledSlotCount as tsGetPlayerInventoryFilledSlotCount,
  findFirstEmptyPlayerInventorySlotIndex as tsFindFirstEmptyPlayerInventorySlotIndex,
  type PlayerInventory,
  type PlayerInventoryItem
} from '../playerInventory'
import {
  getPlayerJobDisplayName as tsGetPlayerJobDisplayName,
  isPlayerJobPromotionAvailable as tsIsPlayerJobPromotionAvailable,
  isPlayerAtMaxLevel as tsIsPlayerAtMaxLevel,
  getPlayerJobPrimaryStatId as tsGetPlayerJobPrimaryStatId,
  getPlayerJobPrimaryStatLabel as tsGetPlayerJobPrimaryStatLabel,
  type PlayerProfile,
  type PlayerSkillSlot,
  type PlayerStatId
} from '../playerProfile'
import {
  grantPlayerLevelUpRewards as tsGrantPlayerLevelUpRewards,
  grantPlayerSkillPoints as tsGrantPlayerSkillPoints,
  spendPlayerStatPoint as tsSpendPlayerStatPoint,
  spendPlayerSkillPoint as tsSpendPlayerSkillPoint,
  getPlayerSkillPointCost as tsGetPlayerSkillPointCost,
  getPlayerSkillLevelLabel as tsGetPlayerSkillLevelLabel,
  getPlayerSkillUserLevel as tsGetPlayerSkillUserLevel,
  getPlayerMaxManaBySkillUserLevel as tsGetPlayerMaxManaBySkillUserLevel,
  getPlayerMaxManaForProfile as tsGetPlayerMaxManaForProfile
} from '../playerProgression'
import {
  createInitialPlayerEquipment as tsCreateInitialPlayerEquipment,
  getPlayerEquipmentSlotLabelById as tsGetPlayerEquipmentSlotLabelById,
  createPlayerEquipmentItemFromDefinition as tsCreatePlayerEquipmentItemFromDefinition,
  getPlayerEquipmentItemDefinitionById as tsGetPlayerEquipmentItemDefinitionById,
  getPlayerEquipmentItemDefinitionBySlotId as tsGetPlayerEquipmentItemDefinitionBySlotId,
  getPlayerEquipmentSlotById as tsGetPlayerEquipmentSlotById,
  findPlayerEquipmentSlotIndexById as tsFindPlayerEquipmentSlotIndexById,
  setPlayerEquipmentSlot as tsSetPlayerEquipmentSlot,
  clearPlayerEquipmentSlot as tsClearPlayerEquipmentSlot,
  type PlayerEquipment,
  type PlayerEquipmentItem,
  type PlayerEquipmentItemDefinition,
  type PlayerEquipmentSlot,
  type PlayerEquipmentSlotId
} from '../playerEquipment'
import {
  createInitialPotionInventory as tsCreateInitialPotionInventory,
  getPotionShopItemDefinitionById as tsGetPotionShopItemDefinitionById,
  getPotionShopBuyPriceById as tsGetPotionShopBuyPriceById,
  getPotionShopSellPriceById as tsGetPotionShopSellPriceById,
  buyPotionShopItem as tsBuyPotionShopItem,
  sellPotionShopItem as tsSellPotionShopItem,
  type PotionShopInventory,
  type PotionShopItemDefinition,
  type PotionShopTransactionResult
} from '../potionShop'
import {
  createInitialPlayerQuickslots as tsCreateInitialPlayerQuickslots,
  setPlayerQuickslotAssignment as tsSetPlayerQuickslotAssignment,
  clearPlayerQuickslotAssignment as tsClearPlayerQuickslotAssignment,
  findPlayerQuickslotIndexByInventorySlotIndex as tsFindPlayerQuickslotIndexByInventorySlotIndex,
  getPlayerQuickslotAssignment as tsGetPlayerQuickslotAssignment,
  type PlayerQuickslots,
  type PlayerQuickslotAssignment
} from '../playerQuickslots'
import {
  createInitialPlayerSkillSlots as tsCreateInitialPlayerSkillSlots,
  setPlayerSkillSlotAssignment as tsSetPlayerSkillSlotAssignment,
  clearPlayerSkillSlotAssignment as tsClearPlayerSkillSlotAssignment,
  findPlayerSkillSlotIndexBySkillId as tsFindPlayerSkillSlotIndexBySkillId,
  getPlayerSkillSlotAssignment as tsGetPlayerSkillSlotAssignment,
  getPlayerSkillSlotIndexFromCode as tsGetPlayerSkillSlotIndexFromCode,
  type PlayerSkillSlots,
  type PlayerSkillSlotAssignment
} from '../playerSkillSlots'
import {
  isPlayerSkillUnlockedInProfile as tsIsPlayerSkillUnlockedInProfile,
  getPlayerSkillLevelById as tsGetPlayerSkillLevelById,
  getPlayerSkillDamageById as tsGetPlayerSkillDamageById,
  getPlayerSkillDamageBonusById as tsGetPlayerSkillDamageBonusById,
  getPlayerSkillManaCostById as tsGetPlayerSkillManaCostById,
  getPlayerProtectSkillDurationByLevel as tsGetPlayerProtectSkillDurationByLevel
} from '../playerSkills'
import {
  usePlayerInventoryConsumable as tsUsePlayerInventoryConsumable,
  usePlayerQuickslotConsumable as tsUsePlayerQuickslotConsumable,
  type UsePlayerInventoryConsumableResult,
  type UsePlayerQuickslotConsumableResult
} from '../playerConsumables'
import {
  unequipPlayerEquipmentSlot as tsUnequipPlayerEquipmentSlot,
  equipPlayerInventorySlot as tsEquipPlayerInventorySlot,
  type PlayerLoadoutState
} from '../playerLoadout'
import {
  createInitialBlacksmithInventory as tsCreateInitialBlacksmithInventory,
  getBlacksmithShopItemPriceById as tsGetBlacksmithShopItemPriceById,
  getBlacksmithShopBuyPriceById as tsGetBlacksmithShopBuyPriceById,
  getBlacksmithShopSellPriceById as tsGetBlacksmithShopSellPriceById,
  buyBlacksmithShopItem as tsBuyBlacksmithShopItem,
  sellBlacksmithShopItem as tsSellBlacksmithShopItem,
  type BlacksmithInventory,
  type BlacksmithShopTransactionResult
} from '../blacksmithShop'
import {
  createInitialQuestLog as tsCreateInitialQuestLog,
  getQuestProgress as tsGetQuestProgress,
  getQuestDefinition as tsGetQuestDefinition,
  startQuest as tsStartQuest,
  recordQuestObjectiveProgress as tsRecordQuestObjectiveProgress,
  recordMonsterDefeatQuestProgress as tsRecordMonsterDefeatQuestProgress,
  recordItemUseQuestProgress as tsRecordItemUseQuestProgress,
  recordShopOpenQuestProgress as tsRecordShopOpenQuestProgress,
  recordSceneEnterQuestProgress as tsRecordSceneEnterQuestProgress,
  recordTalkQuestProgress as tsRecordTalkQuestProgress,
  setQuestTrackerVisible as tsSetQuestTrackerVisible,
  hideVisibleQuestTrackers as tsHideVisibleQuestTrackers,
  abandonQuest as tsAbandonQuest,
  completeQuest as tsCompleteQuest,
  getVisibleQuestTrackers as tsGetVisibleQuestTrackers,
  getQuestNpcBadgeKind as tsGetQuestNpcBadgeKind,
  getQuestNpcBadgeKindForNpc as tsGetQuestNpcBadgeKindForNpc,
  getNextQuestInteractionForNpc as tsGetNextQuestInteractionForNpc,
  formatQuestText as tsFormatQuestText,
  formatQuestTextLines as tsFormatQuestTextLines,
  type QuestLogState,
  type QuestProgress,
  type QuestDefinition,
  type QuestNpcBadgeKind,
  type NpcQuestInteraction,
  type CompleteQuestResult,
  type QuestTrackerItem,
  type QuestTextFormatContext
} from '../questLog'
import {
  createInitialPlayerControlBindings as tsCreateInitialPlayerControlBindings,
  normalizeStoredPlayerControlBindings as tsNormalizeStoredPlayerControlBindings,
  setPlayerControlBinding as tsSetPlayerControlBinding,
  getPlayerControlBindingDisplayText as tsGetPlayerControlBindingDisplayText,
  getPlayerControlMovementDirectionFromCode as tsGetPlayerControlMovementDirectionFromCode,
  getPlayerControlActionFromCode as tsGetPlayerControlActionFromCode,
  getPlayerControlQuickslotIndexFromCode as tsGetPlayerControlQuickslotIndexFromCode,
  isPlayerControlPauseKey as tsIsPlayerControlPauseKey,
  isPlayerControlCaptureModifierKey as tsIsPlayerControlCaptureModifierKey,
  getPlayerControlBindingDefinitionById as tsGetPlayerControlBindingDefinitionById,
  getPlayerControlBindingDefinitions as tsGetPlayerControlBindingDefinitions,
  type PlayerControlBindingDefinition,
  type PlayerControlBindingId,
  type PlayerControlBindings
} from '../playerControls'
import {
  serializePlayerSaveState as tsSerializePlayerSaveState,
  normalizeStoredPlayerSaveState as tsNormalizeStoredPlayerSaveState,
  parseStoredPlayerSaveState as tsParseStoredPlayerSaveState,
  type PlayerSaveState,
  type PlayerSaveStateInput
} from '../playerSaveState'
import {
  getPlayerRollDirectionVector as tsGetPlayerRollDirectionVector,
  normalizePlayerRollVector as tsNormalizePlayerRollVector,
  getPlayerRollProgress as tsGetPlayerRollProgress,
  getPlayerRollDistanceTiles as tsGetPlayerRollDistanceTiles,
  getPlayerRollVisualState as tsGetPlayerRollVisualState,
  type PlayerRollVector,
  type PlayerRollVisualState
} from '../playerRoll'
import {
  getPlayerSmashSkillDirectionVector as tsGetPlayerSmashSkillDirectionVector,
  getPlayerSmashSkillSegmentDistancePixels as tsGetPlayerSmashSkillSegmentDistancePixels,
  getPlayerSmashSkillSegmentPlacement as tsGetPlayerSmashSkillSegmentPlacement,
  type PlayerSmashSkillSegmentPlacement
} from '../playerSmashSkill'
import {
  createHolidayDialogueEventValidationErrors as tsCreateHolidayDialogueEventValidationErrors,
  isHolidayDialogueEventValid as tsIsHolidayDialogueEventValid,
  createTiledNpcEventObject as tsCreateTiledNpcEventObject,
  HOLIDAY_DIALOGUE_EVENT_TYPE,
  TALK_EVENT_TRIGGER_TYPE,
  type HolidayDialogueEventSpec,
  type TiledNpcEventObject
} from '../eventGeneration'
import { createHolidayDialogueEventDraftFromText as tsCreateHolidayDialogueEventDraftFromText } from '../eventDrafting'
import {
  createKeyboardCharacterController as tsCreateKeyboardCharacterController,
  createIdleNpcCharacterController as tsCreateIdleNpcCharacterController,
  createLuaCharacterController as tsCreateLuaCharacterController,
  createInitialPlayerCharacter as tsCreateInitialPlayerCharacter,
  createNpcCharacter as tsCreateNpcCharacter,
  getCharacterMoveDirectionFromKey as tsGetCharacterMoveDirectionFromKey,
  getCharacterActionFromKey as tsGetCharacterActionFromKey,
  getCharacterControllerIntent as tsGetCharacterControllerIntent,
  moveCharacterState as tsMoveCharacterState,
  type CharacterAction,
  type CharacterControllerIntent,
  type CharacterMoveDirection,
  type CharacterState,
  type KeyboardCharacterController,
  type LuaCharacterController,
  type LuaCharacterControllerConfig,
  type NpcCharacterController
} from '../characterState'
import {
  createMonsterPatrolState as tsCreateMonsterPatrolState,
  type MonsterPatrolState
} from '../monsterPatrol'
import { getSceneIntroMessage as tsGetSceneIntroMessage } from '../sceneIntro'

let host: LuaLogicHost | undefined
let statEffects: PlayerStatEffectsLua | undefined
let inventory: PlayerInventoryLua | undefined
let profile: PlayerProfileLua | undefined
let progression: PlayerProgressionLua | undefined
let equipmentLua: PlayerEquipmentLua | undefined
let potionShop: PlayerPotionShopLua | undefined
let quickslots: PlayerQuickslotsLua | undefined
let skillSlots: PlayerSkillSlotsLua | undefined
let skills: PlayerSkillsLua | undefined
let loadout: PlayerLoadoutLua | undefined
let consumables: PlayerConsumablesLua | undefined
let blacksmith: BlacksmithShopLua | undefined
let questLogLua: QuestLogLua | undefined
let experience: PlayerExperienceLua | undefined
let controls: PlayerControlsLua | undefined
let saveState: PlayerSaveStateLua | undefined
let roll: PlayerRollLua | undefined
let smashSkill: PlayerSmashSkillLua | undefined
let eventGeneration: EventGenerationLua | undefined
let eventDrafting: EventDraftingLua | undefined
let characterState: CharacterStateLua | undefined
let monsterPatrol: MonsterPatrolLua | undefined
let sceneIntro: SceneIntroLua | undefined

// 부팅 시 1회 호출 — Lua VM을 띄우고 변환된 모듈들을 모두 로드한다. 이후 아래 함수들이 Lua를 쓴다.
export const initLuaGameLogic = async (
  input: CreateLuaLogicHostInput = {}
): Promise<void> => {
  const next = await createLuaLogicHost(input)
  next.runModule(monsterRewardsSource, '@monster-rewards.lua')
  next.runModule(monsterCombatSource, '@monster-combat.lua')
  next.runModule(monsterDisplayNameSource, '@monster-display-name.lua')
  next.runModule(playerExperienceSource, '@player-experience.lua')
  next.runModule(monsterEquipmentDropsSource, '@monster-equipment-drops.lua')
  host = next

  // 플레이어 시스템 모듈은 검증된 래퍼로 위임하되, 같은 단일 호스트(next)를 공유한다
  // (VM 인스턴스를 여러 개 만들지 않음 — 모든 .lua가 한 Lua 상태에 로드된다).
  statEffects = await createPlayerStatEffectsLua({ host: next })
  inventory = await createPlayerInventoryLua({ host: next })
  profile = await createPlayerProfileLua({ host: next })
  progression = await createPlayerProgressionLua({ host: next })
  // experience 의 grant 는 progression 전역(progression_grant_level_up_rewards)에 의존하므로
  // progression 이 로드된 이후에 만든다(이미 .lua 가 로드돼 있다고 가정).
  experience = await createPlayerExperienceLua({ host: next })
  equipmentLua = await createPlayerEquipmentLua({ host: next })
  // potionShop 은 인벤토리 전역(inventory_*)에 의존하므로 inventory 로드 이후에 만든다.
  potionShop = await createPlayerPotionShopLua({ host: next })
  quickslots = await createPlayerQuickslotsLua({ host: next })
  skillSlots = await createPlayerSkillSlotsLua({ host: next })
  skills = await createPlayerSkillsLua({ host: next })
  loadout = await createPlayerLoadoutLua({ host: next })
  // consumables 는 quickslots 전역이 이미 로드된 이후에 만든다.
  consumables = await createPlayerConsumablesLua({ host: next })
  blacksmith = await createBlacksmithShopLua({ host: next })
  questLogLua = await createQuestLogLua({ host: next })

  // 추가 자급식 모듈들(상호 의존 없음 — 임의 순서). 같은 단일 호스트(next)를 공유한다.
  controls = await createPlayerControlsLua({ host: next })
  saveState = await createPlayerSaveStateLua({ host: next })
  roll = await createPlayerRollLua({ host: next })
  smashSkill = await createPlayerSmashSkillLua({ host: next })
  characterState = await createCharacterStateLua({ host: next })
  monsterPatrol = await createMonsterPatrolLua({ host: next })
  sceneIntro = await createSceneIntroLua({ host: next })
  // eventDrafting 의 .lua 는 두 전역 상수(HOLIDAY_DIALOGUE_EVENT_TYPE, TALK_EVENT_TRIGGER_TYPE)에
  // 의존한다. event-generation.lua 는 이 값들을 local 로만 갖고 전역으로 노출하지 않으므로,
  // 공유 호스트에선 TS(단일 출처)에서 가져온 값을 전역으로 주입한 뒤 eventDrafting 을 만든다.
  eventGeneration = await createEventGenerationLua({ host: next })
  next.runModule(
    [
      `HOLIDAY_DIALOGUE_EVENT_TYPE = ${JSON.stringify(HOLIDAY_DIALOGUE_EVENT_TYPE)}`,
      `TALK_EVENT_TRIGGER_TYPE = ${JSON.stringify(TALK_EVENT_TRIGGER_TYPE)}`
    ].join('\n'),
    '@event-generation.constants.lua'
  )
  eventDrafting = await createEventDraftingLua({ host: next })
}

export const isLuaGameLogicReady = (): boolean => host !== undefined

// 테스트/정리용 — 호스트를 닫고 폴백(TS) 상태로 되돌린다.
export const closeLuaGameLogic = (): void => {
  host?.close()
  host = undefined
  statEffects?.close()
  statEffects = undefined
  inventory?.close()
  inventory = undefined
  profile?.close()
  profile = undefined
  progression?.close()
  progression = undefined
  equipmentLua?.close()
  equipmentLua = undefined
  potionShop?.close()
  potionShop = undefined
  quickslots?.close(); quickslots = undefined
  skillSlots?.close(); skillSlots = undefined
  skills?.close(); skills = undefined
  loadout?.close(); loadout = undefined
  consumables?.close(); consumables = undefined
  blacksmith?.close(); blacksmith = undefined
  questLogLua?.close(); questLogLua = undefined
  experience?.close(); experience = undefined
  controls?.close(); controls = undefined
  saveState?.close(); saveState = undefined
  roll?.close(); roll = undefined
  smashSkill?.close(); smashSkill = undefined
  eventGeneration?.close(); eventGeneration = undefined
  eventDrafting?.close(); eventDrafting = undefined
  characterState?.close(); characterState = undefined
  monsterPatrol?.close(); monsterPatrol = undefined
  sceneIntro?.close(); sceneIntro = undefined
}

// ── monsterRewards ──
export const getMonsterGoldDropAmount = (monsterLevel: number): number =>
  host
    ? host.callNumber('monster_rewards_gold', monsterLevel)
    : tsGetMonsterGoldDropAmount(monsterLevel)

export const getMonsterExperienceDropAmount = (monsterLevel: number): number =>
  host
    ? host.callNumber('monster_rewards_experience', monsterLevel)
    : tsGetMonsterExperienceDropAmount(monsterLevel)

export const getMonsterSkillPointDropAmount = (monsterLevel: number): number =>
  host
    ? host.callNumber('monster_rewards_skill_point', monsterLevel)
    : tsGetMonsterSkillPointDropAmount(monsterLevel)

// ── monsterCombat ──
export const createMonsterCombatState = (
  monsterLevel = 1,
  options: { hpMultiplier?: number; damageMultiplier?: number } = {}
): MonsterCombatState => {
  if (!host) {
    return tsCreateMonsterCombatState(monsterLevel, options)
  }

  const [maxHp, currentHp, contactDamage] = host.callNumbers(
    'monster_combat_create',
    [monsterLevel, options.hpMultiplier ?? 1, options.damageMultiplier ?? 1],
    3
  )

  return { maxHp, currentHp, contactDamage }
}

export const applyMonsterDamage = (
  combatState: MonsterCombatState,
  damage: number
): MonsterCombatState => {
  if (!host) {
    return tsApplyMonsterDamage(combatState, damage)
  }

  const nextCurrentHp = host.callNumber(
    'monster_combat_apply_damage',
    combatState.currentHp,
    damage
  )

  // TS와 동일하게 변화 없으면 같은 참조를 돌려준다(불필요한 리렌더 방지).
  return nextCurrentHp === combatState.currentHp
    ? combatState
    : { ...combatState, currentHp: nextCurrentHp }
}

export const isMonsterDefeated = (combatState: MonsterCombatState): boolean =>
  host
    ? host.callNumber('monster_combat_is_defeated', combatState.currentHp) === 1
    : tsIsMonsterDefeated(combatState)

// ── monsterDisplayName ──
export const getMonsterDisplayName = (args: {
  id: string
  displayText?: string
}): string =>
  host
    ? host.callString('monster_display_name', args.id, args.displayText ?? '')
    : tsGetMonsterDisplayName(args)

// ── playerExperience (다음 레벨 필요 경험치) ──
export const getPlayerExperienceToNextLevel = (level: number): number =>
  host
    ? host.callNumber(
        'player_experience_to_next_level',
        level,
        PLAYER_MAX_LEVEL,
        PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP,
        PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL
      )
    : tsGetPlayerExperienceToNextLevel(level)

// grant 는 레벨업 보상을 progression 전역에 위임하므로 확장된 experience 래퍼로 라우팅한다.
export const grantPlayerExperience = (
  profile: PlayerProfile,
  experienceAmount: number
): GrantPlayerExperienceResult =>
  experience
    ? experience.grantPlayerExperience(profile, experienceAmount)
    : tsGrantPlayerExperience(profile, experienceAmount)

// ── monsterEquipmentDrops ──
// 확률 게이트와 드롭 정의는 TS, 인덱스 계산만 Lua(난수 호출 순서는 원본과 동일).
export const rollMonsterEquipmentDrop = (
  random: () => number = Math.random
): MonsterEquipmentDropDefinition | undefined => {
  if (!host) {
    return tsRollMonsterEquipmentDrop(random)
  }

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
}

// ── playerStatEffects (파생 스탯) ──
export const getPlayerPhysicalAttackPower = (
  player: Pick<PlayerProfile, 'stats'>
): number =>
  statEffects
    ? statEffects.getPlayerPhysicalAttackPower(player)
    : tsGetPlayerPhysicalAttackPower(player)

export const getPlayerMovementSpeedTilesPerSecond = (
  player: Pick<PlayerProfile, 'stats'>
): number =>
  statEffects
    ? statEffects.getPlayerMovementSpeedTilesPerSecond(player)
    : tsGetPlayerMovementSpeedTilesPerSecond(player)

export const getPlayerEvadeChance = (
  player: Pick<PlayerProfile, 'stats'>
): number =>
  statEffects
    ? statEffects.getPlayerEvadeChance(player)
    : tsGetPlayerEvadeChance(player)

export const shouldPlayerEvadeDamage = (
  player: Pick<PlayerProfile, 'stats'>,
  randomValue = Math.random()
): boolean =>
  statEffects
    ? statEffects.shouldPlayerEvadeDamage(player, randomValue)
    : tsShouldPlayerEvadeDamage(player, randomValue)

// ── playerInventory ──
export const createInitialPlayerInventory = (input?: {
  slotCount?: number
  gold?: number
}): PlayerInventory =>
  inventory
    ? inventory.createInitialPlayerInventory(input)
    : tsCreateInitialPlayerInventory(input)

export const setPlayerInventorySlot = (input: {
  inventory: PlayerInventory
  slotIndex: number
  item: PlayerInventoryItem
}): PlayerInventory =>
  inventory
    ? inventory.setPlayerInventorySlot(input)
    : tsSetPlayerInventorySlot(input)

export const clearPlayerInventorySlot = (input: {
  inventory: PlayerInventory
  slotIndex: number
}): PlayerInventory =>
  inventory
    ? inventory.clearPlayerInventorySlot(input)
    : tsClearPlayerInventorySlot(input)

export const getPlayerInventoryFilledSlotCount = (
  playerInventory: PlayerInventory
): number =>
  inventory
    ? inventory.getPlayerInventoryFilledSlotCount(playerInventory)
    : tsGetPlayerInventoryFilledSlotCount(playerInventory)

export const findFirstEmptyPlayerInventorySlotIndex = (
  playerInventory: PlayerInventory
): number | undefined =>
  inventory
    ? inventory.findFirstEmptyPlayerInventorySlotIndex(playerInventory)
    : tsFindFirstEmptyPlayerInventorySlotIndex(playerInventory)

// ── playerProfile (직업/레벨 헬퍼) ──
export const getPlayerJobDisplayName = (
  input: Pick<PlayerProfile, 'job' | 'level'>
): string =>
  profile ? profile.getPlayerJobDisplayName(input) : tsGetPlayerJobDisplayName(input)

export const isPlayerJobPromotionAvailable = (
  input: Pick<PlayerProfile, 'level'>
): boolean =>
  profile
    ? profile.isPlayerJobPromotionAvailable(input)
    : tsIsPlayerJobPromotionAvailable(input)

export const isPlayerAtMaxLevel = (
  input: Pick<PlayerProfile, 'level'>
): boolean =>
  profile ? profile.isPlayerAtMaxLevel(input) : tsIsPlayerAtMaxLevel(input)

export const getPlayerJobPrimaryStatId = (
  job: string
): PlayerStatId | undefined =>
  profile
    ? profile.getPlayerJobPrimaryStatId(job)
    : tsGetPlayerJobPrimaryStatId(job)

export const getPlayerJobPrimaryStatLabel = (
  job: string
): string | undefined =>
  profile
    ? profile.getPlayerJobPrimaryStatLabel(job)
    : tsGetPlayerJobPrimaryStatLabel(job)

// ── playerProgression (레벨업/스탯/스킬) ──
export const grantPlayerLevelUpRewards = (
  playerProfile: PlayerProfile,
  levels?: number
): PlayerProfile =>
  progression
    ? progression.grantPlayerLevelUpRewards(playerProfile, levels)
    : tsGrantPlayerLevelUpRewards(playerProfile, levels)

export const grantPlayerSkillPoints = (
  playerProfile: PlayerProfile,
  gainedSkillPoints: number
): PlayerProfile =>
  progression
    ? progression.grantPlayerSkillPoints(playerProfile, gainedSkillPoints)
    : tsGrantPlayerSkillPoints(playerProfile, gainedSkillPoints)

export const spendPlayerStatPoint = (
  playerProfile: PlayerProfile,
  statId: PlayerStatId
): PlayerProfile | undefined =>
  progression
    ? progression.spendPlayerStatPoint(playerProfile, statId)
    : tsSpendPlayerStatPoint(playerProfile, statId)

export const spendPlayerSkillPoint = (
  playerProfile: PlayerProfile,
  skillIndex: number
): PlayerProfile | undefined =>
  progression
    ? progression.spendPlayerSkillPoint(playerProfile, skillIndex)
    : tsSpendPlayerSkillPoint(playerProfile, skillIndex)

export const getPlayerSkillPointCost = (skill: PlayerSkillSlot): number =>
  progression
    ? progression.getPlayerSkillPointCost(skill)
    : tsGetPlayerSkillPointCost(skill)

export const getPlayerSkillLevelLabel = (skill: PlayerSkillSlot): string =>
  progression
    ? progression.getPlayerSkillLevelLabel(skill)
    : tsGetPlayerSkillLevelLabel(skill)

export const getPlayerSkillUserLevel = (
  totalSkillPointsEarned: number
): number =>
  progression
    ? progression.getPlayerSkillUserLevel(totalSkillPointsEarned)
    : tsGetPlayerSkillUserLevel(totalSkillPointsEarned)

export const getPlayerMaxManaBySkillUserLevel = (
  skillUserLevel: number
): number =>
  progression
    ? progression.getPlayerMaxManaBySkillUserLevel(skillUserLevel)
    : tsGetPlayerMaxManaBySkillUserLevel(skillUserLevel)

export const getPlayerMaxManaForProfile = (
  playerProfile: Pick<PlayerProfile, 'stats' | 'totalSkillPointsEarned'>
): number =>
  progression
    ? progression.getPlayerMaxManaForProfile(playerProfile)
    : tsGetPlayerMaxManaForProfile(playerProfile)

// ── playerEquipment ──
export const createInitialPlayerEquipment = (): PlayerEquipment =>
  equipmentLua
    ? equipmentLua.createInitialPlayerEquipment()
    : tsCreateInitialPlayerEquipment()

export const getPlayerEquipmentSlotLabelById = (
  slotId: PlayerEquipmentSlotId
): string =>
  equipmentLua
    ? equipmentLua.getPlayerEquipmentSlotLabelById(slotId)
    : tsGetPlayerEquipmentSlotLabelById(slotId)

export const createPlayerEquipmentItemFromDefinition = (
  definition: PlayerEquipmentItemDefinition
): PlayerEquipmentItem =>
  equipmentLua
    ? equipmentLua.createPlayerEquipmentItemFromDefinition(definition)
    : tsCreatePlayerEquipmentItemFromDefinition(definition)

export const getPlayerEquipmentItemDefinitionById = (
  itemId: string
): PlayerEquipmentItemDefinition | undefined =>
  equipmentLua
    ? equipmentLua.getPlayerEquipmentItemDefinitionById(itemId)
    : tsGetPlayerEquipmentItemDefinitionById(itemId)

export const getPlayerEquipmentItemDefinitionBySlotId = (
  slotId: PlayerEquipmentSlotId
): PlayerEquipmentItemDefinition | undefined =>
  equipmentLua
    ? equipmentLua.getPlayerEquipmentItemDefinitionBySlotId(slotId)
    : tsGetPlayerEquipmentItemDefinitionBySlotId(slotId)

export const getPlayerEquipmentSlotById = (
  equipment: PlayerEquipment,
  slotId: PlayerEquipmentSlotId
): PlayerEquipmentSlot | undefined =>
  equipmentLua
    ? equipmentLua.getPlayerEquipmentSlotById(equipment, slotId)
    : tsGetPlayerEquipmentSlotById(equipment, slotId)

export const findPlayerEquipmentSlotIndexById = (
  equipment: PlayerEquipment,
  slotId: PlayerEquipmentSlotId
): number =>
  equipmentLua
    ? equipmentLua.findPlayerEquipmentSlotIndexById(equipment, slotId)
    : tsFindPlayerEquipmentSlotIndexById(equipment, slotId)

export const setPlayerEquipmentSlot = (input: {
  equipment: PlayerEquipment
  slotId: PlayerEquipmentSlotId
  item: PlayerEquipmentItem
}): PlayerEquipment =>
  equipmentLua
    ? equipmentLua.setPlayerEquipmentSlot(input)
    : tsSetPlayerEquipmentSlot(input)

export const clearPlayerEquipmentSlot = (input: {
  equipment: PlayerEquipment
  slotId: PlayerEquipmentSlotId
}): PlayerEquipment =>
  equipmentLua
    ? equipmentLua.clearPlayerEquipmentSlot(input)
    : tsClearPlayerEquipmentSlot(input)

// ── potionShop (포션 상점 — 인벤토리 전역에 의존, 같은 호스트 공유) ──
export const createInitialPotionInventory = (input?: {
  slotCount?: number
  gold?: number
}): PotionShopInventory =>
  potionShop
    ? potionShop.createInitialPotionInventory(input)
    : tsCreateInitialPotionInventory(input)

export const getPotionShopItemDefinitionById = (
  itemId: string
): PotionShopItemDefinition | undefined =>
  potionShop
    ? potionShop.getPotionShopItemDefinitionById(itemId)
    : tsGetPotionShopItemDefinitionById(itemId)

export const getPotionShopBuyPriceById = (
  itemId: string
): number | undefined =>
  potionShop
    ? potionShop.getPotionShopBuyPriceById(itemId)
    : tsGetPotionShopBuyPriceById(itemId)

export const getPotionShopSellPriceById = (
  itemId: string
): number | undefined =>
  potionShop
    ? potionShop.getPotionShopSellPriceById(itemId)
    : tsGetPotionShopSellPriceById(itemId)

export const buyPotionShopItem = (input: {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  merchantSlotIndex: number
  quantity?: number
}): PotionShopTransactionResult =>
  potionShop
    ? potionShop.buyPotionShopItem(input)
    : tsBuyPotionShopItem(input)

export const sellPotionShopItem = (input: {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  playerSlotIndex: number
  quantity?: number
}): PotionShopTransactionResult =>
  potionShop
    ? potionShop.sellPotionShopItem(input)
    : tsSellPotionShopItem(input)

// ── playerQuickslots ──
export const createInitialPlayerQuickslots = (input?: {
  slotCount?: number
}): PlayerQuickslots =>
  quickslots
    ? quickslots.createInitialPlayerQuickslots(input)
    : tsCreateInitialPlayerQuickslots(input)

export const setPlayerQuickslotAssignment = (input: {
  quickslots: PlayerQuickslots
  quickslotIndex: number
  inventorySlotIndex: number
}): PlayerQuickslots =>
  quickslots
    ? quickslots.setPlayerQuickslotAssignment(input)
    : tsSetPlayerQuickslotAssignment(input)

export const clearPlayerQuickslotAssignment = (input: {
  quickslots: PlayerQuickslots
  quickslotIndex: number
}): PlayerQuickslots =>
  quickslots
    ? quickslots.clearPlayerQuickslotAssignment(input)
    : tsClearPlayerQuickslotAssignment(input)

export const findPlayerQuickslotIndexByInventorySlotIndex = (
  qs: PlayerQuickslots,
  inventorySlotIndex: number
): number | undefined =>
  quickslots
    ? quickslots.findPlayerQuickslotIndexByInventorySlotIndex(qs, inventorySlotIndex)
    : tsFindPlayerQuickslotIndexByInventorySlotIndex(qs, inventorySlotIndex)

export const getPlayerQuickslotAssignment = (
  qs: PlayerQuickslots,
  quickslotIndex: number
): PlayerQuickslotAssignment | undefined =>
  quickslots
    ? quickslots.getPlayerQuickslotAssignment(qs, quickslotIndex)
    : tsGetPlayerQuickslotAssignment(qs, quickslotIndex)

// ── playerSkillSlots ──
export const createInitialPlayerSkillSlots = (input?: {
  slotCount?: number
  initialSkillIds?: readonly (string | undefined)[]
}): PlayerSkillSlots =>
  skillSlots
    ? skillSlots.createInitialPlayerSkillSlots(input)
    : tsCreateInitialPlayerSkillSlots(input)

export const setPlayerSkillSlotAssignment = (input: {
  skillSlots: PlayerSkillSlots
  skillSlotIndex: number
  skillId: string
}): PlayerSkillSlots =>
  skillSlots
    ? skillSlots.setPlayerSkillSlotAssignment(input)
    : tsSetPlayerSkillSlotAssignment(input)

export const clearPlayerSkillSlotAssignment = (input: {
  skillSlots: PlayerSkillSlots
  skillSlotIndex: number
}): PlayerSkillSlots =>
  skillSlots
    ? skillSlots.clearPlayerSkillSlotAssignment(input)
    : tsClearPlayerSkillSlotAssignment(input)

export const findPlayerSkillSlotIndexBySkillId = (
  ss: PlayerSkillSlots,
  skillId: string
): number | undefined =>
  skillSlots
    ? skillSlots.findPlayerSkillSlotIndexBySkillId(ss, skillId)
    : tsFindPlayerSkillSlotIndexBySkillId(ss, skillId)

export const getPlayerSkillSlotAssignment = (
  ss: PlayerSkillSlots,
  skillSlotIndex: number
): PlayerSkillSlotAssignment | undefined =>
  skillSlots
    ? skillSlots.getPlayerSkillSlotAssignment(ss, skillSlotIndex)
    : tsGetPlayerSkillSlotAssignment(ss, skillSlotIndex)

export const getPlayerSkillSlotIndexFromCode = (
  code: string
): number | undefined =>
  skillSlots
    ? skillSlots.getPlayerSkillSlotIndexFromCode(code)
    : tsGetPlayerSkillSlotIndexFromCode(code)

export { PLAYER_SKILL_SLOT_HOTKEY_CODES } from '../playerSkillSlots'

// ── playerSkills ──
export const isPlayerSkillUnlockedInProfile = (
  profile: Pick<PlayerProfile, 'skills'>,
  skillId: string
): boolean =>
  skills
    ? skills.isPlayerSkillUnlockedInProfile(profile, skillId)
    : tsIsPlayerSkillUnlockedInProfile(profile, skillId)

export const getPlayerSkillLevelById = (
  profile: Pick<PlayerProfile, 'skills'>,
  skillId: string
): number | undefined =>
  skills
    ? skills.getPlayerSkillLevelById(profile, skillId)
    : tsGetPlayerSkillLevelById(profile, skillId)

export const getPlayerSkillDamageById = (
  profile: Pick<PlayerProfile, 'skills'>,
  skillId: string
): number =>
  skills
    ? skills.getPlayerSkillDamageById(profile, skillId)
    : tsGetPlayerSkillDamageById(profile, skillId)

export const getPlayerSkillDamageBonusById = (
  profile: Pick<PlayerProfile, 'skills'>,
  skillId: string
): number =>
  skills
    ? skills.getPlayerSkillDamageBonusById(profile, skillId)
    : tsGetPlayerSkillDamageBonusById(profile, skillId)

export const getPlayerSkillManaCostById = (
  profile: Pick<PlayerProfile, 'skills'>,
  skillId: string
): number =>
  skills
    ? skills.getPlayerSkillManaCostById(profile, skillId)
    : tsGetPlayerSkillManaCostById(profile, skillId)

export const getPlayerProtectSkillDurationByLevel = (
  skillLevel: number
): number =>
  skills
    ? skills.getPlayerProtectSkillDurationByLevel(skillLevel)
    : tsGetPlayerProtectSkillDurationByLevel(skillLevel)

// getPlayerSkillDisplayInfoById 는 Lua 변환 대상 아님 (Vite 에셋 URL) — 원본 그대로 재내보냄.
export { getPlayerSkillDisplayInfoById } from '../playerSkills'

// ── playerConsumables ──
export const usePlayerInventoryConsumable = (input: {
  profile: PlayerProfile
  inventory: PlayerInventory
  slotIndex: number
}): UsePlayerInventoryConsumableResult | undefined =>
  consumables
    ? consumables.usePlayerInventoryConsumable(input)
    : tsUsePlayerInventoryConsumable(input)

export const usePlayerQuickslotConsumable = (input: {
  profile: PlayerProfile
  inventory: PlayerInventory
  quickslots: { slots: ({ inventorySlotIndex: number } | undefined)[] }
  quickslotIndex: number
}): UsePlayerQuickslotConsumableResult | undefined =>
  consumables
    ? consumables.usePlayerQuickslotConsumable(input)
    : tsUsePlayerQuickslotConsumable(input)

// ── playerLoadout ──
export const unequipPlayerEquipmentSlot = (input: {
  equipment: PlayerEquipment
  inventory: PlayerInventory
  slotId: PlayerEquipmentSlotId
}): PlayerLoadoutState | undefined =>
  loadout
    ? loadout.unequipPlayerEquipmentSlot(input)
    : tsUnequipPlayerEquipmentSlot(input)

export const equipPlayerInventorySlot = (input: {
  equipment: PlayerEquipment
  inventory: PlayerInventory
  slotIndex: number
}): PlayerLoadoutState | undefined =>
  loadout
    ? loadout.equipPlayerInventorySlot(input)
    : tsEquipPlayerInventorySlot(input)

// ── blacksmithShop ──
export const createInitialBlacksmithInventory = (input?: {
  slotCount?: number
  gold?: number
}): BlacksmithInventory =>
  blacksmith
    ? blacksmith.createInitialBlacksmithInventory(input)
    : tsCreateInitialBlacksmithInventory(input)

export const getBlacksmithShopItemPriceById = (
  itemId: string
): number | undefined =>
  blacksmith
    ? blacksmith.getBlacksmithShopBuyPriceById(itemId)
    : tsGetBlacksmithShopItemPriceById(itemId)

export const getBlacksmithShopBuyPriceById = (
  itemId: string
): number | undefined =>
  blacksmith
    ? blacksmith.getBlacksmithShopBuyPriceById(itemId)
    : tsGetBlacksmithShopBuyPriceById(itemId)

export const getBlacksmithShopSellPriceById = (
  itemId: string
): number | undefined =>
  blacksmith
    ? blacksmith.getBlacksmithShopSellPriceById(itemId)
    : tsGetBlacksmithShopSellPriceById(itemId)

export const buyBlacksmithShopItem = (input: {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  merchantSlotIndex: number
}): BlacksmithShopTransactionResult =>
  blacksmith
    ? blacksmith.buyBlacksmithShopItem(input)
    : tsBuyBlacksmithShopItem(input)

export const sellBlacksmithShopItem = (input: {
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
  playerSlotIndex: number
}): BlacksmithShopTransactionResult =>
  blacksmith
    ? blacksmith.sellBlacksmithShopItem(input)
    : tsSellBlacksmithShopItem(input)

// ── questLog ──
// 상수·데이터는 TS 소유 — 정적 값이므로 Lua 위임 없이 원본에서 직접 재내보냄.
export {
  QUEST_DEFINITIONS,
  FIRST_SLIME_HUNT_QUEST_ID,
  POTION_SURVIVAL_BASICS_QUEST_ID,
  PIG_TROUBLE_QUEST_ID,
  BLACKSMITH_PREPARATION_QUEST_ID,
  CAVE_ENTRANCE_INVESTIGATION_QUEST_ID,
  SLIME_BOSS_SHADOW_QUEST_ID,
  FINAL_SUPPLIES_QUEST_ID,
  PIG_BOSS_THREAT_QUEST_ID,
  FIRST_SLIME_HUNT_OBJECTIVE_ID,
  FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS,
  FIRST_SLIME_HUNT_REWARD_GOLD,
  FIRST_SLIME_HUNT_REWARD_EXPERIENCE,
  WIZARD_NPC_ID,
  POTION_MERCHANT_NPC_ID,
  BLACKSMITH_NPC_ID,
  QUEST_GIVER_NPC_IDS
} from '../questLog'

export const createInitialQuestLog = (): QuestLogState =>
  questLogLua
    ? questLogLua.createInitialQuestLog()
    : tsCreateInitialQuestLog()

export const getQuestProgress = (
  questLog: QuestLogState,
  questId: string
): QuestProgress =>
  questLogLua
    ? questLogLua.getQuestProgress(questLog, questId)
    : tsGetQuestProgress(questLog, questId)

export const getQuestDefinition = (questId: string): QuestDefinition =>
  questLogLua
    ? questLogLua.getQuestDefinition(questId)
    : tsGetQuestDefinition(questId)

export const startQuest = (
  questLog: QuestLogState,
  questId: string
): QuestLogState =>
  questLogLua
    ? questLogLua.startQuest(questLog, questId)
    : tsStartQuest(questLog, questId)

export const recordQuestObjectiveProgress = (
  questLog: QuestLogState,
  questId: string,
  objectiveId: string,
  amount?: number
): QuestLogState =>
  questLogLua
    ? questLogLua.recordQuestObjectiveProgress(questLog, questId, objectiveId, amount)
    : tsRecordQuestObjectiveProgress(questLog, questId, objectiveId, amount)

export const recordMonsterDefeatQuestProgress = (
  questLog: QuestLogState,
  target: { sceneId: string; appearanceType: string }
): QuestLogState =>
  questLogLua
    ? questLogLua.recordMonsterDefeatQuestProgress(questLog, target)
    : tsRecordMonsterDefeatQuestProgress(questLog, target)

export const recordItemUseQuestProgress = (
  questLog: QuestLogState,
  itemId: string
): QuestLogState =>
  questLogLua
    ? questLogLua.recordItemUseQuestProgress(questLog, itemId)
    : tsRecordItemUseQuestProgress(questLog, itemId)

export const recordShopOpenQuestProgress = (
  questLog: QuestLogState,
  shopId: string
): QuestLogState =>
  questLogLua
    ? questLogLua.recordShopOpenQuestProgress(questLog, shopId)
    : tsRecordShopOpenQuestProgress(questLog, shopId)

export const recordSceneEnterQuestProgress = (
  questLog: QuestLogState,
  sceneId: string
): QuestLogState =>
  questLogLua
    ? questLogLua.recordSceneEnterQuestProgress(questLog, sceneId)
    : tsRecordSceneEnterQuestProgress(questLog, sceneId)

export const recordTalkQuestProgress = (
  questLog: QuestLogState,
  npcId: string
): QuestLogState =>
  questLogLua
    ? questLogLua.recordTalkQuestProgress(questLog, npcId)
    : tsRecordTalkQuestProgress(questLog, npcId)

export const setQuestTrackerVisible = (
  questLog: QuestLogState,
  questId: string,
  trackerVisible: boolean
): QuestLogState =>
  questLogLua
    ? questLogLua.setQuestTrackerVisible(questLog, questId, trackerVisible)
    : tsSetQuestTrackerVisible(questLog, questId, trackerVisible)

export const hideVisibleQuestTrackers = (
  questLog: QuestLogState
): QuestLogState =>
  questLogLua
    ? questLogLua.hideVisibleQuestTrackers(questLog)
    : tsHideVisibleQuestTrackers(questLog)

export const abandonQuest = (
  questLog: QuestLogState,
  questId: string
): QuestLogState =>
  questLogLua
    ? questLogLua.abandonQuest(questLog, questId)
    : tsAbandonQuest(questLog, questId)

export const completeQuest = (
  questLog: QuestLogState,
  questId: string
): CompleteQuestResult =>
  questLogLua
    ? questLogLua.completeQuest(questLog, questId)
    : tsCompleteQuest(questLog, questId)

export const getVisibleQuestTrackers = (
  questLog: QuestLogState
): QuestTrackerItem[] =>
  questLogLua
    ? questLogLua.getVisibleQuestTrackers(questLog)
    : tsGetVisibleQuestTrackers(questLog)

export const getQuestNpcBadgeKind = (
  questLog: QuestLogState,
  questId: string
): QuestNpcBadgeKind | undefined =>
  questLogLua
    ? questLogLua.getQuestNpcBadgeKind(questLog, questId)
    : tsGetQuestNpcBadgeKind(questLog, questId)

export const getQuestNpcBadgeKindForNpc = (
  questLog: QuestLogState,
  npcId: string
): QuestNpcBadgeKind | undefined =>
  questLogLua
    ? questLogLua.getQuestNpcBadgeKindForNpc(questLog, npcId)
    : tsGetQuestNpcBadgeKindForNpc(questLog, npcId)

export const getNextQuestInteractionForNpc = (
  questLog: QuestLogState,
  npcId: string
): NpcQuestInteraction | undefined =>
  questLogLua
    ? questLogLua.getNextQuestInteractionForNpc(questLog, npcId)
    : tsGetNextQuestInteractionForNpc(questLog, npcId)

export const formatQuestText = (
  text: string,
  context: QuestTextFormatContext
): string =>
  questLogLua
    ? questLogLua.formatQuestText(text, context)
    : tsFormatQuestText(text, context)

export const formatQuestTextLines = (
  lines: string[],
  context: QuestTextFormatContext
): string[] =>
  questLogLua
    ? questLogLua.formatQuestTextLines(lines, context)
    : tsFormatQuestTextLines(lines, context)

// ── playerControls ──
export const createInitialPlayerControlBindings = (): PlayerControlBindings =>
  controls
    ? controls.createInitialPlayerControlBindings()
    : tsCreateInitialPlayerControlBindings()

export const normalizeStoredPlayerControlBindings = (
  nextBindings: unknown
): PlayerControlBindings =>
  controls
    ? controls.normalizeStoredPlayerControlBindings(nextBindings)
    : tsNormalizeStoredPlayerControlBindings(nextBindings)

export const setPlayerControlBinding = (input: {
  bindings: PlayerControlBindings
  bindingId: PlayerControlBindingId
  nextCode: string
}): PlayerControlBindings =>
  controls
    ? controls.setPlayerControlBinding(input)
    : tsSetPlayerControlBinding(input)

export const getPlayerControlBindingDisplayText = (code: string): string =>
  controls
    ? controls.getPlayerControlBindingDisplayText(code)
    : tsGetPlayerControlBindingDisplayText(code)

export const getPlayerControlMovementDirectionFromCode = (
  bindings: PlayerControlBindings,
  code: string
): CharacterMoveDirection | undefined =>
  controls
    ? controls.getPlayerControlMovementDirectionFromCode(bindings, code)
    : tsGetPlayerControlMovementDirectionFromCode(bindings, code)

export const getPlayerControlActionFromCode = (
  bindings: PlayerControlBindings,
  code: string
): CharacterAction | undefined =>
  controls
    ? controls.getPlayerControlActionFromCode(bindings, code)
    : tsGetPlayerControlActionFromCode(bindings, code)

export const getPlayerControlQuickslotIndexFromCode = (
  bindings: PlayerControlBindings,
  code: string
): number | undefined =>
  controls
    ? controls.getPlayerControlQuickslotIndexFromCode(bindings, code)
    : tsGetPlayerControlQuickslotIndexFromCode(bindings, code)

export const isPlayerControlPauseKey = (
  bindings: PlayerControlBindings,
  code: string
): boolean =>
  controls
    ? controls.isPlayerControlPauseKey(bindings, code)
    : tsIsPlayerControlPauseKey(bindings, code)

export const isPlayerControlCaptureModifierKey = (code: string): boolean =>
  controls
    ? controls.isPlayerControlCaptureModifierKey(code)
    : tsIsPlayerControlCaptureModifierKey(code)

export const getPlayerControlBindingDefinitionById = (
  bindingId: PlayerControlBindingId
): PlayerControlBindingDefinition =>
  controls
    ? controls.getPlayerControlBindingDefinitionById(bindingId)
    : tsGetPlayerControlBindingDefinitionById(bindingId)

export const getPlayerControlBindingDefinitions =
  (): readonly PlayerControlBindingDefinition[] =>
    controls
      ? controls.getPlayerControlBindingDefinitions()
      : tsGetPlayerControlBindingDefinitions()

export const findPlayerControlBindingIdByCode = (
  bindings: PlayerControlBindings,
  code: string
): PlayerControlBindingId | undefined =>
  controls
    ? controls.findPlayerControlBindingIdByCode(bindings, code)
    : findPlayerControlBindingIdByCodeFallback(bindings, code)

// 원본 findPlayerControlBindingIdByCode 는 모듈 내부 const(미export)라 폴백을 여기 재현한다.
const findPlayerControlBindingIdByCodeFallback = (
  bindings: PlayerControlBindings,
  code: string
): PlayerControlBindingId | undefined => {
  for (const [bindingId, bindingCode] of Object.entries(bindings)) {
    if (bindingCode === code) {
      return bindingId as PlayerControlBindingId
    }
  }

  return undefined
}

export { PLAYER_CONTROL_BINDINGS_STORAGE_KEY } from '../playerControls'

// ── playerSaveState ──
export const serializePlayerSaveState = (
  input: PlayerSaveStateInput
): string =>
  saveState
    ? saveState.serializePlayerSaveState(input)
    : tsSerializePlayerSaveState(input)

export const normalizeStoredPlayerSaveState = (
  value: unknown
): PlayerSaveState | undefined =>
  saveState
    ? saveState.normalizeStoredPlayerSaveState(value)
    : tsNormalizeStoredPlayerSaveState(value)

export const parseStoredPlayerSaveState = (
  raw: string | null | undefined
): PlayerSaveState | undefined =>
  saveState
    ? saveState.parseStoredPlayerSaveState(raw)
    : tsParseStoredPlayerSaveState(raw)

export {
  PLAYER_SAVE_STATE_STORAGE_KEY,
  PLAYER_SAVE_STATE_VERSION
} from '../playerSaveState'

// ── playerRoll ──
export const getPlayerRollDirectionVector = (
  direction: CharacterMoveDirection
): PlayerRollVector =>
  roll
    ? roll.getPlayerRollDirectionVector(direction)
    : tsGetPlayerRollDirectionVector(direction)

export const normalizePlayerRollVector = (
  vector: PlayerRollVector
): PlayerRollVector | undefined =>
  roll
    ? roll.normalizePlayerRollVector(vector)
    : tsNormalizePlayerRollVector(vector)

export const getPlayerRollProgress = (input: {
  nowMilliseconds: number
  startedAtMilliseconds: number
}): number =>
  roll ? roll.getPlayerRollProgress(input) : tsGetPlayerRollProgress(input)

export const getPlayerRollDistanceTiles = (progress: number): number =>
  roll
    ? roll.getPlayerRollDistanceTiles(progress)
    : tsGetPlayerRollDistanceTiles(progress)

export const getPlayerRollVisualState = (input: {
  vector: PlayerRollVector
  progress: number
}): PlayerRollVisualState =>
  roll
    ? roll.getPlayerRollVisualState(input)
    : tsGetPlayerRollVisualState(input)

// ── playerSmashSkill ──
export const getPlayerSmashSkillDirectionVector = (
  facing: CharacterMoveDirection
): { x: number; y: number } =>
  smashSkill
    ? smashSkill.getPlayerSmashSkillDirectionVector(facing)
    : tsGetPlayerSmashSkillDirectionVector(facing)

export const getPlayerSmashSkillSegmentDistancePixels = (
  segmentIndex: number,
  progress = 0
): number =>
  smashSkill
    ? smashSkill.getPlayerSmashSkillSegmentDistancePixels(segmentIndex, progress)
    : tsGetPlayerSmashSkillSegmentDistancePixels(segmentIndex, progress)

export const getPlayerSmashSkillSegmentPlacement = (input: {
  characterCenterX: number
  characterCenterY: number
  facing: CharacterMoveDirection
  segmentIndex: number
  progress?: number
}): PlayerSmashSkillSegmentPlacement =>
  smashSkill
    ? smashSkill.getPlayerSmashSkillSegmentPlacement(input)
    : tsGetPlayerSmashSkillSegmentPlacement(input)

// ── eventGeneration ──
export const createHolidayDialogueEventValidationErrors = (
  event: HolidayDialogueEventSpec
): string[] =>
  eventGeneration
    ? eventGeneration.createHolidayDialogueEventValidationErrors(event)
    : tsCreateHolidayDialogueEventValidationErrors(event)

export const isHolidayDialogueEventValid = (
  event: HolidayDialogueEventSpec
): boolean =>
  eventGeneration
    ? eventGeneration.isHolidayDialogueEventValid(event)
    : tsIsHolidayDialogueEventValid(event)

export const createTiledNpcEventObject = (
  event: HolidayDialogueEventSpec
): TiledNpcEventObject =>
  eventGeneration
    ? eventGeneration.createTiledNpcEventObject(event)
    : tsCreateTiledNpcEventObject(event)

// ── eventDrafting ──
export const createHolidayDialogueEventDraftFromText = (
  prompt: string
): HolidayDialogueEventSpec | undefined =>
  eventDrafting
    ? eventDrafting.createHolidayDialogueEventDraftFromText(prompt)
    : tsCreateHolidayDialogueEventDraftFromText(prompt)

// ── characterState ──
export const createKeyboardCharacterController = (input?: {
  moveSpeedTilesPerSecond?: number
}): KeyboardCharacterController =>
  characterState
    ? characterState.createKeyboardCharacterController(input)
    : tsCreateKeyboardCharacterController(input)

export const createIdleNpcCharacterController = (input?: {
  moveSpeedTilesPerSecond?: number
  dialogueLines?: string[]
  messageDurationMilliseconds?: number
}): NpcCharacterController =>
  characterState
    ? characterState.createIdleNpcCharacterController(input)
    : tsCreateIdleNpcCharacterController(input)

export const createLuaCharacterController = (input: {
  scriptId: string
  radiusInTiles: number
  config?: LuaCharacterControllerConfig
  moveSpeedTilesPerSecond?: number
}): LuaCharacterController =>
  characterState
    ? characterState.createLuaCharacterController(input)
    : tsCreateLuaCharacterController(input)

export const createInitialPlayerCharacter = (input: {
  mapWidth: number
  mapHeight: number
}): CharacterState =>
  characterState
    ? characterState.createInitialPlayerCharacter(input)
    : tsCreateInitialPlayerCharacter(input)

export const createNpcCharacter = (input: {
  id: string
  appearanceType: string
  level?: number
  displayText?: string
  position: { x: number; y: number }
  facing?: CharacterMoveDirection
  collisionSize: { width: number; height: number }
  blocksMovement?: boolean
  controller?: CharacterState['controller']
}): CharacterState =>
  characterState
    ? characterState.createNpcCharacter(input)
    : tsCreateNpcCharacter(input)

export const getCharacterMoveDirectionFromKey = (
  key: string
): CharacterMoveDirection | undefined =>
  characterState
    ? characterState.getCharacterMoveDirectionFromKey(key)
    : tsGetCharacterMoveDirectionFromKey(key)

export const getCharacterActionFromKey = (
  key: string
): CharacterAction | undefined =>
  characterState
    ? characterState.getCharacterActionFromKey(key)
    : tsGetCharacterActionFromKey(key)

export const getCharacterControllerIntent = (input: {
  character: CharacterState
  deltaMilliseconds: number
  pressedDirections?: ReadonlySet<CharacterMoveDirection>
  triggeredActions?: ReadonlySet<CharacterAction>
}): CharacterControllerIntent | undefined =>
  characterState
    ? characterState.getCharacterControllerIntent(input)
    : tsGetCharacterControllerIntent(input)

export const moveCharacterState = (input: {
  character: CharacterState
  delta: { x: number; y: number }
  mapWidth: number
  mapHeight: number
}): CharacterState =>
  characterState
    ? characterState.moveCharacterState(input)
    : tsMoveCharacterState(input)

// ── monsterPatrol (createMonsterPatrolState 만 Lua — stepMonsterPatrol 은 가변 난수라 TS 유지) ──
export const createMonsterPatrolState = (
  character: CharacterState
): MonsterPatrolState =>
  monsterPatrol
    ? monsterPatrol.createMonsterPatrolState(character)
    : tsCreateMonsterPatrolState(character)

// ── sceneIntro ──
export const getSceneIntroMessage = (sceneId: string): string =>
  sceneIntro
    ? sceneIntro.getSceneIntroMessage(sceneId)
    : tsGetSceneIntroMessage(sceneId)
