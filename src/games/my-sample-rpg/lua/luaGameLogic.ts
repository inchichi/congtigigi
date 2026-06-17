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
  PLAYER_BASE_EXPERIENCE_TO_LEVEL_UP,
  PLAYER_EXPERIENCE_TO_LEVEL_UP_PER_LEVEL
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

let host: LuaLogicHost | undefined
let statEffects: PlayerStatEffectsLua | undefined
let inventory: PlayerInventoryLua | undefined
let profile: PlayerProfileLua | undefined
let progression: PlayerProgressionLua | undefined
let equipmentLua: PlayerEquipmentLua | undefined
let potionShop: PlayerPotionShopLua | undefined

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
  equipmentLua = await createPlayerEquipmentLua({ host: next })
  // potionShop 은 인벤토리 전역(inventory_*)에 의존하므로 inventory 로드 이후에 만든다.
  potionShop = await createPlayerPotionShopLua({ host: next })
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
