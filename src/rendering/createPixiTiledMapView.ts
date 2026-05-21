import { CompositeTilemap } from '@pixi/tilemap'
import {
  Application,
  AnimatedSprite,
  Assets,
  Container,
  Graphics,
  NineSliceSprite,
  Rectangle,
  Sprite,
  Text,
  TextStyle,
  Texture,
  UPDATE_PRIORITY
} from 'pixi.js'

import {
  PLAYER_CHARACTER_ID,
  moveCharacterState
} from '../game/characterState'
import type {
  CharacterAction,
  CharacterMoveDirection,
  CharacterState
} from '../game/characterState'
import type { CharacterControllerRuntime } from '../game/createCharacterControllerRuntime'
import {
  createGameEventQueue,
  type GameEvent
} from '../game/events/createGameEventQueue'
import { processInteractionEvents } from '../game/interaction/processInteractionEvents'
import { usePlayerQuickslotConsumable } from '../game/playerConsumables'
import {
  getPlayerEquipmentItemDefinitionById,
  type PlayerEquipment,
  type PlayerEquipmentSlotId
} from '../game/playerEquipment'
import {
  findFirstEmptyPlayerInventorySlotIndex,
  setPlayerInventorySlot,
  type PlayerInventory,
  type PlayerInventoryItem
} from '../game/playerInventory'
import type { PlayerProfile } from '../game/playerProfile'
import {
  clearPlayerQuickslotAssignment,
  type PlayerQuickslots
} from '../game/playerQuickslots'
import {
  createInitialPlayerControlBindings,
  getPlayerControlActionFromCode,
  getPlayerControlMovementDirectionFromCode,
  getPlayerControlQuickslotIndexFromCode,
  isPlayerControlCaptureModifierKey,
  isPlayerControlPauseKey,
  setPlayerControlBinding,
  type PlayerControlBindingId,
  type PlayerControlBindings
} from '../game/playerControls'
import {
  QUEST_GIVER_NPC_IDS,
  completeQuest,
  formatQuestTextLines,
  getNextQuestInteractionForNpc,
  getQuestNpcBadgeKindForNpc,
  recordItemUseQuestProgress,
  recordMonsterDefeatQuestProgress,
  recordShopOpenQuestProgress,
  recordTalkQuestProgress,
  startQuest,
  type CompleteQuestResult,
  type QuestItemReward,
  type QuestLogState
} from '../game/questLog'
import {
  getPlayerMovementSpeedTilesPerSecond,
  getPlayerPhysicalAttackPower,
  shouldPlayerEvadeDamage
} from '../game/playerStatEffects'
import { grantPlayerExperience } from '../game/playerExperience'
import { rollMonsterEquipmentDrop } from '../game/monsterEquipmentDrops'
import {
  createMonsterPatrolState,
  stepMonsterPatrol,
  type MonsterPatrolState
} from '../game/monsterPatrol'
import {
  applyMonsterDamage,
  createMonsterCombatState,
  isMonsterDefeated,
  type MonsterCombatState
} from '../game/monsterCombat'
import {
  getMonsterExperienceDropAmount,
  getMonsterGoldDropAmount
} from '../game/monsterRewards'
import { resolveCharacterInteractionTarget } from '../game/interaction/resolveCharacterInteractionTarget'
import {
  createMapPortalsFromEventLayers,
  type MapPortal
} from '../game/tiled/createMapPortalsFromEventLayers'
import {
  createWallTileLookup,
  isWallTileAt
} from '../game/tiled/createWallTileLookup'
import type {
  ParsedTiledMap,
  ParsedTiledTile,
  ParsedTiledTileset
} from '../game/tiled/parseTiledMap'
import {
  getSpriteTransformForTile,
  hasTileTransform
} from './tiledSpriteTransform'
import { createMapOverlay } from './createMapOverlay'
import { createBlacksmithShopOverlay } from './createBlacksmithShopOverlay'
import { createPotionShopOverlay } from './createPotionShopOverlay'
import { createPlayerEquipmentOverlay } from './createPlayerEquipmentOverlay'
import { createPlayerHudOverlay } from './createPlayerHudOverlay'
import { createPlayerInventoryOverlay } from './createPlayerInventoryOverlay'
import { createPlayerStatOverlay } from './createPlayerStatOverlay'
import { createPlayerSkillOverlay } from './createPlayerSkillOverlay'
import type { MonsterAnimationTextures } from './monsterAnimationTextures'
import { loadMonsterPigAnimationTextures } from './loadMonsterPigAnimationTextures'
import { loadMonsterSlimeAnimationTextures } from './loadMonsterSlimeAnimationTextures'
import { createGameSoundEffects } from './createGameSoundEffects'
import {
  createPauseMenuOverlay,
  type AudioSettings
} from './createPauseMenuOverlay'
import { createQuestLogOverlay } from './createQuestLogOverlay'
import { createQuestTrackerOverlay } from './createQuestTrackerOverlay'

type CreatePixiTiledMapViewInput = {
  mountElement: HTMLElement
  map: ParsedTiledMap
  characters: CharacterState[]
  playerProfile: PlayerProfile
  playerEquipment: PlayerEquipment
  playerInventory: PlayerInventory
  playerQuickslots: PlayerQuickslots
  playerControlBindings: PlayerControlBindings
  questLog: QuestLogState
  merchantInventory: PlayerInventory
  potionMerchantInventory: PlayerInventory
  sceneId: string
  sceneIntroMessage: string
  cameraTargetCharacterId: string
  characterSpriteSheet: {
    tileset: ParsedTiledTileset
    scale: number
  }
  imageUrls: Record<string, string>
  controllerRuntime: CharacterControllerRuntime
  onPlayerInventoryChange: (nextInventory: PlayerInventory) => void
  onPlayerEquipmentChange: (nextEquipment: PlayerEquipment) => void
  onPlayerQuickslotsChange: (nextQuickslots: PlayerQuickslots) => void
  onPlayerControlBindingsChange: (
    nextControlBindings: PlayerControlBindings
  ) => void
  onQuestLogChange: (nextQuestLog: QuestLogState) => void
  onMerchantInventoryChange: (nextInventory: PlayerInventory) => void
  onPotionMerchantInventoryChange: (nextInventory: PlayerInventory) => void
  audioSettings: AudioSettings
  onAudioSettingsChange: (nextAudioSettings: AudioSettings) => void
  onRequestSceneChange: (request: SceneTransitionRequest) => void
}

export type SceneTransitionRequest = {
  sceneId: string
  spawn: {
    x: number
    y: number
  }
  facing?: CharacterMoveDirection
}

type TilesetRenderResources = {
  imageTexture: Texture
  tileTextures: Texture[]
}

type SlashVfxRenderResources = {
  horizontalTextures: Texture[]
  verticalTextures: Texture[]
}

type ResolvedCharacterAppearanceTexture = {
  texture: Texture
  renderScale: number
}

type TileTextureFrameSource = {
  columns: number
  margin: number
  spacing: number
  tileWidth: number
  tileHeight: number
}

type CollisionRect = {
  x: number
  y: number
  width: number
  height: number
}

type ActiveCharacterMessage = {
  container: Container
  panel: NineSliceSprite
  text: Text
  expiresAt: number
}

type ActiveCharacterDamageText = {
  container: Container
  text: Text
  startedAt: number
  durationMilliseconds: number
  expiresAt: number
}

type MonsterGoldDrop = {
  id: string
  container: Container
  coin: Graphics
  amountText: Text
  amount: number
  position: {
    x: number
    y: number
  }
  createdAt: number
}

type MonsterEquipmentDrop = {
  id: string
  dropId: string
  itemId: string
  label: string
  container: Container
  sprite: Sprite
  labelText: Text
  position: {
    x: number
    y: number
  }
  createdAt: number
}

type RenderedCharacterNode = {
  container: Container
  sprite: Sprite
  playerArmorSprite?: Sprite
  playerHelmetSprite?: Sprite
  questBadge?: Sprite
  playerHealthBar?: {
    container: Container
    track: Graphics
    fill: Graphics
  }
  playerManaBar?: {
    container: Container
    track: Graphics
    fill: Graphics
  }
  playerNameBadge?: Text
  displayLabelPanel?: NineSliceSprite
  displayLabel?: Text
  levelBadge?: Text
  monsterHealthBar?: {
    container: Container
    track: Graphics
    fill: Graphics
  }
}

type RenderedPortalNode = {
  container: Container
  sprite: Sprite
}

type MonsterPigAnimationMode = 'idle' | 'run' | 'hit' | 'attack'

type MonsterPigBehaviorState = {
  isAggroed: boolean
  nextAttackAtMilliseconds: number
  attackUntilMilliseconds: number
  hitReactionUntilMilliseconds: number
}

type MonsterBehaviorConfig = {
  renderScale: number
  aggroRangeTiles: number
  deAggroRangeTiles: number
  chaseSpeedTilesPerSecond: number
  patrolSpeedTilesPerSecond: number
  attackRangeTiles: number
  attackIntervalMilliseconds: number
  attackDurationMilliseconds: number
  hitReactionDurationMilliseconds: number
  idleAnimationSpeed: number
  runAnimationSpeed: number
  hitAnimationSpeed: number
  attackAnimationSpeed: number
  usesRunAnimation: boolean
  runMotionBobPixels: number
  runMotionSwayPixels: number
}

type PlayerHitReactionState = {
  directionX: number
  directionY: number
  startedAtMilliseconds: number
  expiresAtMilliseconds: number
}

type PlayerVisualEquipmentSlotId = Extract<PlayerEquipmentSlotId, 'armor' | 'hat'>

type PlayerEquipmentAppearanceConfig = {
  slotId: PlayerVisualEquipmentSlotId
  imageUrl: string
  width: number
  height: number
  position: {
    x: number
    y: number
  }
  zIndex: number
}

type PlayerWeaponAppearanceConfig = {
  imageUrl: string
  worldScale: number
  idleOffsetX: number
  idleOffsetY: number
}

const DEPTH_SORTED_LAYER_NAME = 'object'
const TINY_DUNGEON_TILESET_IMAGE_URL = new URL(
  '../assets/tilesets/tiny-dungeon-16.png',
  import.meta.url
).href
const MONSTER_EQUIPMENT_DROP_IMAGE_URL_BY_DROP_ID: Record<string, string> = {
  'iron-sword_drop': new URL(
    '../assets/weapons/weapon-sword.png',
    import.meta.url
  ).href,
  'battle-axe_drop': new URL(
    '../assets/weapons/weapon-axe.png',
    import.meta.url
  ).href,
  'long-spear_drop': new URL(
    '../assets/weapons/weapon-spear.png',
    import.meta.url
  ).href,
  'quick-dagger_drop': new URL(
    '../assets/weapons/weapon-dagger.png',
    import.meta.url
  ).href,
  'spiked-mace_drop': new URL(
    '../assets/weapons/weapon-mace.png',
    import.meta.url
  ).href,
  'magic-staff_drop': new URL(
    '../assets/weapons/weapon-staff.png',
    import.meta.url
  ).href,
  Leather_Armor_drop: new URL(
    '../assets/armor/dropimage/Leather_Armor_drop.png',
    import.meta.url
  ).href,
  Leather_Helmet_drop: new URL(
    '../assets/armor/dropimage/Leather_Helmet_drop.png',
    import.meta.url
  ).href,
  Chain_Armor_drop: new URL(
    '../assets/armor/dropimage/Chain_Armor_drop.png',
    import.meta.url
  ).href,
  Chain_Helmet_drop: new URL(
    '../assets/armor/dropimage/Chain_Helmet_drop.png',
    import.meta.url
  ).href,
  Iron_Armor_drop: new URL(
    '../assets/armor/dropimage/Iron_Armor_drop.png',
    import.meta.url
  ).href,
  Iron_Helmet_drop: new URL(
    '../assets/armor/dropimage/Iron_Helmet_drop.png',
    import.meta.url
  ).href
}
const MESSAGE_PANEL_BORDER_SIZE = 8
const MESSAGE_PANEL_PADDING_X = 12
const MESSAGE_PANEL_PADDING_Y = 8
const MESSAGE_PANEL_MIN_WIDTH = 64
const MESSAGE_PANEL_MIN_HEIGHT = 28
const MESSAGE_TEXT_MAX_WIDTH = 188
const MESSAGE_OFFSET_Y = 10
const MESSAGE_TEXT_STYLE = new TextStyle({
  align: 'center',
  breakWords: true,
  fill: 0x2e2313,
  fontFamily: '"Jersey 25", NeoDunggeunmo, monospace',
  fontSize: 14,
  lineHeight: 18,
  padding: 2,
  wordWrap: true,
  wordWrapWidth: MESSAGE_TEXT_MAX_WIDTH
})
const DAMAGE_TEXT_STYLE = new TextStyle({
  align: 'center',
  fill: 0xff5b5b,
  fontFamily: '"Jersey 25", NeoDunggeunmo, monospace',
  fontSize: 16,
  lineHeight: 18,
  stroke: {
    color: 0x2a0909,
    width: 3
  }
})
const EVADE_TEXT_STYLE = new TextStyle({
  align: 'center',
  fill: 0xe6f79c,
  fontFamily: '"Jersey 25", NeoDunggeunmo, monospace',
  fontSize: 16,
  lineHeight: 18,
  stroke: {
    color: 0x324b12,
    width: 3
  }
})
const LEVEL_UP_TEXT_STYLE = new TextStyle({
  align: 'center',
  fill: 0xffdf7a,
  fontFamily: '"Jersey 25", NeoDunggeunmo, monospace',
  fontSize: 16,
  lineHeight: 18,
  stroke: {
    color: 0x3b2600,
    width: 3
  }
})
const BLACKSMITH_SHOP_NPC_ID = 'blacksmith'
const POTION_SHOP_NPC_ID = 'potion_merchant'
const SIGN_POST_APPEARANCE_TYPE = 'sign_inn'
const MONSTER_PIG_APPEARANCE_TYPE = 'monster_pig'
const MONSTER_SLIME_APPEARANCE_TYPE = 'monster_slime'
const GROUND_LAYER_NAME = 'ground'
const GRASS_TILE_TYPES = new Set(['garden_round_mid_01'])
const GAME_VIEWPORT_WIDTH = 960
const GAME_VIEWPORT_HEIGHT = 540
const CAMERA_DEFAULT_ZOOM = 1.1
const CAMERA_MIN_ZOOM = 0.8
const CAMERA_MAX_ZOOM = 2
const CAMERA_ZOOM_WHEEL_SPEED = 0.0015
const MONSTER_PIG_WORLD_SCALE = 0.315
const MONSTER_SLIME_WORLD_SCALE = 0.287
const MONSTER_PIG_CHASE_SPEED_TILES_PER_SECOND = 4.4
const MONSTER_PIG_IDLE_ANIMATION_SPEED = 0.08
const MONSTER_PIG_RUN_ANIMATION_SPEED = 0.22
const MONSTER_PIG_HIT_ANIMATION_SPEED = 0.18
const MONSTER_PIG_ATTACK_ANIMATION_SPEED = 0.14
const MONSTER_PIG_ATTACK_INTERVAL_MILLISECONDS = 5000
const MONSTER_PIG_ATTACK_DURATION_MILLISECONDS = 720
const MONSTER_PIG_ATTACK_RANGE_TILES = 1.2
const MONSTER_PIG_AGGRO_RANGE_TILES = 4.8
const MONSTER_PIG_DE_AGGRO_RANGE_TILES = 7.2
const MONSTER_PIG_HIT_REACTION_DURATION_MILLISECONDS = 260
const MONSTER_PIG_RESPAWN_DELAY_MILLISECONDS = 8000
const MONSTER_CONTACT_DAMAGE_COOLDOWN_MILLISECONDS = 900
const PLAYER_ATTACK_PROBE_DISTANCE_IN_TILES = 1.2
const DAMAGE_TEXT_FLOAT_DISTANCE = 16
const DAMAGE_TEXT_DURATION_MILLISECONDS = 1000
const EVADE_TEXT_DURATION_MILLISECONDS = 700
const DAMAGE_TEXT_OFFSET_Y = 8
const MONSTER_CONTACT_DAMAGE_TOUCH_TOLERANCE_TILES = 0.14
const MONSTER_ATTACK_RANGE_TOUCH_TOLERANCE_TILES = 0.14
const PLAYER_RESPAWN_DELAY_MILLISECONDS = 3000
const PLAYER_HIT_REACTION_DURATION_MILLISECONDS = 180
const PLAYER_HIT_REACTION_MAX_OFFSET_PIXELS = 6
const MONSTER_GOLD_DROP_ICON_RADIUS = 7
const MONSTER_GOLD_DROP_ICON_SHINE_RADIUS = 2
const MONSTER_GOLD_DROP_AMOUNT_TEXT_STYLE = new TextStyle({
  align: 'center',
  fill: 0xffd86b,
  fontFamily: '"Jersey 25", NeoDunggeunmo, monospace',
  fontSize: 12,
  lineHeight: 14,
  stroke: {
    color: 0x4f3200,
    width: 3
  }
})
const MONSTER_GOLD_DROP_PICKUP_WIDTH = 14
const MONSTER_GOLD_DROP_PICKUP_HEIGHT = 14
const MONSTER_EQUIPMENT_DROP_RENDER_SIZE = 24
const MONSTER_EQUIPMENT_DROP_PICKUP_WIDTH = 20
const MONSTER_EQUIPMENT_DROP_PICKUP_HEIGHT = 20
const MONSTER_LEVEL_BADGE_STYLE = new TextStyle({
  align: 'center',
  fill: 0xf4e7c5,
  fontFamily: '"Jersey 25", NeoDunggeunmo, monospace',
  fontSize: 12,
  stroke: {
    color: 0x2e2313,
    width: 3
  }
})
const PLAYER_NAME_BADGE_STYLE = new TextStyle({
  align: 'center',
  fill: 0xf4e7c5,
  fontFamily: '"Jersey 25", NeoDunggeunmo, monospace',
  fontSize: 12,
  lineHeight: 13,
  padding: 1,
  stroke: {
    color: 0x2e2313,
    width: 3
  }
})
const PLAYER_HEALTH_BAR_WIDTH = 38
const PLAYER_HEALTH_BAR_HEIGHT = 5
const PLAYER_HEALTH_BAR_TRACK_COLOR = 0x2e2313
const PLAYER_HEALTH_BAR_FILL_COLOR = 0xd06b5d
const PLAYER_HEALTH_BAR_BORDER_COLOR = 0xf4e7c5
const PLAYER_HEALTH_BAR_GAP = 5
const PLAYER_MANA_BAR_WIDTH = 38
const PLAYER_MANA_BAR_HEIGHT = 5
const PLAYER_MANA_BAR_TRACK_COLOR = 0x2e2313
const PLAYER_MANA_BAR_FILL_COLOR = 0x5b86d6
const PLAYER_MANA_BAR_BORDER_COLOR = 0xf4e7c5
const PLAYER_MANA_BAR_GAP = 2
const SIGN_POST_LABEL_STYLE = new TextStyle({
  align: 'center',
  breakWords: true,
  fill: 0xf4e7c5,
  fontFamily: '"Jersey 25", NeoDunggeunmo, monospace',
  fontSize: 9,
  lineHeight: 10,
  padding: 0,
  stroke: {
    color: 0x2e2313,
    width: 2
  },
  wordWrap: true,
  wordWrapWidth: 128
})
const MONSTER_HEALTH_BAR_WIDTH = 34
const MONSTER_HEALTH_BAR_HEIGHT = 5
const MONSTER_HEALTH_BAR_TRACK_COLOR = 0x2e2313
const MONSTER_HEALTH_BAR_FILL_COLOR = 0x7dc96d
const MONSTER_HEALTH_BAR_BORDER_COLOR = 0xf4e7c5
const MONSTER_HEALTH_BAR_GAP = 4
const PLAYER_WEAPON_TILE_LOCAL_ID = 117
const PLAYER_WEAPON_TILE_FRAME_SOURCE: TileTextureFrameSource = {
  columns: 12,
  margin: 0,
  spacing: 0,
  tileWidth: 16,
  tileHeight: 16
}
const WHITE_SLASH_WIDE_FRAME_URLS = [
  new URL(
    '../assets/vfx/Sword Slashes/White Slash Wide/File1.png',
    import.meta.url
  ).href,
  new URL(
    '../assets/vfx/Sword Slashes/White Slash Wide/File2.png',
    import.meta.url
  ).href,
  new URL(
    '../assets/vfx/Sword Slashes/White Slash Wide/File3.png',
    import.meta.url
  ).href,
  new URL(
    '../assets/vfx/Sword Slashes/White Slash Wide/File4.png',
    import.meta.url
  ).href,
  new URL(
    '../assets/vfx/Sword Slashes/White Slash Wide/File5.png',
    import.meta.url
  ).href,
  new URL(
    '../assets/vfx/Sword Slashes/White Slash Wide/File6.png',
    import.meta.url
  ).href
]
const WHITE_SLASH_WIDE_FRAME_BOUNDS: CollisionRect[] = [
  { x: 63, y: 25, width: 465, height: 345 },
  { x: 75, y: 165, width: 463, height: 196 },
  { x: 33, y: 285, width: 373, height: 87 },
  { x: 11, y: 186, width: 119, height: 181 },
  { x: 11, y: 103, width: 41, height: 212 },
  { x: 29, y: 63, width: 23, height: 55 }
]
const SLASH_VFX_HIT_PADDING_PIXELS = 2
const PLAYER_WEAPON_WORLD_SCALE = 1.35
const PLAYER_ATTACK_TRAIL_PROGRESS_STEP = 0.12
const PLAYER_ATTACK_TRAIL_ALPHA = [0.42, 0.28, 0.18, 0.1]
const PLAYER_ATTACK_TRAIL_SPRITE_COUNT = PLAYER_ATTACK_TRAIL_ALPHA.length
const PLAYER_ATTACK_SWING_X_OFFSET = 4
const PLAYER_ATTACK_LIFT_Y_OFFSET = 3
const PLAYER_NAME_BADGE_FOOT_OFFSET = 6
const PLAYER_STATUS_STACK_CLEARANCE = 6
const PLAYER_ATTACK_ROTATION_OFFSET = 1.15
const PLAYER_ATTACK_SCALE_BOOST = 0.06
const PLAYER_ATTACK_SLASH_EFFECT_SCALE_X = 0.23
const PLAYER_ATTACK_SLASH_EFFECT_SCALE_Y = 0.23
const PLAYER_ATTACK_SLASH_EFFECT_ANIMATION_SPEED = 0.6
const PLAYER_WEAPON_PLACEMENT_RIGHT = {
  x: 23,
  y: 21,
  rotation: 0.75
}
const PLAYER_WEAPON_PLACEMENT_LEFT = {
  x: 9,
  y: 21,
  rotation: -0.75
}
const PLAYER_WEAPON_APPEARANCE_CONFIG_BY_ITEM_ID: Record<
  string,
  PlayerWeaponAppearanceConfig
> = {
  'iron-sword': {
    imageUrl: new URL('../assets/weapons/weapon-sword.png', import.meta.url).href,
    worldScale: 0.085,
    idleOffsetX: -3,
    idleOffsetY: 2
  },
  'battle-axe': {
    imageUrl: new URL('../assets/weapons/weapon-axe.png', import.meta.url).href,
    worldScale: 0.085,
    idleOffsetX: -5,
    idleOffsetY: 4
  },
  'long-spear': {
    imageUrl: new URL('../assets/weapons/weapon-spear.png', import.meta.url).href,
    worldScale: 0.085,
    idleOffsetX: -2,
    idleOffsetY: 3
  },
  'quick-dagger': {
    imageUrl: new URL('../assets/weapons/weapon-dagger.png', import.meta.url).href,
    worldScale: 0.085,
    idleOffsetX: -2,
    idleOffsetY: 1
  },
  'spiked-mace': {
    imageUrl: new URL('../assets/weapons/weapon-mace.png', import.meta.url).href,
    worldScale: 0.085,
    idleOffsetX: -4,
    idleOffsetY: 4
  },
  'magic-staff': {
    imageUrl: new URL('../assets/weapons/weapon-staff.png', import.meta.url).href,
    worldScale: 0.085,
    idleOffsetX: -2,
    idleOffsetY: 3
  }
}
const PLAYER_ARMOR_EQUIPMENT_CONFIG = {
  width: 24,
  height: 9,
  position: {
    x: 16,
    y: 24
  },
  zIndex: 12
}
const PLAYER_HELMET_EQUIPMENT_CONFIG = {
  width: 22,
  height: 15,
  position: {
    x: 16,
    y: 9
  },
  zIndex: 13
}
const PLAYER_EQUIPMENT_APPEARANCE_CONFIG_BY_ITEM_ID: Record<
  string,
  PlayerEquipmentAppearanceConfig
> = {
  Leather_Armor: {
    slotId: 'armor',
    imageUrl: new URL('../assets/armor/Leather_Armor.png', import.meta.url).href,
    ...PLAYER_ARMOR_EQUIPMENT_CONFIG
  },
  Leather_Helmet: {
    slotId: 'hat',
    imageUrl: new URL('../assets/armor/Leather_Helmet.png', import.meta.url).href,
    ...PLAYER_HELMET_EQUIPMENT_CONFIG
  },
  Chain_Armor: {
    slotId: 'armor',
    imageUrl: new URL('../assets/armor/Chain_Armor.png', import.meta.url).href,
    ...PLAYER_ARMOR_EQUIPMENT_CONFIG
  },
  Chain_Helmet: {
    slotId: 'hat',
    imageUrl: new URL('../assets/armor/Chain_Helmet.png', import.meta.url).href,
    ...PLAYER_HELMET_EQUIPMENT_CONFIG
  },
  Iron_Armor: {
    slotId: 'armor',
    imageUrl: new URL('../assets/armor/Iron_Armor.png', import.meta.url).href,
    ...PLAYER_ARMOR_EQUIPMENT_CONFIG
  },
  Iron_Helmet: {
    slotId: 'hat',
    imageUrl: new URL('../assets/armor/Iron_Helmet.png', import.meta.url).href,
    ...PLAYER_HELMET_EQUIPMENT_CONFIG
  }
}
const PORTAL_INSIDE_IMAGE_URL = new URL(
  '../assets/tilesets/portal_inside.png',
  import.meta.url
).href
const PORTAL_INSIDE_WORLD_SCALE = 0.08
const SCENE_INTRO_VISIBLE_DURATION_MILLISECONDS = 3000
const PLAYER_ATTACK_DURATION_MILLISECONDS = 320
const PLAYER_ATTACK_COOLDOWN_MILLISECONDS = 300
const QUEST_DIALOGUE_DURATION_MILLISECONDS = 3600
const QUEST_START_TEXT = '퀘스트 시작!'
const QUEST_OBJECTIVE_COMPLETE_TEXT = '퀘스트 목표 완료!'
const QUEST_COMPLETE_TEXT = '퀘스트 완료!'
const QUEST_BADGE_SCALE = 0.16
const QUEST_BADGE_Y_OFFSET = 10
type MonsterAppearanceType =
  | typeof MONSTER_PIG_APPEARANCE_TYPE
  | typeof MONSTER_SLIME_APPEARANCE_TYPE

const MONSTER_BEHAVIOR_CONFIG_BY_APPEARANCE_TYPE: Record<
  MonsterAppearanceType,
  MonsterBehaviorConfig
> = {
  [MONSTER_PIG_APPEARANCE_TYPE]: {
    renderScale: MONSTER_PIG_WORLD_SCALE,
    aggroRangeTiles: MONSTER_PIG_AGGRO_RANGE_TILES,
    deAggroRangeTiles: MONSTER_PIG_DE_AGGRO_RANGE_TILES,
    chaseSpeedTilesPerSecond: MONSTER_PIG_CHASE_SPEED_TILES_PER_SECOND,
    patrolSpeedTilesPerSecond: 2.4,
    attackRangeTiles: MONSTER_PIG_ATTACK_RANGE_TILES,
    attackIntervalMilliseconds: MONSTER_PIG_ATTACK_INTERVAL_MILLISECONDS,
    attackDurationMilliseconds: MONSTER_PIG_ATTACK_DURATION_MILLISECONDS,
    hitReactionDurationMilliseconds: MONSTER_PIG_HIT_REACTION_DURATION_MILLISECONDS,
    idleAnimationSpeed: MONSTER_PIG_IDLE_ANIMATION_SPEED,
    runAnimationSpeed: MONSTER_PIG_RUN_ANIMATION_SPEED,
    hitAnimationSpeed: MONSTER_PIG_HIT_ANIMATION_SPEED,
    attackAnimationSpeed: MONSTER_PIG_ATTACK_ANIMATION_SPEED,
    usesRunAnimation: true,
    runMotionBobPixels: 0,
    runMotionSwayPixels: 0
  },
  [MONSTER_SLIME_APPEARANCE_TYPE]: {
    renderScale: MONSTER_SLIME_WORLD_SCALE,
    aggroRangeTiles: 4.4,
    deAggroRangeTiles: 6.8,
    chaseSpeedTilesPerSecond: 3.1,
    patrolSpeedTilesPerSecond: 1.8,
    attackRangeTiles: 1.0,
    attackIntervalMilliseconds: 5400,
    attackDurationMilliseconds: 760,
    hitReactionDurationMilliseconds: 240,
    idleAnimationSpeed: 0.06,
    runAnimationSpeed: 0.16,
    hitAnimationSpeed: 0.16,
    attackAnimationSpeed: 0.12,
    usesRunAnimation: true,
    runMotionBobPixels: 0,
    runMotionSwayPixels: 0
  }
}

const getMonsterBehaviorConfig = (
  character: CharacterState
): MonsterBehaviorConfig =>
  MONSTER_BEHAVIOR_CONFIG_BY_APPEARANCE_TYPE[
    character.appearanceType as MonsterAppearanceType
  ] ?? MONSTER_BEHAVIOR_CONFIG_BY_APPEARANCE_TYPE[MONSTER_PIG_APPEARANCE_TYPE]

const createMonsterHealthBar = (): NonNullable<
  RenderedCharacterNode['monsterHealthBar']
> => {
  const container = new Container()
  const track = new Graphics()
  const fill = new Graphics()

  container.sortableChildren = true
  track.roundPixels = true
  fill.roundPixels = true
  track.zIndex = 0
  fill.zIndex = 1
  container.addChild(track, fill)

  return {
    container,
    track,
    fill
  }
}

const createQuestBadgeSprite = (texture: Texture): Sprite => {
  const sprite = new Sprite(texture)

  sprite.anchor.set(0.5, 1)
  sprite.scale.set(QUEST_BADGE_SCALE)
  sprite.roundPixels = true
  sprite.visible = false

  return sprite
}

const createPlayerResourceBar = (): NonNullable<
  RenderedCharacterNode['playerHealthBar']
> => {
  const container = new Container()
  const track = new Graphics()
  const fill = new Graphics()

  container.sortableChildren = true
  track.roundPixels = true
  fill.roundPixels = true
  track.zIndex = 0
  fill.zIndex = 1
  container.addChild(track, fill)

  return {
    container,
    track,
    fill
  }
}

let messageFontsReadyPromise: Promise<void> | undefined

export const createPixiTiledMapView = async ({
  mountElement,
  map,
  characters,
  playerProfile,
  playerEquipment,
  playerInventory,
  playerQuickslots,
  playerControlBindings,
  questLog,
  merchantInventory,
  potionMerchantInventory,
  sceneId,
  sceneIntroMessage,
  cameraTargetCharacterId,
  characterSpriteSheet,
  imageUrls,
  controllerRuntime,
  onPlayerInventoryChange,
  onPlayerEquipmentChange,
  onPlayerQuickslotsChange,
  onPlayerControlBindingsChange,
  onQuestLogChange,
  onMerchantInventoryChange,
  onPotionMerchantInventoryChange,
  audioSettings,
  onAudioSettingsChange,
  onRequestSceneChange
}: CreatePixiTiledMapViewInput): Promise<{ destroy: () => void }> => {
  const app = new Application()
  let cameraZoom = CAMERA_DEFAULT_ZOOM
  let scaledMapPixelWidth = Math.round(map.pixelWidth * cameraZoom)
  let scaledMapPixelHeight = Math.round(map.pixelHeight * cameraZoom)
  const [
    portalInsideTexture,
    tinyDungeonWeaponImageTexture,
    slashVfxTextures,
    monsterPigAnimationTextures,
    monsterSlimeAnimationTextures
  ] = await Promise.all([
    Assets.load<Texture>(PORTAL_INSIDE_IMAGE_URL),
    Assets.load<Texture>(TINY_DUNGEON_TILESET_IMAGE_URL),
    loadSlashVfxTextures(),
    loadMonsterPigAnimationTextures(),
    loadMonsterSlimeAnimationTextures()
  ])
  const playerWeaponAppearanceTexturesByItemId = new Map(
    await Promise.all(
      Object.entries(PLAYER_WEAPON_APPEARANCE_CONFIG_BY_ITEM_ID).map(
        async ([itemId, config]) => [
          itemId,
          await Assets.load<Texture>(config.imageUrl)
        ] as const
      )
    )
  )
  const monsterEquipmentDropTexturesByDropId = new Map(
    await Promise.all(
      Object.entries(MONSTER_EQUIPMENT_DROP_IMAGE_URL_BY_DROP_ID).map(
        async ([dropId, imageUrl]) => [
          dropId,
          await Assets.load<Texture>(imageUrl)
        ] as const
      )
    )
  )
  const playerEquipmentTexturesByItemId = new Map(
    await Promise.all(
      Object.entries(PLAYER_EQUIPMENT_APPEARANCE_CONFIG_BY_ITEM_ID).map(
        async ([itemId, config]) => [
          itemId,
          await Assets.load<Texture>(config.imageUrl)
        ] as const
      )
    )
  )
  const messagePanelTexture = createMessagePanelTexture()

  const monsterAnimationTexturesByAppearanceType: Record<
    MonsterAppearanceType,
    MonsterAnimationTextures
  > = {
    [MONSTER_PIG_APPEARANCE_TYPE]: monsterPigAnimationTextures,
    [MONSTER_SLIME_APPEARANCE_TYPE]: monsterSlimeAnimationTextures
  }

  tinyDungeonWeaponImageTexture.source.scaleMode = 'nearest'
  tinyDungeonWeaponImageTexture.source.addressMode = 'clamp-to-edge'
  portalInsideTexture.source.scaleMode = 'nearest'
  portalInsideTexture.source.addressMode = 'clamp-to-edge'
  for (const texture of playerWeaponAppearanceTexturesByItemId.values()) {
    texture.source.scaleMode = 'nearest'
    texture.source.addressMode = 'clamp-to-edge'
  }
  for (const texture of monsterEquipmentDropTexturesByDropId.values()) {
    texture.source.scaleMode = 'nearest'
    texture.source.addressMode = 'clamp-to-edge'
  }
  for (const texture of playerEquipmentTexturesByItemId.values()) {
    texture.source.scaleMode = 'nearest'
    texture.source.addressMode = 'clamp-to-edge'
  }
  await ensureMessageFontsLoaded()

  await app.init({
    antialias: false,
    autoDensity: true,
    backgroundColor: 0x171311,
    height: scaledMapPixelHeight,
    preference: 'webgl',
    roundPixels: true,
    resolution: window.devicePixelRatio || 1,
    width: scaledMapPixelWidth
  })
  app.ticker.maxFPS = 60

  const viewportElement = document.createElement('div')
  const sceneElement = document.createElement('div')
  const runtimeWarningBannerElement = document.createElement('div')
  const sceneIntroBannerElement = document.createElement('div')
  const sceneIntroPanelElement = document.createElement('div')
  const sceneIntroTextElement = document.createElement('div')

  viewportElement.className = 'game-viewport'
  sceneElement.className = 'game-scene'
  sceneElement.style.width = `${scaledMapPixelWidth}px`
  sceneElement.style.height = `${scaledMapPixelHeight}px`
  sceneElement.append(app.canvas)
  viewportElement.append(sceneElement)
  mountElement.replaceChildren(viewportElement)
  app.canvas.classList.add('game-canvas')

  runtimeWarningBannerElement.setAttribute('role', 'alert')
  runtimeWarningBannerElement.setAttribute('aria-live', 'polite')
  runtimeWarningBannerElement.hidden = true
  runtimeWarningBannerElement.style.position = 'fixed'
  runtimeWarningBannerElement.style.top = '12px'
  runtimeWarningBannerElement.style.left = '50%'
  runtimeWarningBannerElement.style.transform = 'translateX(-50%)'
  runtimeWarningBannerElement.style.zIndex = '9999'
  runtimeWarningBannerElement.style.maxWidth = 'min(720px, calc(100vw - 24px))'
  runtimeWarningBannerElement.style.padding = '8px 12px'
  runtimeWarningBannerElement.style.border = '1px solid #d94b4b'
  runtimeWarningBannerElement.style.background = '#fff1f1'
  runtimeWarningBannerElement.style.color = '#7a1f1f'
  runtimeWarningBannerElement.style.fontFamily =
    'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace'
  runtimeWarningBannerElement.style.fontSize = '0.7rem'
  runtimeWarningBannerElement.style.whiteSpace = 'pre-wrap'
  runtimeWarningBannerElement.style.pointerEvents = 'none'
  document.body.append(runtimeWarningBannerElement)

  sceneIntroBannerElement.className = 'scene-intro-overlay'
  sceneIntroBannerElement.setAttribute('aria-hidden', 'true')
  sceneIntroPanelElement.className = 'scene-intro-overlay__panel'
  sceneIntroTextElement.className = 'scene-intro-overlay__message'
  sceneIntroTextElement.textContent = sceneIntroMessage
  sceneIntroPanelElement.append(sceneIntroTextElement)
  sceneIntroBannerElement.append(sceneIntroPanelElement)
  mountElement.append(sceneIntroBannerElement)

  const world = new Container()
  world.scale.set(cameraZoom)
  const messageLayer = new Container()
  const tilesetResources = new Map<string, TilesetRenderResources>()
  const wallTiles = createWallTileLookup(map)
  const pressedDirections = new Set<CharacterMoveDirection>()
  const pressedActions = new Set<CharacterAction>()
  const triggeredActions = new Set<CharacterAction>()
  const gameEventQueue = createGameEventQueue()
  let currentAudioSettings = audioSettings
  const gameSoundEffects = createGameSoundEffects({
    masterVolume: currentAudioSettings.sfxVolume
  })
  const interactionLockUntilByCharacterPair = new Map<string, number>()
  const activeCharacterMessages = new Map<string, ActiveCharacterMessage>()
  const activeCharacterDamageTexts = new Map<
    string,
    ActiveCharacterDamageText
  >()
  const monsterGoldDrops = new Map<string, MonsterGoldDrop>()
  const monsterEquipmentDrops = new Map<string, MonsterEquipmentDrop>()
  const renderedCharacters = new Map<string, RenderedCharacterNode>()
  const renderedPortals = new Map<string, RenderedPortalNode>()
  const characterPixelWidth =
    characterSpriteSheet.tileset.tileWidth * characterSpriteSheet.scale
  const characterPixelHeight =
    characterSpriteSheet.tileset.tileHeight * characterSpriteSheet.scale
  let currentPlayerEquipment = playerEquipment
  let currentPlayerInventory = playerInventory
  let currentPlayerQuickslots = playerQuickslots
  let currentPlayerControlBindings = playerControlBindings
  let currentQuestLog = questLog
  let currentBlacksmithInventory = merchantInventory
  let currentPotionMerchantInventory = potionMerchantInventory
  let playerAttackStartedAtMilliseconds: number | undefined
  let playerAttackFacing: CharacterMoveDirection | undefined
  let playerWeaponTrailSprites: Sprite[] = []
  let playerWeaponSprite: Sprite | undefined
  let playerArmorSprite: Sprite | undefined
  let playerHelmetSprite: Sprite | undefined
  let playerSlashEffectSprite: AnimatedSprite | undefined
  let syncPlayerCharacterVisual: (nowMilliseconds?: number) => void = () => {}
  let isSceneTransitionPending = false
  let isDestroyed = false
  const isBossMonsterScene = sceneIntroMessage === '동굴'
  const monsterCombatStateOptions = isBossMonsterScene
    ? {
        hpMultiplier: 2,
        damageMultiplier: 2
      }
    : undefined
  const monsterRenderScaleMultiplier = isBossMonsterScene ? 2 : 1
  const clearPressedInputState = () => {
    pressedDirections.clear()
    pressedActions.clear()
    triggeredActions.clear()
  }
  let isPlayerUiOpen = false
  let isPlayerStatOpen = false
  let isPlayerEquipmentOpen = false
  let isPlayerSkillOpen = false
  let isQuestLogOpen = false
  let isBlacksmithShopOpen = false
  let isPotionShopOpen = false
  let isPauseMenuOpen = false
  let pendingControlBindingId: PlayerControlBindingId | undefined
  let playerHudOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let playerInventoryOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let playerEquipmentOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let playerStatOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let playerSkillOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let questLogOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let playerShopOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let potionShopOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let pauseMenuOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  let questTrackerOverlay: {
    syncFrame: () => void
    destroy: () => void
  } = {
    syncFrame: () => {},
    destroy: () => {}
  }
  const monsterPatrolStates = new Map<string, MonsterPatrolState>()
  const monsterSpawnStates = new Map<string, CharacterState>()
  const monsterPigAnimatedSprites = new Map<string, AnimatedSprite>()
  const monsterPigAnimationModes = new Map<string, MonsterPigAnimationMode>()
  const monsterPigBehaviorStates = new Map<string, MonsterPigBehaviorState>()
  const monsterCombatStates = new Map<string, MonsterCombatState>()
  const monsterContactDamageLockedUntilById = new Map<string, number>()
  const monsterRespawnAtById = new Map<string, number>()
  let monsterGoldDropSequence = 0
  let monsterEquipmentDropSequence = 0
  let sceneIntroHideTimeoutId: number | undefined
  let playerRespawnAtMilliseconds: number | undefined
  let playerHitReactionState: PlayerHitReactionState | undefined
  let playerAttackResolvedStartedAtMilliseconds: number | undefined
  let playerAttackReadyAtMilliseconds = 0
  let lastRuntimeErrorMessage: string | undefined
  let depthSortedLayer: Container | undefined
  let characterStates = characters.map((character) => ({
    ...character,
    position: { ...character.position },
    collisionSize: { ...character.collisionSize }
  }))
  const initialPlayerCharacter = characterStates.find(
    (character) => character.id === PLAYER_CHARACTER_ID
  )

  if (!initialPlayerCharacter) {
    throw new Error('Missing player character in scene')
  }

  const playerRespawnState = {
    position: {
      ...initialPlayerCharacter.position
    },
    facing: initialPlayerCharacter.facing
  }
  const mapPortals = createMapPortalsFromEventLayers({ map })
  const grassTiles = createGrassTileLookup(map)
  let handleMapOverlayExpandedChange = (_isExpanded: boolean) => {}
  const mapOverlay = createMapOverlay({
    mountElement,
    cameraElement: viewportElement,
    sourceCanvas: app.canvas,
    mapPixelWidth: map.pixelWidth,
    mapPixelHeight: map.pixelHeight,
    getSceneScale: () => cameraZoom,
    getFocusPoint: () => {
      const focusCharacter = characterStates.find(
        (candidateCharacter) => candidateCharacter.id === cameraTargetCharacterId
      )

      if (!focusCharacter) {
        return {
          x: map.pixelWidth / 2,
          y: map.pixelHeight / 2
        }
      }

      return {
        x: focusCharacter.position.x * map.tileWidth + characterPixelWidth / 2,
        y:
          focusCharacter.position.y * map.tileHeight +
          characterPixelHeight / 2
      }
    },
    onExpandedChange: (isExpanded) => handleMapOverlayExpandedChange(isExpanded)
  })
  const syncViewportDisplayScale = () => {
    const displayScale = Math.max(
      0.1,
      Math.min(
        window.innerWidth / GAME_VIEWPORT_WIDTH,
        window.innerHeight / GAME_VIEWPORT_HEIGHT
      )
    )

    viewportElement.style.transform = `scale(${displayScale})`
  }
  const syncCameraZoomLayout = () => {
    scaledMapPixelWidth = Math.round(map.pixelWidth * cameraZoom)
    scaledMapPixelHeight = Math.round(map.pixelHeight * cameraZoom)
    world.scale.set(cameraZoom)
    app.renderer.resize(scaledMapPixelWidth, scaledMapPixelHeight)
    sceneElement.style.width = `${scaledMapPixelWidth}px`
    sceneElement.style.height = `${scaledMapPixelHeight}px`
  }
  const setCameraZoom = (nextCameraZoom: number) => {
    const clampedCameraZoom = clampCameraZoom(nextCameraZoom)

    if (clampedCameraZoom === cameraZoom) {
      return
    }

    cameraZoom = clampedCameraZoom
    syncCameraZoomLayout()
    centerCameraOnCharacter(getCharacterStateById(cameraTargetCharacterId))
    mapOverlay.syncFrame()
  }
  const syncPlayerUiOverlays = () => {
    syncPlayerDerivedCharacterStats()
    playerHudOverlay.syncFrame()
    playerInventoryOverlay.syncFrame()
    playerEquipmentOverlay.syncFrame()
    playerStatOverlay.syncFrame()
    playerSkillOverlay.syncFrame()
    questLogOverlay.syncFrame()
    playerShopOverlay.syncFrame()
    potionShopOverlay.syncFrame()
    pauseMenuOverlay.syncFrame()
    questTrackerOverlay.syncFrame()
  }
  const syncPlayerDerivedCharacterStats = () => {
    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)

    if (playerCharacter.controller.kind !== 'keyboard') {
      return
    }

    playerCharacter.controller = {
      ...playerCharacter.controller,
      moveSpeedTilesPerSecond:
        getPlayerMovementSpeedTilesPerSecond(playerProfile)
    }
  }
  const showSceneIntroBanner = () => {
    if (!sceneIntroMessage) {
      return
    }

    window.clearTimeout(sceneIntroHideTimeoutId)
    sceneIntroBannerElement.classList.add('scene-intro-overlay--visible')
    sceneIntroHideTimeoutId = window.setTimeout(() => {
      sceneIntroBannerElement.classList.remove('scene-intro-overlay--visible')
    }, SCENE_INTRO_VISIBLE_DURATION_MILLISECONDS)
  }
  const getQuickslotIndexFromKeyboardEvent = (
    event: KeyboardEvent
  ): number | undefined => {
    return getPlayerControlQuickslotIndexFromCode(
      currentPlayerControlBindings,
      event.code
    )
  }
  const triggerPlayerAttack = (now: number) => {
    if (now < playerAttackReadyAtMilliseconds) {
      return
    }

    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)
    playerAttackStartedAtMilliseconds = now
    playerAttackResolvedStartedAtMilliseconds = undefined
    playerAttackReadyAtMilliseconds =
      now + PLAYER_ATTACK_COOLDOWN_MILLISECONDS
    playerAttackFacing = playerCharacter.facing
    playPlayerSlashEffect(playerCharacter)
  }
  const clearPlayerSlashEffectSprite = () => {
    if (!playerSlashEffectSprite) {
      return
    }

    const sprite = playerSlashEffectSprite

    playerSlashEffectSprite = undefined
    sprite.removeFromParent()
    sprite.destroy()
  }
  const playPlayerSlashEffect = (character: CharacterState) => {
    clearPlayerSlashEffectSprite()

    const isHorizontalSlash =
      character.facing !== 'up' && character.facing !== 'down'
    const slashTextures = isHorizontalSlash
      ? slashVfxTextures.horizontalTextures
      : slashVfxTextures.verticalTextures
    const slashSprite = new AnimatedSprite(slashTextures)
    const slashBaseScaleX =
      character.facing === 'left'
        ? -PLAYER_ATTACK_SLASH_EFFECT_SCALE_X
        : PLAYER_ATTACK_SLASH_EFFECT_SCALE_X

    slashSprite.label = 'character:player:slash-effect'
    slashSprite.anchor.set(0.5)
    slashSprite.animationSpeed = PLAYER_ATTACK_SLASH_EFFECT_ANIMATION_SPEED
    slashSprite.loop = false
    slashSprite.roundPixels = true
    slashSprite.rotation = isHorizontalSlash
      ? 0
      : character.facing === 'up'
        ? -Math.PI / 2
        : Math.PI / 2
    slashSprite.position.set(
      character.position.x * map.tileWidth + characterPixelWidth / 2,
      character.position.y * map.tileHeight + characterPixelHeight / 2 - 1
    )
    slashSprite.scale.set(slashBaseScaleX, PLAYER_ATTACK_SLASH_EFFECT_SCALE_Y)
    slashSprite.zIndex =
      getCharacterDepthSortValue(
        character.position.y,
        characterPixelHeight,
        map.tileHeight
      ) + 1
    slashSprite.onComplete = () => {
      if (playerSlashEffectSprite === slashSprite) {
        playerSlashEffectSprite = undefined
      }
      slashSprite.removeFromParent()
      slashSprite.destroy()
    }

    playerSlashEffectSprite = slashSprite
    depthSortedLayer?.addChild(slashSprite)
    depthSortedLayer?.sortChildren()
    slashSprite.play()
  }
  const stopPlayerFootsteps = () => {
    gameSoundEffects.stop('grassFootstep')
  }
  const syncPlayerFootsteps = (didPlayerMove: boolean) => {
    if (
      !didPlayerMove ||
      playerProfile.hp.current === 0 ||
      isPauseMenuOpen ||
      mapOverlay.getIsExpanded()
    ) {
      stopPlayerFootsteps()
      return
    }

    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)

    if (!isCharacterOnGrass(playerCharacter, grassTiles)) {
      stopPlayerFootsteps()
      return
    }

    gameSoundEffects.startLoop('grassFootstep')
  }
  const setPlayerUiOpen = (nextIsOpen: boolean) => {
    if (isPlayerUiOpen === nextIsOpen) {
      return
    }

    isPlayerUiOpen = nextIsOpen
    syncPlayerUiOverlays()
  }
  const setPlayerStatOpen = (nextIsOpen: boolean) => {
    if (isPlayerStatOpen === nextIsOpen) {
      return
    }

    isPlayerStatOpen = nextIsOpen
    syncPlayerUiOverlays()
  }
  const setPlayerEquipmentOpen = (nextIsOpen: boolean) => {
    if (isPlayerEquipmentOpen === nextIsOpen) {
      return
    }

    isPlayerEquipmentOpen = nextIsOpen
    syncPlayerUiOverlays()
  }
  const setPlayerSkillOpen = (nextIsOpen: boolean) => {
    if (isPlayerSkillOpen === nextIsOpen) {
      return
    }

    isPlayerSkillOpen = nextIsOpen
    syncPlayerUiOverlays()
  }
  const setQuestLogOpen = (nextIsOpen: boolean) => {
    if (isQuestLogOpen === nextIsOpen) {
      return
    }

    isQuestLogOpen = nextIsOpen
    syncPlayerUiOverlays()
  }
  const setBlacksmithShopOpen = (nextIsOpen: boolean) => {
    if (isBlacksmithShopOpen === nextIsOpen) {
      return
    }

    if (nextIsOpen) {
      isQuestLogOpen = false
      isPotionShopOpen = false
    }

    isBlacksmithShopOpen = nextIsOpen
    if (nextIsOpen) {
      setQuestLogWithObjectiveFeedback(
        recordShopOpenQuestProgress(currentQuestLog, 'blacksmith')
      )
    }
    syncPlayerUiOverlays()
  }
  const setPotionShopOpen = (nextIsOpen: boolean) => {
    if (isPotionShopOpen === nextIsOpen) {
      return
    }

    if (nextIsOpen) {
      isQuestLogOpen = false
      isBlacksmithShopOpen = false
    }

    isPotionShopOpen = nextIsOpen
    if (nextIsOpen) {
      setQuestLogWithObjectiveFeedback(
        recordShopOpenQuestProgress(currentQuestLog, 'potion')
      )
    }
    syncPlayerUiOverlays()
  }
  const setPauseMenuOpen = (nextIsOpen: boolean) => {
    if (isPauseMenuOpen === nextIsOpen) {
      return
    }

    isPauseMenuOpen = nextIsOpen
    pendingControlBindingId = undefined
    if (nextIsOpen) {
      isPlayerUiOpen = false
      isPlayerStatOpen = false
      isPlayerEquipmentOpen = false
      isPlayerSkillOpen = false
      isQuestLogOpen = false
      isBlacksmithShopOpen = false
      isPotionShopOpen = false
      mapOverlay.setExpanded(false)
      gameSoundEffects.stopAllLoops()
    }
    clearPressedInputState()
    syncPlayerUiOverlays()
  }
  const updateCurrentAudioSettings = (nextAudioSettings: AudioSettings) => {
    currentAudioSettings = nextAudioSettings
    gameSoundEffects.setMasterVolume(currentAudioSettings.sfxVolume)
    onAudioSettingsChange(currentAudioSettings)
    pauseMenuOverlay.syncFrame()
  }
  const setQuestLog = (nextQuestLog: QuestLogState) => {
    if (currentQuestLog === nextQuestLog) {
      return
    }

    currentQuestLog = nextQuestLog
    onQuestLogChange(nextQuestLog)
    questLogOverlay.syncFrame()
    questTrackerOverlay.syncFrame()
    syncQuestNpcBadges()
  }
  const setQuestLogWithObjectiveFeedback = (nextQuestLog: QuestLogState) => {
    const previousQuestLog = currentQuestLog

    if (previousQuestLog === nextQuestLog) {
      return
    }

    const didCompleteObjective = Object.entries(
      previousQuestLog.progressByQuestId
    ).some(([questId, previousQuest]) => {
      const nextQuest = nextQuestLog.progressByQuestId[questId]

      return (
        previousQuest.status === 'active' &&
        nextQuest?.status === 'ready-to-turn-in'
      )
    })

    setQuestLog(nextQuestLog)

    if (didCompleteObjective) {
      showCharacterDamageText(
        PLAYER_CHARACTER_ID,
        QUEST_OBJECTIVE_COMPLETE_TEXT,
        DAMAGE_TEXT_DURATION_MILLISECONDS,
        LEVEL_UP_TEXT_STYLE
      )
    }
  }
  const grantPlayerExperienceReward = (experienceReward: number) => {
    const nextPlayerProgress = grantPlayerExperience(
      playerProfile,
      experienceReward
    )

    if (nextPlayerProgress.nextProfile === playerProfile) {
      return nextPlayerProgress
    }

    Object.assign(playerProfile, nextPlayerProgress.nextProfile)
    syncPlayerUiOverlays()
    if (nextPlayerProgress.levelsGained > 0) {
      gameSoundEffects.play('levelUp')
      showCharacterDamageText(
        PLAYER_CHARACTER_ID,
        nextPlayerProgress.levelsGained > 1
          ? `레벨 업 x${nextPlayerProgress.levelsGained}!`
          : '레벨 업!',
        DAMAGE_TEXT_DURATION_MILLISECONDS,
        LEVEL_UP_TEXT_STYLE
      )
    }

    return nextPlayerProgress
  }
  const grantQuestCompletionRewards = (result: CompleteQuestResult) => {
    if (!result.didComplete) {
      return
    }

    if (result.goldReward > 0) {
      currentPlayerInventory = {
        ...currentPlayerInventory,
        gold: currentPlayerInventory.gold + result.goldReward
      }
      onPlayerInventoryChange(currentPlayerInventory)
    }

    if (result.itemRewards.length > 0) {
      currentPlayerInventory = addQuestItemRewardsToInventory(
        currentPlayerInventory,
        result.itemRewards
      )
      onPlayerInventoryChange(currentPlayerInventory)
    }

    grantPlayerExperienceReward(result.experienceReward)
    showCharacterDamageText(
      PLAYER_CHARACTER_ID,
      QUEST_COMPLETE_TEXT,
      DAMAGE_TEXT_DURATION_MILLISECONDS,
      LEVEL_UP_TEXT_STYLE
    )
    syncPlayerUiOverlays()
  }
  const handleConsumableUsed = (itemId: string) => {
    setQuestLogWithObjectiveFeedback(
      recordItemUseQuestProgress(currentQuestLog, itemId)
    )
  }
  handleMapOverlayExpandedChange = (nextIsExpanded: boolean) => {
    if (nextIsExpanded) {
      isPlayerUiOpen = false
      isPlayerStatOpen = false
      isPlayerEquipmentOpen = false
      isPlayerSkillOpen = false
      isQuestLogOpen = false
      isBlacksmithShopOpen = false
      isPotionShopOpen = false
      isPauseMenuOpen = false
      pendingControlBindingId = undefined
      gameSoundEffects.stopAllLoops()
    }
    clearPressedInputState()
    syncPlayerUiOverlays()
  }
  const closeAllOverlays = (): boolean => {
    if (
      !isPlayerUiOpen &&
      !isPlayerStatOpen &&
      !isPlayerEquipmentOpen &&
      !isPlayerSkillOpen &&
      !isQuestLogOpen &&
      !isBlacksmithShopOpen &&
      !isPotionShopOpen &&
      !isPauseMenuOpen &&
      !mapOverlay.getIsExpanded()
    ) {
      return false
    }

    isPlayerUiOpen = false
    isPlayerStatOpen = false
    isPlayerEquipmentOpen = false
    isPlayerSkillOpen = false
    isQuestLogOpen = false
    isBlacksmithShopOpen = false
    isPotionShopOpen = false
    isPauseMenuOpen = false
    pendingControlBindingId = undefined
    mapOverlay.setExpanded(false)
    gameSoundEffects.stopAllLoops()
    clearPressedInputState()
    syncPlayerUiOverlays()
    return true
  }
  const setControlBindingCaptureTarget = (
    bindingId: PlayerControlBindingId | undefined
  ) => {
    pendingControlBindingId = bindingId
    syncPlayerUiOverlays()
  }
  const updatePlayerControlBindings = (
    nextControlBindings: PlayerControlBindings
  ) => {
    currentPlayerControlBindings = nextControlBindings
    onPlayerControlBindingsChange(nextControlBindings)
  }
  const resetPlayerControlBindings = () => {
    pendingControlBindingId = undefined
    updatePlayerControlBindings(createInitialPlayerControlBindings())
    syncPlayerUiOverlays()
  }
  const requestSceneTransition = (portal: MapPortal) => {
    if (isSceneTransitionPending) {
      return
    }

    isSceneTransitionPending = true
    closeAllOverlays()
    clearPressedInputState()
    stopPlayerFootsteps()
    triggeredActions.clear()
    gameEventQueue.clear()
    onRequestSceneChange({
      sceneId: portal.targetSceneId,
      spawn: {
        x: portal.targetSpawn.x,
        y: portal.targetSpawn.y
      },
      facing: portal.targetFacing
    })
  }
  const findTouchedMapPortal = (character: CharacterState): MapPortal | undefined => {
    const characterRect = createCollisionRectFromCharacter(character)

    return mapPortals.find((portal) =>
      doCollisionRectsIntersect(characterRect, createCollisionRectFromPortal(portal))
    )
  }
  const handleQuestInteractionEvents = (
    events: GameEvent[],
    now: number
  ): GameEvent[] => {
    const unhandledEvents: GameEvent[] = []

    for (const event of events) {
      if (event.kind !== 'interaction-requested') {
        unhandledEvents.push(event)
        continue
      }

      const sourceCharacter = characterStates.find(
        (character) => character.id === event.sourceCharacterId
      )

      if (!sourceCharacter) {
        unhandledEvents.push(event)
        continue
      }

      const targetCharacter = resolveCharacterInteractionTarget({
        sourceCharacter,
        targetCharacters: characterStates,
        canReceiveInteraction: (character) =>
          getNextQuestInteractionForNpc(currentQuestLog, character.id) !==
          undefined
      })

      if (!targetCharacter) {
        unhandledEvents.push(event)
        continue
      }

      const lockKey = `${sourceCharacter.id}:${targetCharacter.id}:quest`
      const lockedUntil = interactionLockUntilByCharacterPair.get(lockKey) ?? 0

      if (lockedUntil > now) {
        continue
      }

      handleQuestNpcInteraction(targetCharacter)
      interactionLockUntilByCharacterPair.set(
        lockKey,
        now + QUEST_DIALOGUE_DURATION_MILLISECONDS
      )
    }

    return unhandledEvents
  }
  const handleQuestNpcInteraction = (targetCharacter: CharacterState) => {
    const interaction = getNextQuestInteractionForNpc(
      currentQuestLog,
      targetCharacter.id
    )

    if (!interaction) {
      return
    }

    switch (interaction.action) {
      case 'start': {
        let nextQuestLog = startQuest(currentQuestLog, interaction.questId)

        nextQuestLog = recordTalkQuestProgress(nextQuestLog, targetCharacter.id)
        setQuestLogWithObjectiveFeedback(nextQuestLog)
        showQuestDialogue(
          targetCharacter.id,
          interaction.definition.startDialogueLines
        )
        showCharacterDamageText(
          PLAYER_CHARACTER_ID,
          QUEST_START_TEXT,
          DAMAGE_TEXT_DURATION_MILLISECONDS,
          LEVEL_UP_TEXT_STYLE
        )
        maybeOpenQuestNpcShop(targetCharacter.id)
        return
      }
      case 'active': {
        setQuestLogWithObjectiveFeedback(
          recordTalkQuestProgress(currentQuestLog, targetCharacter.id)
        )
        showQuestDialogue(
          targetCharacter.id,
          interaction.definition.activeDialogueLines
        )
        maybeOpenQuestNpcShop(targetCharacter.id)
        return
      }
      case 'complete': {
        const result = completeQuest(currentQuestLog, interaction.questId)
        const completionLines = interaction.definition.arcCompletionMessage
          ? [
              ...interaction.definition.completionDialogueLines,
              interaction.definition.arcCompletionMessage
            ]
          : interaction.definition.completionDialogueLines

        setQuestLog(result.nextQuestLog)
        showQuestDialogue(targetCharacter.id, completionLines)
        grantQuestCompletionRewards(result)
      }
    }
  }
  const showQuestDialogue = (characterId: string, lines: string[]) => {
    showCharacterMessage(
      characterId,
      formatQuestTextLines(lines, {
        playerName: playerProfile.name
      }).join('\n'),
      QUEST_DIALOGUE_DURATION_MILLISECONDS
    )
  }
  const maybeOpenQuestNpcShop = (npcId: string) => {
    if (npcId === BLACKSMITH_SHOP_NPC_ID) {
      setBlacksmithShopOpen(true)
    }
  }
  playerHudOverlay = createPlayerHudOverlay({
    mountElement,
    profile: playerProfile,
    getInventory: () => currentPlayerInventory,
    getQuickslots: () => currentPlayerQuickslots,
    onRequestQuickslotChange: (nextQuickslots) => {
      currentPlayerQuickslots = nextQuickslots
      onPlayerQuickslotsChange(nextQuickslots)
      syncPlayerUiOverlays()
    }
  })
  playerInventoryOverlay = createPlayerInventoryOverlay({
    mountElement,
    profile: playerProfile,
    getInventory: () => currentPlayerInventory,
    getQuickslots: () => currentPlayerQuickslots,
    getEquipment: () => currentPlayerEquipment,
    getIsOpen: () => isPlayerUiOpen,
    onRequestOpenChange: setPlayerUiOpen,
    onRequestInventoryChange: (nextInventory) => {
      currentPlayerInventory = nextInventory
      onPlayerInventoryChange(nextInventory)
      syncPlayerUiOverlays()
    },
    onRequestEquipmentChange: (nextEquipment) => {
      currentPlayerEquipment = nextEquipment
      onPlayerEquipmentChange(nextEquipment)
      syncPlayerCharacterVisual()
      syncPlayerUiOverlays()
    },
    onRequestProfileChange: (nextProfile) => {
      Object.assign(playerProfile, nextProfile)
      syncPlayerUiOverlays()
    },
    onConsumableUsed: handleConsumableUsed
  })
  playerEquipmentOverlay = createPlayerEquipmentOverlay({
    mountElement,
    profile: playerProfile,
    getInventory: () => currentPlayerInventory,
    getEquipment: () => currentPlayerEquipment,
    getIsOpen: () => isPlayerEquipmentOpen,
    onRequestOpenChange: setPlayerEquipmentOpen,
    onRequestInventoryChange: (nextInventory) => {
      currentPlayerInventory = nextInventory
      onPlayerInventoryChange(nextInventory)
      syncPlayerUiOverlays()
    },
    onRequestEquipmentChange: (nextEquipment) => {
      currentPlayerEquipment = nextEquipment
      onPlayerEquipmentChange(nextEquipment)
      syncPlayerCharacterVisual()
      syncPlayerUiOverlays()
    }
  })
  playerStatOverlay = createPlayerStatOverlay({
    mountElement,
    profile: playerProfile,
    getIsOpen: () => isPlayerStatOpen,
    onRequestOpenChange: setPlayerStatOpen,
    onRequestProfileChange: (nextProfile) => {
      Object.assign(playerProfile, nextProfile)
      syncPlayerUiOverlays()
    }
  })
  playerSkillOverlay = createPlayerSkillOverlay({
    mountElement,
    profile: playerProfile,
    getIsOpen: () => isPlayerSkillOpen,
    onRequestOpenChange: setPlayerSkillOpen,
    onRequestProfileChange: (nextProfile) => {
      Object.assign(playerProfile, nextProfile)
      syncPlayerUiOverlays()
    }
  })
  playerShopOverlay = createBlacksmithShopOverlay({
    mountElement,
    getPlayerName: () => playerProfile.name,
    getPlayerInventory: () => currentPlayerInventory,
    getMerchantInventory: () => currentBlacksmithInventory,
    getIsOpen: () => isBlacksmithShopOpen,
    onRequestOpenChange: setBlacksmithShopOpen,
    onRequestTradeStateChange: (
      nextPlayerInventory,
      nextMerchantInventory
    ) => {
      currentPlayerInventory = nextPlayerInventory
      currentBlacksmithInventory = nextMerchantInventory
      onPlayerInventoryChange(nextPlayerInventory)
      onMerchantInventoryChange(nextMerchantInventory)
      syncPlayerUiOverlays()
    }
  })
  potionShopOverlay = createPotionShopOverlay({
    mountElement,
    getPlayerName: () => playerProfile.name,
    getPlayerInventory: () => currentPlayerInventory,
    getMerchantInventory: () => currentPotionMerchantInventory,
    getIsOpen: () => isPotionShopOpen,
    onRequestOpenChange: setPotionShopOpen,
    onRequestTradeStateChange: (
      nextPlayerInventory,
      nextMerchantInventory
    ) => {
      currentPlayerInventory = nextPlayerInventory
      currentPotionMerchantInventory = nextMerchantInventory
      onPlayerInventoryChange(nextPlayerInventory)
      onPotionMerchantInventoryChange(nextMerchantInventory)
      syncPlayerUiOverlays()
    }
  })
  pauseMenuOverlay = createPauseMenuOverlay({
    mountElement,
    getIsOpen: () => isPauseMenuOpen,
    getAudioSettings: () => currentAudioSettings,
    getControlBindings: () => currentPlayerControlBindings,
    getControlBindingCaptureTarget: () => pendingControlBindingId,
    onRequestOpenChange: setPauseMenuOpen,
    onAudioSettingsChange: updateCurrentAudioSettings,
    onRequestControlBindingCapture: setControlBindingCaptureTarget,
    onRequestControlBindingsReset: resetPlayerControlBindings
  })
  questLogOverlay = createQuestLogOverlay({
    mountElement,
    getIsOpen: () => isQuestLogOpen,
    getQuestLog: () => currentQuestLog,
    getPlayerName: () => playerProfile.name,
    onRequestOpenChange: setQuestLogOpen,
    onQuestLogChange: setQuestLog
  })
  questTrackerOverlay = createQuestTrackerOverlay({
    mountElement,
    getQuestLog: () => currentQuestLog,
    onQuestLogChange: setQuestLog
  })

  const syncRuntimeWarningBanner = () => {
    const warnings = controllerRuntime.getRuntimeWarnings()

    if (warnings.length === 0) {
      runtimeWarningBannerElement.hidden = true
      runtimeWarningBannerElement.replaceChildren()
      return
    }

    runtimeWarningBannerElement.hidden = false
    const warningBlocks = warnings.map((warning) => {
      const warningElement = document.createElement('div')
      const warningLines = warning.split('\n')

      warningElement.style.display = 'grid'
      warningElement.style.gap = '2px'

      for (const line of warningLines) {
        const lineElement = document.createElement('div')
        const luaSourceReferenceMatch = line.match(/([A-Za-z0-9_./-]+\.lua:\d+)/u)

        if (luaSourceReferenceMatch && luaSourceReferenceMatch.index !== undefined) {
          const prefixElement = document.createElement('span')
          const pathElement = document.createElement('span')
          const suffixElement = document.createElement('span')
          const matchStart = luaSourceReferenceMatch.index
          const matchedPath = luaSourceReferenceMatch[1]

          prefixElement.textContent = line.slice(0, matchStart)
          pathElement.textContent = matchedPath
          pathElement.style.textDecoration = 'underline'
          pathElement.style.textDecorationThickness = '1px'
          suffixElement.textContent = line.slice(matchStart + matchedPath.length)
          lineElement.append(prefixElement, pathElement, suffixElement)
        } else {
          lineElement.textContent = line
        }

        warningElement.append(lineElement)
      }

      return warningElement
    })

    runtimeWarningBannerElement.replaceChildren(...warningBlocks)
  }

  controllerRuntime.syncCharacters(characterStates)
  syncRuntimeWarningBanner()

  messageLayer.label = 'layer:messages'
  messageLayer.sortableChildren = true
  app.stage.addChild(world)

  for (const tileset of map.tilesets) {
    tilesetResources.set(
      tileset.source,
      await loadTilesetRenderResources(tileset, imageUrls)
    )
  }
  const characterTilesetResources = await loadTilesetRenderResources(
    characterSpriteSheet.tileset,
    imageUrls,
    'nearest'
  )
  const playerWeaponTexture = createTileTexture(
    tinyDungeonWeaponImageTexture,
    PLAYER_WEAPON_TILE_FRAME_SOURCE,
    PLAYER_WEAPON_TILE_LOCAL_ID
  )
  const questNewTexture = await Assets.load<Texture>(
    imageUrls['quest_new.png']
  )
  const questFinTexture = await Assets.load<Texture>(
    imageUrls['quest_fin.png']
  )
  const caveEntranceTexture = await Assets.load<Texture>(
    imageUrls['cave1-visible.png']
  )
  caveEntranceTexture.source.addressMode = 'clamp-to-edge'
  const resolveMapPortalTexture = (appearanceType: string): Texture => {
    if (appearanceType === 'cave_entrance') {
      return caveEntranceTexture
    }

    for (const tileset of map.tilesets) {
      const renderResources = tilesetResources.get(tileset.source)

      if (!renderResources) {
        throw new Error(`Missing render resources for tileset ${tileset.source}`)
      }

      try {
        return renderResources.tileTextures[
          resolveTilesetLocalIdByType(tileset, appearanceType)
        ]
      } catch {
        continue
      }
    }

    throw new Error(`Could not resolve portal texture ${appearanceType}`)
  }

  for (const layer of map.layers) {
    if (layer.name.toLowerCase() === DEPTH_SORTED_LAYER_NAME) {
      const nextDepthSortedLayer = new Container()

      nextDepthSortedLayer.label = `layer:${layer.name}:depth`
      nextDepthSortedLayer.sortableChildren = true

      for (const tile of layer.tiles) {
        const tileset = resolveTilesetForTile(tile, map.tilesets)
        const renderResources = tilesetResources.get(tileset.source)

        if (!renderResources) {
          throw new Error(`Missing render resources for tileset ${tileset.source}`)
        }

        const sprite = createDepthSortedTileSprite(
          renderResources.tileTextures[tile.localId],
          tile,
          map.tileWidth,
          map.tileHeight
        )

        sprite.alpha = layer.opacity
        sprite.visible = layer.visible
        sprite.zIndex = getTileDepthSortValue(tile.y, map.tileHeight)
        nextDepthSortedLayer.addChild(sprite)
      }

      depthSortedLayer = nextDepthSortedLayer
      world.addChild(nextDepthSortedLayer)
      continue
    }

    const tilemap = new CompositeTilemap()
    const transformedTileLayer = new Container()

    tilemap.label = `layer:${layer.name}`
    tilemap.alpha = layer.opacity
    tilemap.visible = layer.visible
    transformedTileLayer.label = `layer:${layer.name}:transforms`
    transformedTileLayer.alpha = layer.opacity
    transformedTileLayer.visible = layer.visible

    for (const tile of layer.tiles) {
      const tileset = resolveTilesetForTile(tile, map.tilesets)
      const renderResources = tilesetResources.get(tileset.source)

      if (!renderResources) {
        throw new Error(`Missing render resources for tileset ${tileset.source}`)
      }

      if (hasTileTransform(tile)) {
        transformedTileLayer.addChild(
          createTransformedTileSprite(
            renderResources.tileTextures[tile.localId],
            tile,
            map.tileWidth,
            map.tileHeight
          )
        )
        continue
      }

      tilemap.tile(
        renderResources.tileTextures[tile.localId],
        tile.x * map.tileWidth,
        tile.y * map.tileHeight
      )
    }

    world.addChild(tilemap)
    world.addChild(transformedTileLayer)
  }

  if (!depthSortedLayer) {
    depthSortedLayer = new Container()
    depthSortedLayer.label = 'layer:characters:depth'
    depthSortedLayer.sortableChildren = true
    world.addChild(depthSortedLayer)
  }

  for (const character of characterStates) {
    const container = new Container()
    const isMonsterCharacter = character.appearanceType.startsWith('monster_')
    const isSignPostCharacter =
      character.appearanceType === SIGN_POST_APPEARANCE_TYPE
    const monsterAnimationTextures = isMonsterCharacter
      ? monsterAnimationTexturesByAppearanceType[
          character.appearanceType as MonsterAppearanceType
        ]
      : undefined
    const monsterBehaviorConfig = isMonsterCharacter
      ? getMonsterBehaviorConfig(character)
      : undefined
    const resolvedCharacterAppearanceTexture =
      monsterAnimationTextures === undefined
        ? resolveCharacterTexture(
            isSignPostCharacter ? 'post_tall_base_00' : character.appearanceType,
            characterTilesetResources.tileTextures,
            characterSpriteSheet.tileset,
            map.tilesets,
            tilesetResources,
            map.tileWidth
          )
        : undefined
    const sprite = monsterAnimationTextures
      ? new AnimatedSprite(monsterAnimationTextures.idleLeft)
      : new Sprite(resolvedCharacterAppearanceTexture!.texture)
    const renderScale = monsterBehaviorConfig
      ? monsterBehaviorConfig.renderScale * monsterRenderScaleMultiplier
      : resolvedCharacterAppearanceTexture!.renderScale
    const isPlayer = character.id === PLAYER_CHARACTER_ID
    const playerHealthBar = isPlayer
      ? createPlayerResourceBar()
      : undefined
    const playerManaBar = isPlayer
      ? createPlayerResourceBar()
      : undefined
    const monsterHealthBar = isMonsterCharacter
      ? createMonsterHealthBar()
      : undefined
    const playerNameBadge =
      isPlayer
        ? new Text({
            style: PLAYER_NAME_BADGE_STYLE,
            text: playerProfile.name
          })
        : undefined
    const displayLabel =
      character.displayText === undefined
        ? undefined
        : new Text({
            style: isSignPostCharacter
              ? SIGN_POST_LABEL_STYLE
              : PLAYER_NAME_BADGE_STYLE,
            text: character.displayText
          })
    const displayLabelPanel =
      displayLabel && isSignPostCharacter
        ? new NineSliceSprite({
            texture: messagePanelTexture,
            bottomHeight: MESSAGE_PANEL_BORDER_SIZE,
            leftWidth: MESSAGE_PANEL_BORDER_SIZE,
            rightWidth: MESSAGE_PANEL_BORDER_SIZE,
            topHeight: MESSAGE_PANEL_BORDER_SIZE
          })
        : undefined
    if (displayLabelPanel && displayLabel) {
      displayLabelPanel.roundPixels = true
      displayLabelPanel.setSize(
        Math.max(112, Math.ceil(displayLabel.width) + 18),
        Math.max(28, Math.ceil(displayLabel.height) + 8)
      )
    }
    const levelBadge =
      character.level === undefined
        ? undefined
        : new Text({
            style: MONSTER_LEVEL_BADGE_STYLE,
            text: `Lv ${character.level}`
          })
    const questBadge =
      QUEST_GIVER_NPC_IDS.includes(
        character.id as (typeof QUEST_GIVER_NPC_IDS)[number]
      )
        ? createQuestBadgeSprite(questNewTexture)
        : undefined
    container.label = `character:${character.id}:container`
    container.sortableChildren = true
    sprite.label = `character:${character.id}`
    sprite.scale.set(renderScale)
    sprite.roundPixels = true
    sprite.zIndex = 10
    container.addChild(sprite)
    if (displayLabelPanel) {
      displayLabelPanel.label = `character:${character.id}:display-label-panel`
      displayLabelPanel.zIndex = 16
      container.addChild(displayLabelPanel)
    }
    if (displayLabel) {
      displayLabel.label = `character:${character.id}:display-label`
      displayLabel.roundPixels = true
      displayLabel.zIndex = displayLabelPanel ? 17 : 16
      container.addChild(displayLabel)
    }
    if (playerNameBadge) {
      playerNameBadge.label = `character:${character.id}:name`
      playerNameBadge.roundPixels = true
      playerNameBadge.zIndex = 21
      container.addChild(playerNameBadge)
    }
    if (playerHealthBar) {
      playerHealthBar.container.label = `character:${character.id}:player-health-bar`
      playerHealthBar.container.zIndex = 18
      container.addChild(playerHealthBar.container)
    }
    if (playerManaBar) {
      playerManaBar.container.label = `character:${character.id}:player-mana-bar`
      playerManaBar.container.zIndex = 18.5
      container.addChild(playerManaBar.container)
    }
    if (monsterHealthBar) {
      monsterHealthBar.container.label = `character:${character.id}:monster-health-bar`
      monsterHealthBar.container.zIndex = 15
      container.addChild(monsterHealthBar.container)
    }
    if (levelBadge) {
      levelBadge.label = `character:${character.id}:level`
      levelBadge.roundPixels = true
      levelBadge.zIndex = 20
      container.addChild(levelBadge)
    }
    if (questBadge) {
      questBadge.label = `character:${character.id}:quest-badge`
      questBadge.roundPixels = true
      questBadge.zIndex = 22
      container.addChild(questBadge)
    }

    if (isMonsterCharacter) {
      monsterCombatStates.set(
        character.id,
        createMonsterCombatState(
          character.level ?? 1,
          monsterCombatStateOptions
        )
      )
      monsterSpawnStates.set(character.id, {
        ...character,
        position: {
          ...character.position
        },
        collisionSize: {
          ...character.collisionSize
        }
      })
    }

    if (isPlayer) {
      playerArmorSprite = new Sprite(Texture.EMPTY)
      playerArmorSprite.label = 'character:player:armor'
      playerArmorSprite.anchor.set(0.5)
      playerArmorSprite.visible = false
      playerArmorSprite.roundPixels = true
      playerArmorSprite.zIndex = PLAYER_ARMOR_EQUIPMENT_CONFIG.zIndex
      container.addChild(playerArmorSprite)
      playerHelmetSprite = new Sprite(Texture.EMPTY)
      playerHelmetSprite.label = 'character:player:helmet'
      playerHelmetSprite.anchor.set(0.5)
      playerHelmetSprite.visible = false
      playerHelmetSprite.roundPixels = true
      playerHelmetSprite.zIndex = PLAYER_HELMET_EQUIPMENT_CONFIG.zIndex
      container.addChild(playerHelmetSprite)
      playerWeaponTrailSprites = Array.from(
        { length: PLAYER_ATTACK_TRAIL_SPRITE_COUNT },
        (_, index) => {
          const trailSprite = new Sprite(Texture.EMPTY)

          trailSprite.label = `character:player:weapon-trail:${index}`
          trailSprite.anchor.set(0.5, 1)
          trailSprite.visible = false
          trailSprite.roundPixels = true
          trailSprite.zIndex = index + 1
          container.addChild(trailSprite)

          return trailSprite
        }
      )
      playerWeaponSprite = new Sprite(Texture.EMPTY)
      playerWeaponSprite.label = 'character:player:weapon'
      playerWeaponSprite.anchor.set(0.5, 1)
      playerWeaponSprite.visible = false
      playerWeaponSprite.roundPixels = true
      playerWeaponSprite.zIndex = PLAYER_ATTACK_TRAIL_SPRITE_COUNT + 1
      container.addChild(playerWeaponSprite)
    }

    renderedCharacters.set(character.id, {
      container,
      sprite,
      playerArmorSprite: isPlayer ? playerArmorSprite : undefined,
      playerHelmetSprite: isPlayer ? playerHelmetSprite : undefined,
      playerHealthBar,
      playerManaBar,
      playerNameBadge,
      displayLabelPanel,
      displayLabel,
      levelBadge,
      questBadge,
      monsterHealthBar
    })
    if (monsterAnimationTextures) {
      monsterPigAnimatedSprites.set(character.id, sprite as AnimatedSprite)
      monsterPigBehaviorStates.set(character.id, createMonsterPigBehaviorState())
      syncMonsterAnimation(character.id, 'idle')
    }
    depthSortedLayer.addChild(container)
  }

  for (const portal of mapPortals) {
    const container = new Container()
    const baseSprite = new Sprite(resolveMapPortalTexture(portal.appearanceType))
    const coreSprite = new Sprite(portalInsideTexture)
    container.label = `portal:${portal.id}:container`
    container.sortableChildren = true
    baseSprite.label = `portal:${portal.id}:base`
    baseSprite.roundPixels = true
    baseSprite.zIndex = 0
    coreSprite.label = `portal:${portal.id}:core`
    coreSprite.anchor.set(0.5, 0.5)
    coreSprite.scale.set(PORTAL_INSIDE_WORLD_SCALE)
    coreSprite.roundPixels = true
    coreSprite.zIndex = 1
    const isCaveEntrancePortal = portal.appearanceType === 'cave_entrance'

    if (isCaveEntrancePortal) {
      baseSprite.scale.set(0.24)
      container.position.set(
        portal.position.x * map.tileWidth - 32,
        portal.position.y * map.tileHeight
      )
      container.addChild(baseSprite)
    } else if (portal.appearanceType === 'stairs_stone_step_base_00') {
      container.position.set(
        portal.position.x * map.tileWidth,
        portal.position.y * map.tileHeight
      )
      baseSprite.scale.set(portal.collisionSize.width, portal.collisionSize.height)
      coreSprite.position.set(
        (portal.collisionSize.width * map.tileWidth) / 2,
        (portal.collisionSize.height * map.tileHeight) / 2
      )
      container.addChild(baseSprite, coreSprite)
    } else {
      baseSprite.scale.set(portal.collisionSize.width, portal.collisionSize.height)
      container.position.set(
        portal.position.x * map.tileWidth,
        portal.position.y * map.tileHeight
      )
      container.addChild(baseSprite)
    }
    container.zIndex = Math.round(
      (portal.position.y + portal.collisionSize.height) * map.tileHeight
    )
    renderedPortals.set(portal.id, {
      container,
      sprite: baseSprite
    })
    depthSortedLayer.addChild(container)
  }
  depthSortedLayer.sortChildren()
  world.addChild(messageLayer)

  function getCharacterStateById(characterId: string): CharacterState {
    const character = characterStates.find(
      (candidateCharacter) => candidateCharacter.id === characterId
    )

    if (!character) {
      throw new Error(`Missing character ${characterId}`)
    }

    return character
  }

  syncPlayerCharacterVisual = (now = performance.now()) => {
    syncCharacterSprite(getCharacterStateById(PLAYER_CHARACTER_ID), now)
  }

  const syncCharacterSprite = (
    character: CharacterState,
    now = performance.now()
  ) => {
    const renderNode = renderedCharacters.get(character.id)

    if (!renderNode) {
      throw new Error(`Missing rendered sprite for character ${character.id}`)
    }

    const combatState = monsterCombatStates.get(character.id)

    if (combatState && isMonsterDefeated(combatState)) {
      renderNode.container.visible = false
      return
    }

    if (
      character.id === PLAYER_CHARACTER_ID &&
      playerProfile.hp.current === 0
    ) {
      playerHitReactionState = undefined
      renderNode.container.visible = false
      syncPlayerWeaponSprite(character)
      syncPlayerEquipmentSprites(renderNode)
      return
    }

    let playerHitReactionOffsetX = 0
    let playerHitReactionOffsetY = 0
    let monsterRunMotionOffsetX = 0
    let monsterRunMotionOffsetY = 0

    if (character.id === PLAYER_CHARACTER_ID && playerHitReactionState) {
      if (playerHitReactionState.expiresAtMilliseconds <= now) {
        playerHitReactionState = undefined
      } else {
        const elapsedMilliseconds = now - playerHitReactionState.startedAtMilliseconds
        const progress = Math.min(
          1,
          Math.max(0, elapsedMilliseconds / PLAYER_HIT_REACTION_DURATION_MILLISECONDS)
        )
        const recoilStrength =
          PLAYER_HIT_REACTION_MAX_OFFSET_PIXELS * Math.pow(1 - progress, 2)

        playerHitReactionOffsetX = Math.round(
          playerHitReactionState.directionX * recoilStrength
        )
        playerHitReactionOffsetY = Math.round(
          playerHitReactionState.directionY * recoilStrength
        )
      }
    }

    const isMonsterCharacter = character.appearanceType.startsWith('monster_')
    const monsterBehaviorConfig = isMonsterCharacter
      ? getMonsterBehaviorConfig(character)
      : undefined
    const monsterAnimationMode = isMonsterCharacter
      ? monsterPigAnimationModes.get(character.id)
      : undefined

    if (
      monsterBehaviorConfig &&
      monsterAnimationMode === 'run' &&
      monsterBehaviorConfig.usesRunAnimation &&
      (monsterBehaviorConfig.runMotionBobPixels > 0 ||
        monsterBehaviorConfig.runMotionSwayPixels > 0)
    ) {
      const runMotionPhase =
        now / 110 +
        character.position.x * 0.31 +
        character.position.y * 0.53
      const facingMultiplier = character.facing === 'left' ? -1 : 1

      monsterRunMotionOffsetX = Math.round(
        Math.sin(runMotionPhase * 0.5) *
          monsterBehaviorConfig.runMotionSwayPixels *
          facingMultiplier
      )
      monsterRunMotionOffsetY = Math.round(
        -Math.abs(Math.sin(runMotionPhase)) *
          monsterBehaviorConfig.runMotionBobPixels
      )
    }

    renderNode.container.visible = true
    renderNode.container.position.set(
      character.position.x * map.tileWidth +
        playerHitReactionOffsetX +
        monsterRunMotionOffsetX,
      character.position.y * map.tileHeight +
        playerHitReactionOffsetY +
        monsterRunMotionOffsetY
    )
    renderNode.container.zIndex = getCharacterDepthSortValue(
      character.position.y,
      renderNode.sprite.height,
      map.tileHeight
    ) + (character.appearanceType === SIGN_POST_APPEARANCE_TYPE ? 1 : 0)
    syncCharacterDisplayLabel(renderNode)
    syncPlayerNameBadge(renderNode, character)
    syncCharacterLevelBadge(renderNode, character)
    syncPlayerResourceBars(renderNode, character)
    depthSortedLayer?.sortChildren()

    if (character.id === PLAYER_CHARACTER_ID) {
      syncPlayerEquipmentSprites(renderNode)
      syncPlayerWeaponSprite(character)
    }
  }

  const syncPlayerEquipmentSprites = (renderNode: RenderedCharacterNode) => {
    syncPlayerEquipmentSprite(renderNode, 'armor', renderNode.playerArmorSprite)
    syncPlayerEquipmentSprite(renderNode, 'hat', renderNode.playerHelmetSprite)
  }

  const syncPlayerEquipmentSprite = (
    renderNode: RenderedCharacterNode,
    slotId: PlayerVisualEquipmentSlotId,
    sprite: Sprite | undefined
  ) => {
    if (!sprite) {
      return
    }

    const equippedItem = currentPlayerEquipment.slots.find(
      (slot) => slot.id === slotId
    )?.item
    const config = equippedItem
      ? PLAYER_EQUIPMENT_APPEARANCE_CONFIG_BY_ITEM_ID[equippedItem.id]
      : undefined
    const texture = equippedItem
      ? playerEquipmentTexturesByItemId.get(equippedItem.id)
      : undefined
    const shouldShowEquipment =
      renderNode.container.visible &&
      playerProfile.hp.current > 0 &&
      config?.slotId === slotId &&
      texture !== undefined

    sprite.visible = shouldShowEquipment

    if (!shouldShowEquipment || !config || !texture) {
      return
    }

    sprite.texture = texture
    sprite.position.set(config.position.x, config.position.y)
    sprite.width = config.width
    sprite.height = config.height
    sprite.zIndex = config.zIndex
  }

  const syncCharacterLevelBadge = (
    renderNode: RenderedCharacterNode,
    character: CharacterState
  ) => {
    if (!renderNode.levelBadge) {
      return
    }

    if (character.level === undefined) {
      renderNode.levelBadge.visible = false
      renderNode.levelBadge.text = ''
      if (renderNode.monsterHealthBar) {
        renderNode.monsterHealthBar.container.visible = false
      }
      return
    }

    if (character.id === PLAYER_CHARACTER_ID) {
      renderNode.levelBadge.visible = true
      renderNode.levelBadge.text = `Lv ${character.level}`
      const playerStatusStackHeight =
        renderNode.levelBadge.height +
        PLAYER_HEALTH_BAR_HEIGHT +
        PLAYER_MANA_BAR_HEIGHT +
        PLAYER_HEALTH_BAR_GAP +
        PLAYER_MANA_BAR_GAP +
        PLAYER_STATUS_STACK_CLEARANCE
      renderNode.levelBadge.position.set(
        Math.round((renderNode.sprite.width - renderNode.levelBadge.width) / 2),
        -Math.round(playerStatusStackHeight)
      )
      return
    }

    renderNode.levelBadge.visible = true
    renderNode.levelBadge.text = `Lv ${character.level}`

    if (renderNode.monsterHealthBar && character.appearanceType.startsWith('monster_')) {
      const combatState = monsterCombatStates.get(character.id)

      syncMonsterHealthBar(renderNode.monsterHealthBar, combatState)
      renderNode.levelBadge.position.set(
        Math.round((renderNode.sprite.width - renderNode.levelBadge.width) / 2),
        -Math.round(
          renderNode.levelBadge.height +
            MONSTER_HEALTH_BAR_HEIGHT +
            MONSTER_HEALTH_BAR_GAP +
            6
        )
      )
      renderNode.monsterHealthBar.container.position.set(
        Math.round((renderNode.sprite.width - MONSTER_HEALTH_BAR_WIDTH) / 2),
        -Math.round(MONSTER_HEALTH_BAR_HEIGHT + 4)
      )
      return
    }

    renderNode.levelBadge.position.set(
      Math.round((renderNode.sprite.width - renderNode.levelBadge.width) / 2),
      -Math.round(renderNode.levelBadge.height + 4)
    )
  }

  const syncQuestNpcBadges = () => {
    for (const npcId of QUEST_GIVER_NPC_IDS) {
      const renderNode = renderedCharacters.get(npcId)

      if (!renderNode?.questBadge) {
        continue
      }

      const badgeKind = getQuestNpcBadgeKindForNpc(currentQuestLog, npcId)

      if (!badgeKind) {
        renderNode.questBadge.visible = false
        continue
      }

      renderNode.questBadge.visible = true
      renderNode.questBadge.texture =
        badgeKind === 'new' ? questNewTexture : questFinTexture
      renderNode.questBadge.position.set(
        Math.round(renderNode.sprite.width / 2),
        -QUEST_BADGE_Y_OFFSET
      )
    }
  }

  const syncPlayerNameBadge = (
    renderNode: RenderedCharacterNode,
    character: CharacterState
  ) => {
    if (character.id !== PLAYER_CHARACTER_ID || !renderNode.playerNameBadge) {
      return
    }

    renderNode.playerNameBadge.visible = true
    renderNode.playerNameBadge.text = playerProfile.name

    renderNode.playerNameBadge.position.set(
      Math.round((renderNode.sprite.width - renderNode.playerNameBadge.width) / 2),
      Math.round(renderNode.sprite.height + PLAYER_NAME_BADGE_FOOT_OFFSET)
    )
  }

  const syncPlayerResourceBars = (
    renderNode: RenderedCharacterNode,
    character: CharacterState
  ) => {
    if (
      character.id !== PLAYER_CHARACTER_ID ||
      !renderNode.playerHealthBar ||
      !renderNode.playerManaBar ||
      !renderNode.playerNameBadge
    ) {
      return
    }

    const healthBar = renderNode.playerHealthBar
    const manaBar = renderNode.playerManaBar
    const healthRatio =
      playerProfile.hp.max === 0
        ? 0
        : Math.min(1, Math.max(0, playerProfile.hp.current / playerProfile.hp.max))
    const manaRatio =
      playerProfile.mp.max === 0
        ? 0
        : Math.min(1, Math.max(0, playerProfile.mp.current / playerProfile.mp.max))
    const healthInnerWidth = PLAYER_HEALTH_BAR_WIDTH - 2
    const healthInnerHeight = PLAYER_HEALTH_BAR_HEIGHT - 2
    const manaInnerWidth = PLAYER_MANA_BAR_WIDTH - 2
    const manaInnerHeight = PLAYER_MANA_BAR_HEIGHT - 2
    const healthFilledWidth = Math.max(0, Math.round(healthInnerWidth * healthRatio))
    const manaFilledWidth = Math.max(0, Math.round(manaInnerWidth * manaRatio))
    const statusStackTopY = -Math.round(
      PLAYER_HEALTH_BAR_HEIGHT +
        PLAYER_MANA_BAR_HEIGHT +
        PLAYER_MANA_BAR_GAP +
        PLAYER_STATUS_STACK_CLEARANCE
    )

    healthBar.container.visible = true
    healthBar.track.clear()
    healthBar.track
      .rect(0, 0, PLAYER_HEALTH_BAR_WIDTH, PLAYER_HEALTH_BAR_HEIGHT)
      .fill({ color: PLAYER_HEALTH_BAR_TRACK_COLOR })
      .stroke({ color: PLAYER_HEALTH_BAR_BORDER_COLOR, width: 1 })
    healthBar.fill.clear()

    if (healthFilledWidth > 0) {
      healthBar.fill
        .rect(1, 1, healthFilledWidth, healthInnerHeight)
        .fill({ color: PLAYER_HEALTH_BAR_FILL_COLOR })
    }

    manaBar.container.visible = true
    manaBar.track.clear()
    manaBar.track
      .rect(0, 0, PLAYER_MANA_BAR_WIDTH, PLAYER_MANA_BAR_HEIGHT)
      .fill({ color: PLAYER_MANA_BAR_TRACK_COLOR })
      .stroke({ color: PLAYER_MANA_BAR_BORDER_COLOR, width: 1 })
    manaBar.fill.clear()

    if (manaFilledWidth > 0) {
      manaBar.fill
        .rect(1, 1, manaFilledWidth, manaInnerHeight)
        .fill({ color: PLAYER_MANA_BAR_FILL_COLOR })
    }

    healthBar.container.position.set(
      Math.round((renderNode.sprite.width - PLAYER_HEALTH_BAR_WIDTH) / 2),
      statusStackTopY
    )
    manaBar.container.position.set(
      Math.round((renderNode.sprite.width - PLAYER_MANA_BAR_WIDTH) / 2),
      healthBar.container.position.y + PLAYER_HEALTH_BAR_HEIGHT + PLAYER_MANA_BAR_GAP
    )
  }

  const syncCharacterDisplayLabel = (
    renderNode: RenderedCharacterNode
  ) => {
    if (!renderNode.displayLabel) {
      return
    }

    if (renderNode.displayLabelPanel) {
      renderNode.displayLabelPanel.visible = true
      renderNode.displayLabelPanel.position.set(
        Math.round(
          (renderNode.sprite.width - renderNode.displayLabelPanel.width) / 2
        ),
        -Math.round(renderNode.displayLabelPanel.height - 6)
      )
      renderNode.displayLabel.anchor.set(0.5)
      renderNode.displayLabel.visible = true
      renderNode.displayLabel.position.set(
        Math.round(renderNode.sprite.width / 2),
        Math.round(
          renderNode.displayLabelPanel.position.y +
            renderNode.displayLabelPanel.height / 2
        )
      )
      return
    }

    renderNode.displayLabel.visible = true
    renderNode.displayLabel.anchor.set(0)
    renderNode.displayLabel.position.set(
      Math.round((renderNode.sprite.width - renderNode.displayLabel.width) / 2),
      Math.round(renderNode.sprite.height + PLAYER_NAME_BADGE_FOOT_OFFSET)
    )
  }

  const syncMonsterHealthBar = (
    monsterHealthBar: NonNullable<RenderedCharacterNode['monsterHealthBar']>,
    combatState: MonsterCombatState | undefined
  ) => {
    if (!combatState) {
      monsterHealthBar.container.visible = false
      return
    }

    const ratio =
      combatState.maxHp === 0
        ? 0
        : Math.min(1, Math.max(0, combatState.currentHp / combatState.maxHp))
    const innerWidth = MONSTER_HEALTH_BAR_WIDTH - 2
    const innerHeight = MONSTER_HEALTH_BAR_HEIGHT - 2
    const filledWidth = Math.max(0, Math.round(innerWidth * ratio))

    monsterHealthBar.container.visible = true
    monsterHealthBar.track.clear()
    monsterHealthBar.track
      .rect(0, 0, MONSTER_HEALTH_BAR_WIDTH, MONSTER_HEALTH_BAR_HEIGHT)
      .fill({ color: MONSTER_HEALTH_BAR_TRACK_COLOR })
      .stroke({ color: MONSTER_HEALTH_BAR_BORDER_COLOR, width: 1 })
    monsterHealthBar.fill.clear()

    if (filledWidth > 0) {
      monsterHealthBar.fill
        .rect(1, 1, filledWidth, innerHeight)
        .fill({ color: MONSTER_HEALTH_BAR_FILL_COLOR })
    }
  }

  function syncMonsterAnimation(
    characterId: string,
    mode: MonsterPigAnimationMode,
    options: {
      forceRestart?: boolean
    } = {}
  ) {
    const sprite = monsterPigAnimatedSprites.get(characterId)
    const combatState = monsterCombatStates.get(characterId)
    const character = getCharacterStateById(characterId)
    const monsterAnimationTextures =
      monsterAnimationTexturesByAppearanceType[
        character.appearanceType as MonsterAppearanceType
      ]
    const behaviorConfig = getMonsterBehaviorConfig(character)

    if (
      !sprite ||
      !monsterAnimationTextures ||
      (combatState && isMonsterDefeated(combatState))
    ) {
      return
    }

    const facingKey = character.facing === 'right' ? 'right' : 'left'
    const isRunAnimationEnabled = behaviorConfig.usesRunAnimation
    const resolvedMode =
      mode === 'run' && !isRunAnimationEnabled ? 'idle' : mode
    const nextTextures =
      resolvedMode === 'run'
        ? facingKey === 'right'
          ? monsterAnimationTextures.runRight
          : monsterAnimationTextures.runLeft
        : resolvedMode === 'hit'
          ? facingKey === 'right'
            ? monsterAnimationTextures.hitRight
            : monsterAnimationTextures.hitLeft
          : resolvedMode === 'attack'
            ? facingKey === 'right'
              ? monsterAnimationTextures.attackRight
              : monsterAnimationTextures.attackLeft
            : facingKey === 'right'
              ? monsterAnimationTextures.idleRight
            : monsterAnimationTextures.idleLeft
    const previousMode = monsterPigAnimationModes.get(characterId)
    const currentFrame = sprite.currentFrame

    if (
      !options.forceRestart &&
      previousMode === mode &&
      sprite.textures === nextTextures
    ) {
      return
    }

    monsterPigAnimationModes.set(characterId, mode)
    sprite.textures = nextTextures
    sprite.animationSpeed =
      resolvedMode === 'run'
        ? behaviorConfig.runAnimationSpeed
        : resolvedMode === 'hit'
          ? behaviorConfig.hitAnimationSpeed
          : resolvedMode === 'attack'
            ? behaviorConfig.attackAnimationSpeed
            : behaviorConfig.idleAnimationSpeed
    sprite.loop = resolvedMode === 'idle' || resolvedMode === 'run'
    const shouldPreserveRunPhase =
      resolvedMode === 'run' &&
      isRunAnimationEnabled &&
      previousMode === mode &&
      !options.forceRestart

    sprite.gotoAndPlay(
      shouldPreserveRunPhase
        ? currentFrame % nextTextures.length
        : 0
    )

  }

  function createMonsterPigBehaviorState(): MonsterPigBehaviorState {
    return {
      isAggroed: false,
      nextAttackAtMilliseconds: 0,
      attackUntilMilliseconds: 0,
      hitReactionUntilMilliseconds: 0
    }
  }

  function getMonsterPigBehaviorState(
    characterId: string
  ): MonsterPigBehaviorState {
    const currentBehaviorState = monsterPigBehaviorStates.get(characterId)

    if (currentBehaviorState) {
      return currentBehaviorState
    }

    const nextBehaviorState = createMonsterPigBehaviorState()

    monsterPigBehaviorStates.set(characterId, nextBehaviorState)
    return nextBehaviorState
  }

  function setMonsterPigAggro(
    characterId: string,
    now: number,
    behaviorConfig: MonsterBehaviorConfig
  ): void {
    const currentBehaviorState = getMonsterPigBehaviorState(characterId)

    if (currentBehaviorState.isAggroed) {
      return
    }

    monsterPigBehaviorStates.set(characterId, {
      ...currentBehaviorState,
      isAggroed: true,
      nextAttackAtMilliseconds:
        now + behaviorConfig.attackIntervalMilliseconds
    })
    syncCharacterSprite(getCharacterStateById(characterId), now)
  }

  function setMonsterPigHitReaction(
    characterId: string,
    now: number,
    behaviorConfig: MonsterBehaviorConfig
  ): void {
    const currentBehaviorState = getMonsterPigBehaviorState(characterId)

    monsterPigBehaviorStates.set(characterId, {
      ...currentBehaviorState,
      isAggroed: true,
      nextAttackAtMilliseconds: Math.max(
        currentBehaviorState.nextAttackAtMilliseconds,
        now + behaviorConfig.attackIntervalMilliseconds
      ),
      attackUntilMilliseconds: 0,
      hitReactionUntilMilliseconds:
        now + behaviorConfig.hitReactionDurationMilliseconds
    })
    syncCharacterSprite(getCharacterStateById(characterId), now)
  }

  function setMonsterPigAttackState(
    characterId: string,
    now: number,
    behaviorConfig: MonsterBehaviorConfig
  ): void {
    const currentBehaviorState = getMonsterPigBehaviorState(characterId)

    monsterPigBehaviorStates.set(characterId, {
      ...currentBehaviorState,
      isAggroed: true,
      attackUntilMilliseconds:
        now + behaviorConfig.attackDurationMilliseconds,
      hitReactionUntilMilliseconds: 0,
      nextAttackAtMilliseconds:
        now + behaviorConfig.attackIntervalMilliseconds
    })
    syncCharacterSprite(getCharacterStateById(characterId), now)
  }

  function getKnockbackDirection(
    targetCharacter: CharacterState,
    sourceCharacter: CharacterState
  ): {
    x: number
    y: number
  } {
    const targetCenterX =
      targetCharacter.position.x + targetCharacter.collisionSize.width / 2
    const targetCenterY =
      targetCharacter.position.y + targetCharacter.collisionSize.height / 2
    const sourceCenterX =
      sourceCharacter.position.x + sourceCharacter.collisionSize.width / 2
    const sourceCenterY =
      sourceCharacter.position.y + sourceCharacter.collisionSize.height / 2
    let deltaX = targetCenterX - sourceCenterX
    let deltaY = targetCenterY - sourceCenterY

    if (deltaX === 0 && deltaY === 0) {
      switch (sourceCharacter.facing) {
        case 'up':
          deltaY = -1
          break
        case 'down':
          deltaY = 1
          break
        case 'left':
          deltaX = -1
          break
        case 'right':
          deltaX = 1
          break
      }
    }

    const distance = Math.hypot(deltaX, deltaY) || 1

    return {
      x: deltaX / distance,
      y: deltaY / distance
    }
  }

  function setPlayerHitReaction(
    sourceCharacter: CharacterState,
    now: number
  ): void {
    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)
    const direction = getKnockbackDirection(playerCharacter, sourceCharacter)

    playerHitReactionState = {
      directionX: direction.x,
      directionY: direction.y,
      startedAtMilliseconds: now,
      expiresAtMilliseconds: now + PLAYER_HIT_REACTION_DURATION_MILLISECONDS
    }
  }

  function knockbackCharacterAwayFromCharacter(
    targetCharacterId: string,
    sourceCharacter: CharacterState,
    distanceInTiles: number
  ): void {
    const targetCharacter = getCharacterStateById(targetCharacterId)
    const direction = getKnockbackDirection(targetCharacter, sourceCharacter)

    tryMoveCharacter(
      targetCharacterId,
      direction.x * distanceInTiles,
      direction.y * distanceInTiles,
      { preserveFacing: true }
    )
  }

  function knockbackMonsterAwayFromCharacter(
    characterId: string,
    sourceCharacter: CharacterState,
    distanceInTiles: number
  ): void {
    knockbackCharacterAwayFromCharacter(
      characterId,
      sourceCharacter,
      distanceInTiles
    )
  }

  const syncAllCharacterSprites = () => {
    for (const character of characterStates) {
      syncCharacterSprite(character)
    }
  }

  function isMonsterCharacter(character: CharacterState): boolean {
    return character.appearanceType.startsWith('monster_')
  }

  function isMonsterCombatStateDefeated(characterId: string): boolean {
    const combatState = monsterCombatStates.get(characterId)

    return combatState ? isMonsterDefeated(combatState) : false
  }

  function createSlashEffectHitRect(
    slashSprite: AnimatedSprite
  ): CollisionRect {
    const frameIndex = Math.min(
      Math.max(0, Math.floor(slashSprite.currentFrame)),
      WHITE_SLASH_WIDE_FRAME_BOUNDS.length - 1
    )
    const frameBounds =
      WHITE_SLASH_WIDE_FRAME_BOUNDS[frameIndex] ??
      WHITE_SLASH_WIDE_FRAME_BOUNDS[0]
    const textureWidth = slashSprite.texture.source.pixelWidth
    const textureHeight = slashSprite.texture.source.pixelHeight
    const scaleX = slashSprite.scale.x
    const scaleY = slashSprite.scale.y
    const rotation = slashSprite.rotation
    const cos = Math.cos(rotation)
    const sin = Math.sin(rotation)
    const corners = [
      {
        x: frameBounds.x - textureWidth / 2 - SLASH_VFX_HIT_PADDING_PIXELS,
        y: frameBounds.y - textureHeight / 2 - SLASH_VFX_HIT_PADDING_PIXELS
      },
      {
        x:
          frameBounds.x +
          frameBounds.width -
          textureWidth / 2 +
          SLASH_VFX_HIT_PADDING_PIXELS,
        y: frameBounds.y - textureHeight / 2 - SLASH_VFX_HIT_PADDING_PIXELS
      },
      {
        x: frameBounds.x - textureWidth / 2 - SLASH_VFX_HIT_PADDING_PIXELS,
        y:
          frameBounds.y +
          frameBounds.height -
          textureHeight / 2 +
          SLASH_VFX_HIT_PADDING_PIXELS
      },
      {
        x:
          frameBounds.x +
          frameBounds.width -
          textureWidth / 2 +
          SLASH_VFX_HIT_PADDING_PIXELS,
        y:
          frameBounds.y +
          frameBounds.height -
          textureHeight / 2 +
          SLASH_VFX_HIT_PADDING_PIXELS
      }
    ]
    const worldCorners = corners.map((corner) => {
      const scaledX = corner.x * scaleX
      const scaledY = corner.y * scaleY

      return {
        x:
          slashSprite.position.x +
          scaledX * cos -
          scaledY * sin,
        y:
          slashSprite.position.y +
          scaledX * sin +
          scaledY * cos
      }
    })
    const xCoordinates = worldCorners.map((corner) => corner.x)
    const yCoordinates = worldCorners.map((corner) => corner.y)
    const minX = Math.min(...xCoordinates)
    const maxX = Math.max(...xCoordinates)
    const minY = Math.min(...yCoordinates)
    const maxY = Math.max(...yCoordinates)

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY
    }
  }

  function resolveClosestMonsterInCollisionRect(
    hitRect: CollisionRect
  ): CharacterState | undefined {
    const hitRectCenterX = hitRect.x + hitRect.width / 2
    const hitRectCenterY = hitRect.y + hitRect.height / 2

    return characterStates
      .filter(
        (character) =>
          isMonsterCharacter(character) &&
          !isMonsterCombatStateDefeated(character.id) &&
          doCollisionRectsIntersect(
            hitRect,
            createCollisionRectFromCharacter(character)
          )
      )
      .sort((leftCharacter, rightCharacter) => {
        const leftCenterX =
          leftCharacter.position.x + leftCharacter.collisionSize.width / 2
        const leftCenterY =
          leftCharacter.position.y + leftCharacter.collisionSize.height / 2
        const rightCenterX =
          rightCharacter.position.x + rightCharacter.collisionSize.width / 2
        const rightCenterY =
          rightCharacter.position.y + rightCharacter.collisionSize.height / 2
        const leftDistance =
          (leftCenterX - hitRectCenterX) ** 2 +
          (leftCenterY - hitRectCenterY) ** 2
        const rightDistance =
          (rightCenterX - hitRectCenterX) ** 2 +
          (rightCenterY - hitRectCenterY) ** 2

        if (leftDistance !== rightDistance) {
          return leftDistance - rightDistance
        }

        return leftCharacter.id.localeCompare(rightCharacter.id)
      })[0]
  }

  function maybeRespawnMonster(characterId: string, now: number): boolean {
    const respawnAt = monsterRespawnAtById.get(characterId)

    if (respawnAt === undefined || respawnAt > now) {
      return false
    }

    const spawnCharacter = monsterSpawnStates.get(characterId)

    if (!spawnCharacter) {
      return false
    }

    const nextCharacter: CharacterState = {
      ...spawnCharacter,
      position: {
        ...spawnCharacter.position
      },
      collisionSize: {
        ...spawnCharacter.collisionSize
      }
    }

    characterStates = characterStates.map((character) =>
      character.id === characterId ? nextCharacter : character
    )
    monsterCombatStates.set(
      characterId,
      createMonsterCombatState(
        nextCharacter.level ?? 1,
        monsterCombatStateOptions
      )
    )
    monsterPatrolStates.delete(characterId)
    monsterContactDamageLockedUntilById.delete(characterId)
    monsterPigBehaviorStates.set(
      characterId,
      createMonsterPigBehaviorState()
    )
    monsterPigAnimationModes.delete(characterId)
    monsterRespawnAtById.delete(characterId)
    syncCharacterSprite(nextCharacter)
    syncMonsterAnimation(characterId, 'idle')

    return true
  }

  function spawnMonsterGoldDrop(
    characterId: string,
    amount: number,
    position: {
      x: number
      y: number
    },
    now: number
  ): void {
    const dropId = `${characterId}:${++monsterGoldDropSequence}`
    const container = new Container()
    const coin = new Graphics()
    const coinHighlight = new Graphics()
    const amountText = new Text({
      style: MONSTER_GOLD_DROP_AMOUNT_TEXT_STYLE,
      text: `${amount}원`
    })

    container.label = `monster-gold-drop:${dropId}`
    container.sortableChildren = true
    coin.roundPixels = true
    coinHighlight.roundPixels = true
    amountText.roundPixels = true
    coin.circle(0, 0, MONSTER_GOLD_DROP_ICON_RADIUS)
    coin.fill({ color: 0xf0c24b })
    coin.stroke({ color: 0x8b5a00, width: 2 })
    coinHighlight.circle(
      -MONSTER_GOLD_DROP_ICON_RADIUS * 0.28,
      -MONSTER_GOLD_DROP_ICON_RADIUS * 0.28,
      MONSTER_GOLD_DROP_ICON_SHINE_RADIUS
    )
    coinHighlight.fill({ color: 0xfff1b0 })
    coinHighlight.alpha = 0.9
    amountText.position.set(
      -Math.round(amountText.width / 2),
      MONSTER_GOLD_DROP_ICON_RADIUS + 4
    )
    coin.zIndex = 0
    coinHighlight.zIndex = 1
    amountText.zIndex = 2
    container.addChild(coin, coinHighlight, amountText)
    container.position.set(position.x, position.y)
    container.zIndex = Math.round(position.y + map.tileHeight)
    depthSortedLayer?.addChild(container)
    monsterGoldDrops.set(dropId, {
      id: dropId,
      container,
      coin,
      amountText,
      amount,
      position: {
        x: position.x,
        y: position.y
      },
      createdAt: now
    })
  }

  const syncMonsterGoldDropElement = (
    drop: MonsterGoldDrop,
    now: number
  ) => {
    const bobOffset = Math.sin((now - drop.createdAt) / 220) * 1.5

    drop.container.position.set(drop.position.x, drop.position.y + bobOffset)
    drop.container.zIndex = Math.round(drop.position.y + map.tileHeight)
    drop.amountText.text = `${drop.amount}원`
    drop.amountText.position.set(
      -Math.round(drop.amountText.width / 2),
      MONSTER_GOLD_DROP_ICON_RADIUS + 4
    )
  }

  const syncActiveMonsterGoldDrops = (now: number) => {
    for (const drop of monsterGoldDrops.values()) {
      syncMonsterGoldDropElement(drop, now)
    }

    depthSortedLayer?.sortChildren()
  }

  function spawnMonsterEquipmentDrop(
    characterId: string,
    dropDefinition: ReturnType<typeof rollMonsterEquipmentDrop>,
    position: {
      x: number
      y: number
    },
    now: number
  ): void {
    if (!dropDefinition) {
      return
    }

    const dropTexture = monsterEquipmentDropTexturesByDropId.get(
      dropDefinition.dropId
    )

    if (!dropTexture) {
      return
    }

    const dropId = `${characterId}:equipment:${++monsterEquipmentDropSequence}`
    const container = new Container()
    const sprite = new Sprite(dropTexture)
    const labelText = new Text({
      style: MONSTER_GOLD_DROP_AMOUNT_TEXT_STYLE,
      text: dropDefinition.label
    })

    container.label = `monster-equipment-drop:${dropId}`
    container.sortableChildren = true
    sprite.anchor.set(0.5)
    sprite.width = MONSTER_EQUIPMENT_DROP_RENDER_SIZE
    sprite.height = MONSTER_EQUIPMENT_DROP_RENDER_SIZE
    sprite.roundPixels = true
    labelText.roundPixels = true
    labelText.position.set(
      -Math.round(labelText.width / 2),
      Math.round(MONSTER_EQUIPMENT_DROP_RENDER_SIZE / 2) + 3
    )
    sprite.zIndex = 0
    labelText.zIndex = 1
    container.addChild(sprite, labelText)
    container.position.set(position.x, position.y)
    container.zIndex = Math.round(position.y + map.tileHeight)
    depthSortedLayer?.addChild(container)
    monsterEquipmentDrops.set(dropId, {
      id: dropId,
      dropId: dropDefinition.dropId,
      itemId: dropDefinition.itemId,
      label: dropDefinition.label,
      container,
      sprite,
      labelText,
      position: {
        x: position.x,
        y: position.y
      },
      createdAt: now
    })
  }

  const syncMonsterEquipmentDropElement = (
    drop: MonsterEquipmentDrop,
    now: number
  ) => {
    const bobOffset = Math.sin((now - drop.createdAt) / 240) * 1.75

    drop.container.position.set(drop.position.x, drop.position.y + bobOffset)
    drop.container.zIndex = Math.round(drop.position.y + map.tileHeight)
    drop.labelText.text = drop.label
    drop.labelText.position.set(
      -Math.round(drop.labelText.width / 2),
      Math.round(MONSTER_EQUIPMENT_DROP_RENDER_SIZE / 2) + 3
    )
  }

  const syncActiveMonsterEquipmentDrops = (now: number) => {
    for (const drop of monsterEquipmentDrops.values()) {
      syncMonsterEquipmentDropElement(drop, now)
    }

    depthSortedLayer?.sortChildren()
  }

  const resolveMonsterEquipmentDropPickups = () => {
    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)
    const playerRect = {
      x: playerCharacter.position.x * map.tileWidth,
      y: playerCharacter.position.y * map.tileHeight,
      width: playerCharacter.collisionSize.width * map.tileWidth,
      height: playerCharacter.collisionSize.height * map.tileHeight
    }

    for (const [dropMapId, drop] of monsterEquipmentDrops) {
      const dropRect = {
        x: drop.position.x - MONSTER_EQUIPMENT_DROP_PICKUP_WIDTH / 2,
        y: drop.position.y - MONSTER_EQUIPMENT_DROP_PICKUP_HEIGHT / 2,
        width: MONSTER_EQUIPMENT_DROP_PICKUP_WIDTH,
        height: MONSTER_EQUIPMENT_DROP_PICKUP_HEIGHT
      }

      if (!doCollisionRectsIntersect(playerRect, dropRect)) {
        continue
      }

      const equipmentDefinition = getPlayerEquipmentItemDefinitionById(
        drop.itemId
      )

      if (!equipmentDefinition) {
        continue
      }

      const emptySlotIndex = findFirstEmptyPlayerInventorySlotIndex(
        currentPlayerInventory
      )

      if (emptySlotIndex === undefined) {
        showCharacterDamageText(
          PLAYER_CHARACTER_ID,
          '가방이 가득 찼습니다',
          DAMAGE_TEXT_DURATION_MILLISECONDS
        )
        continue
      }

      currentPlayerInventory = setPlayerInventorySlot({
        inventory: currentPlayerInventory,
        slotIndex: emptySlotIndex,
        item: {
          id: drop.itemId,
          label: equipmentDefinition.label,
          quantity: 1
        }
      })
      onPlayerInventoryChange(currentPlayerInventory)
      syncPlayerUiOverlays()
      showCharacterDamageText(
        PLAYER_CHARACTER_ID,
        `${equipmentDefinition.label} 획득!`,
        DAMAGE_TEXT_DURATION_MILLISECONDS,
        LEVEL_UP_TEXT_STYLE
      )
      drop.container.removeFromParent()
      drop.container.destroy({ children: true })
      monsterEquipmentDrops.delete(dropMapId)
    }
  }

  const resolveMonsterGoldDropPickups = () => {
    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)
    const playerRect = {
      x: playerCharacter.position.x * map.tileWidth,
      y: playerCharacter.position.y * map.tileHeight,
      width: playerCharacter.collisionSize.width * map.tileWidth,
      height: playerCharacter.collisionSize.height * map.tileHeight
    }

    for (const [dropId, drop] of monsterGoldDrops) {
      const dropRect = {
        x: drop.position.x - MONSTER_GOLD_DROP_PICKUP_WIDTH / 2,
        y: drop.position.y - MONSTER_GOLD_DROP_PICKUP_HEIGHT / 2,
        width: MONSTER_GOLD_DROP_PICKUP_WIDTH,
        height: MONSTER_GOLD_DROP_PICKUP_HEIGHT
      }

      if (!doCollisionRectsIntersect(playerRect, dropRect)) {
        continue
      }

      currentPlayerInventory = {
        ...currentPlayerInventory,
        gold: currentPlayerInventory.gold + drop.amount
      }
      onPlayerInventoryChange(currentPlayerInventory)
      syncPlayerUiOverlays()
      drop.container.removeFromParent()
      drop.container.destroy({ children: true })
      monsterGoldDrops.delete(dropId)
    }
  }

  function getMonsterDistanceToPlayer(character: CharacterState): number {
    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)
    const monsterCenterX = character.position.x + character.collisionSize.width / 2
    const monsterCenterY = character.position.y + character.collisionSize.height / 2
    const playerCenterX =
      playerCharacter.position.x + playerCharacter.collisionSize.width / 2
    const playerCenterY =
      playerCharacter.position.y + playerCharacter.collisionSize.height / 2

    return Math.hypot(
      playerCenterX - monsterCenterX,
      playerCenterY - monsterCenterY
    )
  }

  function isMonsterWithinRange(
    character: CharacterState,
    tiles: number
  ): boolean {
    return getMonsterDistanceToPlayer(character) <= tiles
  }

  function isMonsterWithinAttackRange(
    monsterCharacter: CharacterState,
    playerCharacter: CharacterState,
    behaviorConfig: MonsterBehaviorConfig
  ): boolean {
    const monsterCenterX =
      monsterCharacter.position.x + monsterCharacter.collisionSize.width / 2
    const monsterCenterY =
      monsterCharacter.position.y + monsterCharacter.collisionSize.height / 2
    const playerCenterX =
      playerCharacter.position.x + playerCharacter.collisionSize.width / 2
    const playerCenterY =
      playerCharacter.position.y + playerCharacter.collisionSize.height / 2

    return (
      Math.hypot(
        playerCenterX - monsterCenterX,
        playerCenterY - monsterCenterY
      ) <=
      behaviorConfig.attackRangeTiles +
        MONSTER_ATTACK_RANGE_TOUCH_TOLERANCE_TILES
    )
  }

  function beginPlayerDeath(now: number): void {
    if (playerRespawnAtMilliseconds !== undefined) {
      return
    }

    playerRespawnAtMilliseconds =
      now + PLAYER_RESPAWN_DELAY_MILLISECONDS
    playerHitReactionState = undefined
    clearPressedInputState()
    stopPlayerFootsteps()
    playerAttackStartedAtMilliseconds = undefined
    playerAttackResolvedStartedAtMilliseconds = undefined
    playerAttackFacing = undefined
    clearPlayerSlashEffectSprite()
    playerAttackReadyAtMilliseconds = now + PLAYER_RESPAWN_DELAY_MILLISECONDS
    syncCharacterSprite(getCharacterStateById(PLAYER_CHARACTER_ID), now)
  }

  function maybeRespawnPlayer(now: number): boolean {
    if (playerProfile.hp.current > 0) {
      return false
    }

    const respawnAt = playerRespawnAtMilliseconds

    if (respawnAt === undefined || respawnAt > now) {
      return false
    }

    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)

    playerCharacter.position = {
      ...playerRespawnState.position
    }
    playerCharacter.facing = playerRespawnState.facing
    playerProfile.hp.current = playerProfile.hp.max
    playerRespawnAtMilliseconds = undefined
    playerHitReactionState = undefined
    clearPressedInputState()
    playerAttackStartedAtMilliseconds = undefined
    playerAttackResolvedStartedAtMilliseconds = undefined
    playerAttackFacing = undefined
    clearPlayerSlashEffectSprite()
    playerAttackReadyAtMilliseconds = now
    syncCharacterSprite(playerCharacter, now)
    syncPlayerUiOverlays()
    showCharacterDamageText(
      PLAYER_CHARACTER_ID,
      '부활했다!',
      DAMAGE_TEXT_DURATION_MILLISECONDS
    )

    return true
  }

  function applyDamageToPlayer(
    damage: number,
    now: number,
    sourceCharacter?: CharacterState
  ): void {
    const nextDamage = Math.max(0, Math.floor(damage))

    if (nextDamage === 0 || playerProfile.hp.current === 0) {
      return
    }

    if (
      sourceCharacter &&
      shouldPlayerEvadeDamage(playerProfile)
    ) {
      showCharacterDamageText(
        PLAYER_CHARACTER_ID,
        '회피!',
        EVADE_TEXT_DURATION_MILLISECONDS,
        EVADE_TEXT_STYLE
      )
      return
    }

    const nextHp = Math.max(0, playerProfile.hp.current - nextDamage)
    const damageMessage =
      nextHp === 0 ? `-${nextDamage}\n쓰러졌다!` : `-${nextDamage}`

    playerProfile.hp.current = nextHp
    gameSoundEffects.play(nextHp === 0 ? 'playerGameOver' : 'playerDamage')
    showCharacterDamageText(
      PLAYER_CHARACTER_ID,
      damageMessage,
      DAMAGE_TEXT_DURATION_MILLISECONDS
    )
    if (sourceCharacter && nextHp > 0) {
      setPlayerHitReaction(sourceCharacter, now)
    }
    syncPlayerUiOverlays()

    if (nextHp === 0) {
      playerHitReactionState = undefined
      beginPlayerDeath(now)
    }
  }

  function applyDamageToMonster(
    characterId: string,
    damage: number,
    now: number
  ): void {
    const character = getCharacterStateById(characterId)
    const combatState = monsterCombatStates.get(characterId)
    const monsterBehaviorConfig = getMonsterBehaviorConfig(character)

    if (!combatState) {
      return
    }

    const nextCombatState = applyMonsterDamage(combatState, damage)

    if (nextCombatState === combatState) {
      return
    }

    monsterCombatStates.set(characterId, nextCombatState)
    gameSoundEffects.play('playerSwordHit')
    const nextDamage = Math.max(0, Math.floor(damage))
    const damageMessage =
      nextCombatState.currentHp === 0
        ? `-${nextDamage}\n쓰러졌다!`
        : `-${nextDamage}`

    showCharacterDamageText(
      characterId,
      damageMessage,
      DAMAGE_TEXT_DURATION_MILLISECONDS
    )

    if (isMonsterDefeated(nextCombatState)) {
      if (character.appearanceType === MONSTER_SLIME_APPEARANCE_TYPE) {
        gameSoundEffects.play('slimeDeath')
      }
      setQuestLogWithObjectiveFeedback(
        recordMonsterDefeatQuestProgress(currentQuestLog, {
          sceneId,
          appearanceType: character.appearanceType
        })
      )
      character.blocksMovement = false
      monsterPatrolStates.delete(characterId)
      monsterContactDamageLockedUntilById.delete(characterId)
      monsterPigAnimationModes.delete(characterId)
      monsterPigBehaviorStates.delete(characterId)
      const experienceReward = getMonsterExperienceDropAmount(
        character.level ?? 1
      )
      grantPlayerExperienceReward(experienceReward)
      const dropPosition = {
        x:
          character.position.x * map.tileWidth +
          (character.collisionSize.width * map.tileWidth) / 2,
        y:
          character.position.y * map.tileHeight +
          (character.collisionSize.height * map.tileHeight) / 2
      }

      const equipmentDrop = rollMonsterEquipmentDrop(Math.random)

      if (equipmentDrop) {
        spawnMonsterEquipmentDrop(characterId, equipmentDrop, dropPosition, now)
      } else {
        spawnMonsterGoldDrop(
          characterId,
          getMonsterGoldDropAmount(character.level ?? 1),
          dropPosition,
          now
        )
      }
      monsterRespawnAtById.set(
        characterId,
        now + MONSTER_PIG_RESPAWN_DELAY_MILLISECONDS
      )
      const renderNode = renderedCharacters.get(characterId)

      if (renderNode) {
        renderNode.container.visible = false
      }
      return
    }

    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)

    monsterContactDamageLockedUntilById.set(
      characterId,
      now + monsterBehaviorConfig.hitReactionDurationMilliseconds
    )
    setMonsterPigHitReaction(characterId, now, monsterBehaviorConfig)
    knockbackMonsterAwayFromCharacter(characterId, playerCharacter, 0.45)
    monsterPigAnimationModes.delete(characterId)
    syncMonsterAnimation(characterId, 'hit', { forceRestart: true })
    syncCharacterSprite(getCharacterStateById(characterId), now)
  }

  function resolvePlayerAttackDamage(now: number): void {
    if (playerProfile.hp.current === 0) {
      return
    }

    if (
      playerAttackStartedAtMilliseconds === undefined ||
      playerAttackResolvedStartedAtMilliseconds ===
        playerAttackStartedAtMilliseconds
    ) {
      return
    }

    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)
    const targetCharacter = playerSlashEffectSprite
      ? resolveClosestMonsterInCollisionRect(
          createSlashEffectHitRect(playerSlashEffectSprite)
        )
      : resolveCharacterInteractionTarget({
          sourceCharacter: playerCharacter,
          targetCharacters: characterStates,
          canReceiveInteraction: (character) =>
            isMonsterCharacter(character) &&
            !isMonsterCombatStateDefeated(character.id),
          interactionProbeDistanceInTiles: PLAYER_ATTACK_PROBE_DISTANCE_IN_TILES
        })

    if (targetCharacter) {
      applyDamageToMonster(
        targetCharacter.id,
        getPlayerPhysicalAttackPower(playerProfile),
        now
      )
      playerAttackResolvedStartedAtMilliseconds =
        playerAttackStartedAtMilliseconds
    }
  }

  function resolveMonsterContactDamage(now: number): void {
    if (playerProfile.hp.current === 0) {
      return
    }

    const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)
    const playerRect = createCollisionRectFromCharacter(playerCharacter)
    const expandedPlayerRect = {
      x: playerRect.x - MONSTER_CONTACT_DAMAGE_TOUCH_TOLERANCE_TILES,
      y: playerRect.y - MONSTER_CONTACT_DAMAGE_TOUCH_TOLERANCE_TILES,
      width: playerRect.width + MONSTER_CONTACT_DAMAGE_TOUCH_TOLERANCE_TILES * 2,
      height:
        playerRect.height + MONSTER_CONTACT_DAMAGE_TOUCH_TOLERANCE_TILES * 2
    }

    for (const monsterCharacter of characterStates) {
      if (!isMonsterCharacter(monsterCharacter)) {
        continue
      }

      const combatState = monsterCombatStates.get(monsterCharacter.id)

      if (!combatState || isMonsterDefeated(combatState)) {
        continue
      }

      if (
        !doCollisionRectsIntersect(
          expandedPlayerRect,
          createCollisionRectFromCharacter(monsterCharacter)
        )
      ) {
        continue
      }

      const lockedUntil =
        monsterContactDamageLockedUntilById.get(monsterCharacter.id) ?? 0

      if (lockedUntil > now) {
        continue
      }

      monsterContactDamageLockedUntilById.set(
        monsterCharacter.id,
        now + MONSTER_CONTACT_DAMAGE_COOLDOWN_MILLISECONDS
      )
      applyDamageToPlayer(combatState.contactDamage, now, monsterCharacter)
      knockbackCharacterAwayFromCharacter(PLAYER_CHARACTER_ID, monsterCharacter, 0.18)
    }
  }

  const syncPlayerWeaponSprite = (character: CharacterState) => {
    if (!playerWeaponSprite) {
      return
    }

    if (playerProfile.hp.current === 0) {
      playerWeaponSprite.visible = false
      for (const trailSprite of playerWeaponTrailSprites) {
        trailSprite.visible = false
      }
      return
    }

    const weaponSlot = currentPlayerEquipment.slots.find(
      (slot) => slot.id === 'weapon'
    )
    const weaponItem = weaponSlot?.item

    if (!weaponItem) {
      playerWeaponSprite.visible = false
      for (const trailSprite of playerWeaponTrailSprites) {
        trailSprite.visible = false
      }
      return
    }

    const weaponAppearance = PLAYER_WEAPON_APPEARANCE_CONFIG_BY_ITEM_ID[
      weaponItem.id
    ]
    const weaponTexture = weaponAppearance
      ? playerWeaponAppearanceTexturesByItemId.get(weaponItem.id) ??
        playerWeaponTexture
      : playerWeaponTexture

    if (!weaponTexture) {
      playerWeaponSprite.visible = false
      for (const trailSprite of playerWeaponTrailSprites) {
        trailSprite.visible = false
      }
      return
    }

    const attackFacing = playerAttackFacing ?? character.facing
    const placement =
      attackFacing === 'left'
        ? PLAYER_WEAPON_PLACEMENT_LEFT
        : PLAYER_WEAPON_PLACEMENT_RIGHT
    const weaponWorldScale =
      weaponAppearance?.worldScale ?? PLAYER_WEAPON_WORLD_SCALE
    const now = performance.now()
    const attackElapsedMilliseconds =
      playerAttackStartedAtMilliseconds === undefined
        ? undefined
        : now - playerAttackStartedAtMilliseconds
    const attackProgress =
      attackElapsedMilliseconds === undefined ||
      attackElapsedMilliseconds < 0 ||
      attackElapsedMilliseconds >= PLAYER_ATTACK_DURATION_MILLISECONDS
        ? undefined
        : attackElapsedMilliseconds / PLAYER_ATTACK_DURATION_MILLISECONDS
    const facingMultiplier = attackFacing === 'left' ? -1 : 1
    const createPose = (progress: number | undefined) => {
      if (progress === undefined) {
        return {
          x: placement.x + (weaponAppearance?.idleOffsetX ?? 0),
          y: placement.y + (weaponAppearance?.idleOffsetY ?? 0),
          rotation: placement.rotation,
          scale: weaponWorldScale
        }
      }

      const swingAmount = Math.sin(progress * Math.PI)
      const liftAmount = Math.sin(progress * Math.PI * 0.5)

      return {
        x:
          placement.x +
          facingMultiplier * PLAYER_ATTACK_SWING_X_OFFSET * swingAmount,
        y: placement.y - PLAYER_ATTACK_LIFT_Y_OFFSET * liftAmount,
        rotation:
          placement.rotation +
          facingMultiplier * PLAYER_ATTACK_ROTATION_OFFSET * swingAmount,
        scale: weaponWorldScale + PLAYER_ATTACK_SCALE_BOOST * swingAmount
      }
    }
    const applyPose = (
      sprite: Sprite,
      pose: {
        x: number
        y: number
        rotation: number
        scale: number
      },
      alpha: number
    ) => {
      sprite.texture = weaponTexture
      sprite.visible = true
      sprite.position.set(pose.x, pose.y)
      sprite.rotation = pose.rotation
      sprite.scale.set(pose.scale)
      sprite.alpha = alpha
    }

    applyPose(playerWeaponSprite, createPose(attackProgress), 1)

    if (
      attackProgress === undefined &&
      playerAttackStartedAtMilliseconds !== undefined
    ) {
      playerAttackStartedAtMilliseconds = undefined
      playerAttackResolvedStartedAtMilliseconds = undefined
      playerAttackFacing = undefined
    }

    for (let index = 0; index < playerWeaponTrailSprites.length; index += 1) {
      const trailSprite = playerWeaponTrailSprites[index]
      const trailProgress =
        attackProgress === undefined
          ? undefined
          : attackProgress - (index + 1) * PLAYER_ATTACK_TRAIL_PROGRESS_STEP

      if (trailProgress === undefined || trailProgress <= 0) {
        trailSprite.visible = false
        continue
      }

      applyPose(
        trailSprite,
        createPose(trailProgress),
        PLAYER_ATTACK_TRAIL_ALPHA[index] ?? 0.1
      )
    }
  }

  const syncCharacterMessageElement = (characterId: string) => {
    const activeMessage = activeCharacterMessages.get(characterId)

    if (!activeMessage) {
      return
    }

    const character = characterStates.find(
      (candidateCharacter) => candidateCharacter.id === characterId
    )

    if (!character) {
      activeMessage.container.removeFromParent()
      activeMessage.container.destroy({ children: true })
      activeCharacterMessages.delete(characterId)
      return
    }

    activeMessage.container.position.set(
      Math.round(
        character.position.x * map.tileWidth +
          characterPixelWidth / 2 -
          activeMessage.panel.width / 2
      ),
      Math.round(
        character.position.y * map.tileHeight -
          activeMessage.panel.height -
          MESSAGE_OFFSET_Y
      )
    )
    activeMessage.container.zIndex = getCharacterDepthSortValue(
      character.position.y,
      characterPixelHeight,
      map.tileHeight
    )
    messageLayer.sortChildren()
  }

  const syncActiveCharacterMessages = () => {
    for (const characterId of activeCharacterMessages.keys()) {
      syncCharacterMessageElement(characterId)
    }
  }

  const showCharacterMessage = (
    characterId: string,
    message: string,
    durationMilliseconds: number
  ) => {
    let activeMessage = activeCharacterMessages.get(characterId)

    if (!activeMessage) {
      const container = new Container()
      const panel = new NineSliceSprite({
        texture: messagePanelTexture,
        bottomHeight: MESSAGE_PANEL_BORDER_SIZE,
        leftWidth: MESSAGE_PANEL_BORDER_SIZE,
        rightWidth: MESSAGE_PANEL_BORDER_SIZE,
        topHeight: MESSAGE_PANEL_BORDER_SIZE
      })
      const text = new Text({
        style: MESSAGE_TEXT_STYLE,
        text: ''
      })

      panel.roundPixels = true
      text.roundPixels = true
      container.addChild(panel, text)
      messageLayer.addChild(container)
      activeMessage = {
        container,
        panel,
        text,
        expiresAt: 0
      }
      activeCharacterMessages.set(characterId, activeMessage)
    }

    activeMessage.text.text = message
    const panelWidth = Math.max(
      MESSAGE_PANEL_MIN_WIDTH,
      Math.ceil(activeMessage.text.width) + MESSAGE_PANEL_PADDING_X * 2
    )
    const panelHeight = Math.max(
      MESSAGE_PANEL_MIN_HEIGHT,
      Math.ceil(activeMessage.text.height) + MESSAGE_PANEL_PADDING_Y * 2
    )

    activeMessage.panel.setSize(panelWidth, panelHeight)
    activeMessage.text.position.set(
      Math.round((panelWidth - activeMessage.text.width) / 2),
      Math.round((panelHeight - activeMessage.text.height) / 2)
    )
    activeMessage.expiresAt = performance.now() + durationMilliseconds
    syncCharacterMessageElement(characterId)
  }

  const hideCharacterMessage = (characterId: string) => {
    const activeMessage = activeCharacterMessages.get(characterId)

    if (!activeMessage) {
      return
    }

    activeMessage.container.removeFromParent()
    activeMessage.container.destroy({ children: true })
    activeCharacterMessages.delete(characterId)
  }

  const pruneExpiredCharacterMessages = (now: number) => {
    for (const [characterId, activeMessage] of activeCharacterMessages) {
      if (activeMessage.expiresAt > now) {
        continue
      }

      activeMessage.container.removeFromParent()
      activeMessage.container.destroy({ children: true })
      activeCharacterMessages.delete(characterId)
    }
  }

  const syncCharacterDamageTextElement = (
    characterId: string,
    now: number
  ) => {
    const activeDamageText = activeCharacterDamageTexts.get(characterId)

    if (!activeDamageText) {
      return
    }

    const character = characterStates.find(
      (candidateCharacter) => candidateCharacter.id === characterId
    )

    if (!character) {
      activeDamageText.container.removeFromParent()
      activeDamageText.container.destroy({ children: true })
      activeCharacterDamageTexts.delete(characterId)
      return
    }

    const elapsedMilliseconds = now - activeDamageText.startedAt
    const progress = Math.min(
      1,
      Math.max(
        0,
        elapsedMilliseconds / activeDamageText.durationMilliseconds
      )
    )
    const floatOffset = Math.round(progress * DAMAGE_TEXT_FLOAT_DISTANCE)

    activeDamageText.container.position.set(
      Math.round(
        character.position.x * map.tileWidth +
          characterPixelWidth / 2 -
          activeDamageText.text.width / 2
      ),
      Math.round(
        character.position.y * map.tileHeight -
          activeDamageText.text.height -
          DAMAGE_TEXT_OFFSET_Y -
          floatOffset
      )
    )
    activeDamageText.container.alpha = 1 - progress
    activeDamageText.container.zIndex = getCharacterDepthSortValue(
      character.position.y,
      characterPixelHeight,
      map.tileHeight
    )
    messageLayer.sortChildren()
  }

  const syncActiveCharacterDamageTexts = (now: number) => {
    for (const characterId of activeCharacterDamageTexts.keys()) {
      syncCharacterDamageTextElement(characterId, now)
    }
  }

  const showCharacterDamageText = (
    characterId: string,
    message: string,
    durationMilliseconds: number,
    style: TextStyle = DAMAGE_TEXT_STYLE
  ) => {
    let activeDamageText = activeCharacterDamageTexts.get(characterId)

    if (!activeDamageText) {
      const container = new Container()
      const text = new Text({
        style: DAMAGE_TEXT_STYLE,
        text: ''
      })

      text.roundPixels = true
      container.addChild(text)
      messageLayer.addChild(container)
      activeDamageText = {
        container,
        text,
        startedAt: 0,
        durationMilliseconds,
        expiresAt: 0
      }
      activeCharacterDamageTexts.set(characterId, activeDamageText)
    }

    activeDamageText.text.style = style
    activeDamageText.text.text = message
    activeDamageText.startedAt = performance.now()
    activeDamageText.durationMilliseconds = durationMilliseconds
    activeDamageText.expiresAt =
      activeDamageText.startedAt + durationMilliseconds
    activeDamageText.container.alpha = 1
    syncCharacterDamageTextElement(characterId, activeDamageText.startedAt)
  }

  const pruneExpiredCharacterDamageTexts = (now: number) => {
    for (const [characterId, activeDamageText] of activeCharacterDamageTexts) {
      if (activeDamageText.expiresAt > now) {
        continue
      }

      activeDamageText.container.removeFromParent()
      activeDamageText.container.destroy({ children: true })
      activeCharacterDamageTexts.delete(characterId)
    }
  }

  const centerCameraOnCharacter = (character: CharacterState) => {
    const characterCenterX =
      (character.position.x * map.tileWidth + characterPixelWidth / 2) *
      cameraZoom
    const characterCenterY =
      (character.position.y * map.tileHeight + characterPixelHeight / 2) *
      cameraZoom
    const nextScrollLeft = clampScrollOffset(
      characterCenterX - viewportElement.clientWidth / 2,
      scaledMapPixelWidth - viewportElement.clientWidth
    )
    const nextScrollTop = clampScrollOffset(
      characterCenterY - viewportElement.clientHeight / 2,
      scaledMapPixelHeight - viewportElement.clientHeight
    )

    viewportElement.scrollTo({
      left: nextScrollLeft,
      top: nextScrollTop
    })
  }

  const getBlockingCollisionRects = (
    excludedCharacterId: string
  ): CollisionRect[] =>
    characterStates
      .filter(
        (character) =>
          character.blocksMovement && character.id !== excludedCharacterId
      )
      .map((character) => createCollisionRectFromCharacter(character))

  const tryMoveCharacter = (
    characterId: string,
    deltaX: number,
    deltaY: number,
    options: {
      preserveFacing?: boolean
    } = {}
  ): boolean => {
    const currentCharacter = getCharacterStateById(characterId)
    const desiredFacing = moveCharacterState({
      character: currentCharacter,
      delta: {
        x: deltaX,
        y: deltaY
      },
      mapWidth: map.width,
      mapHeight: map.height
    }).facing
    const nextFacing = options.preserveFacing
      ? currentCharacter.facing
      : desiredFacing
    const blockingRects = getBlockingCollisionRects(characterId)
    let nextCharacter =
      nextFacing === currentCharacter.facing
        ? currentCharacter
        : {
            ...currentCharacter,
            facing: nextFacing
          }

    if (deltaX !== 0) {
      const nextXCharacter = moveCharacterState({
        character: nextCharacter,
        delta: {
          x: deltaX,
          y: 0
        },
        mapWidth: map.width,
        mapHeight: map.height
      })

      if (
        !isCharacterPositionBlocked(
          wallTiles,
          blockingRects,
          nextXCharacter.position.x,
          nextXCharacter.position.y,
          nextXCharacter.collisionSize.width,
          nextXCharacter.collisionSize.height
        )
      ) {
        nextCharacter = nextXCharacter
      }
    }

    if (deltaY !== 0) {
      const nextYCharacter = moveCharacterState({
        character: nextCharacter,
        delta: {
          x: 0,
          y: deltaY
        },
        mapWidth: map.width,
        mapHeight: map.height
      })

      if (
        !isCharacterPositionBlocked(
          wallTiles,
          blockingRects,
          nextYCharacter.position.x,
          nextYCharacter.position.y,
          nextYCharacter.collisionSize.width,
          nextYCharacter.collisionSize.height
        )
      ) {
        nextCharacter = nextYCharacter
      }
    }

    if (options.preserveFacing) {
      nextCharacter = {
        ...nextCharacter,
        facing: currentCharacter.facing
      }
    } else if (nextCharacter.facing !== desiredFacing) {
      nextCharacter = {
        ...nextCharacter,
        facing: desiredFacing
      }
    }

    if (
      nextCharacter.position.x === currentCharacter.position.x &&
      nextCharacter.position.y === currentCharacter.position.y &&
      nextCharacter.facing === currentCharacter.facing
    ) {
      return false
    }

    characterStates = characterStates.map((character) =>
      character.id === nextCharacter.id ? nextCharacter : character
    )

    const didPositionChange =
      nextCharacter.position.x !== currentCharacter.position.x ||
      nextCharacter.position.y !== currentCharacter.position.y
    const didFacingChange = nextCharacter.facing !== currentCharacter.facing

    if (didPositionChange || didFacingChange) {
      syncCharacterSprite(nextCharacter)
    }

    if (nextCharacter.id === PLAYER_CHARACTER_ID && didPositionChange) {
      const touchedPortal = findTouchedMapPortal(nextCharacter)

      if (touchedPortal) {
        requestSceneTransition(touchedPortal)
        return didPositionChange
      }
    }

    if (nextCharacter.id === cameraTargetCharacterId && didPositionChange) {
      centerCameraOnCharacter(nextCharacter)
    }

    return didPositionChange
  }

  const drainControllerRuntimeEventsIntoQueue = () => {
    for (const event of controllerRuntime.drainEvents()) {
      gameEventQueue.enqueue(event)
    }
  }

  const updateCharacters = () => {
    try {
      const now = performance.now()

      if (isPauseMenuOpen) {
        clearPressedInputState()
        stopPlayerFootsteps()
        triggeredActions.clear()
        lastRuntimeErrorMessage = undefined
        return
      }

      controllerRuntime.syncCharacters(characterStates)
      drainControllerRuntimeEventsIntoQueue()
      maybeRespawnPlayer(now)
      let didPlayerMoveThisFrame = false

      for (const character of [...characterStates]) {
        if (character.id === PLAYER_CHARACTER_ID && playerProfile.hp.current === 0) {
          continue
        }

        const intent = controllerRuntime.getIntent({
          character,
          deltaMilliseconds: app.ticker.deltaMS,
          pressedDirections,
          triggeredActions
        })

        if (intent) {
          if (intent.movement) {
            const didMove = tryMoveCharacter(
              character.id,
              intent.movement.x,
              intent.movement.y
            )

            if (character.id === PLAYER_CHARACTER_ID && didMove) {
              didPlayerMoveThisFrame = true
            }
          }

          if (isSceneTransitionPending) {
            stopPlayerFootsteps()
            return
          }

          drainControllerRuntimeEventsIntoQueue()

          if (intent.actions?.includes('interact')) {
            gameEventQueue.enqueue({
              kind: 'interaction-requested',
              sourceCharacterId: character.id
            })
          }

          if (
            intent.actions?.includes('attack') &&
            character.id === PLAYER_CHARACTER_ID
          ) {
            triggerPlayerAttack(now)
          }
        }

        if (
          character.appearanceType.startsWith('monster_') &&
          monsterAnimationTexturesByAppearanceType[
            character.appearanceType as MonsterAppearanceType
          ]
        ) {
          if (maybeRespawnMonster(character.id, now)) {
            continue
          }

          if (isMonsterCombatStateDefeated(character.id)) {
            continue
          }

          const monsterCharacter = getCharacterStateById(character.id)
          const monsterBehaviorConfig =
            getMonsterBehaviorConfig(monsterCharacter)
          let behaviorState = getMonsterPigBehaviorState(character.id)
          const monsterDistanceToPlayer =
            getMonsterDistanceToPlayer(monsterCharacter)

          if (
            behaviorState.isAggroed &&
            monsterDistanceToPlayer > monsterBehaviorConfig.deAggroRangeTiles
          ) {
            monsterPigBehaviorStates.set(character.id, {
              ...behaviorState,
              isAggroed: false,
              nextAttackAtMilliseconds: 0,
              attackUntilMilliseconds: 0,
              hitReactionUntilMilliseconds: 0
            })
            behaviorState = getMonsterPigBehaviorState(character.id)
            syncMonsterAnimation(character.id, 'idle', {
              forceRestart: true
            })
          }

          if (behaviorState.hitReactionUntilMilliseconds > now) {
            syncMonsterAnimation(character.id, 'hit')
            continue
          }

          if (behaviorState.attackUntilMilliseconds > now) {
            syncMonsterAnimation(character.id, 'attack')
            continue
          }

          if (
            !behaviorState.isAggroed &&
            isMonsterWithinRange(
              monsterCharacter,
              monsterBehaviorConfig.aggroRangeTiles
            )
          ) {
            setMonsterPigAggro(character.id, now, monsterBehaviorConfig)
            behaviorState = getMonsterPigBehaviorState(character.id)
          }

          if (behaviorState.isAggroed) {
            const playerCharacter = getCharacterStateById(PLAYER_CHARACTER_ID)
            const monsterCenterX =
              monsterCharacter.position.x +
              monsterCharacter.collisionSize.width / 2
            const monsterCenterY =
              monsterCharacter.position.y +
              monsterCharacter.collisionSize.height / 2
            const playerCenterX =
              playerCharacter.position.x +
              playerCharacter.collisionSize.width / 2
            const playerCenterY =
              playerCharacter.position.y + playerCharacter.collisionSize.height / 2
            const deltaX = playerCenterX - monsterCenterX
            const deltaY = playerCenterY - monsterCenterY
            const distance = Math.hypot(deltaX, deltaY)

            if (
              behaviorState.nextAttackAtMilliseconds <= now &&
              isMonsterWithinAttackRange(
                monsterCharacter,
                playerCharacter,
                monsterBehaviorConfig
              )
            ) {
              const combatState = monsterCombatStates.get(character.id)

              if (combatState) {
                setMonsterPigAttackState(
                  character.id,
                  now,
                  monsterBehaviorConfig
                )
                if (
                  monsterCharacter.appearanceType ===
                  MONSTER_SLIME_APPEARANCE_TYPE
                ) {
                  gameSoundEffects.play('slimeAttack')
                }
                monsterContactDamageLockedUntilById.set(
                  character.id,
                  now + monsterBehaviorConfig.attackDurationMilliseconds
                )
                applyDamageToPlayer(
                  Math.max(1, combatState.contactDamage + 1),
                  now,
                  monsterCharacter
                )
                knockbackCharacterAwayFromCharacter(
                  PLAYER_CHARACTER_ID,
                  monsterCharacter,
                  0.12
                )
                syncMonsterAnimation(character.id, 'attack')
                continue
              }
            }

            if (distance > 0) {
              const stepDistance =
                (monsterBehaviorConfig.chaseSpeedTilesPerSecond *
                  app.ticker.deltaMS) /
                1000

              tryMoveCharacter(
                character.id,
                (deltaX / distance) * stepDistance,
                (deltaY / distance) * stepDistance
              )
            }

            syncMonsterAnimation(character.id, 'run')
            continue
          }

          const patrolState =
            monsterPatrolStates.get(character.id) ??
            createMonsterPatrolState(monsterCharacter)
          const nextPatrolStep = stepMonsterPatrol({
            character: monsterCharacter,
            patrolState,
            deltaMilliseconds: app.ticker.deltaMS,
            nowMilliseconds: now,
            mapWidth: map.width,
            mapHeight: map.height,
            speedTilesPerSecond: monsterBehaviorConfig.patrolSpeedTilesPerSecond,
            random: Math.random
          })

          monsterPatrolStates.set(character.id, nextPatrolStep.patrolState)

          if (nextPatrolStep.movement) {
            tryMoveCharacter(
              character.id,
              nextPatrolStep.movement.x,
              nextPatrolStep.movement.y
            )
            syncMonsterAnimation(character.id, 'run')
          } else {
            syncMonsterAnimation(character.id, 'idle')
          }
        }
      }

      resolvePlayerAttackDamage(now)
      resolveMonsterContactDamage(now)
      resolveMonsterGoldDropPickups()
      resolveMonsterEquipmentDropPickups()
      syncActiveMonsterGoldDrops(now)
      syncActiveMonsterEquipmentDrops(now)

      const interactionEvents = handleQuestInteractionEvents(
        gameEventQueue.drain(),
        now
      )

      const emittedEvents = processInteractionEvents({
        events: interactionEvents,
        characters: characterStates,
        controllerRuntime,
        now,
        interactionLockUntilByCharacterPair
      })

      for (const event of emittedEvents) {
        if (event.kind !== 'show-character-message') {
          continue
        }

        if (event.characterId === POTION_SHOP_NPC_ID) {
          hideCharacterMessage(event.characterId)
          setPotionShopOpen(true)
          continue
        }

        showCharacterMessage(
          event.characterId,
          event.message,
          event.durationMilliseconds
        )

        if (
          event.message === '!' &&
          getCharacterStateById(event.characterId).appearanceType.startsWith(
            'monster_'
          )
        ) {
          setMonsterPigAggro(
            event.characterId,
            now,
            getMonsterBehaviorConfig(getCharacterStateById(event.characterId))
          )
        }

        if (event.characterId === BLACKSMITH_SHOP_NPC_ID) {
          setBlacksmithShopOpen(true)
        }
      }

      pruneExpiredCharacterMessages(now)
      pruneExpiredCharacterDamageTexts(now)
      syncActiveCharacterMessages()
      syncActiveCharacterDamageTexts(now)
      syncPlayerCharacterVisual(now)
      syncPlayerFootsteps(didPlayerMoveThisFrame)
      triggeredActions.clear()
      lastRuntimeErrorMessage = undefined
    } catch (error) {
      stopPlayerFootsteps()
      gameEventQueue.clear()
      triggeredActions.clear()

      const message = error instanceof Error ? error.message : String(error)

      if (message !== lastRuntimeErrorMessage) {
        console.error('Runtime update failed.', error)
        lastRuntimeErrorMessage = message
      }
    } finally {
      syncRuntimeWarningBanner()
    }
  }

  const handleKeyDown = (event: KeyboardEvent) => {
    const code = event.code
    const isInventoryToggleKey = code === currentPlayerControlBindings.inventory
    const isStatToggleKey = code === currentPlayerControlBindings.stat
    const isEquipmentToggleKey = code === currentPlayerControlBindings.equipment
    const isSkillToggleKey = code === currentPlayerControlBindings.skill
    const isQuestLogToggleKey = code === currentPlayerControlBindings.quest
    const isMapToggleKey = code === currentPlayerControlBindings.map
    const isPauseKey = isPlayerControlPauseKey(
      currentPlayerControlBindings,
      code
    )

    if (pendingControlBindingId) {
      event.preventDefault()

      if (event.repeat || isPlayerControlCaptureModifierKey(code)) {
        return
      }

      updatePlayerControlBindings(
        setPlayerControlBinding({
          bindings: currentPlayerControlBindings,
          bindingId: pendingControlBindingId,
          nextCode: code
        })
      )
      pendingControlBindingId = undefined
      syncPlayerUiOverlays()
      return
    }

    if (isPauseKey) {
      event.preventDefault()

      if (event.repeat) {
        return
      }

      if (!closeAllOverlays()) {
        setPauseMenuOpen(true)
      }

      return
    }

    if (isPauseMenuOpen) {
      if (
        !(event.target instanceof HTMLInputElement) ||
        event.target.type !== 'range'
      ) {
        event.preventDefault()
      }

      return
    }

    if (isMapToggleKey) {
      if (!event.repeat) {
        event.preventDefault()

        if (mapOverlay.getIsExpanded()) {
          mapOverlay.setExpanded(false)
        } else {
          mapOverlay.toggleVisible()
        }
      }

      return
    }

    if (mapOverlay.getIsExpanded()) {
      event.preventDefault()
      return
    }

    if (isInventoryToggleKey) {
      if (!event.repeat) {
        event.preventDefault()
        setPlayerUiOpen(!isPlayerUiOpen)
      }

      return
    }

    if (isStatToggleKey) {
      if (!event.repeat) {
        event.preventDefault()
        setPlayerStatOpen(!isPlayerStatOpen)
      }

      return
    }

    if (isEquipmentToggleKey) {
      if (!event.repeat) {
        event.preventDefault()
        setPlayerEquipmentOpen(!isPlayerEquipmentOpen)
      }

      return
    }

    if (isSkillToggleKey) {
      if (!event.repeat) {
        event.preventDefault()
        setPlayerSkillOpen(!isPlayerSkillOpen)
      }

      return
    }

    if (isQuestLogToggleKey) {
      if (!event.repeat) {
        event.preventDefault()
        setQuestLogOpen(!isQuestLogOpen)
      }

      return
    }

    if (playerProfile.hp.current === 0) {
      return
    }

    const quickslotIndex = getQuickslotIndexFromKeyboardEvent(event)

    if (quickslotIndex !== undefined) {
      if (event.repeat) {
        return
      }

      const quickslotAssignment =
        currentPlayerQuickslots.slots[quickslotIndex]

      if (!quickslotAssignment) {
        return
      }

      const assignedInventorySlotIndex = quickslotAssignment.inventorySlotIndex
      const assignedItem =
        currentPlayerInventory.slots[assignedInventorySlotIndex]

      if (!assignedItem) {
        currentPlayerQuickslots = clearPlayerQuickslotAssignment({
          quickslots: currentPlayerQuickslots,
          quickslotIndex
        })
        onPlayerQuickslotsChange(currentPlayerQuickslots)
        syncPlayerUiOverlays()
        return
      }

      const nextState = usePlayerQuickslotConsumable({
        profile: playerProfile,
        inventory: currentPlayerInventory,
        quickslots: currentPlayerQuickslots,
        quickslotIndex
      })

      if (!nextState) {
        return
      }

      event.preventDefault()
      currentPlayerInventory = nextState.inventory
      Object.assign(playerProfile, nextState.profile)
      onPlayerInventoryChange(nextState.inventory)
      handleConsumableUsed(assignedItem.id)

      if (nextState.inventory.slots[assignedInventorySlotIndex] === undefined) {
        currentPlayerQuickslots = clearPlayerQuickslotAssignment({
          quickslots: currentPlayerQuickslots,
          quickslotIndex
        })
        onPlayerQuickslotsChange(currentPlayerQuickslots)
      }

      syncPlayerUiOverlays()
      return
    }

    const action = getPlayerControlActionFromCode(
      currentPlayerControlBindings,
      code
    )

    if (action) {
      event.preventDefault()

      if (!pressedActions.has(action)) {
        triggeredActions.add(action)
      }

      pressedActions.add(action)
      return
    }

    const direction = getPlayerControlMovementDirectionFromCode(
      currentPlayerControlBindings,
      code
    )

    if (!direction) {
      return
    }

    event.preventDefault()
    pressedDirections.add(direction)
  }

  const handleKeyUp = (event: KeyboardEvent) => {
    const code = event.code
    const action = getPlayerControlActionFromCode(
      currentPlayerControlBindings,
      code
    )

    if (action) {
      pressedActions.delete(action)
      return
    }

    const direction = getPlayerControlMovementDirectionFromCode(
      currentPlayerControlBindings,
      code
    )

    if (!direction) {
      return
    }

    pressedDirections.delete(direction)
  }

  const handleWindowBlur = () => {
    clearPressedInputState()
    stopPlayerFootsteps()
  }

  const handleViewportWheel = (event: WheelEvent) => {
    event.preventDefault()

    if (event.deltaY === 0) {
      return
    }

    setCameraZoom(cameraZoom * Math.exp(-event.deltaY * CAMERA_ZOOM_WHEEL_SPEED))
  }

  const handleWindowResize = () => {
    syncViewportDisplayScale()
    centerCameraOnCharacter(getCharacterStateById(cameraTargetCharacterId))
    mapOverlay.syncFrame()
  }

  const handleVisibilityChange = () => {
    if (document.hidden) {
      handleWindowBlur()
      app.stop()
      return
    }

    app.start()
  }

  window.addEventListener('keydown', handleKeyDown)
  window.addEventListener('keyup', handleKeyUp)
  window.addEventListener('blur', handleWindowBlur)
  window.addEventListener('resize', handleWindowResize)
  viewportElement.addEventListener('wheel', handleViewportWheel, {
    passive: false
  })
  document.addEventListener('visibilitychange', handleVisibilityChange)
  app.ticker.add(updateCharacters)
  app.ticker.add(mapOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  app.ticker.add(playerHudOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  app.ticker.add(
    playerInventoryOverlay.syncFrame,
    undefined,
    UPDATE_PRIORITY.UTILITY
  )
  app.ticker.add(
    playerEquipmentOverlay.syncFrame,
    undefined,
    UPDATE_PRIORITY.UTILITY
  )
  app.ticker.add(playerStatOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  app.ticker.add(
    playerSkillOverlay.syncFrame,
    undefined,
    UPDATE_PRIORITY.UTILITY
  )
  app.ticker.add(questLogOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  app.ticker.add(playerShopOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  app.ticker.add(pauseMenuOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  app.ticker.add(questTrackerOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  syncAllCharacterSprites()
  syncQuestNpcBadges()
  syncViewportDisplayScale()
  centerCameraOnCharacter(getCharacterStateById(cameraTargetCharacterId))
  showSceneIntroBanner()
  mapOverlay.syncFrame()
  playerHudOverlay.syncFrame()
  playerInventoryOverlay.syncFrame()
  playerEquipmentOverlay.syncFrame()
  playerStatOverlay.syncFrame()
  playerSkillOverlay.syncFrame()
  questLogOverlay.syncFrame()
  playerShopOverlay.syncFrame()
  pauseMenuOverlay.syncFrame()
  questTrackerOverlay.syncFrame()
  handleVisibilityChange()

  const destroy = () => {
    if (isDestroyed) {
      return
    }

    isDestroyed = true
    window.removeEventListener('keydown', handleKeyDown)
    window.removeEventListener('keyup', handleKeyUp)
    window.removeEventListener('blur', handleWindowBlur)
    window.removeEventListener('resize', handleWindowResize)
    viewportElement.removeEventListener('wheel', handleViewportWheel)
    document.removeEventListener('visibilitychange', handleVisibilityChange)
    app.ticker.remove(updateCharacters)
    app.ticker.remove(mapOverlay.syncFrame)
    app.ticker.remove(playerHudOverlay.syncFrame)
    app.ticker.remove(playerInventoryOverlay.syncFrame)
    app.ticker.remove(playerEquipmentOverlay.syncFrame)
    app.ticker.remove(playerStatOverlay.syncFrame)
    app.ticker.remove(playerSkillOverlay.syncFrame)
    app.ticker.remove(questLogOverlay.syncFrame)
    app.ticker.remove(playerShopOverlay.syncFrame)
    app.ticker.remove(pauseMenuOverlay.syncFrame)
    app.ticker.remove(questTrackerOverlay.syncFrame)
    gameEventQueue.clear()
    monsterPatrolStates.clear()
    monsterSpawnStates.clear()
    monsterCombatStates.clear()
    monsterContactDamageLockedUntilById.clear()
    monsterRespawnAtById.clear()
    for (const monsterGoldDrop of monsterGoldDrops.values()) {
      monsterGoldDrop.container.destroy({ children: true })
    }
    monsterGoldDrops.clear()
    for (const monsterEquipmentDrop of monsterEquipmentDrops.values()) {
      monsterEquipmentDrop.container.destroy({ children: true })
    }
    monsterEquipmentDrops.clear()
    monsterPigAnimatedSprites.clear()
    monsterPigAnimationModes.clear()
    monsterPigBehaviorStates.clear()
    for (const activeDamageText of activeCharacterDamageTexts.values()) {
      activeDamageText.container.destroy({ children: true })
    }
    activeCharacterDamageTexts.clear()
    window.clearTimeout(sceneIntroHideTimeoutId)
    for (const activeMessage of activeCharacterMessages.values()) {
      activeMessage.container.destroy({ children: true })
    }
    activeCharacterMessages.clear()
    runtimeWarningBannerElement.remove()
    sceneIntroBannerElement.remove()
    mapOverlay.destroy()
    playerHudOverlay.destroy()
    playerInventoryOverlay.destroy()
    playerEquipmentOverlay.destroy()
    playerStatOverlay.destroy()
    playerSkillOverlay.destroy()
    questLogOverlay.destroy()
    playerShopOverlay.destroy()
    potionShopOverlay.destroy()
    pauseMenuOverlay.destroy()
    questTrackerOverlay.destroy()
    gameSoundEffects.destroy()
    controllerRuntime.destroy()
    app.destroy({ removeView: true }, { children: true })
    viewportElement.remove()
  }

  if (import.meta.hot) {
    import.meta.hot.dispose(destroy)
  }

  return {
    destroy
  }
}

const createMessagePanelTexture = (): Texture => {
  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d')

  if (!context) {
    throw new Error('Could not create canvas context for message panel')
  }

  canvas.width = 32
  canvas.height = 32
  context.imageSmoothingEnabled = false
  context.clearRect(0, 0, canvas.width, canvas.height)

  context.fillStyle = '#c9a271'
  context.fillRect(0, 0, canvas.width, canvas.height)
  context.fillStyle = '#fffaf0'
  context.fillRect(1, 1, canvas.width - 2, canvas.height - 2)
  context.fillStyle = '#f4e5c8'
  context.fillRect(1, 1, canvas.width - 2, 1)
  context.fillRect(1, 1, 1, canvas.height - 2)
  context.fillStyle = '#a98256'
  context.fillRect(1, canvas.height - 2, canvas.width - 2, 1)
  context.fillRect(canvas.width - 2, 1, 1, canvas.height - 2)
  context.fillStyle = '#fffdf8'
  context.fillRect(2, 2, canvas.width - 4, canvas.height - 4)
  context.fillStyle = '#ecd8b8'
  context.fillRect(2, 2, canvas.width - 4, 1)
  context.fillRect(2, 2, 1, canvas.height - 4)
  context.fillStyle = '#d9c29a'
  context.fillRect(2, canvas.height - 3, canvas.width - 4, 1)
  context.fillRect(canvas.width - 3, 2, 1, canvas.height - 4)

  const texture = Texture.from(canvas)

  texture.source.scaleMode = 'nearest'
  texture.source.addressMode = 'clamp-to-edge'

  return texture
}

  const loadSlashVfxTextures = async (): Promise<SlashVfxRenderResources> => {
  const textures = await Promise.all(
    WHITE_SLASH_WIDE_FRAME_URLS.map((frameUrl) =>
      Assets.load<Texture>(frameUrl)
    )
  )

  textures.forEach((texture) => {
    texture.source.scaleMode = 'nearest'
  })

  return {
    horizontalTextures: textures,
    verticalTextures: textures
  }
}

const ensureMessageFontsLoaded = async (): Promise<void> => {
  if (messageFontsReadyPromise) {
    return messageFontsReadyPromise
  }

  if (!document.fonts) {
    return
  }

  messageFontsReadyPromise = Promise.all([
    document.fonts.load('400 14px "Jersey 25"'),
    document.fonts.load('400 14px "NeoDunggeunmo"')
  ]).then(() => undefined)

  return messageFontsReadyPromise
}

const loadTilesetRenderResources = async (
  tileset: ParsedTiledTileset,
  imageUrls: Record<string, string>,
  scaleMode?: 'nearest' | 'linear'
): Promise<TilesetRenderResources> => {
  const imageUrl = imageUrls[tileset.image.source]

  if (!imageUrl) {
    throw new Error(`Missing image URL for ${tileset.image.source}`)
  }

  const imageTexture = await Assets.load<Texture>(imageUrl)

  if (scaleMode) {
    imageTexture.source.scaleMode = scaleMode
  }
  imageTexture.source.addressMode = 'clamp-to-edge'
  const tileTextures = Array.from(
    { length: tileset.tileCount },
    (_, localId) => createTileTexture(imageTexture, tileset, localId)
  )

  return {
    imageTexture,
    tileTextures
  }
}

const createTileTexture = (
  imageTexture: Texture,
  tileset: TileTextureFrameSource,
  localId: number
): Texture => {
  const columnIndex = localId % tileset.columns
  const rowIndex = Math.floor(localId / tileset.columns)
  const frameX =
    tileset.margin + columnIndex * (tileset.tileWidth + tileset.spacing)
  const frameY =
    tileset.margin + rowIndex * (tileset.tileHeight + tileset.spacing)

  return new Texture({
    source: imageTexture.source,
    frame: new Rectangle(
      frameX,
      frameY,
      tileset.tileWidth,
      tileset.tileHeight
    ),
    orig: new Rectangle(0, 0, tileset.tileWidth, tileset.tileHeight)
  })
}

const resolveTilesetForTile = (
  tile: ParsedTiledTile,
  tilesets: ParsedTiledTileset[]
): ParsedTiledTileset => {
  for (let index = tilesets.length - 1; index >= 0; index -= 1) {
    const tileset = tilesets[index]

    if (tileset.firstGid <= tile.gid) {
      return tileset
    }
  }

  throw new Error(`Could not resolve tileset for gid ${tile.gid}`)
}

const clampScrollOffset = (value: number, max: number): number =>
  Math.max(0, Math.min(Math.round(value), Math.max(0, max)))

const clampCameraZoom = (value: number): number =>
  Math.max(CAMERA_MIN_ZOOM, Math.min(value, CAMERA_MAX_ZOOM))

const isCharacterPositionBlocked = (
  wallTiles: Set<string>,
  blockingRects: CollisionRect[],
  x: number,
  y: number,
  width: number,
  height: number
): boolean => {
  const epsilon = 1e-6
  const minTileX = Math.floor(x + epsilon)
  const maxTileX = Math.floor(x + width - epsilon)
  const minTileY = Math.floor(y + epsilon)
  const maxTileY = Math.floor(y + height - epsilon)

  for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
    for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
      if (isWallTileAt(wallTiles, tileX, tileY)) {
        return true
      }
    }
  }

  return blockingRects.some((blockingRect) =>
    doCollisionRectsIntersect(
      blockingRect,
      {
        x,
        y,
        width,
        height
      }
    )
  )
}

const createGrassTileLookup = (map: ParsedTiledMap): Set<string> => {
  const groundLayer = map.layers.find(
    (layer) => layer.name.toLowerCase() === GROUND_LAYER_NAME
  )
  const grassTileKeys = new Set<string>()

  if (!groundLayer) {
    return grassTileKeys
  }

  for (const tile of groundLayer.tiles) {
    const tileset = resolveTilesetForTile(tile, map.tilesets)
    const tileType = tileset.tileTypes[tile.localId]

    if (tileType && GRASS_TILE_TYPES.has(tileType)) {
      grassTileKeys.add(createTileLookupKey(tile.x, tile.y))
    }
  }

  return grassTileKeys
}

const isCharacterOnGrass = (
  character: CharacterState,
  grassTiles: Set<string>
): boolean => {
  const tileX = Math.floor(
    character.position.x + character.collisionSize.width / 2
  )
  const tileY = Math.floor(
    character.position.y + character.collisionSize.height / 2
  )

  return grassTiles.has(createTileLookupKey(tileX, tileY))
}

const createTileLookupKey = (tileX: number, tileY: number): string =>
  `${tileX},${tileY}`

const createCollisionRectFromCharacter = (
  character: CharacterState
): CollisionRect => ({
  x: character.position.x,
  y: character.position.y,
  width: character.collisionSize.width,
  height: character.collisionSize.height
})

const createCollisionRectFromPortal = (portal: MapPortal): CollisionRect => ({
  x: portal.position.x,
  y: portal.position.y,
  width: portal.collisionSize.width,
  height: portal.collisionSize.height
})

const getTileDepthSortValue = (tileY: number, tileHeight: number): number =>
  (tileY + 1) * tileHeight

const getCharacterDepthSortValue = (
  characterY: number,
  characterPixelHeight: number,
  tileHeight: number
): number => characterY * tileHeight + characterPixelHeight

const doCollisionRectsIntersect = (
  left: CollisionRect,
  right: CollisionRect
): boolean =>
  left.x < right.x + right.width &&
  left.x + left.width > right.x &&
  left.y < right.y + right.height &&
  left.y + left.height > right.y

const resolveCharacterTexture = (
  appearanceType: string,
  tileTextures: Texture[],
  tileset: ParsedTiledTileset,
  fallbackTilesets: ParsedTiledTileset[],
  fallbackTileTextureResources: Map<string, TilesetRenderResources>,
  mapTileWidth: number
): ResolvedCharacterAppearanceTexture => {
  const characterTexture = resolveTextureByAppearanceType(
    appearanceType,
    tileTextures,
    tileset
  )

  if (characterTexture) {
    return {
      texture: characterTexture,
      renderScale: mapTileWidth / tileset.tileWidth
    }
  }

  for (const fallbackTileset of fallbackTilesets) {
    const fallbackTileTextureResource = fallbackTileTextureResources.get(
      fallbackTileset.source
    )

    if (!fallbackTileTextureResource) {
      throw new Error(
        `Missing render resources for tileset ${fallbackTileset.source}`
      )
    }

    const fallbackTexture = resolveTextureByAppearanceType(
      appearanceType,
      fallbackTileTextureResource.tileTextures,
      fallbackTileset
    )

    if (fallbackTexture) {
      return {
        texture: fallbackTexture,
        renderScale: mapTileWidth / fallbackTileset.tileWidth
      }
    }
  }

  throw new Error(`Could not resolve tileset tile type ${appearanceType}`)
}

const resolveTextureByAppearanceType = (
  appearanceType: string,
  tileTextures: Texture[],
  tileset: ParsedTiledTileset
): Texture | undefined => {
  try {
    const localId = resolveTilesetLocalIdByType(tileset, appearanceType)

    return tileTextures[localId]
  } catch {
    return undefined
  }
}

const resolveTilesetLocalIdByType = (
  tileset: ParsedTiledTileset,
  tileType: string
): number => {
  const entry = Object.entries(tileset.tileTypes).find(
    ([, candidateType]) => candidateType === tileType
  )

  if (!entry) {
    throw new Error(`Could not resolve tileset tile type ${tileType}`)
  }

  return Number(entry[0])
}

const STACKABLE_QUEST_REWARD_ITEM_IDS = new Set([
  'health-potion',
  'mana-potion'
])

const addQuestItemRewardsToInventory = (
  inventory: PlayerInventory,
  itemRewards: QuestItemReward[]
): PlayerInventory => {
  let nextInventory = inventory

  for (const itemReward of itemRewards) {
    nextInventory = addQuestItemRewardToInventory(nextInventory, itemReward)
  }

  return nextInventory
}

const addQuestItemRewardToInventory = (
  inventory: PlayerInventory,
  itemReward: QuestItemReward
): PlayerInventory => {
  if (STACKABLE_QUEST_REWARD_ITEM_IDS.has(itemReward.id)) {
    const stackSlotIndex = inventory.slots.findIndex(
      (item) => item?.id === itemReward.id
    )

    if (stackSlotIndex >= 0) {
      const slots = [...inventory.slots]
      const stack = slots[stackSlotIndex] as PlayerInventoryItem

      slots[stackSlotIndex] = {
        ...stack,
        quantity: stack.quantity + itemReward.quantity
      }

      return {
        ...inventory,
        slots
      }
    }

    return addQuestRewardAsNewInventorySlot(inventory, itemReward)
  }

  return addQuestRewardAsNewInventorySlots(inventory, itemReward)
}

const addQuestRewardAsNewInventorySlot = (
  inventory: PlayerInventory,
  itemReward: QuestItemReward
): PlayerInventory => {
  const emptySlotIndex = inventory.slots.findIndex((item) => item === undefined)

  if (emptySlotIndex < 0) {
    return inventory
  }

  const slots = [...inventory.slots]

  slots[emptySlotIndex] = {
    id: itemReward.id,
    label: itemReward.label,
    quantity: itemReward.quantity
  }

  return {
    ...inventory,
    slots
  }
}

const addQuestRewardAsNewInventorySlots = (
  inventory: PlayerInventory,
  itemReward: QuestItemReward
): PlayerInventory => {
  let nextInventory = inventory

  for (let quantity = 0; quantity < itemReward.quantity; quantity += 1) {
    const emptySlotIndex = nextInventory.slots.findIndex(
      (item) => item === undefined
    )

    if (emptySlotIndex < 0) {
      return nextInventory
    }

    const slots = [...nextInventory.slots]

    slots[emptySlotIndex] = {
      id: itemReward.id,
      label: itemReward.label,
      quantity: 1
    }
    nextInventory = {
      ...nextInventory,
      slots
    }
  }

  return nextInventory
}

const createDepthSortedTileSprite = (
  texture: Texture,
  tile: ParsedTiledTile,
  tileWidth: number,
  tileHeight: number
): Sprite => {
  if (hasTileTransform(tile)) {
    return createTransformedTileSprite(texture, tile, tileWidth, tileHeight)
  }

  const sprite = new Sprite(texture)

  sprite.position.set(tile.x * tileWidth, tile.y * tileHeight)
  sprite.roundPixels = true

  return sprite
}

const createTransformedTileSprite = (
  texture: Texture,
  tile: ParsedTiledTile,
  tileWidth: number,
  tileHeight: number
): Sprite => {
  const sprite = new Sprite(texture)
  const transform = getSpriteTransformForTile(tile)

  sprite.anchor.set(0.5)
  sprite.position.set(
    tile.x * tileWidth + tileWidth / 2,
    tile.y * tileHeight + tileHeight / 2
  )
  sprite.rotation = transform.rotation
  sprite.scale.set(transform.scaleX, transform.scaleY)
  sprite.roundPixels = true

  return sprite
}
