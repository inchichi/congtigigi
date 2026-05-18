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
  Spritesheet,
  Text,
  TextStyle,
  Texture,
  UPDATE_PRIORITY
} from 'pixi.js'

import {
  PLAYER_CHARACTER_ID,
  getCharacterActionFromKey,
  getCharacterMoveDirectionFromKey,
  moveCharacterState
} from '../game/characterState'
import type {
  CharacterAction,
  CharacterMoveDirection,
  CharacterState
} from '../game/characterState'
import type { CharacterControllerRuntime } from '../game/createCharacterControllerRuntime'
import { createGameEventQueue } from '../game/events/createGameEventQueue'
import { processInteractionEvents } from '../game/interaction/processInteractionEvents'
import type { PlayerEquipment } from '../game/playerEquipment'
import type { PlayerInventory } from '../game/playerInventory'
import type { PlayerProfile } from '../game/playerProfile'
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
import { getMonsterGoldDropAmount } from '../game/monsterRewards'
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
import { createPlayerHudOverlay } from './createPlayerHudOverlay'
import { createPlayerInventoryOverlay } from './createPlayerInventoryOverlay'
import type { MonsterAnimationTextures } from './monsterAnimationTextures'
import { loadMonsterPigAnimationTextures } from './loadMonsterPigAnimationTextures'
import { loadMonsterSlimeAnimationTextures } from './loadMonsterSlimeAnimationTextures'

type CreatePixiTiledMapViewInput = {
  mountElement: HTMLElement
  map: ParsedTiledMap
  characters: CharacterState[]
  playerProfile: PlayerProfile
  playerEquipment: PlayerEquipment
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
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
  onMerchantInventoryChange: (nextInventory: PlayerInventory) => void
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

type RenderedCharacterNode = {
  container: Container
  sprite: Sprite
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

const DEPTH_SORTED_LAYER_NAME = 'object'
const UI_SPRITESHEET_URL = new URL(
  '../assets/spritesheets/uipack_rpg_sheet.json',
  import.meta.url
).href
const TINY_DUNGEON_TILESET_IMAGE_URL = new URL(
  '../assets/tilesets/tiny-dungeon-16.png',
  import.meta.url
).href
const MESSAGE_PANEL_TEXTURE_NAME = 'panelInset_beige.png'
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
const BLACKSMITH_SHOP_NPC_ID = 'blacksmith'
const MONSTER_PIG_APPEARANCE_TYPE = 'monster_pig'
const MONSTER_SLIME_APPEARANCE_TYPE = 'monster_slime'
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
const PLAYER_WEAPON_WORLD_SCALE = 1.35
const PLAYER_ATTACK_TRAIL_PROGRESS_STEP = 0.12
const PLAYER_ATTACK_TRAIL_ALPHA = [0.42, 0.28, 0.18, 0.1]
const PLAYER_ATTACK_TRAIL_SPRITE_COUNT = PLAYER_ATTACK_TRAIL_ALPHA.length
const PLAYER_ATTACK_SWING_X_OFFSET = 4
const PLAYER_ATTACK_LIFT_Y_OFFSET = 3
const PLAYER_ATTACK_ROTATION_OFFSET = 1.15
const PLAYER_ATTACK_SCALE_BOOST = 0.06
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
const SCENE_INTRO_VISIBLE_DURATION_MILLISECONDS = 3000
const PLAYER_ATTACK_DURATION_MILLISECONDS = 320
const PLAYER_ATTACK_COOLDOWN_MILLISECONDS = 420
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

let messageFontsReadyPromise: Promise<void> | undefined

export const createPixiTiledMapView = async ({
  mountElement,
  map,
  characters,
  playerProfile,
  playerEquipment,
  playerInventory,
  merchantInventory,
  sceneIntroMessage,
  cameraTargetCharacterId,
  characterSpriteSheet,
  imageUrls,
  controllerRuntime,
  onPlayerInventoryChange,
  onPlayerEquipmentChange,
  onMerchantInventoryChange,
  onRequestSceneChange
}: CreatePixiTiledMapViewInput): Promise<{ destroy: () => void }> => {
  const app = new Application()
  const sceneScale = Math.max(
    1,
    Math.ceil(
      Math.min(window.innerWidth / map.pixelWidth, window.innerHeight / map.pixelHeight)
    )
  )
  const scaledMapPixelWidth = map.pixelWidth * sceneScale
  const scaledMapPixelHeight = map.pixelHeight * sceneScale
  const [
    messagePanelTexture,
    tinyDungeonWeaponImageTexture,
    monsterPigAnimationTextures,
    monsterSlimeAnimationTextures
  ] = await Promise.all([
    loadMessagePanelTexture(),
    Assets.load<Texture>(TINY_DUNGEON_TILESET_IMAGE_URL),
    loadMonsterPigAnimationTextures(),
    loadMonsterSlimeAnimationTextures()
  ])

  const monsterAnimationTexturesByAppearanceType: Record<
    MonsterAppearanceType,
    MonsterAnimationTextures
  > = {
    [MONSTER_PIG_APPEARANCE_TYPE]: monsterPigAnimationTextures,
    [MONSTER_SLIME_APPEARANCE_TYPE]: monsterSlimeAnimationTextures
  }

  tinyDungeonWeaponImageTexture.source.scaleMode = 'nearest'
  tinyDungeonWeaponImageTexture.source.addressMode = 'clamp-to-edge'
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

  const sceneElement = document.createElement('div')
  const runtimeWarningBannerElement = document.createElement('div')
  const sceneIntroBannerElement = document.createElement('div')
  const sceneIntroPanelElement = document.createElement('div')
  const sceneIntroTextElement = document.createElement('div')

  sceneElement.className = 'game-scene'
  sceneElement.style.width = `${scaledMapPixelWidth}px`
  sceneElement.style.height = `${scaledMapPixelHeight}px`
  sceneElement.append(app.canvas)
  mountElement.replaceChildren(sceneElement)
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
  world.scale.set(sceneScale)
  const messageLayer = new Container()
  const tilesetResources = new Map<string, TilesetRenderResources>()
  const wallTiles = createWallTileLookup(map)
  const pressedDirections = new Set<CharacterMoveDirection>()
  const pressedActions = new Set<CharacterAction>()
  const triggeredActions = new Set<CharacterAction>()
  const gameEventQueue = createGameEventQueue()
  const interactionLockUntilByCharacterPair = new Map<string, number>()
  const activeCharacterMessages = new Map<string, ActiveCharacterMessage>()
  const activeCharacterDamageTexts = new Map<
    string,
    ActiveCharacterDamageText
  >()
  const monsterGoldDrops = new Map<string, MonsterGoldDrop>()
  const renderedCharacters = new Map<string, RenderedCharacterNode>()
  const renderedPortals = new Map<string, RenderedPortalNode>()
  const characterPixelWidth =
    characterSpriteSheet.tileset.tileWidth * characterSpriteSheet.scale
  const characterPixelHeight =
    characterSpriteSheet.tileset.tileHeight * characterSpriteSheet.scale
  let currentPlayerEquipment = playerEquipment
  let currentPlayerInventory = playerInventory
  let currentBlacksmithInventory = merchantInventory
  let playerAttackStartedAtMilliseconds: number | undefined
  let playerWeaponTrailSprites: Sprite[] = []
  let playerWeaponSprite: Sprite | undefined
  let syncPlayerCharacterVisual: (nowMilliseconds?: number) => void = () => {}
  let isSceneTransitionPending = false
  let isDestroyed = false
  const clearPressedInputState = () => {
    pressedDirections.clear()
    pressedActions.clear()
    triggeredActions.clear()
  }
  let isPlayerUiOpen = false
  let isBlacksmithShopOpen = false
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
  let playerShopOverlay: {
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
  const mapOverlay = createMapOverlay({
    mountElement,
    sourceCanvas: app.canvas,
    mapPixelWidth: map.pixelWidth,
    mapPixelHeight: map.pixelHeight,
    sceneScale,
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
    }
  })
  const syncPlayerUiOverlays = () => {
    playerHudOverlay.syncFrame()
    playerInventoryOverlay.syncFrame()
    playerShopOverlay.syncFrame()
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
  const isAttackKey = (event: KeyboardEvent): boolean =>
    event.code === 'KeyA' || event.key.toLowerCase() === 'a'
  const triggerPlayerAttack = (now: number) => {
    if (now < playerAttackReadyAtMilliseconds) {
      return
    }

    playerAttackStartedAtMilliseconds = now
    playerAttackResolvedStartedAtMilliseconds = undefined
    playerAttackReadyAtMilliseconds =
      now + PLAYER_ATTACK_COOLDOWN_MILLISECONDS
  }
  const setPlayerUiOpen = (nextIsOpen: boolean) => {
    if (isPlayerUiOpen === nextIsOpen) {
      return
    }

    isPlayerUiOpen = nextIsOpen
    if (nextIsOpen) {
      isBlacksmithShopOpen = false
    }
    clearPressedInputState()
    syncPlayerUiOverlays()
  }
  const setBlacksmithShopOpen = (nextIsOpen: boolean) => {
    if (isBlacksmithShopOpen === nextIsOpen) {
      return
    }

    isBlacksmithShopOpen = nextIsOpen
    if (nextIsOpen) {
      isPlayerUiOpen = false
    }
    clearPressedInputState()
    syncPlayerUiOverlays()
  }
  const closeAllOverlays = () => {
    if (!isPlayerUiOpen && !isBlacksmithShopOpen) {
      return
    }

    isPlayerUiOpen = false
    isBlacksmithShopOpen = false
    clearPressedInputState()
    syncPlayerUiOverlays()
  }
  const requestSceneTransition = (portal: MapPortal) => {
    if (isSceneTransitionPending) {
      return
    }

    isSceneTransitionPending = true
    closeAllOverlays()
    clearPressedInputState()
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
  playerHudOverlay = createPlayerHudOverlay({
    mountElement,
    profile: playerProfile,
    getIsInventoryOpen: () => isPlayerUiOpen,
    onRequestInventoryOpenChange: setPlayerUiOpen
  })
  playerInventoryOverlay = createPlayerInventoryOverlay({
    mountElement,
    profile: playerProfile,
    getInventory: () => currentPlayerInventory,
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
    }
  })
  playerShopOverlay = createBlacksmithShopOverlay({
    mountElement,
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
  const resolveMapPortalTexture = (appearanceType: string): Texture => {
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
    const monsterAnimationTextures = isMonsterCharacter
      ? monsterAnimationTexturesByAppearanceType[
          character.appearanceType as MonsterAppearanceType
        ]
      : undefined
    const monsterBehaviorConfig = isMonsterCharacter
      ? getMonsterBehaviorConfig(character)
      : undefined
    const sprite = monsterAnimationTextures
      ? new AnimatedSprite(monsterAnimationTextures.idleLeft)
      : new Sprite(
          resolveCharacterTexture(
            characterTilesetResources.tileTextures,
            characterSpriteSheet.tileset,
            character.appearanceType
          )
        )
    const renderScale = monsterBehaviorConfig
      ? monsterBehaviorConfig.renderScale
      : characterSpriteSheet.scale
    const isPlayer = character.id === PLAYER_CHARACTER_ID
    const monsterHealthBar = isMonsterCharacter
      ? createMonsterHealthBar()
      : undefined
    const levelBadge =
      character.level === undefined
        ? undefined
        : new Text({
            style: MONSTER_LEVEL_BADGE_STYLE,
            text: `Lv ${character.level}`
          })
    container.label = `character:${character.id}:container`
    container.sortableChildren = true
    sprite.label = `character:${character.id}`
    sprite.scale.set(renderScale)
    sprite.roundPixels = true
    sprite.zIndex = 10
    container.addChild(sprite)
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

    if (isMonsterCharacter) {
      monsterCombatStates.set(
        character.id,
        createMonsterCombatState(character.level ?? 1)
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
      playerWeaponTrailSprites = Array.from(
        { length: PLAYER_ATTACK_TRAIL_SPRITE_COUNT },
        (_, index) => {
          const trailSprite = new Sprite(playerWeaponTexture)

          trailSprite.label = `character:player:weapon-trail:${index}`
          trailSprite.anchor.set(0.5, 1)
          trailSprite.visible = false
          trailSprite.roundPixels = true
          trailSprite.zIndex = index + 1
          container.addChild(trailSprite)

          return trailSprite
        }
      )
      playerWeaponSprite = new Sprite(playerWeaponTexture)
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
      levelBadge,
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
    const sprite = new Sprite(resolveMapPortalTexture(portal.appearanceType))

    container.label = `portal:${portal.id}:container`
    container.sortableChildren = true
    sprite.label = `portal:${portal.id}`
    sprite.scale.set(portal.collisionSize.width, portal.collisionSize.height)
    sprite.roundPixels = true
    sprite.zIndex = 0
    container.position.set(
      portal.position.x * map.tileWidth,
      portal.position.y * map.tileHeight
    )
    container.zIndex = Math.round(
      (portal.position.y + portal.collisionSize.height) * map.tileHeight
    )
    container.addChild(sprite)
    renderedPortals.set(portal.id, {
      container,
      sprite
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
    )
    syncCharacterLevelBadge(renderNode, character)
    depthSortedLayer?.sortChildren()

    if (character.id === PLAYER_CHARACTER_ID) {
      syncPlayerWeaponSprite(character)
    }
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
      createMonsterCombatState(nextCharacter.level ?? 1)
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
    playerAttackStartedAtMilliseconds = undefined
    playerAttackResolvedStartedAtMilliseconds = undefined
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

    const nextHp = Math.max(0, playerProfile.hp.current - nextDamage)
    const damageMessage =
      nextHp === 0 ? `-${nextDamage}\n쓰러졌다!` : `-${nextDamage}`

    playerProfile.hp.current = nextHp
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
      character.blocksMovement = false
      monsterPatrolStates.delete(characterId)
      monsterContactDamageLockedUntilById.delete(characterId)
      monsterPigAnimationModes.delete(characterId)
      monsterPigBehaviorStates.delete(characterId)
      spawnMonsterGoldDrop(
        characterId,
        getMonsterGoldDropAmount(character.level ?? 1),
        {
          x:
            character.position.x * map.tileWidth +
            (character.collisionSize.width * map.tileWidth) / 2,
          y:
            character.position.y * map.tileHeight +
            (character.collisionSize.height * map.tileHeight) / 2
        },
        now
      )
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
    const targetCharacter = resolveCharacterInteractionTarget({
      sourceCharacter: playerCharacter,
      targetCharacters: characterStates,
      canReceiveInteraction: (character) =>
        isMonsterCharacter(character) &&
        !isMonsterCombatStateDefeated(character.id),
      interactionProbeDistanceInTiles: PLAYER_ATTACK_PROBE_DISTANCE_IN_TILES
    })

    if (targetCharacter) {
      applyDamageToMonster(targetCharacter.id, playerProfile.stats.attack, now)
    }

    playerAttackResolvedStartedAtMilliseconds = playerAttackStartedAtMilliseconds
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

    const placement =
      character.facing === 'left'
        ? PLAYER_WEAPON_PLACEMENT_LEFT
        : PLAYER_WEAPON_PLACEMENT_RIGHT
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
    const facingMultiplier = character.facing === 'left' ? -1 : 1
    const createPose = (progress: number | undefined) => {
      if (progress === undefined) {
        return {
          x: placement.x,
          y: placement.y,
          rotation: placement.rotation,
          scale: PLAYER_WEAPON_WORLD_SCALE
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
        scale: PLAYER_WEAPON_WORLD_SCALE + PLAYER_ATTACK_SCALE_BOOST * swingAmount
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
      sprite.visible = true
      sprite.position.set(pose.x, pose.y)
      sprite.rotation = pose.rotation
      sprite.scale.set(pose.scale)
      sprite.alpha = alpha
    }

    applyPose(playerWeaponSprite, createPose(attackProgress), 1)

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
    durationMilliseconds: number
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

  const centerViewportOnCharacter = (character: CharacterState) => {
    const characterCenterX =
      (character.position.x * map.tileWidth + characterPixelWidth / 2) *
      sceneScale
    const characterCenterY =
      (character.position.y * map.tileHeight + characterPixelHeight / 2) *
      sceneScale
    const nextScrollLeft = clampScrollOffset(
      characterCenterX - mountElement.clientWidth / 2,
      scaledMapPixelWidth - mountElement.clientWidth
    )
    const nextScrollTop = clampScrollOffset(
      characterCenterY - mountElement.clientHeight / 2,
      scaledMapPixelHeight - mountElement.clientHeight
    )

    mountElement.scrollTo({
      left: nextScrollLeft,
      top: nextScrollTop
    })
  }

  const keepCharacterVisible = (character: CharacterState) => {
    const characterLeft = character.position.x * map.tileWidth * sceneScale
    const characterTop = character.position.y * map.tileHeight * sceneScale
    const characterRight = characterLeft + characterPixelWidth * sceneScale
    const characterBottom = characterTop + characterPixelHeight * sceneScale
    const viewportLeft = mountElement.scrollLeft
    const viewportTop = mountElement.scrollTop
    const viewportWidth = mountElement.clientWidth
    const viewportHeight = mountElement.clientHeight
    const viewportRight = viewportLeft + viewportWidth
    const viewportBottom = viewportTop + viewportHeight
    const horizontalMargin = Math.floor(viewportWidth * 0.3)
    const verticalMargin = Math.floor(viewportHeight * 0.3)
    const deadZoneLeft = viewportLeft + horizontalMargin
    const deadZoneRight = viewportRight - horizontalMargin
    const deadZoneTop = viewportTop + verticalMargin
    const deadZoneBottom = viewportBottom - verticalMargin
    let nextScrollLeft = viewportLeft
    let nextScrollTop = viewportTop

    if (characterLeft < deadZoneLeft) {
      nextScrollLeft = clampScrollOffset(
        characterLeft - horizontalMargin,
        scaledMapPixelWidth - viewportWidth
      )
    } else if (characterRight > deadZoneRight) {
      nextScrollLeft = clampScrollOffset(
        characterRight + horizontalMargin - viewportWidth,
        scaledMapPixelWidth - viewportWidth
      )
    }

    if (characterTop < deadZoneTop) {
      nextScrollTop = clampScrollOffset(
        characterTop - verticalMargin,
        scaledMapPixelHeight - viewportHeight
      )
    } else if (characterBottom > deadZoneBottom) {
      nextScrollTop = clampScrollOffset(
        characterBottom + verticalMargin - viewportHeight,
        scaledMapPixelHeight - viewportHeight
      )
    }

    if (nextScrollLeft !== viewportLeft || nextScrollTop !== viewportTop) {
      mountElement.scrollTo({
        left: nextScrollLeft,
        top: nextScrollTop
      })
    }
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
  ) => {
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
      return
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
        return
      }
    }

    if (nextCharacter.id === cameraTargetCharacterId && didPositionChange) {
      keepCharacterVisible(nextCharacter)
    }
  }

  const drainControllerRuntimeEventsIntoQueue = () => {
    for (const event of controllerRuntime.drainEvents()) {
      gameEventQueue.enqueue(event)
    }
  }

  const updateCharacters = () => {
    try {
      controllerRuntime.syncCharacters(characterStates)
      drainControllerRuntimeEventsIntoQueue()
      const now = performance.now()

      maybeRespawnPlayer(now)

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
            tryMoveCharacter(character.id, intent.movement.x, intent.movement.y)
          }

          if (isSceneTransitionPending) {
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
      syncActiveMonsterGoldDrops(now)

      const emittedEvents = processInteractionEvents({
        events: gameEventQueue.drain(),
        characters: characterStates,
        controllerRuntime,
        now,
        interactionLockUntilByCharacterPair
      })

      for (const event of emittedEvents) {
        if (event.kind !== 'show-character-message') {
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
      triggeredActions.clear()
      lastRuntimeErrorMessage = undefined
    } catch (error) {
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
    const isInventoryToggleKey =
      event.code === 'KeyI' || event.key.toLowerCase() === 'i'

    if (isInventoryToggleKey) {
      if (!event.repeat) {
        event.preventDefault()
        setPlayerUiOpen(!isPlayerUiOpen)
      }

      return
    }

    if (playerProfile.hp.current === 0) {
      return
    }

    if (isPlayerUiOpen || isBlacksmithShopOpen) {
      const action = getCharacterActionFromKey(event.key)
      const direction = getCharacterMoveDirectionFromKey(event.key)

      if (event.key === 'Escape' || action || direction || isAttackKey(event)) {
        event.preventDefault()
      }

      if (event.key === 'Escape') {
        closeAllOverlays()
      }

      return
    }

    const action = getCharacterActionFromKey(event.key)

    if (action || isAttackKey(event)) {
      event.preventDefault()

      const nextAction = action ?? 'attack'

      if (!pressedActions.has(nextAction)) {
        triggeredActions.add(nextAction)
      }

      pressedActions.add(nextAction)
      return
    }

    const direction = getCharacterMoveDirectionFromKey(event.key)

    if (!direction) {
      return
    }

    event.preventDefault()
    pressedDirections.add(direction)
  }

  const handleKeyUp = (event: KeyboardEvent) => {
    const action = getCharacterActionFromKey(event.key)

    if (action) {
      pressedActions.delete(action)
      return
    }

    if (isAttackKey(event)) {
      pressedActions.delete('attack')
      return
    }

    const direction = getCharacterMoveDirectionFromKey(event.key)

    if (!direction) {
      return
    }

    pressedDirections.delete(direction)
  }

  const handleWindowBlur = () => {
    clearPressedInputState()
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
  document.addEventListener('visibilitychange', handleVisibilityChange)
  app.ticker.add(updateCharacters)
  app.ticker.add(mapOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  app.ticker.add(playerHudOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  app.ticker.add(
    playerInventoryOverlay.syncFrame,
    undefined,
    UPDATE_PRIORITY.UTILITY
  )
  app.ticker.add(playerShopOverlay.syncFrame, undefined, UPDATE_PRIORITY.UTILITY)
  syncAllCharacterSprites()
  centerViewportOnCharacter(getCharacterStateById(cameraTargetCharacterId))
  showSceneIntroBanner()
  mapOverlay.syncFrame()
  playerHudOverlay.syncFrame()
  playerInventoryOverlay.syncFrame()
  playerShopOverlay.syncFrame()
  handleVisibilityChange()

  const destroy = () => {
    if (isDestroyed) {
      return
    }

    isDestroyed = true
    window.removeEventListener('keydown', handleKeyDown)
    window.removeEventListener('keyup', handleKeyUp)
    window.removeEventListener('blur', handleWindowBlur)
    document.removeEventListener('visibilitychange', handleVisibilityChange)
    app.ticker.remove(updateCharacters)
    app.ticker.remove(mapOverlay.syncFrame)
    app.ticker.remove(playerHudOverlay.syncFrame)
    app.ticker.remove(playerInventoryOverlay.syncFrame)
    app.ticker.remove(playerShopOverlay.syncFrame)
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
    playerShopOverlay.destroy()
    controllerRuntime.destroy()
    app.destroy({ removeView: true }, { children: true })
    sceneElement.remove()
  }

  if (import.meta.hot) {
    import.meta.hot.dispose(destroy)
  }

  return {
    destroy
  }
}

const loadMessagePanelTexture = async (): Promise<Texture> => {
  const uiSpritesheet = await Assets.load<Spritesheet>(UI_SPRITESHEET_URL)

  uiSpritesheet.textureSource.scaleMode = 'nearest'

  const panelTexture = uiSpritesheet.textures[MESSAGE_PANEL_TEXTURE_NAME]

  if (!panelTexture) {
    throw new Error(
      `Missing ${MESSAGE_PANEL_TEXTURE_NAME} in ${UI_SPRITESHEET_URL}`
    )
  }

  return panelTexture
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
  tileTextures: Texture[],
  tileset: ParsedTiledTileset,
  appearanceType: string
): Texture => tileTextures[resolveTilesetLocalIdByType(tileset, appearanceType)]

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
