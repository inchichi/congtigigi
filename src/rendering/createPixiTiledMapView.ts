import { CompositeTilemap } from '@pixi/tilemap'
import {
  Application,
  Assets,
  Container,
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

type CreatePixiTiledMapViewInput = {
  mountElement: HTMLElement
  map: ParsedTiledMap
  characters: CharacterState[]
  playerProfile: PlayerProfile
  playerEquipment: PlayerEquipment
  playerInventory: PlayerInventory
  merchantInventory: PlayerInventory
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

type RenderedCharacterNode = {
  container: Container
  sprite: Sprite
}

type RenderedPortalNode = {
  container: Container
  sprite: Sprite
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
const BLACKSMITH_SHOP_NPC_ID = 'blacksmith'
const PLAYER_WEAPON_TILE_LOCAL_ID = 117
const PLAYER_WEAPON_TILE_FRAME_SOURCE: TileTextureFrameSource = {
  columns: 12,
  margin: 0,
  spacing: 0,
  tileWidth: 16,
  tileHeight: 16
}
const PLAYER_WEAPON_WORLD_SCALE = 1.35
const PLAYER_ATTACK_DURATION_MILLISECONDS = 220
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
let messageFontsReadyPromise: Promise<void> | undefined

export const createPixiTiledMapView = async ({
  mountElement,
  map,
  characters,
  playerProfile,
  playerEquipment,
  playerInventory,
  merchantInventory,
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
  const [messagePanelTexture, tinyDungeonWeaponImageTexture] = await Promise.all([
    loadMessagePanelTexture(),
    Assets.load<Texture>(TINY_DUNGEON_TILESET_IMAGE_URL)
  ])

  tinyDungeonWeaponImageTexture.source.scaleMode = 'nearest'
  tinyDungeonWeaponImageTexture.source.addressMode = 'clamp-to-edge'
  await ensureMessageFontsLoaded()

  await app.init({
    antialias: false,
    autoDensity: true,
    backgroundColor: 0x171311,
    height: map.pixelHeight,
    preference: 'webgl',
    roundPixels: true,
    resolution: window.devicePixelRatio || 1,
    width: map.pixelWidth
  })
  app.ticker.maxFPS = 60

  const sceneElement = document.createElement('div')
  const runtimeWarningBannerElement = document.createElement('div')

  sceneElement.className = 'game-scene'
  sceneElement.style.width = `${map.pixelWidth}px`
  sceneElement.style.height = `${map.pixelHeight}px`
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

  const world = new Container()
  const messageLayer = new Container()
  const tilesetResources = new Map<string, TilesetRenderResources>()
  const wallTiles = createWallTileLookup(map)
  const pressedDirections = new Set<CharacterMoveDirection>()
  const pressedActions = new Set<CharacterAction>()
  const triggeredActions = new Set<CharacterAction>()
  const gameEventQueue = createGameEventQueue()
  const interactionLockUntilByCharacterPair = new Map<string, number>()
  const activeCharacterMessages = new Map<string, ActiveCharacterMessage>()
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
  let syncPlayerCharacterVisual = () => {}
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
  let lastRuntimeErrorMessage: string | undefined
  let depthSortedLayer: Container | undefined
  let characterStates = characters.map((character) => ({
    ...character,
    position: { ...character.position },
    collisionSize: { ...character.collisionSize }
  }))
  const mapPortals = createMapPortalsFromEventLayers({ map })
  const mapOverlay = createMapOverlay({
    mountElement,
    sourceCanvas: app.canvas,
    mapPixelWidth: map.pixelWidth,
    mapPixelHeight: map.pixelHeight,
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
  const isAttackKey = (event: KeyboardEvent): boolean =>
    event.code === 'KeyA' || event.key.toLowerCase() === 'a'
  const triggerPlayerAttack = (now: number) => {
    playerAttackStartedAtMilliseconds = now
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
    const sprite = new Sprite(
      resolveCharacterTexture(
        characterTilesetResources.tileTextures,
        characterSpriteSheet.tileset,
        character.appearanceType
      )
    )
    const isPlayer = character.id === PLAYER_CHARACTER_ID

    container.label = `character:${character.id}:container`
    container.sortableChildren = true
    sprite.label = `character:${character.id}`
    sprite.scale.set(characterSpriteSheet.scale)
    sprite.roundPixels = true
    sprite.zIndex = 0
    container.addChild(sprite)

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
      sprite
    })
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

  const getCharacterStateById = (characterId: string): CharacterState => {
    const character = characterStates.find(
      (candidateCharacter) => candidateCharacter.id === characterId
    )

    if (!character) {
      throw new Error(`Missing character ${characterId}`)
    }

    return character
  }

  syncPlayerCharacterVisual = () => {
    syncCharacterSprite(getCharacterStateById(PLAYER_CHARACTER_ID))
  }

  const syncCharacterSprite = (character: CharacterState) => {
    const renderNode = renderedCharacters.get(character.id)

    if (!renderNode) {
      throw new Error(`Missing rendered sprite for character ${character.id}`)
    }

    renderNode.container.position.set(
      character.position.x * map.tileWidth,
      character.position.y * map.tileHeight
    )
    renderNode.container.zIndex = getCharacterDepthSortValue(
      character.position.y,
      characterPixelHeight,
      map.tileHeight
    )
    depthSortedLayer?.sortChildren()

    if (character.id === PLAYER_CHARACTER_ID) {
      syncPlayerWeaponSprite(character)
    }
  }

  const syncAllCharacterSprites = () => {
    for (const character of characterStates) {
      syncCharacterSprite(character)
    }
  }

  const syncPlayerWeaponSprite = (character: CharacterState) => {
    if (!playerWeaponSprite) {
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

  const centerViewportOnCharacter = (character: CharacterState) => {
    const characterCenterX =
      character.position.x * map.tileWidth + characterPixelWidth / 2
    const characterCenterY =
      character.position.y * map.tileHeight + characterPixelHeight / 2
    const nextScrollLeft = clampScrollOffset(
      characterCenterX - mountElement.clientWidth / 2,
      map.pixelWidth - mountElement.clientWidth
    )
    const nextScrollTop = clampScrollOffset(
      characterCenterY - mountElement.clientHeight / 2,
      map.pixelHeight - mountElement.clientHeight
    )

    mountElement.scrollTo({
      left: nextScrollLeft,
      top: nextScrollTop
    })
  }

  const keepCharacterVisible = (character: CharacterState) => {
    const characterLeft = character.position.x * map.tileWidth
    const characterTop = character.position.y * map.tileHeight
    const characterRight = characterLeft + characterPixelWidth
    const characterBottom = characterTop + characterPixelHeight
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
        map.pixelWidth - viewportWidth
      )
    } else if (characterRight > deadZoneRight) {
      nextScrollLeft = clampScrollOffset(
        characterRight + horizontalMargin - viewportWidth,
        map.pixelWidth - viewportWidth
      )
    }

    if (characterTop < deadZoneTop) {
      nextScrollTop = clampScrollOffset(
        characterTop - verticalMargin,
        map.pixelHeight - viewportHeight
      )
    } else if (characterBottom > deadZoneBottom) {
      nextScrollTop = clampScrollOffset(
        characterBottom + verticalMargin - viewportHeight,
        map.pixelHeight - viewportHeight
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
    deltaY: number
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
    const blockingRects = getBlockingCollisionRects(characterId)
    let nextCharacter =
      desiredFacing === currentCharacter.facing
        ? currentCharacter
        : {
            ...currentCharacter,
            facing: desiredFacing
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

    if (nextCharacter.facing !== desiredFacing) {
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

      for (const character of [...characterStates]) {
        const intent = controllerRuntime.getIntent({
          character,
          deltaMilliseconds: app.ticker.deltaMS,
          pressedDirections,
          triggeredActions
        })

        if (!intent) {
          continue
        }

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

        if (event.characterId === BLACKSMITH_SHOP_NPC_ID) {
          setBlacksmithShopOpen(true)
        }
      }

      pruneExpiredCharacterMessages(now)
      syncActiveCharacterMessages()
      syncPlayerWeaponSprite(getCharacterStateById(PLAYER_CHARACTER_ID))
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
    for (const activeMessage of activeCharacterMessages.values()) {
      activeMessage.container.destroy({ children: true })
    }
    activeCharacterMessages.clear()
    runtimeWarningBannerElement.remove()
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
