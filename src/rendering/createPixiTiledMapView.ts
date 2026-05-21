import { CompositeTilemap } from '@pixi/tilemap'
import {
  Application,
  Assets,
  AnimatedSprite,
  Container,
  NineSliceSprite,
  Rectangle,
  Sprite,
  Text,
  TextStyle,
  Texture
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

type CreatePixiTiledMapViewInput = {
  mountElement: HTMLElement
  map: ParsedTiledMap
  characters: CharacterState[]
  cameraTargetCharacterId: string
  characterSpriteSheet: {
    tileset: ParsedTiledTileset
    scale: number
  }
  imageUrls: Record<string, string>
  controllerRuntime: CharacterControllerRuntime
}

type TilesetRenderResources = {
  imageTexture: Texture
  tileTextures: Texture[]
}

type SmearVfxRenderResources = {
  horizontalTextures: Texture[]
  verticalTextures: Texture[]
}

type PlayerEquipmentTier = 'leather' | 'chain' | 'iron'

type PlayerEquipmentTextureSet = {
  armor: Texture
  helmet: Texture
}

type PlayerEquipmentSprites = {
  armor: Sprite
  helmet: Sprite
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

const DEPTH_SORTED_LAYER_NAME = 'object'
const UI_SPRITESHEET_IMAGE_URL = new URL(
  '../assets/spritesheets/uipack_rpg_sheet.png',
  import.meta.url
).href
const PLAYER_WEAPON_TILE_LOCAL_ID = 117
const PLAYER_WEAPON_TILE_FRAME_SOURCE: TileTextureFrameSource = {
  columns: 12,
  margin: 0,
  spacing: 0,
  tileWidth: 16,
  tileHeight: 16
}
const SMEAR_VFX_HORIZONTAL_SPRITESHEET_URL = new URL(
  '../assets/vfx/smear-vfx-01/smear-vfx-01-horizontal-1.png',
  import.meta.url
).href
const SMEAR_VFX_VERTICAL_SPRITESHEET_URL = new URL(
  '../assets/vfx/smear-vfx-01/smear-vfx-01-vertical-1.png',
  import.meta.url
).href
const PLAYER_EQUIPMENT_ASSET_URLS = {
  leather: {
    armor: new URL(
      '../assets/spritesheets/player-equipment/leather-armor-overlay.png',
      import.meta.url
    ).href,
    helmet: new URL(
      '../assets/spritesheets/player-equipment/leather-helmet-overlay.png',
      import.meta.url
    ).href
  },
  chain: {
    armor: new URL(
      '../assets/spritesheets/player-equipment/chain-armor-overlay.png',
      import.meta.url
    ).href,
    helmet: new URL(
      '../assets/spritesheets/player-equipment/chain-helmet-overlay.png',
      import.meta.url
    ).href
  },
  iron: {
    armor: new URL(
      '../assets/spritesheets/player-equipment/iron-armor-overlay.png',
      import.meta.url
    ).href,
    helmet: new URL(
      '../assets/spritesheets/player-equipment/iron-helmet-overlay.png',
      import.meta.url
    ).href
  }
} satisfies Record<PlayerEquipmentTier, Record<'armor' | 'helmet', string>>
const PLAYER_EQUIPMENT_TIER_BY_KEY: Record<string, PlayerEquipmentTier> = {
  Digit1: 'leather',
  Digit2: 'chain',
  Digit3: 'iron'
}
const PLAYER_DEFAULT_EQUIPMENT_TIER: PlayerEquipmentTier = 'leather'
const SMEAR_VFX_FRAME_SIZE = 48
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
const MESSAGE_PANEL_TEXTURE_NAME = 'panelInset_beige.png'
const MESSAGE_PANEL_TEXTURE_FRAME = new Rectangle(200, 294, 93, 94)
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
const PLAYER_ATTACK_SLASH_EFFECT_SCALE_X = 1.55
const PLAYER_ATTACK_SLASH_EFFECT_SCALE_Y = 1.35
const PLAYER_ATTACK_SLASH_EFFECT_WIDE_FRAME_INDEX = 1
const PLAYER_ATTACK_SLASH_EFFECT_WIDE_FRAME_X_MULTIPLIER = 1.2
let messageFontsReadyPromise: Promise<void> | undefined

export const createPixiTiledMapView = async ({
  mountElement,
  map,
  characters,
  cameraTargetCharacterId,
  characterSpriteSheet,
  imageUrls,
  controllerRuntime
}: CreatePixiTiledMapViewInput): Promise<Application> => {
  const app = new Application()
  const [messagePanelTexture, smearVfxTextures, playerEquipmentTextures] =
    await Promise.all([
    loadMessagePanelTexture(),
    loadSmearVfxTextures(),
    loadPlayerEquipmentTextures()
  ])

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
  const renderedCharacters = new Map<string, Sprite>()
  const characterPixelWidth =
    characterSpriteSheet.tileset.tileWidth * characterSpriteSheet.scale
  const characterPixelHeight =
    characterSpriteSheet.tileset.tileHeight * characterSpriteSheet.scale
  let playerAttackStartedAtMilliseconds: number | undefined
  let playerAttackFacing: CharacterMoveDirection | undefined
  let playerWeaponTrailSprites: Sprite[] = []
  let playerWeaponSprite: Sprite | undefined
  let playerEquipmentSprites: PlayerEquipmentSprites | undefined
  let activePlayerEquipmentTier: PlayerEquipmentTier = PLAYER_DEFAULT_EQUIPMENT_TIER
  let playerSlashEffectSprite: AnimatedSprite | undefined
  let lastRuntimeErrorMessage: string | undefined
  let depthSortedLayer: Container | undefined
  let characterStates = characters.map((character) => ({
    ...character,
    position: { ...character.position },
    collisionSize: { ...character.collisionSize }
  }))

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
    characterTilesetResources.imageTexture,
    PLAYER_WEAPON_TILE_FRAME_SOURCE,
    PLAYER_WEAPON_TILE_LOCAL_ID
  )

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
    const sprite = new Sprite(
      resolveCharacterTexture(
        characterTilesetResources.tileTextures,
        characterSpriteSheet.tileset,
        character.appearanceType
      )
    )

    sprite.label = `character:${character.id}`
    sprite.scale.set(characterSpriteSheet.scale)
    sprite.roundPixels = true
    renderedCharacters.set(character.id, sprite)
    depthSortedLayer.addChild(sprite)

    if (character.id === PLAYER_CHARACTER_ID) {
      playerEquipmentSprites = createPlayerEquipmentSprites(
        playerEquipmentTextures[activePlayerEquipmentTier]
      )
      depthSortedLayer.addChild(
        playerEquipmentSprites.armor,
        playerEquipmentSprites.helmet
      )
      playerWeaponTrailSprites = Array.from(
        { length: PLAYER_ATTACK_TRAIL_SPRITE_COUNT },
        (_, index) => {
          const trailSprite = new Sprite(playerWeaponTexture)

          trailSprite.label = `character:player:weapon-trail:${index}`
          trailSprite.anchor.set(0.5, 1)
          trailSprite.visible = false
          trailSprite.roundPixels = true
          trailSprite.zIndex = index + 1
          depthSortedLayer.addChild(trailSprite)

          return trailSprite
        }
      )
      playerWeaponSprite = new Sprite(playerWeaponTexture)
      playerWeaponSprite.label = 'character:player:weapon'
      playerWeaponSprite.anchor.set(0.5, 1)
      playerWeaponSprite.visible = false
      playerWeaponSprite.roundPixels = true
      playerWeaponSprite.zIndex = PLAYER_ATTACK_TRAIL_SPRITE_COUNT + 1
      depthSortedLayer.addChild(playerWeaponSprite)
    }
  }
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

  const syncCharacterSprite = (character: CharacterState) => {
    const sprite = renderedCharacters.get(character.id)

    if (!sprite) {
      throw new Error(`Missing rendered sprite for character ${character.id}`)
    }

    sprite.position.set(
      character.position.x * map.tileWidth,
      character.position.y * map.tileHeight
    )
    sprite.zIndex = getCharacterDepthSortValue(
      character.position.y,
      characterPixelHeight,
      map.tileHeight
    )

    if (character.id === PLAYER_CHARACTER_ID && playerEquipmentSprites) {
      syncPlayerEquipmentSprites(character)
    }

    depthSortedLayer?.sortChildren()
  }

  const syncAllCharacterSprites = () => {
    for (const character of characterStates) {
      syncCharacterSprite(character)
    }
  }

  const syncPlayerEquipmentSprites = (character: CharacterState) => {
    if (!playerEquipmentSprites) {
      return
    }

    const characterLeft = character.position.x * map.tileWidth
    const characterTop = character.position.y * map.tileHeight
    const characterDepth = getCharacterDepthSortValue(
      character.position.y,
      characterPixelHeight,
      map.tileHeight
    )

    playerEquipmentSprites.armor.position.set(
      characterLeft,
      characterTop
    )
    playerEquipmentSprites.armor.zIndex = characterDepth + 0.1
    playerEquipmentSprites.helmet.position.set(
      characterLeft,
      characterTop
    )
    playerEquipmentSprites.helmet.zIndex = characterDepth + 0.2
  }

  const updatePlayerEquipmentTier = (tier: PlayerEquipmentTier) => {
    activePlayerEquipmentTier = tier

    if (!playerEquipmentSprites) {
      return
    }

    const textures = playerEquipmentTextures[tier]

    playerEquipmentSprites.armor.texture = textures.armor
    playerEquipmentSprites.helmet.texture = textures.helmet
    syncPlayerEquipmentSprites(getCharacterStateById(PLAYER_CHARACTER_ID))
    depthSortedLayer?.sortChildren()
  }

  const clearPlayerAttackSprites = () => {
    playerAttackStartedAtMilliseconds = undefined
    playerAttackFacing = undefined

    if (playerWeaponSprite) {
      playerWeaponSprite.visible = false
    }

    for (const trailSprite of playerWeaponTrailSprites) {
      trailSprite.visible = false
    }
  }

  const syncPlayerWeaponSprite = (character: CharacterState, now: number) => {
    if (!playerWeaponSprite) {
      return
    }

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

    if (attackProgress === undefined) {
      clearPlayerAttackSprites()
      return
    }

    const attackFacing = playerAttackFacing ?? character.facing
    const placement =
      attackFacing === 'left'
        ? PLAYER_WEAPON_PLACEMENT_LEFT
        : PLAYER_WEAPON_PLACEMENT_RIGHT
    const facingMultiplier = attackFacing === 'left' ? -1 : 1
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
      sprite.position.set(
        character.position.x * map.tileWidth + pose.x,
        character.position.y * map.tileHeight + pose.y
      )
      sprite.rotation = pose.rotation
      sprite.scale.set(pose.scale)
      sprite.alpha = alpha
    }

    applyPose(playerWeaponSprite, createPose(attackProgress), 1)

    for (let index = 0; index < playerWeaponTrailSprites.length; index += 1) {
      const trailSprite = playerWeaponTrailSprites[index]
      const trailProgress =
        attackProgress - (index + 1) * PLAYER_ATTACK_TRAIL_PROGRESS_STEP

      if (trailProgress <= 0) {
        trailSprite.visible = false
        continue
      }

      applyPose(
        trailSprite,
        createPose(trailProgress),
        PLAYER_ATTACK_TRAIL_ALPHA[index] ?? 0.1
      )
    }

    depthSortedLayer?.sortChildren()
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

    if (
      nextCharacter.position.x !== currentCharacter.position.x ||
      nextCharacter.position.y !== currentCharacter.position.y
    ) {
      syncCharacterSprite(nextCharacter)
    }

    if (
      nextCharacter.id === cameraTargetCharacterId &&
      (nextCharacter.position.x !== currentCharacter.position.x ||
        nextCharacter.position.y !== currentCharacter.position.y)
    ) {
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

        drainControllerRuntimeEventsIntoQueue()

        if (intent.actions?.includes('interact')) {
          gameEventQueue.enqueue({
            kind: 'interaction-requested',
            sourceCharacterId: character.id
          })
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
      }

      pruneExpiredCharacterMessages(now)
      syncActiveCharacterMessages()
      syncPlayerWeaponSprite(getCharacterStateById(PLAYER_CHARACTER_ID), now)
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

  const isAttackKey = (event: KeyboardEvent): boolean =>
    event.code === 'KeyA' || event.key.toLowerCase() === 'a'
  const triggerPlayerAttack = (character: CharacterState, now: number) => {
    playerAttackStartedAtMilliseconds = now
    playerAttackFacing = character.facing
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
      ? smearVfxTextures.horizontalTextures
      : smearVfxTextures.verticalTextures
    const slashSprite = new AnimatedSprite(slashTextures)
    const slashBaseScaleX =
      character.facing === 'left'
        ? -PLAYER_ATTACK_SLASH_EFFECT_SCALE_X
        : PLAYER_ATTACK_SLASH_EFFECT_SCALE_X
    const applySlashFrameScale = (frameIndex: number) => {
      const frameScaleX =
        isHorizontalSlash &&
        frameIndex === PLAYER_ATTACK_SLASH_EFFECT_WIDE_FRAME_INDEX
          ? PLAYER_ATTACK_SLASH_EFFECT_WIDE_FRAME_X_MULTIPLIER
          : 1

      slashSprite.scale.set(
        slashBaseScaleX * frameScaleX,
        PLAYER_ATTACK_SLASH_EFFECT_SCALE_Y
      )
    }

    slashSprite.label = 'character:player:slash-effect'
    slashSprite.anchor.set(0.5)
    slashSprite.animationSpeed = 0.8
    slashSprite.loop = false
    slashSprite.roundPixels = true
    slashSprite.position.set(
      character.position.x * map.tileWidth + characterPixelWidth / 2,
      character.position.y * map.tileHeight + characterPixelHeight / 2 - 1
    )
    applySlashFrameScale(0)
    slashSprite.onFrameChange = (currentFrame) => {
      applySlashFrameScale(currentFrame)
    }
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
  const handleKeyDown = (event: KeyboardEvent) => {
    const equipmentTier = PLAYER_EQUIPMENT_TIER_BY_KEY[event.code]

    if (equipmentTier) {
      event.preventDefault()
      updatePlayerEquipmentTier(equipmentTier)
      return
    }

    if (isAttackKey(event)) {
      event.preventDefault()
      if (!event.repeat) {
        triggerPlayerAttack(
          getCharacterStateById(PLAYER_CHARACTER_ID),
          performance.now()
        )
        playPlayerSlashEffect(getCharacterStateById(PLAYER_CHARACTER_ID))
      }
      return
    }

    const action = getCharacterActionFromKey(event.key)

    if (action) {
      event.preventDefault()

      if (!pressedActions.has(action)) {
        triggeredActions.add(action)
      }

      pressedActions.add(action)
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
    if (isAttackKey(event)) {
      return
    }

    const action = getCharacterActionFromKey(event.key)

    if (action) {
      pressedActions.delete(action)
      return
    }

    const direction = getCharacterMoveDirectionFromKey(event.key)

    if (!direction) {
      return
    }

    pressedDirections.delete(direction)
  }

  const handleWindowBlur = () => {
    pressedDirections.clear()
    pressedActions.clear()
    triggeredActions.clear()
    clearPlayerAttackSprites()
    clearPlayerSlashEffectSprite()
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
  syncAllCharacterSprites()
  centerViewportOnCharacter(getCharacterStateById(cameraTargetCharacterId))
  handleVisibilityChange()

  if (import.meta.hot) {
    import.meta.hot.dispose(() => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', handleWindowBlur)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      app.ticker.remove(updateCharacters)
      gameEventQueue.clear()
      for (const activeMessage of activeCharacterMessages.values()) {
        activeMessage.container.destroy({ children: true })
      }
      activeCharacterMessages.clear()
      runtimeWarningBannerElement.remove()
      controllerRuntime.destroy()
      app.destroy({ removeView: true }, { children: true })
    })
  }

  return app
}

const loadMessagePanelTexture = async (): Promise<Texture> => {
  const uiSpritesheetTexture = await Assets.load<Texture>(UI_SPRITESHEET_IMAGE_URL)

  uiSpritesheetTexture.source.scaleMode = 'nearest'

  const panelTexture = new Texture({
    source: uiSpritesheetTexture.source,
    frame: MESSAGE_PANEL_TEXTURE_FRAME,
    orig: new Rectangle(
      0,
      0,
      MESSAGE_PANEL_TEXTURE_FRAME.width,
      MESSAGE_PANEL_TEXTURE_FRAME.height
    )
  })

  if (!panelTexture) {
    throw new Error(
      `Missing ${MESSAGE_PANEL_TEXTURE_NAME} in ${UI_SPRITESHEET_IMAGE_URL}`
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

const loadSmearVfxTextures = async (): Promise<SmearVfxRenderResources> => {
  const horizontalSpritesheet = await Assets.load<Texture>(
    SMEAR_VFX_HORIZONTAL_SPRITESHEET_URL
  )
  const verticalSpritesheet = await Assets.load<Texture>(
    SMEAR_VFX_VERTICAL_SPRITESHEET_URL
  )

  horizontalSpritesheet.source.scaleMode = 'nearest'
  verticalSpritesheet.source.scaleMode = 'nearest'

  return {
    horizontalTextures: createSmearVfxFrameTextures(horizontalSpritesheet),
    verticalTextures: createSmearVfxFrameTextures(verticalSpritesheet)
  }
}

const createSmearVfxFrameTextures = (imageTexture: Texture): Texture[] => {
  const frameCount = Math.floor(imageTexture.source.pixelWidth / SMEAR_VFX_FRAME_SIZE)

  if (frameCount < 1) {
    throw new Error(
      `Expected at least one smear VFX frame in ${imageTexture.source.label ?? 'texture'}`
    )
  }

  return Array.from({ length: frameCount }, (_, frameIndex) =>
    new Texture({
      source: imageTexture.source,
      frame: new Rectangle(
        frameIndex * SMEAR_VFX_FRAME_SIZE,
        0,
        SMEAR_VFX_FRAME_SIZE,
        SMEAR_VFX_FRAME_SIZE
      ),
      orig: new Rectangle(0, 0, SMEAR_VFX_FRAME_SIZE, SMEAR_VFX_FRAME_SIZE)
    })
  )
}

const loadPlayerEquipmentTextures = async (): Promise<
  Record<PlayerEquipmentTier, PlayerEquipmentTextureSet>
> => ({
  leather: {
    armor: await loadNearestTexture(PLAYER_EQUIPMENT_ASSET_URLS.leather.armor),
    helmet: await loadNearestTexture(PLAYER_EQUIPMENT_ASSET_URLS.leather.helmet)
  },
  chain: {
    armor: await loadNearestTexture(PLAYER_EQUIPMENT_ASSET_URLS.chain.armor),
    helmet: await loadNearestTexture(PLAYER_EQUIPMENT_ASSET_URLS.chain.helmet)
  },
  iron: {
    armor: await loadNearestTexture(PLAYER_EQUIPMENT_ASSET_URLS.iron.armor),
    helmet: await loadNearestTexture(PLAYER_EQUIPMENT_ASSET_URLS.iron.helmet)
  }
})

const loadNearestTexture = async (imageUrl: string): Promise<Texture> => {
  const texture = await Assets.load<Texture>(imageUrl)

  texture.source.scaleMode = 'nearest'

  return texture
}

const createPlayerEquipmentSprites = (
  textures: PlayerEquipmentTextureSet
): PlayerEquipmentSprites => {
  const armor = new Sprite(textures.armor)
  const helmet = new Sprite(textures.helmet)

  armor.label = 'character:player:armor'
  helmet.label = 'character:player:helmet'
  armor.roundPixels = true
  helmet.roundPixels = true

  return {
    armor,
    helmet
  }
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
