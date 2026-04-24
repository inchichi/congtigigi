import { CompositeTilemap } from '@pixi/tilemap'
import {
  Application,
  Assets,
  Container,
  Rectangle,
  Sprite,
  Texture
} from 'pixi.js'

import {
  getCharacterMoveDirectionFromKey,
  moveCharacterState
} from '../game/characterState'
import type {
  CharacterMoveDirection,
  CharacterState
} from '../game/characterState'
import type { CharacterControllerRuntime } from '../game/createCharacterControllerRuntime'
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

const DEPTH_SORTED_LAYER_NAME = 'object'

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

  mountElement.replaceChildren(app.canvas)
  app.canvas.classList.add('game-canvas')

  const world = new Container()
  const tilesetResources = new Map<string, TilesetRenderResources>()
  const wallTiles = createWallTileLookup(map)
  const pressedDirections = new Set<CharacterMoveDirection>()
  const renderedCharacters = new Map<string, Sprite>()
  const characterPixelWidth =
    characterSpriteSheet.tileset.tileWidth * characterSpriteSheet.scale
  const characterPixelHeight =
    characterSpriteSheet.tileset.tileHeight * characterSpriteSheet.scale
  let depthSortedLayer: Container | undefined
  let characterStates = characters.map((character) => ({
    ...character,
    position: { ...character.position },
    collisionSize: { ...character.collisionSize }
  }))

  controllerRuntime.syncCharacters(characterStates)

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
  }

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
    depthSortedLayer?.sortChildren()
  }

  const syncAllCharacterSprites = () => {
    for (const character of characterStates) {
      syncCharacterSprite(character)
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
    const blockingRects = getBlockingCollisionRects(characterId)
    let nextCharacter = currentCharacter

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

    if (
      nextCharacter.position.x === currentCharacter.position.x &&
      nextCharacter.position.y === currentCharacter.position.y
    ) {
      return
    }

    characterStates = characterStates.map((character) =>
      character.id === nextCharacter.id ? nextCharacter : character
    )
    syncCharacterSprite(nextCharacter)

    if (nextCharacter.id === cameraTargetCharacterId) {
      keepCharacterVisible(nextCharacter)
    }
  }

  const updateCharacters = () => {
    controllerRuntime.syncCharacters(characterStates)

    for (const character of [...characterStates]) {
      const delta = controllerRuntime.getMovementDelta({
        character,
        deltaMilliseconds: app.ticker.deltaMS,
        pressedDirections
      })

      if (!delta) {
        continue
      }

      tryMoveCharacter(character.id, delta.x, delta.y)
    }
  }

  const handleKeyDown = (event: KeyboardEvent) => {
    const direction = getCharacterMoveDirectionFromKey(event.key)

    if (!direction) {
      return
    }

    event.preventDefault()
    pressedDirections.add(direction)
  }

  const handleKeyUp = (event: KeyboardEvent) => {
    const direction = getCharacterMoveDirectionFromKey(event.key)

    if (!direction) {
      return
    }

    pressedDirections.delete(direction)
  }

  const handleWindowBlur = () => {
    pressedDirections.clear()
  }

  window.addEventListener('keydown', handleKeyDown)
  window.addEventListener('keyup', handleKeyUp)
  window.addEventListener('blur', handleWindowBlur)
  app.ticker.add(updateCharacters)
  syncAllCharacterSprites()
  centerViewportOnCharacter(getCharacterStateById(cameraTargetCharacterId))

  if (import.meta.hot) {
    import.meta.hot.dispose(() => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', handleWindowBlur)
      app.ticker.remove(updateCharacters)
      controllerRuntime.destroy()
      app.destroy({ removeView: true }, { children: true })
    })
  }

  return app
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
  imageTexture.source.wrapMode = 'clamp-to-edge'
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
