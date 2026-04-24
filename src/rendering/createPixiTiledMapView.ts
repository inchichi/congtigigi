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
  PLAYER_MOVE_SPEED_TILES_PER_SECOND,
  getPlayerMoveDirectionFromKey,
  movePlayerState
} from '../game/playerState'
import type { PlayerMoveDirection, PlayerState } from '../game/playerState'
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
  player: PlayerState
  playerSpriteSheet: {
    imageUrl: string
    tileWidth: number
    tileHeight: number
    columns: number
    localId: number
    scale: number
  }
  eventSpriteSheet: {
    tileset: ParsedTiledTileset
    scale: number
  }
  imageUrls: Record<string, string>
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
  player,
  playerSpriteSheet,
  eventSpriteSheet,
  imageUrls
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

  app.stage.addChild(world)

  for (const tileset of map.tilesets) {
    tilesetResources.set(
      tileset.source,
      await loadTilesetRenderResources(tileset, imageUrls)
    )
  }
  const eventTilesetResources = await loadTilesetRenderResources(
    eventSpriteSheet.tileset,
    imageUrls
  )

  const playerTexture = await loadStandaloneTileTexture({
    imageUrl: playerSpriteSheet.imageUrl,
    tileWidth: playerSpriteSheet.tileWidth,
    tileHeight: playerSpriteSheet.tileHeight,
    columns: playerSpriteSheet.columns,
    localId: playerSpriteSheet.localId,
    scaleMode: 'nearest'
  })
  const playerSprite = new Sprite(playerTexture)
  const playerPixelWidth = playerSpriteSheet.tileWidth * playerSpriteSheet.scale
  const playerPixelHeight = playerSpriteSheet.tileHeight * playerSpriteSheet.scale
  const playerWidthInTiles = playerPixelWidth / map.tileWidth
  const playerHeightInTiles = playerPixelHeight / map.tileHeight
  const eventPixelWidth = eventSpriteSheet.tileset.tileWidth * eventSpriteSheet.scale
  const eventPixelHeight = eventSpriteSheet.tileset.tileHeight * eventSpriteSheet.scale
  const pressedDirections = new Set<PlayerMoveDirection>()
  let depthSortedLayer: Container | undefined
  let playerState = player

  playerSprite.label = 'player'
  playerSprite.scale.set(playerSpriteSheet.scale)
  playerSprite.roundPixels = true

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

  const eventCharacters = map.eventLayers.flatMap((eventLayer) =>
    eventLayer.events.flatMap((event) => {
      if (!event.visible || event.className !== 'character' || !event.appearanceType) {
        return []
      }

      const localId = resolveTilesetLocalIdByType(
        eventSpriteSheet.tileset,
        event.appearanceType
      )
      const texture = eventTilesetResources.tileTextures[localId]
      const sprite = new Sprite(texture)
      const collisionRect = createCollisionRectFromBottomCenter(
        event.x,
        event.y,
        event.width || eventPixelWidth,
        event.height || eventPixelHeight,
        map.tileWidth,
        map.tileHeight
      )

      sprite.label = `event:${event.name}`
      sprite.anchor.set(0.5, 1)
      sprite.position.set(event.x, event.y)
      sprite.scale.set(eventSpriteSheet.scale)
      sprite.roundPixels = true
      sprite.zIndex = event.y

      return [
        {
          collisionRect,
          sprite
        }
      ]
    })
  )
  const eventCollisionRects = eventCharacters.map(
    (eventCharacter) => eventCharacter.collisionRect
  )

  if (!depthSortedLayer && eventCharacters.length > 0) {
    depthSortedLayer = new Container()
    depthSortedLayer.label = 'layer:events:depth'
    depthSortedLayer.sortableChildren = true
    world.addChild(depthSortedLayer)
  }

  if (depthSortedLayer) {
    depthSortedLayer.addChild(playerSprite)

    for (const eventCharacter of eventCharacters) {
      depthSortedLayer.addChild(eventCharacter.sprite)
    }
  } else {
    world.addChild(playerSprite)
  }

  const syncPlayerSpritePosition = () => {
    playerSprite.position.set(
      playerState.position.x * map.tileWidth,
      playerState.position.y * map.tileHeight
    )
    playerSprite.zIndex = getPlayerDepthSortValue(
      playerState.position.y,
      playerPixelHeight,
      map.tileHeight
    )
    depthSortedLayer?.sortChildren()
  }

  const centerViewportOnPlayer = () => {
    const playerCenterX =
      playerState.position.x * map.tileWidth + playerPixelWidth / 2
    const playerCenterY =
      playerState.position.y * map.tileHeight + playerPixelHeight / 2
    const nextScrollLeft = clampScrollOffset(
      playerCenterX - mountElement.clientWidth / 2,
      map.pixelWidth - mountElement.clientWidth
    )
    const nextScrollTop = clampScrollOffset(
      playerCenterY - mountElement.clientHeight / 2,
      map.pixelHeight - mountElement.clientHeight
    )

    mountElement.scrollTo({
      left: nextScrollLeft,
      top: nextScrollTop
    })
  }

  const keepPlayerVisible = () => {
    const playerLeft = playerState.position.x * map.tileWidth
    const playerTop = playerState.position.y * map.tileHeight
    const playerRight = playerLeft + playerPixelWidth
    const playerBottom = playerTop + playerPixelHeight
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

    if (playerLeft < deadZoneLeft) {
      nextScrollLeft = clampScrollOffset(
        playerLeft - horizontalMargin,
        map.pixelWidth - viewportWidth
      )
    } else if (playerRight > deadZoneRight) {
      nextScrollLeft = clampScrollOffset(
        playerRight + horizontalMargin - viewportWidth,
        map.pixelWidth - viewportWidth
      )
    }

    if (playerTop < deadZoneTop) {
      nextScrollTop = clampScrollOffset(
        playerTop - verticalMargin,
        map.pixelHeight - viewportHeight
      )
    } else if (playerBottom > deadZoneBottom) {
      nextScrollTop = clampScrollOffset(
        playerBottom + verticalMargin - viewportHeight,
        map.pixelHeight - viewportHeight
      )
    }

    if (
      nextScrollLeft !== viewportLeft ||
      nextScrollTop !== viewportTop
    ) {
      mountElement.scrollTo({
        left: nextScrollLeft,
        top: nextScrollTop
      })
    }
  }

  const tryMovePlayer = (deltaX: number, deltaY: number) => {
    let nextPlayerState = playerState

    if (deltaX !== 0) {
      const nextXState = movePlayerState({
        player: nextPlayerState,
        delta: {
          x: deltaX,
          y: 0
        },
        mapWidth: map.width,
        mapHeight: map.height,
        playerWidth: playerWidthInTiles,
        playerHeight: playerHeightInTiles
      })

      if (
        !isPlayerPositionBlocked(
          wallTiles,
          eventCollisionRects,
          nextXState.position.x,
          nextXState.position.y,
          playerWidthInTiles,
          playerHeightInTiles
        )
      ) {
        nextPlayerState = nextXState
      }
    }

    if (deltaY !== 0) {
      const nextYState = movePlayerState({
        player: nextPlayerState,
        delta: {
          x: 0,
          y: deltaY
        },
        mapWidth: map.width,
        mapHeight: map.height,
        playerWidth: playerWidthInTiles,
        playerHeight: playerHeightInTiles
      })

      if (
        !isPlayerPositionBlocked(
          wallTiles,
          eventCollisionRects,
          nextYState.position.x,
          nextYState.position.y,
          playerWidthInTiles,
          playerHeightInTiles
        )
      ) {
        nextPlayerState = nextYState
      }
    }

    if (
      nextPlayerState.position.x === playerState.position.x &&
      nextPlayerState.position.y === playerState.position.y
    ) {
      return
    }

    playerState = nextPlayerState
    syncPlayerSpritePosition()
    keepPlayerVisible()
  }

  const updatePlayer = () => {
    const movement = getNormalizedMovementVector(pressedDirections)

    if (!movement) {
      return
    }

    const distanceInTiles =
      (PLAYER_MOVE_SPEED_TILES_PER_SECOND * app.ticker.deltaMS) / 1000

    tryMovePlayer(
      movement.x * distanceInTiles,
      movement.y * distanceInTiles
    )
  }

  const handleKeyDown = (event: KeyboardEvent) => {
    const direction = getPlayerMoveDirectionFromKey(event.key)

    if (!direction) {
      return
    }

    event.preventDefault()
    pressedDirections.add(direction)
  }

  const handleKeyUp = (event: KeyboardEvent) => {
    const direction = getPlayerMoveDirectionFromKey(event.key)

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
  app.ticker.add(updatePlayer)
  syncPlayerSpritePosition()
  centerViewportOnPlayer()

  if (import.meta.hot) {
    import.meta.hot.dispose(() => {
      window.removeEventListener('keydown', handleKeyDown)
      window.removeEventListener('keyup', handleKeyUp)
      window.removeEventListener('blur', handleWindowBlur)
      app.ticker.remove(updatePlayer)
      app.destroy({ removeView: true }, { children: true })
    })
  }

  return app
}

const loadTilesetRenderResources = async (
  tileset: ParsedTiledTileset,
  imageUrls: Record<string, string>
): Promise<TilesetRenderResources> => {
  const imageUrl = imageUrls[tileset.image.source]

  if (!imageUrl) {
    throw new Error(`Missing image URL for ${tileset.image.source}`)
  }

  const imageTexture = await Assets.load<Texture>(imageUrl)

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

const loadStandaloneTileTexture = async ({
  imageUrl,
  tileWidth,
  tileHeight,
  columns,
  localId,
  scaleMode
}: {
  imageUrl: string
  tileWidth: number
  tileHeight: number
  columns: number
  localId: number
  scaleMode: 'nearest' | 'linear'
}): Promise<Texture> => {
  const imageTexture = await Assets.load<Texture>(imageUrl)

  imageTexture.source.scaleMode = scaleMode
  imageTexture.source.wrapMode = 'clamp-to-edge'

  return createTileTexture(
    imageTexture,
    {
      columns,
      margin: 0,
      spacing: 0,
      tileWidth,
      tileHeight
    },
    localId
  )
}

const getNormalizedMovementVector = (
  pressedDirections: Set<PlayerMoveDirection>
): { x: number; y: number } | undefined => {
  let x = 0
  let y = 0

  if (pressedDirections.has('left')) {
    x -= 1
  }

  if (pressedDirections.has('right')) {
    x += 1
  }

  if (pressedDirections.has('up')) {
    y -= 1
  }

  if (pressedDirections.has('down')) {
    y += 1
  }

  if (x === 0 && y === 0) {
    return undefined
  }

  const magnitude = Math.hypot(x, y)

  return {
    x: x / magnitude,
    y: y / magnitude
  }
}

const isPlayerPositionBlocked = (
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

const getTileDepthSortValue = (tileY: number, tileHeight: number): number =>
  (tileY + 1) * tileHeight

const getPlayerDepthSortValue = (
  playerY: number,
  playerPixelHeight: number,
  tileHeight: number
): number => playerY * tileHeight + playerPixelHeight

const createCollisionRectFromBottomCenter = (
  x: number,
  y: number,
  width: number,
  height: number,
  tileWidth: number,
  tileHeight: number
): CollisionRect => ({
  x: (x - width / 2) / tileWidth,
  y: (y - height) / tileHeight,
  width: width / tileWidth,
  height: height / tileHeight
})

const doCollisionRectsIntersect = (
  left: CollisionRect,
  right: CollisionRect
): boolean =>
  left.x < right.x + right.width &&
  left.x + left.width > right.x &&
  left.y < right.y + right.height &&
  left.y + left.height > right.y

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
