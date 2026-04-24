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
  getPlayerMoveDirectionFromKey,
  movePlayerState
} from '../game/playerState'
import type { PlayerState } from '../game/playerState'
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

export const createPixiTiledMapView = async ({
  mountElement,
  map,
  player,
  playerSpriteSheet,
  imageUrls
}: CreatePixiTiledMapViewInput): Promise<Application> => {
  const app = new Application()

  await app.init({
    antialias: false,
    autoDensity: true,
    backgroundColor: 0x171311,
    preference: 'webgl',
    roundPixels: true,
    resizeTo: mountElement,
    resolution: window.devicePixelRatio || 1
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

  for (const layer of map.layers) {
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

  const playerTexture = await loadStandaloneTileTexture({
    imageUrl: playerSpriteSheet.imageUrl,
    tileWidth: playerSpriteSheet.tileWidth,
    tileHeight: playerSpriteSheet.tileHeight,
    columns: playerSpriteSheet.columns,
    localId: playerSpriteSheet.localId,
    scaleMode: 'nearest'
  })
  const playerSprite = new Sprite(playerTexture)
  let playerState = player

  playerSprite.label = 'player'
  playerSprite.scale.set(playerSpriteSheet.scale)
  playerSprite.roundPixels = true
  world.addChild(playerSprite)

  const syncPlayerSpritePosition = () => {
    playerSprite.position.set(
      playerState.position.x * map.tileWidth,
      playerState.position.y * map.tileHeight
    )
  }

  const handleKeyDown = (event: KeyboardEvent) => {
    const direction = getPlayerMoveDirectionFromKey(event.key)

    if (!direction) {
      return
    }

    event.preventDefault()
    const nextPlayerState = movePlayerState({
      player: playerState,
      direction,
      mapWidth: map.width,
      mapHeight: map.height
    })

    if (isWallTileAt(wallTiles, nextPlayerState.position.x, nextPlayerState.position.y)) {
      return
    }

    playerState = nextPlayerState
    syncPlayerSpritePosition()
  }

  window.addEventListener('keydown', handleKeyDown)
  syncPlayerSpritePosition()

  const layoutWorld = () => {
    const availableWidth = app.screen.width
    const availableHeight = app.screen.height
    const fittedScale = Math.min(
      availableWidth / map.pixelWidth,
      availableHeight / map.pixelHeight
    )
    const scale = fittedScale >= 1 ? Math.max(1, Math.floor(fittedScale)) : fittedScale
    const scaledWidth = map.pixelWidth * scale
    const scaledHeight = map.pixelHeight * scale

    world.scale.set(scale)
    world.position.set(
      Math.round((availableWidth - scaledWidth) / 2),
      Math.round((availableHeight - scaledHeight) / 2)
    )
  }

  const resizeObserver = new ResizeObserver(() => {
    layoutWorld()
  })

  resizeObserver.observe(mountElement)
  layoutWorld()

  if (import.meta.hot) {
    import.meta.hot.dispose(() => {
      window.removeEventListener('keydown', handleKeyDown)
      resizeObserver.disconnect()
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

  imageTexture.source.scaleMode = 'nearest'
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
