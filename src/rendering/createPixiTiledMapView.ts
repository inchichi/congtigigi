import { CompositeTilemap } from '@pixi/tilemap'
import {
  Application,
  Assets,
  Container,
  Rectangle,
  Texture,
  groupD8
} from 'pixi.js'

import type {
  ParsedTiledMap,
  ParsedTiledTile,
  ParsedTiledTileset
} from '../game/tiled/parseTiledMap'

type CreatePixiTiledMapViewInput = {
  mountElement: HTMLElement
  map: ParsedTiledMap
  imageUrls: Record<string, string>
}

type TilesetRenderResources = {
  imageTexture: Texture
  tileTextures: Texture[]
}

export const createPixiTiledMapView = async ({
  mountElement,
  map,
  imageUrls
}: CreatePixiTiledMapViewInput): Promise<Application> => {
  const app = new Application()

  await app.init({
    antialias: false,
    autoDensity: true,
    backgroundColor: 0x171311,
    preference: 'webgl',
    resizeTo: mountElement,
    resolution: window.devicePixelRatio || 1
  })

  mountElement.replaceChildren(app.canvas)
  app.canvas.classList.add('game-canvas')

  const world = new Container()
  const tilesetResources = new Map<string, TilesetRenderResources>()

  app.stage.addChild(world)

  for (const tileset of map.tilesets) {
    tilesetResources.set(
      tileset.source,
      await loadTilesetRenderResources(tileset, imageUrls)
    )
  }

  for (const layer of map.layers) {
    const tilemap = new CompositeTilemap()

    tilemap.label = `layer:${layer.name}`
    tilemap.alpha = layer.opacity
    tilemap.visible = layer.visible

    for (const tile of layer.tiles) {
      const tileset = resolveTilesetForTile(tile, map.tilesets)
      const renderResources = tilesetResources.get(tileset.source)

      if (!renderResources) {
        throw new Error(`Missing render resources for tileset ${tileset.source}`)
      }

      tilemap.tile(
        renderResources.tileTextures[tile.localId],
        tile.x * map.tileWidth,
        tile.y * map.tileHeight,
        {
          rotate: toPixiTileRotation(tile)
        }
      )
    }

    world.addChild(tilemap)
  }

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
  tileset: ParsedTiledTileset,
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

const toPixiTileRotation = (tile: ParsedTiledTile): number => {
  if (tile.flipDiagonally) {
    if (tile.flipHorizontally && tile.flipVertically) {
      return groupD8.REVERSE_DIAGONAL
    }

    if (tile.flipHorizontally) {
      return groupD8.S
    }

    if (tile.flipVertically) {
      return groupD8.N
    }

    return groupD8.MAIN_DIAGONAL
  }

  if (tile.flipHorizontally && tile.flipVertically) {
    return groupD8.W
  }

  if (tile.flipHorizontally) {
    return groupD8.MIRROR_HORIZONTAL
  }

  if (tile.flipVertically) {
    return groupD8.MIRROR_VERTICAL
  }

  return groupD8.E
}
