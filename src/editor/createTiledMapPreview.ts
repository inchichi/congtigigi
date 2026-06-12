import { CompositeTilemap } from '@pixi/tilemap'
import { Application, Container, Sprite, Texture } from 'pixi.js'

import type {
  ParsedTiledMap,
  ParsedTiledTile
} from '../games/my-sample-rpg/tiled/parseTiledMap'
import {
  createTileTexture,
  resolveTilesetForTile,
  type TilesetRenderResources
} from '../games/my-sample-rpg/rendering/tiledMapRenderResources'
import {
  getSpriteTransformForTile,
  hasTileTransform
} from '../games/my-sample-rpg/rendering/tiledSpriteTransform'

// 에디터 오른쪽 패널용 경량 Tiled 맵 뷰어. 본게임(createPixiTiledMapView)이 쓰는 것과 똑같은
// 파서/타일 텍스처/CompositeTilemap 렌더 경로를 쓰되, 플레이어·몬스터·전투 같은 게임 로직은
// 빼고 "맵을 그려서 보여주는" 부분만 남긴다. Love2D 게임처럼 브라우저에서 런타임을 돌릴 수
// 없는 게임도, 적어도 실제 맵 모습을 라이브로 확인할 수 있게 한다.

export type CreateTiledMapPreviewInput = {
  mountElement: HTMLElement
  map: ParsedTiledMap
  // 타일셋 이미지 source(예: "town-32.png") → 로드 가능한 URL(blob/object URL 포함).
  imageUrls: Record<string, string>
}

export type TiledMapPreview = {
  destroy: () => void
}

const BACKGROUND_COLOR = 0x101014
const MIN_ZOOM_FACTOR = 0.4
const MAX_ZOOM_FACTOR = 12

// Pixi의 Assets.load는 URL 확장자로 로더를 고르는데, 폴더에서 만든 blob: URL은 확장자가 없어
// 텍스처 로더를 못 고른다. 그래서 HTMLImageElement로 직접 디코드한 뒤 Texture.from으로 만든다.
const loadImageTexture = async (url: string): Promise<Texture> => {
  const image = new Image()
  image.src = url
  await image.decode()
  const texture = Texture.from(image)
  // 픽셀 아트가 흐려지지 않도록 nearest + 가장자리 클램프(본게임과 동일).
  texture.source.scaleMode = 'nearest'
  texture.source.addressMode = 'clamp-to-edge'
  return texture
}

const loadTilesetResources = async (
  map: ParsedTiledMap,
  imageUrls: Record<string, string>
): Promise<Map<string, TilesetRenderResources>> => {
  const resources = new Map<string, TilesetRenderResources>()

  for (const tileset of map.tilesets) {
    const imageUrl = imageUrls[tileset.image.source]

    if (!imageUrl) {
      throw new Error(`타일셋 이미지를 찾지 못했습니다: ${tileset.image.source}`)
    }

    const imageTexture = await loadImageTexture(imageUrl)
    const tileTextures = Array.from({ length: tileset.tileCount }, (_, localId) =>
      createTileTexture(imageTexture, tileset, localId)
    )

    resources.set(tileset.source, { imageTexture, tileTextures })
  }

  return resources
}

const buildWorld = (
  map: ParsedTiledMap,
  tilesetResources: Map<string, TilesetRenderResources>
): Container => {
  const world = new Container()

  for (const layer of map.layers) {
    const tilemap = new CompositeTilemap()
    const transformedTiles = new Container()

    tilemap.alpha = layer.opacity
    tilemap.visible = layer.visible
    transformedTiles.alpha = layer.opacity
    transformedTiles.visible = layer.visible

    for (const tile of layer.tiles) {
      const tileset = resolveTilesetForTile(tile, map.tilesets)
      const resources = tilesetResources.get(tileset.source)

      if (!resources) {
        continue
      }

      const texture = resources.tileTextures[tile.localId]

      if (!texture) {
        continue
      }

      if (hasTileTransform(tile)) {
        transformedTiles.addChild(
          createTransformedTileSprite(texture, tile, map.tileWidth, map.tileHeight)
        )
        continue
      }

      tilemap.tile(texture, tile.x * map.tileWidth, tile.y * map.tileHeight)
    }

    world.addChild(tilemap)
    world.addChild(transformedTiles)
  }

  return world
}

// flip/회전된 타일은 CompositeTilemap이 표현 못 해서 본게임처럼 개별 Sprite로 그린다.
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

export const createTiledMapPreview = async ({
  mountElement,
  map,
  imageUrls
}: CreateTiledMapPreviewInput): Promise<TiledMapPreview> => {
  const tilesetResources = await loadTilesetResources(map, imageUrls)

  const app = new Application()
  await app.init({
    resizeTo: mountElement,
    antialias: false,
    roundPixels: true,
    backgroundColor: BACKGROUND_COLOR
  })

  // destroy를 두 번 불러도 안전하도록 가드한다(맵 전환 시 이전 프리뷰 정리 + 패널 모드 전환).
  let isDestroyed = false
  const canvas = app.canvas
  mountElement.append(canvas)

  const world = buildWorld(map, tilesetResources)
  app.stage.addChild(world)

  let baseScale = 1

  const fitToView = (): void => {
    const viewWidth = app.screen.width
    const viewHeight = app.screen.height

    if (viewWidth === 0 || viewHeight === 0) {
      return
    }

    baseScale = Math.min(
      viewWidth / map.pixelWidth,
      viewHeight / map.pixelHeight
    )
    world.scale.set(baseScale)
    world.position.set(
      (viewWidth - map.pixelWidth * baseScale) / 2,
      (viewHeight - map.pixelHeight * baseScale) / 2
    )
  }

  fitToView()

  // 휠 줌(포인터 위치 고정).
  const handleWheel = (event: WheelEvent): void => {
    event.preventDefault()
    const rect = canvas.getBoundingClientRect()
    const pointerX = event.clientX - rect.left
    const pointerY = event.clientY - rect.top
    const worldX = (pointerX - world.x) / world.scale.x
    const worldY = (pointerY - world.y) / world.scale.y
    const zoomFactor = event.deltaY < 0 ? 1.1 : 1 / 1.1
    const nextScale = Math.max(
      baseScale * MIN_ZOOM_FACTOR,
      Math.min(world.scale.x * zoomFactor, baseScale * MAX_ZOOM_FACTOR)
    )

    world.scale.set(nextScale)
    world.x = pointerX - worldX * nextScale
    world.y = pointerY - worldY * nextScale
  }

  // 드래그 팬.
  let isPanning = false
  let panPointerId = -1
  let lastPointerX = 0
  let lastPointerY = 0

  const handlePointerDown = (event: PointerEvent): void => {
    isPanning = true
    panPointerId = event.pointerId
    lastPointerX = event.clientX
    lastPointerY = event.clientY
    canvas.setPointerCapture(event.pointerId)
  }

  const handlePointerMove = (event: PointerEvent): void => {
    if (!isPanning || event.pointerId !== panPointerId) {
      return
    }

    world.x += event.clientX - lastPointerX
    world.y += event.clientY - lastPointerY
    lastPointerX = event.clientX
    lastPointerY = event.clientY
  }

  const endPan = (event: PointerEvent): void => {
    if (event.pointerId !== panPointerId) {
      return
    }

    isPanning = false
    panPointerId = -1
  }

  canvas.addEventListener('wheel', handleWheel, { passive: false })
  canvas.addEventListener('pointerdown', handlePointerDown)
  canvas.addEventListener('pointermove', handlePointerMove)
  canvas.addEventListener('pointerup', endPan)
  canvas.addEventListener('pointercancel', endPan)

  // 패널 크기가 바뀌면 다시 화면에 맞춘다.
  const resizeObserver = new ResizeObserver(() => {
    fitToView()
  })
  resizeObserver.observe(mountElement)

  const destroy = (): void => {
    if (isDestroyed) {
      return
    }

    isDestroyed = true
    resizeObserver.disconnect()
    canvas.removeEventListener('wheel', handleWheel)
    canvas.removeEventListener('pointerdown', handlePointerDown)
    canvas.removeEventListener('pointermove', handlePointerMove)
    canvas.removeEventListener('pointerup', endPan)
    canvas.removeEventListener('pointercancel', endPan)
    app.destroy({ removeView: true }, { children: true })
  }

  return { destroy }
}
