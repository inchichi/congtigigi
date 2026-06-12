import { Rectangle, Texture } from 'pixi.js'

import type {
  ParsedTiledTile,
  ParsedTiledTileset
} from '../tiled/parseTiledMap'

// 타일셋 PNG에서 개별 타일 텍스처를 잘라내는 데 필요한 최소 정보. ParsedTiledTileset이
// 이 형태를 만족하므로 그대로 넘길 수 있고, 무기 스프라이트처럼 타일셋이 아닌 프레임에도 쓴다.
export type TileTextureFrameSource = {
  columns: number
  margin: number
  spacing: number
  tileWidth: number
  tileHeight: number
}

// 타일셋 이미지 한 장을 로드한 뒤, 각 localId에 해당하는 프레임 텍스처들을 함께 들고 다닌다.
export type TilesetRenderResources = {
  imageTexture: Texture
  tileTextures: Texture[]
}

// 타일셋 이미지(아틀라스)에서 localId 칸 하나를 Rectangle 프레임으로 잘라 텍스처로 만든다.
export const createTileTexture = (
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
    frame: new Rectangle(frameX, frameY, tileset.tileWidth, tileset.tileHeight),
    orig: new Rectangle(0, 0, tileset.tileWidth, tileset.tileHeight)
  })
}

// gid로 어느 타일셋에 속하는지 찾는다(firstGid가 작거나 같은 것 중 가장 큰 것).
export const resolveTilesetForTile = (
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
