import { groupD8 } from 'pixi.js'

import type { ParsedTiledTile } from '../tiled/parseTiledMap'

export type TiledSpriteTransform = {
  rotation: number
  scaleX: number
  scaleY: number
}

export const hasTileTransform = (tile: ParsedTiledTile): boolean =>
  tile.flipHorizontally || tile.flipVertically || tile.flipDiagonally

export const getSpriteTransformForTile = (
  tile: ParsedTiledTile
): TiledSpriteTransform => {
  const symmetry = groupD8.inv(getTiledTileSymmetry(tile))

  switch (symmetry) {
    case groupD8.S:
      return {
        rotation: Math.PI / 2,
        scaleX: 1,
        scaleY: 1
      }
    case groupD8.W:
      return {
        rotation: Math.PI,
        scaleX: 1,
        scaleY: 1
      }
    case groupD8.N:
      return {
        rotation: -Math.PI / 2,
        scaleX: 1,
        scaleY: 1
      }
    case groupD8.MIRROR_VERTICAL:
      return {
        rotation: 0,
        scaleX: 1,
        scaleY: -1
      }
    case groupD8.MAIN_DIAGONAL:
      return {
        rotation: Math.PI / 2,
        scaleX: 1,
        scaleY: -1
      }
    case groupD8.MIRROR_HORIZONTAL:
      return {
        rotation: 0,
        scaleX: -1,
        scaleY: 1
      }
    case groupD8.REVERSE_DIAGONAL:
      return {
        rotation: -Math.PI / 2,
        scaleX: 1,
        scaleY: -1
      }
    default:
      return {
        rotation: 0,
        scaleX: 1,
        scaleY: 1
      }
  }
}

const getTiledTileSymmetry = (tile: ParsedTiledTile): number => {
  if (tile.flipDiagonally) {
    if (tile.flipHorizontally && tile.flipVertically) {
      return groupD8.REVERSE_DIAGONAL
    }

    if (tile.flipHorizontally) {
      return groupD8.N
    }

    if (tile.flipVertically) {
      return groupD8.S
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
