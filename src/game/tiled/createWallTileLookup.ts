import type { ParsedTiledMap } from './parseTiledMap'

const toTileKey = (x: number, y: number): string => `${x},${y}`

export const createWallTileLookup = (map: ParsedTiledMap): Set<string> => {
  const wallTiles = new Set<string>()
  const collisionLayer = map.layers.find(
    (layer) => layer.name.toLowerCase() === 'object'
  )

  if (!collisionLayer) {
    return wallTiles
  }

  for (const tile of collisionLayer.tiles) {
    wallTiles.add(toTileKey(tile.x, tile.y))
  }

  return wallTiles
}

export const isWallTileAt = (wallTiles: Set<string>, x: number, y: number): boolean =>
  wallTiles.has(toTileKey(x, y))
