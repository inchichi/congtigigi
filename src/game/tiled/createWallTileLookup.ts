import type { ParsedTiledMap, ParsedTiledTile, ParsedTiledTileset } from './parseTiledMap'

const toTileKey = (x: number, y: number): string => `${x},${y}`

export const createWallTileLookup = (map: ParsedTiledMap): Set<string> => {
  const wallTiles = new Set<string>()

  for (const layer of map.layers) {
    for (const tile of layer.tiles) {
      const tileset = resolveTilesetForTile(tile, map.tilesets)

      if (tileset.tileProperties[tile.localId]?.wall === true) {
        wallTiles.add(toTileKey(tile.x, tile.y))
      }
    }
  }

  return wallTiles
}

export const isWallTileAt = (wallTiles: Set<string>, x: number, y: number): boolean =>
  wallTiles.has(toTileKey(x, y))

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
