export type GameState = {
  mapId: string
  playerSpawnTile: {
    x: number
    y: number
  }
  scriptRuntime: 'lua'
}

export const createInitialGameState = (): GameState => ({
  mapId: 'sample-map',
  playerSpawnTile: {
    x: 4,
    y: 6
  },
  scriptRuntime: 'lua'
})
