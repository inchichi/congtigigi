import { describe, expect, it } from 'vitest'

import { createInitialGameState } from './createInitialGameState'

describe('createInitialGameState', () => {
  it('returns the default startup state for the prototype', () => {
    expect(createInitialGameState()).toEqual({
      mapId: 'sample-map',
      playerSpawnTile: {
        x: 4,
        y: 6
      },
      scriptRuntime: 'lua'
    })
  })
})
