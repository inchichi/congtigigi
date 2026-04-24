import { describe, expect, it } from 'vitest'

import { createGameEventQueue } from './createGameEventQueue'

describe('createGameEventQueue', () => {
  it('drains queued events in insertion order', () => {
    const queue = createGameEventQueue()

    queue.enqueue({
      kind: 'interaction-requested',
      sourceCharacterId: 'player'
    })
    queue.enqueue({
      kind: 'show-character-message',
      characterId: 'blacksmith',
      message: 'Hello there.',
      durationMilliseconds: 1500
    })

    expect(queue.drain()).toEqual([
      {
        kind: 'interaction-requested',
        sourceCharacterId: 'player'
      },
      {
        kind: 'show-character-message',
        characterId: 'blacksmith',
        message: 'Hello there.',
        durationMilliseconds: 1500
      }
    ])
    expect(queue.drain()).toEqual([])
  })
})
