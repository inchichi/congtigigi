import { describe, expect, it, vi } from 'vitest'

import { createInitialPlayerCharacter, createNpcCharacter } from '../characterState'
import { processInteractionEvents } from './processInteractionEvents'

describe('processInteractionEvents', () => {
  it('turns an interaction request into a message event from the target', () => {
    const player = {
      ...createInitialPlayerCharacter({
        mapWidth: 20,
        mapHeight: 20
      }),
      position: {
        x: 5,
        y: 5
      },
      facing: 'right' as const
    }
    const blacksmith = createNpcCharacter({
      id: 'blacksmith',
      appearanceType: 'character_bearded_apron_man',
      position: {
        x: 6,
        y: 5
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })
    const interactionLockUntilBySourceCharacterId = new Map<string, number>()

    expect(
      processInteractionEvents({
        events: [
          {
            kind: 'interaction-requested',
            sourceCharacterId: player.id
          }
        ],
        characters: [player, blacksmith],
        controllerRuntime: {
          canReceiveInteraction: (character) => character.id === blacksmith.id,
          drainEvents: vi.fn(() => []),
          handleInteraction: vi.fn(() => ({
            kind: 'message' as const,
            message: 'Need any tools?',
            durationMilliseconds: 1200
          }))
        },
        now: 100,
        interactionLockUntilBySourceCharacterId
      })
    ).toEqual([
      {
        kind: 'show-character-message',
        characterId: 'blacksmith',
        message: 'Need any tools?',
        durationMilliseconds: 1200
      }
    ])
    expect(interactionLockUntilBySourceCharacterId.get(player.id)).toBe(1300)
  })

  it('ignores repeated interaction requests while the source is locked', () => {
    const player = {
      ...createInitialPlayerCharacter({
        mapWidth: 20,
        mapHeight: 20
      }),
      position: {
        x: 5,
        y: 5
      },
      facing: 'right' as const
    }
    const blacksmith = createNpcCharacter({
      id: 'blacksmith',
      appearanceType: 'character_bearded_apron_man',
      position: {
        x: 6,
        y: 5
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })

    expect(
      processInteractionEvents({
        events: [
          {
            kind: 'interaction-requested',
            sourceCharacterId: player.id
          }
        ],
        characters: [player, blacksmith],
        controllerRuntime: {
          canReceiveInteraction: () => true,
          drainEvents: vi.fn(() => []),
          handleInteraction: vi.fn(() => ({
            kind: 'message' as const,
            message: 'Need any tools?',
            durationMilliseconds: 1200
          }))
        },
        now: 500,
        interactionLockUntilBySourceCharacterId: new Map([[player.id, 1000]])
      })
    ).toEqual([])
  })

  it('passes through queued message events and prefers runtime-emitted interaction messages', () => {
    const player = {
      ...createInitialPlayerCharacter({
        mapWidth: 20,
        mapHeight: 20
      }),
      position: {
        x: 5,
        y: 5
      },
      facing: 'right' as const
    }
    const blacksmith = createNpcCharacter({
      id: 'blacksmith',
      appearanceType: 'character_bearded_apron_man',
      position: {
        x: 6,
        y: 5
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })
    const interactionLockUntilBySourceCharacterId = new Map<string, number>()

    expect(
      processInteractionEvents({
        events: [
          {
            kind: 'show-character-message',
            characterId: 'sign',
            message: 'Town square',
            durationMilliseconds: 900
          },
          {
            kind: 'interaction-requested',
            sourceCharacterId: player.id
          }
        ],
        characters: [player, blacksmith],
        controllerRuntime: {
          canReceiveInteraction: () => true,
          drainEvents: vi.fn(() => [
            {
              kind: 'show-character-message' as const,
              characterId: blacksmith.id,
              message: 'Crafting takes time.',
              durationMilliseconds: 1600
            }
          ]),
          handleInteraction: vi.fn(() => ({
            kind: 'message' as const,
            message: 'Legacy reply',
            durationMilliseconds: 1200
          }))
        },
        now: 250,
        interactionLockUntilBySourceCharacterId
      })
    ).toEqual([
      {
        kind: 'show-character-message',
        characterId: 'sign',
        message: 'Town square',
        durationMilliseconds: 900
      },
      {
        kind: 'show-character-message',
        characterId: blacksmith.id,
        message: 'Crafting takes time.',
        durationMilliseconds: 1600
      }
    ])
    expect(interactionLockUntilBySourceCharacterId.get(player.id)).toBe(1850)
  })
})
