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
    const interactionLockUntilByCharacterPair = new Map<string, number>()

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
        interactionLockUntilByCharacterPair
      })
    ).toEqual([
      {
        kind: 'show-character-message',
        characterId: 'blacksmith',
        message: 'Need any tools?',
        durationMilliseconds: 1200
      }
    ])
    expect(interactionLockUntilByCharacterPair.get(`${player.id}:blacksmith`)).toBe(
      1300
    )
  })

  it('ignores repeated interaction requests while the same source-target pair is locked', () => {
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
        interactionLockUntilByCharacterPair: new Map([
          [`${player.id}:blacksmith`, 1000]
        ])
      })
    ).toEqual([])
  })

  it('allows interactions with a different target while another target is locked', () => {
    const player = {
      ...createInitialPlayerCharacter({
        mapWidth: 20,
        mapHeight: 20
      }),
      position: {
        x: 5,
        y: 5
      },
      facing: 'down' as const
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
    const villager = createNpcCharacter({
      id: 'villager',
      appearanceType: 'character_blue_hood',
      position: {
        x: 5,
        y: 6
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
        characters: [player, blacksmith, villager],
        controllerRuntime: {
          canReceiveInteraction: () => true,
          drainEvents: vi.fn(() => []),
          handleInteraction: vi.fn(() => ({
            kind: 'message' as const,
            message: 'Hello there.',
            durationMilliseconds: 900
          }))
        },
        now: 500,
        interactionLockUntilByCharacterPair: new Map([
          [`${player.id}:blacksmith`, 1000]
        ])
      })
    ).toEqual([
      {
        kind: 'show-character-message',
        characterId: 'villager',
        message: 'Hello there.',
        durationMilliseconds: 900
      }
    ])
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
    const interactionLockUntilByCharacterPair = new Map<string, number>()

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
        interactionLockUntilByCharacterPair
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
    expect(interactionLockUntilByCharacterPair.get(`${player.id}:blacksmith`)).toBe(
      1850
    )
  })
})
