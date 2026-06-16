import { describe, expect, it, vi } from 'vitest'

import {
  createIdleNpcCharacterController,
  createInitialPlayerCharacter,
  createLuaCharacterController,
  createNpcCharacter
} from './characterState'
import { createCharacterControllerRuntime } from './createCharacterControllerRuntime'
import type { LuaCharacterControllerRuntime } from './lua/createLuaCharacterControllerRuntime'

describe('createCharacterControllerRuntime', () => {
  it('resolves keyboard movement through the shared runtime', () => {
    const runtime = createCharacterControllerRuntime()

    expect(
      runtime.getIntent({
        character: createInitialPlayerCharacter({
          mapWidth: 32,
          mapHeight: 20
        }),
        deltaMilliseconds: 250,
        pressedDirections: new Set(['right'])
      })
    ).toEqual({
      movement: {
        x: 2,
        y: 0
      }
    })
  })

  it('calls lua attach and detach hooks when a controller is added or removed', () => {
    const luaRuntime = createLuaRuntimeStub()
    const runtime = createCharacterControllerRuntime({
      luaControllerRuntime: luaRuntime
    })
    const luaCharacter = {
      ...createNpcCharacter({
        id: 'villager_1',
        appearanceType: 'character_commoner_tan_tunic',
        position: {
          x: 10,
          y: 12
        },
        collisionSize: {
          width: 1,
          height: 1
        }
      }),
      controller: createLuaCharacterController({
        scriptId: 'wander-near-home',
        radiusInTiles: 2,
        moveSpeedTilesPerSecond: 1.5
      })
    }
    const idleCharacter = {
      ...luaCharacter,
      controller: createIdleNpcCharacterController()
    }

    runtime.syncCharacters([luaCharacter])
    runtime.syncCharacters([idleCharacter])

    expect(luaRuntime.attachCharacter).toHaveBeenCalledWith(
      luaCharacter,
      luaCharacter.controller
    )
    expect(luaRuntime.detachCharacter).toHaveBeenCalledWith(
      luaCharacter,
      luaCharacter.controller
    )
  })

  it('forwards lua movement, interactions, and script updates to the lua runtime', () => {
    const luaRuntime = createLuaRuntimeStub({
      movementDelta: {
        x: -0.5,
        y: 0.25
      },
      interactionResponse: {
        message: 'Hello there.',
        durationMilliseconds: 1800
      },
      emittedEvents: [
        {
          kind: 'show-character-message',
          characterId: 'villager_1',
          message: 'Queued event',
          durationMilliseconds: 900
        }
      ]
    })
    const runtime = createCharacterControllerRuntime({
      luaControllerRuntime: luaRuntime
    })
    const luaCharacter = {
      ...createNpcCharacter({
        id: 'villager_1',
        appearanceType: 'character_commoner_tan_tunic',
        position: {
          x: 10,
          y: 12
        },
        collisionSize: {
          width: 1,
          height: 1
        }
      }),
      controller: createLuaCharacterController({
        scriptId: 'wander-near-home',
        radiusInTiles: 2,
        moveSpeedTilesPerSecond: 1.5
      })
    }

    runtime.syncCharacters([luaCharacter])

    expect(
      runtime.getIntent({
        character: luaCharacter,
        deltaMilliseconds: 250
      })
    ).toEqual({
      movement: {
        x: -0.1875,
        y: 0.09375
      }
    })
    expect(runtime.canReceiveInteraction(luaCharacter)).toBe(true)
    expect(
      runtime.handleInteraction({
        targetCharacter: luaCharacter,
        sourceCharacter: createInitialPlayerCharacter({
          mapWidth: 32,
          mapHeight: 20
        })
      })
    ).toEqual({
      kind: 'message',
      message: 'Hello there.',
      durationMilliseconds: 1800
    })
    expect(runtime.drainEvents()).toEqual([
      {
        kind: 'show-character-message',
        characterId: 'villager_1',
        message: 'Queued event',
        durationMilliseconds: 900
      }
    ])
    expect(runtime.getRuntimeWarnings()).toEqual(['Lua warning'])

    runtime.updateLuaControllerScript('wander-near-home', {
      source: 'return 1'
    })

    expect(luaRuntime.updateScript).toHaveBeenCalledWith('wander-near-home', {
      source: 'return 1'
    })
  })

  it('lets monster npc characters receive interaction and return an alert', () => {
    const runtime = createCharacterControllerRuntime()
    const monster = createNpcCharacter({
      id: '꿀꿀이',
      appearanceType: 'monster_pig',
      position: {
        x: 10,
        y: 8
      },
      collisionSize: {
        width: 1,
        height: 1
      }
    })

    expect(runtime.canReceiveInteraction(monster)).toBe(true)
    expect(
      runtime.handleInteraction({
        targetCharacter: monster,
        sourceCharacter: createInitialPlayerCharacter({
          mapWidth: 32,
          mapHeight: 20
        })
      })
    ).toEqual({
      kind: 'message',
      message: '!',
      durationMilliseconds: 600
    })
  })

  it('ignores defeated monster corpses for interaction', () => {
    const runtime = createCharacterControllerRuntime()
    const monster = createNpcCharacter({
      id: '꿀꿀이',
      appearanceType: 'monster_pig',
      position: {
        x: 10,
        y: 8
      },
      collisionSize: {
        width: 1,
        height: 1
      },
      blocksMovement: false
    })

    expect(runtime.canReceiveInteraction(monster)).toBe(false)
    expect(
      runtime.handleInteraction({
        targetCharacter: monster,
        sourceCharacter: createInitialPlayerCharacter({
          mapWidth: 32,
          mapHeight: 20
        })
      })
    ).toBeUndefined()
  })
})

const createLuaRuntimeStub = ({
  movementDelta = undefined,
  interactionResponse = undefined,
  emittedEvents = [],
  activeErrorMessages = ['Lua warning']
}: {
  movementDelta?: { x: number; y: number } | undefined
  interactionResponse?:
    | { message: string; durationMilliseconds: number }
    | undefined
  emittedEvents?: Array<{
    kind: 'show-character-message'
    characterId: string
    message: string
    durationMilliseconds: number
  }>
  activeErrorMessages?: string[]
} = {}): LuaCharacterControllerRuntime => ({
  attachCharacter: vi.fn(),
  detachCharacter: vi.fn(),
  getMovementDelta: vi.fn(() => movementDelta),
  canReceiveInteraction: vi.fn(() => interactionResponse !== undefined),
  handleInteraction: vi.fn(() => interactionResponse),
  drainEvents: vi.fn(() => emittedEvents),
  getActiveErrorMessages: vi.fn(() => activeErrorMessages),
  updateScript: vi.fn(),
  pushSnapshot: vi.fn(),
  destroy: vi.fn()
})
