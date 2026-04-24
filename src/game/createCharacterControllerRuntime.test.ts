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
      runtime.getMovementDelta({
        character: createInitialPlayerCharacter({
          mapWidth: 32,
          mapHeight: 20
        }),
        deltaMilliseconds: 250,
        pressedDirections: new Set(['right'])
      })
    ).toEqual({
      x: 2,
      y: 0
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

  it('forwards lua movement and script updates to the lua runtime', () => {
    const luaRuntime = createLuaRuntimeStub({
      movementDelta: {
        x: -0.5,
        y: 0.25
      }
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
      runtime.getMovementDelta({
        character: luaCharacter,
        deltaMilliseconds: 250
      })
    ).toEqual({
      x: -0.5,
      y: 0.25
    })

    runtime.updateLuaControllerScript('wander-near-home', {
      registerFunctionName: 'register_wander_controller',
      unregisterFunctionName: 'unregister_wander_controller',
      stepFunctionName: 'step_wander_controller',
      source: 'return 1'
    })

    expect(luaRuntime.updateScript).toHaveBeenCalledWith('wander-near-home', {
      registerFunctionName: 'register_wander_controller',
      unregisterFunctionName: 'unregister_wander_controller',
      stepFunctionName: 'step_wander_controller',
      source: 'return 1'
    })
  })
})

const createLuaRuntimeStub = ({
  movementDelta = undefined
}: {
  movementDelta?: { x: number; y: number } | undefined
} = {}): LuaCharacterControllerRuntime => ({
  attachCharacter: vi.fn(),
  detachCharacter: vi.fn(),
  getMovementDelta: vi.fn(() => movementDelta),
  updateScript: vi.fn(),
  destroy: vi.fn()
})
