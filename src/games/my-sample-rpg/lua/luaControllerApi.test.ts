import { describe, expect, it } from 'vitest'

import {
  createLuaCharacterController,
  createNpcCharacter
} from '../characterState'
import {
  DEFAULT_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS,
  LUA_CONTROLLER_INTERACT_METHOD_NAME,
  LUA_CONTROLLER_PUBLIC_API_NAME,
  LUA_CONTROLLER_REGISTER_METHOD_NAME,
  LUA_CONTROLLER_STEP_METHOD_NAME,
  LUA_CONTROLLER_UNREGISTER_METHOD_NAME,
  MIN_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS,
  createLuaControllerInteractInput,
  createLuaControllerInteractionResponse,
  createLuaControllerMoveIntent,
  createLuaControllerRegisterInput,
  createLuaControllerStepInput,
  createLuaControllerUnregisterInput,
  getLuaControllerMessageDurationMilliseconds,
  getLuaControllerInteractFunctionArguments,
  getLuaControllerRegisterFunctionArguments,
  getLuaControllerStepFunctionArguments,
  getLuaControllerUnregisterFunctionArguments
} from './luaControllerApi'

describe('luaControllerApi', () => {
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
      config: {
        dialogueLines: ['Hello there.']
      },
      moveSpeedTilesPerSecond: 1.5
    })
  }

  it('creates the register input from the current character snapshot', () => {
    expect(
      createLuaControllerRegisterInput(
        luaCharacter,
        luaCharacter.controller
      )
    ).toEqual({
      character: {
        id: 'villager_1',
        position: {
          x: 10,
          y: 12
        }
      },
      radiusInTiles: 2
    })
    expect(luaCharacter.controller.config).toEqual({
      dialogueLines: ['Hello there.']
    })
  })

  it('creates the step input from the current character snapshot', () => {
    expect(createLuaControllerStepInput(luaCharacter, 250)).toEqual({
      character: {
        id: 'villager_1',
        position: {
          x: 10,
          y: 12
        }
      },
      deltaSeconds: 0.25
    })
  })

  it('serializes lifecycle inputs into the Lua function arguments', () => {
    expect(
      getLuaControllerRegisterFunctionArguments({
        character: {
          id: 'villager_1',
          position: {
            x: 10,
            y: 12
          }
        },
        radiusInTiles: 2
      })
    ).toEqual(['villager_1', 10, 12, 2])
    expect(
      getLuaControllerStepFunctionArguments({
        character: {
          id: 'villager_1',
          position: {
            x: 9.5,
            y: 13
          }
        },
        deltaSeconds: 0.125
      })
    ).toEqual(['villager_1', 0.125, 9.5, 13])
    expect(
      getLuaControllerInteractFunctionArguments(
        createLuaControllerInteractInput(
          luaCharacter,
          createNpcCharacter({
            id: 'player',
            appearanceType: 'character_adventurer_brown_hair',
            position: {
              x: 8,
              y: 12
            },
            collisionSize: {
              width: 1,
              height: 1
            }
          })
        )
      )
    ).toEqual(['villager_1', 'player'])
    expect(
      getLuaControllerUnregisterFunctionArguments(
        createLuaControllerUnregisterInput(luaCharacter)
      )
    ).toEqual(['villager_1'])
  })

  it('treats a zero vector as no movement intent', () => {
    expect(createLuaControllerMoveIntent(0, 0)).toBeUndefined()
    expect(createLuaControllerMoveIntent(1, -0.5)).toEqual({
      moveX: 1,
      moveY: -0.5
    })
  })

  it('creates a message interaction response with a safe default duration', () => {
    expect(
      createLuaControllerInteractionResponse('Hello there.', undefined)
    ).toEqual({
      message: 'Hello there.',
      durationMilliseconds:
        DEFAULT_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS
    })
    expect(
      createLuaControllerInteractionResponse('Hello there.', 1.75)
    ).toEqual({
      message: 'Hello there.',
      durationMilliseconds: 1750
    })
    expect(getLuaControllerMessageDurationMilliseconds(0.1)).toBe(
      MIN_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS
    )
    expect(createLuaControllerInteractionResponse('', 1)).toBeUndefined()
  })

  it('keeps the public Lua api under the engine namespace', () => {
    expect(LUA_CONTROLLER_PUBLIC_API_NAME).toBe('engine')
  })

  it('keeps reserved controller method names stable', () => {
    expect(LUA_CONTROLLER_REGISTER_METHOD_NAME).toBe('register')
    expect(LUA_CONTROLLER_UNREGISTER_METHOD_NAME).toBe('unregister')
    expect(LUA_CONTROLLER_STEP_METHOD_NAME).toBe('step')
    expect(LUA_CONTROLLER_INTERACT_METHOD_NAME).toBe('interact')
  })
})
