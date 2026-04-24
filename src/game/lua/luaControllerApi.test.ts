import { describe, expect, it } from 'vitest'

import {
  createLuaCharacterController,
  createNpcCharacter
} from '../characterState'
import {
  createLuaControllerMoveIntent,
  createLuaControllerRegisterInput,
  createLuaControllerStepInput,
  createLuaControllerUnregisterInput,
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
})
