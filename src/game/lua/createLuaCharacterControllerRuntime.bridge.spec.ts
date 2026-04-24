import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  createInitialPlayerCharacter,
  createLuaCharacterController,
  createNpcCharacter
} from '../characterState'
import { createLuaCharacterControllerRuntime } from './createLuaCharacterControllerRuntime'

const LUA_MODULE_JS_URL = new URL(
  '../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)
const BRIDGE_SCRIPT_ID = 'bridge-test'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('createLuaCharacterControllerRuntime bridge', () => {
  it('round-trips movement and interaction results through the real wasm bridge', async () => {
    const runtime = await createBridgeRuntime({
      source: `
local controllers = {}

function register_bridge_controller(id, home_x, home_y, radius)
  controllers[id] = {
    greeting = "Hello",
    radius = radius
  }
end

function unregister_bridge_controller(id)
  controllers[id] = nil
end

function step_bridge_controller(id, dt, x, y)
  if controllers[id] == nil then
    error("missing controller state")
  end

  return 0.5, -0.25
end

function interact_bridge_controller(id, source_id)
  if controllers[id] == nil then
    error("missing controller state")
  end

  return controllers[id].greeting .. ", " .. source_id .. "!", 1.75
end
`
    })
    const character = createBridgeCharacter()
    const player = createInitialPlayerCharacter({
      mapWidth: 20,
      mapHeight: 20
    })

    try {
      runtime.attachCharacter(character, character.controller)

      expect(
        runtime.getMovementDelta(character, character.controller, 250)
      ).toEqual({
        x: 0.5,
        y: -0.25
      })
      expect(
        runtime.handleInteraction(character, character.controller, player)
      ).toEqual({
        message: 'Hello, player!',
        durationMilliseconds: 1750
      })
    } finally {
      runtime.destroy()
    }
  })

  it('reloads attached character scripts through updateScript', async () => {
    const runtime = await createBridgeRuntime({
      source: `
local controllers = {}

function register_bridge_controller(id, home_x, home_y, radius)
  controllers[id] = true
end

function unregister_bridge_controller(id)
  controllers[id] = nil
end

function step_bridge_controller(id, dt, x, y)
  return 0.25, 0
end

function interact_bridge_controller(id, source_id)
  return "Old reply", 1.0
end
`
    })
    const character = createBridgeCharacter()
    const player = createInitialPlayerCharacter({
      mapWidth: 20,
      mapHeight: 20
    })

    try {
      runtime.attachCharacter(character, character.controller)

      expect(
        runtime.handleInteraction(character, character.controller, player)
      ).toEqual({
        message: 'Old reply',
        durationMilliseconds: 1000
      })

      runtime.updateScript(BRIDGE_SCRIPT_ID, {
        registerFunctionName: 'register_bridge_controller',
        unregisterFunctionName: 'unregister_bridge_controller',
        stepFunctionName: 'step_bridge_controller',
        interactFunctionName: 'interact_bridge_controller',
        source: `
local controllers = {}

function register_bridge_controller(id, home_x, home_y, radius)
  controllers[id] = true
end

function unregister_bridge_controller(id)
  controllers[id] = nil
end

function step_bridge_controller(id, dt, x, y)
  return -1, 0.5
end

function interact_bridge_controller(id, source_id)
  return "New reply", 2.5
end
`
      })

      expect(
        runtime.getMovementDelta(character, character.controller, 250)
      ).toEqual({
        x: -1,
        y: 0.5
      })
      expect(
        runtime.handleInteraction(character, character.controller, player)
      ).toEqual({
        message: 'New reply',
        durationMilliseconds: 2500
      })
    } finally {
      runtime.destroy()
    }
  })

  it('drains events emitted through the public engine api', async () => {
    const runtime = await createBridgeRuntime({
      source: `
local controllers = {}

function register_bridge_controller(id, home_x, home_y, radius)
  controllers[id] = true
end

function unregister_bridge_controller(id)
  controllers[id] = nil
end

function step_bridge_controller(id, dt, x, y)
  engine.ui.show_message("step:" .. id, 1.25)
  return 0, 0
end

function interact_bridge_controller(id, source_id)
  engine.ui.show_message("Hello, " .. source_id .. "!", 2.25)
end
`
    })
    const character = createBridgeCharacter()
    const player = createInitialPlayerCharacter({
      mapWidth: 20,
      mapHeight: 20
    })

    try {
      runtime.attachCharacter(character, character.controller)

      expect(
        runtime.getMovementDelta(character, character.controller, 250)
      ).toBeUndefined()
      expect(runtime.drainEvents()).toEqual([
        {
          kind: 'show-character-message',
          characterId: character.id,
          message: `step:${character.id}`,
          durationMilliseconds: 1250
        }
      ])

      expect(
        runtime.handleInteraction(character, character.controller, player)
      ).toBeUndefined()
      expect(runtime.drainEvents()).toEqual([
        {
          kind: 'show-character-message',
          characterId: character.id,
          message: 'Hello, player!',
          durationMilliseconds: 2250
        }
      ])
    } finally {
      runtime.destroy()
    }
  })

  it('contains register, step, and interact Lua errors inside the bridge', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const runtime = await createBridgeRuntime({
      source: `
function register_bridge_controller(id, home_x, home_y, radius)
  error("register broke")
end

function unregister_bridge_controller(id)
  error("unregister broke")
end

function step_bridge_controller(id, dt, x, y)
  error("step broke")
end

function interact_bridge_controller(id, source_id)
  error("interact broke")
end
`
    })
    const character = createBridgeCharacter()
    const player = createInitialPlayerCharacter({
      mapWidth: 20,
      mapHeight: 20
    })

    expect(() => runtime.attachCharacter(character, character.controller)).not.toThrow()
    expect(
      runtime.getMovementDelta(character, character.controller, 250)
    ).toBeUndefined()
    expect(
      runtime.handleInteraction(character, character.controller, player)
    ).toBeUndefined()
    expect(() => runtime.detachCharacter(character, character.controller)).not.toThrow()
    expect(() => runtime.destroy()).not.toThrow()

    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        `Lua runtime error [${BRIDGE_SCRIPT_ID}:attach:${character.id}]`
      ),
      expect.any(Error)
    )
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        `Lua runtime error [${BRIDGE_SCRIPT_ID}:step:${character.id}]`
      ),
      expect.any(Error)
    )
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        `Lua runtime error [${BRIDGE_SCRIPT_ID}:interact:${character.id}]`
      ),
      expect.any(Error)
    )
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        `Lua runtime error [${BRIDGE_SCRIPT_ID}:detach:${character.id}]`
      ),
      expect.any(Error)
    )
  })
})

const createBridgeRuntime = async ({ source }: { source: string }) => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createLuaCharacterControllerRuntime({
    scriptsById: {
      [BRIDGE_SCRIPT_ID]: {
        registerFunctionName: 'register_bridge_controller',
        unregisterFunctionName: 'unregister_bridge_controller',
        stepFunctionName: 'step_bridge_controller',
        interactFunctionName: 'interact_bridge_controller',
        source
      }
    },
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: {
      wasmBinary
    }
  })
}

const createBridgeCharacter = () => ({
  ...createNpcCharacter({
    id: 'bridge-npc',
    appearanceType: 'character_villager_brown_tunic',
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
    scriptId: BRIDGE_SCRIPT_ID,
    radiusInTiles: 2,
    moveSpeedTilesPerSecond: 1.5
  })
})
