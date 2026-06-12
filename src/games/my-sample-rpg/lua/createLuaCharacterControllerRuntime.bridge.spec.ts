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
const createControllerModuleSource = (methodsSource: string): string => `
local controller = {}

${methodsSource}

return controller
`

afterEach(() => {
  vi.restoreAllMocks()
})

describe('createLuaCharacterControllerRuntime bridge', () => {
  it('round-trips movement and interaction results through the real wasm bridge', async () => {
    const runtime = await createBridgeRuntime({
      source: createControllerModuleSource(`
local controllers = {}

function controller.register(id, home_x, home_y, radius)
  controllers[id] = {
    greeting = "Hello",
    radius = radius
  }
end

function controller.unregister(id)
  controllers[id] = nil
end

function controller.step(id, dt, x, y)
  if controllers[id] == nil then
    error("missing controller state")
  end

  return 0.5, -0.25
end

function controller.interact(id, source_id)
  if controllers[id] == nil then
    error("missing controller state")
  end

  return controllers[id].greeting .. ", " .. source_id .. "!", 1.75
end
`)
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
      source: createControllerModuleSource(`
local controllers = {}

function controller.register(id, home_x, home_y, radius)
  controllers[id] = true
end

function controller.unregister(id)
  controllers[id] = nil
end

function controller.step(id, dt, x, y)
  return 0.25, 0
end

function controller.interact(id, source_id)
  return "Old reply", 1.0
end
`)
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
        source: createControllerModuleSource(`
local controllers = {}

function controller.register(id, home_x, home_y, radius)
  controllers[id] = true
end

function controller.unregister(id)
  controllers[id] = nil
end

function controller.step(id, dt, x, y)
  return -1, 0.5
end

function controller.interact(id, source_id)
  return "New reply", 2.5
end
`)
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
      source: createControllerModuleSource(`
local controllers = {}

function controller.register(id, home_x, home_y, radius)
  controllers[id] = true
end

function controller.unregister(id)
  controllers[id] = nil
end

function controller.step(id, dt, x, y)
  engine.ui.show_message("step:" .. id, 1.25)
  return 0, 0
end

function controller.interact(id, source_id)
  engine.ui.show_message("Hello, " .. source_id .. "!", 2.25)
end
`)
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

  it('exposes TMX-backed controller config through engine.self', async () => {
    const runtime = await createBridgeRuntime({
      source: createControllerModuleSource(`
local controllers = {}

function controller.register(id, home_x, home_y, radius)
  local config = engine.self.get_controller_config()

  controllers[id] = {
    first_line = config.dialogueLines[1],
    second_line = config.dialogueLines[2],
    first_delay = config.patrolDelaysSeconds[1],
    second_delay = config.patrolDelaysSeconds[2],
    first_flag = config.patrolFlags[1],
    second_flag = config.patrolFlags[2],
    duration = config.messageDurationSeconds,
    role = config.role
  }
end

function controller.step(id, dt, x, y)
  return 0, 0
end

function controller.interact(id, source_id)
  local controller_state = controllers[id]

  return controller_state.first_line
    .. " / "
    .. controller_state.second_line
    .. " / "
    .. tostring(controller_state.first_delay)
    .. " / "
    .. tostring(controller_state.second_delay)
    .. " / "
    .. tostring(controller_state.first_flag)
    .. " / "
    .. tostring(controller_state.second_flag)
    .. " / "
    .. controller_state.role,
    controller_state.duration
end
`)
    })
    const character = createBridgeCharacter({
      controller: createLuaCharacterController({
        scriptId: BRIDGE_SCRIPT_ID,
        radiusInTiles: 2,
        moveSpeedTilesPerSecond: 1.5,
        config: {
          dialogueLines: ['Need any tools?', 'Best steel in town.'],
          patrolDelaysSeconds: [0.5, 1.25],
          patrolFlags: [true, false],
          messageDurationSeconds: 2.5,
          role: 'blacksmith'
        }
      })
    })
    const player = createInitialPlayerCharacter({
      mapWidth: 20,
      mapHeight: 20
    })

    try {
      runtime.attachCharacter(character, character.controller)

      expect(
        runtime.handleInteraction(character, character.controller, player)
      ).toEqual({
        message:
          'Need any tools? / Best steel in town. / 0.5 / 1.25 / true / false / blacksmith',
        durationMilliseconds: 2500
      })
    } finally {
      runtime.destroy()
    }
  })

  it('contains register, step, and interact Lua errors inside the bridge', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const runtime = await createBridgeRuntime({
      source: createControllerModuleSource(`
function controller.register(id, home_x, home_y, radius)
  error("register broke")
end

function controller.unregister(id)
  error("unregister broke")
end

function controller.step(id, dt, x, y)
  error("step broke")
end

function controller.interact(id, source_id)
  error("interact broke")
end
`)
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
        `Lua controller error [${BRIDGE_SCRIPT_ID}:attach:${character.id}]`
      ),
      expect.any(Error)
    )
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        `Lua controller error [${BRIDGE_SCRIPT_ID}:step:${character.id}]`
      ),
      expect.any(Error)
    )
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        `Lua controller error [${BRIDGE_SCRIPT_ID}:interact:${character.id}]`
      ),
      expect.any(Error)
    )
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(
        `Lua controller error [${BRIDGE_SCRIPT_ID}:detach:${character.id}]`
      ),
      expect.any(Error)
    )
  })

  it('rejects invalid controller modules before they become active', async () => {
    await expect(
      createBridgeRuntime({
        source: createControllerModuleSource(`
function controller.register(id, home_x, home_y, radius)
end
`)
      })
    ).rejects.toThrow(
      `Lua controller validation failed for "${BRIDGE_SCRIPT_ID}": missing required method controller.step(id, delta_seconds, x, y)`
    )
  })

  it('keeps the previous script when a hot update fails validation', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const runtime = await createBridgeRuntime({
      source: createControllerModuleSource(`
function controller.step(id, dt, x, y)
  return 0.25, 0
end
`)
    })
    const character = createBridgeCharacter()

    try {
      runtime.attachCharacter(character, character.controller)

      expect(
        runtime.getMovementDelta(character, character.controller, 250)
      ).toEqual({
        x: 0.25,
        y: 0
      })

      runtime.updateScript(BRIDGE_SCRIPT_ID, {
        source: createControllerModuleSource(`
function controller.register(id, home_x, home_y, radius)
end
`)
      })

      expect(runtime.getActiveErrorMessages()).toHaveLength(1)
      expect(runtime.getActiveErrorMessages()[0]).toContain(
        'Lua controller validation failed'
      )
      expect(
        runtime.getMovementDelta(character, character.controller, 250)
      ).toEqual({
        x: 0.25,
        y: 0
      })
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining(`Lua controller error [${BRIDGE_SCRIPT_ID}:reload]`),
        expect.objectContaining({
          message: expect.stringContaining(
            `Lua controller validation failed for "${BRIDGE_SCRIPT_ID}"`
          )
        })
      )

      runtime.updateScript(BRIDGE_SCRIPT_ID, {
        source: createControllerModuleSource(`
function controller.step(id, dt, x, y)
  return 0.5, 0
end
`)
      })

      expect(runtime.getActiveErrorMessages()).toEqual([])
    } finally {
      runtime.destroy()
    }
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
        source
      }
    },
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: {
      wasmBinary
    }
  })
}

const createBridgeCharacter = (
  overrides: Partial<ReturnType<typeof createNpcCharacter>> & {
    controller?: ReturnType<typeof createLuaCharacterController>
  } = {}
) => ({
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
  ...overrides,
  controller:
    overrides.controller ??
    createLuaCharacterController({
      scriptId: BRIDGE_SCRIPT_ID,
      radiusInTiles: 2,
      moveSpeedTilesPerSecond: 1.5
    })
})
