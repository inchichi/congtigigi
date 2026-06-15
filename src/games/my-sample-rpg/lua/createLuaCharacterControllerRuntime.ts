import type {
  CharacterState,
  LuaCharacterController,
  LuaCharacterControllerConfig,
  LuaCharacterControllerConfigValue
} from '../characterState'
import {
  DEFAULT_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS,
  LUA_CONTROLLER_INTERACT_METHOD_NAME,
  LUA_CONTROLLER_REGISTER_METHOD_NAME,
  LUA_CONTROLLER_STEP_METHOD_NAME,
  LUA_CONTROLLER_UNREGISTER_METHOD_NAME,
  LUA_CONTROLLER_PUBLIC_API_NAME,
  MIN_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS,
  createLuaControllerInteractInput,
  createLuaControllerInteractionResponse,
  createLuaControllerMoveIntent,
  createLuaControllerRegisterInput,
  createLuaControllerStepInput,
  createLuaControllerUnregisterInput,
  getLuaControllerInteractFunctionArguments,
  getLuaControllerRegisterFunctionArguments,
  getLuaControllerStepFunctionArguments,
  getLuaControllerUnregisterFunctionArguments,
  type LuaControllerMethodName,
  type LuaControllerInteractionResponse,
  type LuaControllerRuntimeEvent
} from './luaControllerApi'

export type LuaControllerScriptSource = {
  source: string
}

type LuaModuleInitOptions = {
  locateFile?: (path: string) => string
  wasmBinary?: ArrayBuffer | Uint8Array
  print?: (value: string) => void
  printErr?: (value: string) => void
}

type CreateLuaCharacterControllerRuntimeInput = {
  scriptsById: Record<string, LuaControllerScriptSource>
  createLuaModuleFactory?: () => Promise<LuaModuleFactory>
  createLuaModuleOptions?: LuaModuleInitOptions
}

type LuaFunctionArgument = number | string

type LuaModule = {
  _luaL_newstate: () => number
  _luaL_openlibs: (luaState: number) => void
  _lua_close: (luaState: number) => void
  _lua_type: (luaState: number, index: number) => number
  _luaL_loadbufferx: (
    luaState: number,
    sourcePointer: number,
    sourceLength: number,
    chunkNamePointer: number,
    modePointer: number
  ) => number
  _lua_getglobal: (luaState: number, namePointer: number) => number
  _lua_pcallk: (
    luaState: number,
    argumentCount: number,
    resultCount: number,
    messageHandlerIndex: number,
    context: number,
    continuationPointer: number
  ) => number
  _lua_gettop: (luaState: number) => number
  _lua_settop: (luaState: number, index: number) => void
  _lua_pushnumber: (luaState: number, value: number) => void
  _lua_pushlstring: (
    luaState: number,
    valuePointer: number,
    valueLength: number
  ) => number
  _lua_tonumberx: (
    luaState: number,
    index: number,
    isNumberPointer: number
  ) => number
  _lua_tolstring: (
    luaState: number,
    index: number,
    lengthPointer: number
  ) => number
  _malloc: (size: number) => number
  _free: (pointer: number) => void
  lengthBytesUTF8: (value: string) => number
  stringToUTF8: (value: string, pointer: number, maxBytesToWrite: number) => void
  UTF8ToString: (pointer: number) => string
}

export type LuaCharacterControllerRuntime = {
  attachCharacter: (
    character: CharacterState,
    controller: LuaCharacterController
  ) => void
  detachCharacter: (
    character: CharacterState,
    controller: LuaCharacterController
  ) => void
  getMovementDelta: (
    character: CharacterState,
    controller: LuaCharacterController,
    deltaMilliseconds: number
  ) => { x: number; y: number } | undefined
  canReceiveInteraction: (
    character: CharacterState,
    controller: LuaCharacterController
  ) => boolean
  handleInteraction: (
    character: CharacterState,
    controller: LuaCharacterController,
    sourceCharacter: CharacterState
  ) => LuaControllerInteractionResponse | undefined
  drainEvents: () => LuaControllerRuntimeEvent[]
  getActiveErrorMessages: () => string[]
  updateScript: (scriptId: string, script: LuaControllerScriptSource) => void
  destroy: () => void
}

type LuaModuleFactory = (
  moduleArg?: LuaModuleInitOptions
) => Promise<LuaModule>

type AttachedLuaController = {
  character: CharacterState
  controller: LuaCharacterController
}

type LuaControllerCallContext = {
  characterId: string
  sourceCharacterId?: string
}

const LUA_MODULE_PATH = '/vendor/lua/lua-5.3.6.mjs'
const LUA_CONTROLLER_LOAD_HELPER_FUNCTION_NAME = '__engine_load_controller'
const LUA_CONTROLLER_CALL_HELPER_FUNCTION_NAME = '__engine_call_controller'
const LUA_CONTROLLER_HAS_METHOD_HELPER_FUNCTION_NAME =
  '__engine_has_controller_method'
const LUA_CONTROLLER_VALIDATE_HELPER_FUNCTION_NAME =
  '__engine_validate_controller'
const LUA_CONTROLLER_RESET_CONFIG_HELPER_FUNCTION_NAME =
  '__engine_reset_controller_config'
const LUA_CONTROLLER_SET_CONFIG_STRING_HELPER_FUNCTION_NAME =
  '__engine_set_controller_config_string'
const LUA_CONTROLLER_SET_CONFIG_NUMBER_HELPER_FUNCTION_NAME =
  '__engine_set_controller_config_number'
const LUA_CONTROLLER_SET_CONFIG_BOOLEAN_HELPER_FUNCTION_NAME =
  '__engine_set_controller_config_boolean'
const LUA_CONTROLLER_APPEND_CONFIG_STRING_LIST_ITEM_HELPER_FUNCTION_NAME =
  '__engine_append_controller_config_string_list_item'
const LUA_CONTROLLER_APPEND_CONFIG_NUMBER_LIST_ITEM_HELPER_FUNCTION_NAME =
  '__engine_append_controller_config_number_list_item'
const LUA_CONTROLLER_APPEND_CONFIG_BOOLEAN_LIST_ITEM_HELPER_FUNCTION_NAME =
  '__engine_append_controller_config_boolean_list_item'
const LUA_CONTROLLER_REMOVE_CONFIG_HELPER_FUNCTION_NAME =
  '__engine_remove_controller_config'
const LUA_CONTROLLER_DRAIN_EVENTS_FUNCTION_NAME = '__engine_drain_events_json'
const LUA_CONTROLLER_RUNTIME_HOST_API_SOURCE = `
local runtime = {
  current_character_id = nil,
  current_source_character_id = nil,
  queued_events = {},
  controllers_by_script_id = {},
  controller_config_by_character_id = {}
}

local function escape_json_string(value)
  local replacements = {
    ['\\\\'] = '\\\\\\\\',
    ['"'] = '\\\\\\"',
    ['\\b'] = '\\\\b',
    ['\\f'] = '\\\\f',
    ['\\n'] = '\\\\n',
    ['\\r'] = '\\\\r',
    ['\\t'] = '\\\\t'
  }

  return '"' .. string.gsub(value, '[%z\\1-\\31\\\\"]', function(character)
    local replacement = replacements[character]

    if replacement ~= nil then
      return replacement
    end

    return string.format('\\\\u%04x', string.byte(character))
  end) .. '"'
end

local function encode_runtime_event(event)
  if event.kind == 'show-character-message' then
    return '{"kind":"show-character-message","characterId":'
      .. escape_json_string(event.character_id)
      .. ',"message":'
      .. escape_json_string(event.message)
      .. ',"durationMilliseconds":'
      .. tostring(event.duration_milliseconds)
      .. '}'
  end

  error('Unsupported engine runtime event kind: ' .. tostring(event.kind))
end

local function get_duration_milliseconds(duration_seconds)
  if
    type(duration_seconds) ~= 'number'
    or duration_seconds ~= duration_seconds
    or duration_seconds == math.huge
    or duration_seconds == -math.huge
  then
    return ${DEFAULT_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS}
  end

  local rounded = math.floor(duration_seconds * 1000 + 0.5)

  if rounded < ${MIN_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS} then
    return ${MIN_LUA_CONTROLLER_MESSAGE_DURATION_MILLISECONDS}
  end

  return rounded
end

local function require_current_character_id()
  if
    type(runtime.current_character_id) ~= 'string'
    or runtime.current_character_id == ''
  then
    error('${LUA_CONTROLLER_PUBLIC_API_NAME} API called outside controller context')
  end

  return runtime.current_character_id
end

local function clone_table(value)
  local clone = {}

  for key, entry in pairs(value) do
    if type(entry) == 'table' then
      clone[key] = clone_table(entry)
    else
      clone[key] = entry
    end
  end

  return clone
end

function ${LUA_CONTROLLER_LOAD_HELPER_FUNCTION_NAME}(script_id, source, chunk_name)
  local chunk, load_error = load(source, chunk_name)

  if type(chunk) ~= 'function' then
    error(
      'Lua controller syntax error in "'
        .. tostring(script_id)
        .. '": '
        .. tostring(load_error)
    )
  end

  local results = table.pack(pcall(chunk))

  if not results[1] then
    error(results[2])
  end

  local controller = results[2]

  if type(controller) ~= 'table' then
    error(
      'Lua controller validation failed for "'
        .. tostring(script_id)
        .. '": the module must return a controller table'
    )
  end

  runtime.controllers_by_script_id[script_id] = controller
end

function ${LUA_CONTROLLER_HAS_METHOD_HELPER_FUNCTION_NAME}(script_id, method_name)
  local controller = runtime.controllers_by_script_id[script_id]

  return type(controller) == 'table' and type(controller[method_name]) == 'function' and 1 or 0
end

function ${LUA_CONTROLLER_RESET_CONFIG_HELPER_FUNCTION_NAME}(character_id)
  runtime.controller_config_by_character_id[character_id] = {}
end

function ${LUA_CONTROLLER_SET_CONFIG_STRING_HELPER_FUNCTION_NAME}(character_id, key, value)
  runtime.controller_config_by_character_id[character_id][key] = tostring(value)
end

function ${LUA_CONTROLLER_SET_CONFIG_NUMBER_HELPER_FUNCTION_NAME}(character_id, key, value)
  runtime.controller_config_by_character_id[character_id][key] = value
end

function ${LUA_CONTROLLER_SET_CONFIG_BOOLEAN_HELPER_FUNCTION_NAME}(character_id, key, numeric_boolean)
  runtime.controller_config_by_character_id[character_id][key] = numeric_boolean ~= 0
end

function ${LUA_CONTROLLER_APPEND_CONFIG_STRING_LIST_ITEM_HELPER_FUNCTION_NAME}(character_id, key, value)
  local config = runtime.controller_config_by_character_id[character_id]
  local list = config[key]

  if type(list) ~= 'table' then
    list = {}
    config[key] = list
  end

  list[#list + 1] = tostring(value)
end

function ${LUA_CONTROLLER_APPEND_CONFIG_NUMBER_LIST_ITEM_HELPER_FUNCTION_NAME}(character_id, key, value)
  local config = runtime.controller_config_by_character_id[character_id]
  local list = config[key]

  if type(list) ~= 'table' then
    list = {}
    config[key] = list
  end

  list[#list + 1] = value
end

function ${LUA_CONTROLLER_APPEND_CONFIG_BOOLEAN_LIST_ITEM_HELPER_FUNCTION_NAME}(character_id, key, numeric_boolean)
  local config = runtime.controller_config_by_character_id[character_id]
  local list = config[key]

  if type(list) ~= 'table' then
    list = {}
    config[key] = list
  end

  list[#list + 1] = numeric_boolean ~= 0
end

function ${LUA_CONTROLLER_REMOVE_CONFIG_HELPER_FUNCTION_NAME}(character_id)
  runtime.controller_config_by_character_id[character_id] = nil
end

function ${LUA_CONTROLLER_VALIDATE_HELPER_FUNCTION_NAME}(script_id, source, chunk_name)
  local env = setmetatable({
    ${LUA_CONTROLLER_PUBLIC_API_NAME} = ${LUA_CONTROLLER_PUBLIC_API_NAME}
  }, {
    __index = _G
  })
  local chunk, load_error = load(source, chunk_name, 't', env)

  if type(chunk) ~= 'function' then
    error(
      'Lua controller syntax error in "'
        .. tostring(script_id)
        .. '": '
        .. tostring(load_error)
    )
  end

  local results = table.pack(pcall(chunk))

  if not results[1] then
    error(results[2])
  end

  local controller = results[2]

  if type(controller) ~= 'table' then
    error(
      'Lua controller validation failed for "'
        .. tostring(script_id)
        .. '": the module must return a controller table'
    )
  end

  if type(controller.${LUA_CONTROLLER_STEP_METHOD_NAME}) ~= 'function' then
    error(
      'Lua controller validation failed for "'
        .. tostring(script_id)
        .. '": missing required method controller.${LUA_CONTROLLER_STEP_METHOD_NAME}(id, delta_seconds, x, y)'
    )
  end

  if controller.${LUA_CONTROLLER_REGISTER_METHOD_NAME} ~= nil
    and type(controller.${LUA_CONTROLLER_REGISTER_METHOD_NAME}) ~= 'function'
  then
    error(
      'Lua controller validation failed for "'
        .. tostring(script_id)
        .. '": controller.${LUA_CONTROLLER_REGISTER_METHOD_NAME} must be a function when present'
    )
  end

  if controller.${LUA_CONTROLLER_UNREGISTER_METHOD_NAME} ~= nil
    and type(controller.${LUA_CONTROLLER_UNREGISTER_METHOD_NAME}) ~= 'function'
  then
    error(
      'Lua controller validation failed for "'
        .. tostring(script_id)
        .. '": controller.${LUA_CONTROLLER_UNREGISTER_METHOD_NAME} must be a function when present'
    )
  end

  if controller.${LUA_CONTROLLER_INTERACT_METHOD_NAME} ~= nil
    and type(controller.${LUA_CONTROLLER_INTERACT_METHOD_NAME}) ~= 'function'
  then
    error(
      'Lua controller validation failed for "'
        .. tostring(script_id)
        .. '": controller.${LUA_CONTROLLER_INTERACT_METHOD_NAME} must be a function when present'
    )
  end
end

function ${LUA_CONTROLLER_CALL_HELPER_FUNCTION_NAME}(script_id, method_name, character_id, source_character_id, ...)
  local controller = runtime.controllers_by_script_id[script_id]

  if type(controller) ~= 'table' then
    error('Missing Lua controller module ' .. tostring(script_id))
  end

  local controller_fn = controller[method_name]

  if type(controller_fn) ~= 'function' then
    error(
      'Missing Lua controller method '
        .. tostring(method_name)
        .. ' in '
        .. tostring(script_id)
    )
  end

  runtime.current_character_id = character_id
  runtime.current_source_character_id = source_character_id

  local results = table.pack(pcall(controller_fn, ...))

  runtime.current_character_id = nil
  runtime.current_source_character_id = nil

  if not results[1] then
    error(results[2])
  end

  return table.unpack(results, 2, results.n)
end

function ${LUA_CONTROLLER_DRAIN_EVENTS_FUNCTION_NAME}()
  if #runtime.queued_events == 0 then
    return '[]'
  end

  local encoded_events = {}

  for index, event in ipairs(runtime.queued_events) do
    encoded_events[index] = encode_runtime_event(event)
  end

  runtime.queued_events = {}

  return '[' .. table.concat(encoded_events, ',') .. ']'
end

${LUA_CONTROLLER_PUBLIC_API_NAME} = ${LUA_CONTROLLER_PUBLIC_API_NAME} or {}
${LUA_CONTROLLER_PUBLIC_API_NAME}.self = ${LUA_CONTROLLER_PUBLIC_API_NAME}.self or {}
${LUA_CONTROLLER_PUBLIC_API_NAME}.ui = ${LUA_CONTROLLER_PUBLIC_API_NAME}.ui or {}

function ${LUA_CONTROLLER_PUBLIC_API_NAME}.self.get_controller_config()
  local config = runtime.controller_config_by_character_id[require_current_character_id()]

  if type(config) ~= 'table' then
    return {}
  end

  return clone_table(config)
end

function ${LUA_CONTROLLER_PUBLIC_API_NAME}.ui.show_message(message, duration_seconds)
  if message == nil then
    return
  end

  local normalized_message = tostring(message)

  if normalized_message == '' then
    return
  end

  runtime.queued_events[#runtime.queued_events + 1] = {
    kind = 'show-character-message',
    character_id = require_current_character_id(),
    message = normalized_message,
    duration_milliseconds = get_duration_milliseconds(duration_seconds)
  }
end
`

export const createLuaCharacterControllerRuntime = async ({
  scriptsById,
  createLuaModuleFactory,
  createLuaModuleOptions
}: CreateLuaCharacterControllerRuntimeInput): Promise<LuaCharacterControllerRuntime> => {
  const createLuaModule = createLuaModuleFactory
    ? await createLuaModuleFactory()
    : await loadLuaModuleFactory()
  const lua = await createLuaModule(
    createLuaModuleOptions ?? {
      locateFile: (path: string) =>
        new URL(path, getLuaAssetBaseUrl()).href
    }
  )
  const scriptSources = new Map(Object.entries(scriptsById))
  const attachedControllers = new Map<string, AttachedLuaController>()
  const lastLoggedErrorByKey = new Map<string, string>()
  let luaState = createLuaState(lua, scriptSources)
  let isDestroyed = false

  const rebuildLuaState = (
    nextScriptSources: Map<string, LuaControllerScriptSource>
  ) => {
    const nextLuaState = createLuaState(lua, nextScriptSources)

    try {
      for (const attachedController of attachedControllers.values()) {
        attachCharacterToState(
          lua,
          nextLuaState,
          nextScriptSources,
          attachedController
        )
      }
    } catch (error) {
      lua._lua_close(nextLuaState)
      throw error
    }

    lua._lua_close(luaState)
    luaState = nextLuaState
  }

  return {
    attachCharacter: (character, controller) => {
      try {
        const attachedController = {
          character,
          controller
        }

        attachCharacterToState(lua, luaState, scriptSources, attachedController)
        attachedControllers.set(character.id, attachedController)
        clearLoggedError(
          lastLoggedErrorByKey,
          `${controller.scriptId}:attach:${character.id}`
        )
      } catch (error) {
        reportLuaRuntimeError(
          lastLoggedErrorByKey,
          `${controller.scriptId}:attach:${character.id}`,
          error,
          {
            scriptId: controller.scriptId,
            source:
              getOptionalScriptSource(scriptSources, controller.scriptId) ?? ''
          }
        )
      }
    },
    detachCharacter: (character, controller) => {
      try {
        detachCharacterFromState(lua, luaState, scriptSources, {
          character,
          controller
        })
      } catch (error) {
        reportLuaRuntimeError(
          lastLoggedErrorByKey,
          `${controller.scriptId}:detach:${character.id}`,
          error,
          {
            scriptId: controller.scriptId,
            source:
              getOptionalScriptSource(scriptSources, controller.scriptId) ?? ''
          }
        )
      }

      attachedControllers.delete(character.id)
    },
    getMovementDelta: (character, controller, deltaMilliseconds) => {
      try {
        if (attachedControllers.has(character.id)) {
          attachedControllers.set(character.id, {
            character,
            controller
          })
        }

        getRequiredScript(scriptSources, controller.scriptId)
        const [x, y] = callLuaControllerMethod(
          lua,
          luaState,
          controller.scriptId,
          LUA_CONTROLLER_STEP_METHOD_NAME,
          {
            characterId: character.id
          },
          getLuaControllerStepFunctionArguments(
            createLuaControllerStepInput(character, deltaMilliseconds)
          ),
          2
        )
        const moveIntent = createLuaControllerMoveIntent(x, y)

        clearLoggedError(
          lastLoggedErrorByKey,
          `${controller.scriptId}:step:${character.id}`
        )

        if (!moveIntent) {
          return undefined
        }

        return {
          x: moveIntent.moveX,
          y: moveIntent.moveY
        }
      } catch (error) {
        reportLuaRuntimeError(
          lastLoggedErrorByKey,
          `${controller.scriptId}:step:${character.id}`,
          error,
          {
            scriptId: controller.scriptId,
            source:
              getOptionalScriptSource(scriptSources, controller.scriptId) ?? ''
          }
        )
        return undefined
      }
    },
    canReceiveInteraction: (_character, controller) => {
      getRequiredScript(scriptSources, controller.scriptId)

      return hasLuaControllerMethod(
        lua,
        luaState,
        controller.scriptId,
        LUA_CONTROLLER_INTERACT_METHOD_NAME
      )
    },
    handleInteraction: (character, controller, sourceCharacter) => {
      try {
        if (attachedControllers.has(character.id)) {
          attachedControllers.set(character.id, {
            character,
            controller
          })
        }

        getRequiredScript(scriptSources, controller.scriptId)

        if (
          !hasLuaControllerMethod(
            lua,
            luaState,
            controller.scriptId,
            LUA_CONTROLLER_INTERACT_METHOD_NAME
          )
        ) {
          return undefined
        }

        const [message, durationSeconds] =
          callLuaControllerMethodForStringAndNumber(
          lua,
          luaState,
          controller.scriptId,
          LUA_CONTROLLER_INTERACT_METHOD_NAME,
          {
            characterId: character.id,
            sourceCharacterId: sourceCharacter.id
          },
          getLuaControllerInteractFunctionArguments(
            createLuaControllerInteractInput(character, sourceCharacter)
          )
        )

        clearLoggedError(
          lastLoggedErrorByKey,
          `${controller.scriptId}:interact:${character.id}`
        )

        return createLuaControllerInteractionResponse(message, durationSeconds)
      } catch (error) {
        reportLuaRuntimeError(
          lastLoggedErrorByKey,
          `${controller.scriptId}:interact:${character.id}`,
          error,
          {
            scriptId: controller.scriptId,
            source:
              getOptionalScriptSource(scriptSources, controller.scriptId) ?? ''
          }
        )
        return undefined
      }
    },
    drainEvents: () => {
      try {
        clearLoggedError(lastLoggedErrorByKey, 'runtime:drain-events')

        return drainLuaControllerRuntimeEvents(lua, luaState)
      } catch (error) {
        reportLuaRuntimeError(
          lastLoggedErrorByKey,
          'runtime:drain-events',
          error
        )
        return []
      }
    },
    getActiveErrorMessages: () => [...new Set(lastLoggedErrorByKey.values())],
    updateScript: (scriptId, script) => {
      const reloadErrorKey = `${scriptId}:reload`

      try {
        const nextScriptSources = new Map(scriptSources)

        nextScriptSources.set(scriptId, script)
        rebuildLuaState(nextScriptSources)
        scriptSources.clear()

        for (const [nextScriptId, nextScript] of nextScriptSources) {
          scriptSources.set(nextScriptId, nextScript)
        }

        clearLoggedError(lastLoggedErrorByKey, reloadErrorKey)
      } catch (error) {
        reportLuaRuntimeError(lastLoggedErrorByKey, reloadErrorKey, error, {
          scriptId,
          source: script.source
        })
      }
    },
    destroy: () => {
      if (isDestroyed) {
        return
      }

      for (const attachedController of attachedControllers.values()) {
        try {
          detachCharacterFromState(lua, luaState, scriptSources, attachedController)
        } catch (error) {
          reportLuaRuntimeError(
            lastLoggedErrorByKey,
            `${attachedController.controller.scriptId}:destroy:${attachedController.character.id}`,
            error,
            {
              scriptId: attachedController.controller.scriptId,
              source:
                getOptionalScriptSource(
                  scriptSources,
                  attachedController.controller.scriptId
                ) ?? ''
            }
          )
        }
      }

      attachedControllers.clear()
      lua._lua_close(luaState)
      isDestroyed = true
    }
  }
}

const createLuaState = (
  lua: LuaModule,
  scriptSources: Map<string, LuaControllerScriptSource>
): number => {
  for (const [scriptId, script] of scriptSources) {
    validateLuaControllerScript(lua, scriptId, script)
  }

  const luaState = lua._luaL_newstate()

  if (!luaState) {
    throw new Error('Failed to create a Lua state.')
  }

  lua._luaL_openlibs(luaState)

  try {
    runLuaChunk(
      lua,
      luaState,
      LUA_CONTROLLER_RUNTIME_HOST_API_SOURCE,
      '@engine-runtime.lua'
    )

    for (const [scriptId, script] of scriptSources) {
      loadLuaControllerModule(lua, luaState, scriptId, script)
    }
  } catch (error) {
    lua._lua_close(luaState)
    throw error
  }

  return luaState
}

const attachCharacterToState = (
  lua: LuaModule,
  luaState: number,
  scriptSources: Map<string, LuaControllerScriptSource>,
  attachedController: AttachedLuaController
) => {
  getRequiredScript(
    scriptSources,
    attachedController.controller.scriptId
  )
  syncControllerConfigToState(
    lua,
    luaState,
    attachedController.character.id,
    attachedController.controller.config
  )

  if (
    !hasLuaControllerMethod(
      lua,
      luaState,
      attachedController.controller.scriptId,
      LUA_CONTROLLER_REGISTER_METHOD_NAME
    )
  ) {
    return
  }

  callLuaControllerMethod(
    lua,
    luaState,
    attachedController.controller.scriptId,
    LUA_CONTROLLER_REGISTER_METHOD_NAME,
    {
      characterId: attachedController.character.id
    },
    getLuaControllerRegisterFunctionArguments(
      createLuaControllerRegisterInput(
        attachedController.character,
        attachedController.controller
      )
    )
  )
}

const detachCharacterFromState = (
  lua: LuaModule,
  luaState: number,
  scriptSources: Map<string, LuaControllerScriptSource>,
  attachedController: AttachedLuaController
) => {
  getRequiredScript(scriptSources, attachedController.controller.scriptId)

  if (
    !hasLuaControllerMethod(
      lua,
      luaState,
      attachedController.controller.scriptId,
      LUA_CONTROLLER_UNREGISTER_METHOD_NAME
    )
  ) {
    removeControllerConfigFromState(lua, luaState, attachedController.character.id)
    return
  }

  callLuaControllerMethod(
    lua,
    luaState,
    attachedController.controller.scriptId,
    LUA_CONTROLLER_UNREGISTER_METHOD_NAME,
    {
      characterId: attachedController.character.id
    },
    getLuaControllerUnregisterFunctionArguments(
      createLuaControllerUnregisterInput(attachedController.character)
    )
  )

  removeControllerConfigFromState(lua, luaState, attachedController.character.id)
}

const getRequiredScript = (
  scriptSources: Map<string, LuaControllerScriptSource>,
  scriptId: string
): LuaControllerScriptSource => {
  const script = scriptSources.get(scriptId)

  if (!script) {
    throw new Error(`Missing Lua controller script ${scriptId}`)
  }

  return script
}

const syncControllerConfigToState = (
  lua: LuaModule,
  luaState: number,
  characterId: string,
  config: LuaCharacterControllerConfig
) => {
  callLuaFunction(lua, luaState, LUA_CONTROLLER_RESET_CONFIG_HELPER_FUNCTION_NAME, [
    characterId
  ])

  for (const [configKey, configValue] of Object.entries(config)) {
    pushControllerConfigValueToState(lua, luaState, characterId, configKey, configValue)
  }
}

const pushControllerConfigValueToState = (
  lua: LuaModule,
  luaState: number,
  characterId: string,
  configKey: string,
  configValue: LuaCharacterControllerConfigValue
) => {
  if (typeof configValue === 'string') {
    callLuaFunction(lua, luaState, LUA_CONTROLLER_SET_CONFIG_STRING_HELPER_FUNCTION_NAME, [
      characterId,
      configKey,
      configValue
    ])
    return
  }

  if (typeof configValue === 'number') {
    callLuaFunction(lua, luaState, LUA_CONTROLLER_SET_CONFIG_NUMBER_HELPER_FUNCTION_NAME, [
      characterId,
      configKey,
      configValue
    ])
    return
  }

  if (typeof configValue === 'boolean') {
    callLuaFunction(lua, luaState, LUA_CONTROLLER_SET_CONFIG_BOOLEAN_HELPER_FUNCTION_NAME, [
      characterId,
      configKey,
      configValue ? 1 : 0
    ])
    return
  }

  if (configValue.every((item) => typeof item === 'string')) {
    for (const item of configValue) {
      callLuaFunction(
        lua,
        luaState,
        LUA_CONTROLLER_APPEND_CONFIG_STRING_LIST_ITEM_HELPER_FUNCTION_NAME,
        [characterId, configKey, item]
      )
    }

    return
  }

  if (configValue.every((item) => typeof item === 'number')) {
    for (const item of configValue) {
      callLuaFunction(
        lua,
        luaState,
        LUA_CONTROLLER_APPEND_CONFIG_NUMBER_LIST_ITEM_HELPER_FUNCTION_NAME,
        [characterId, configKey, item]
      )
    }

    return
  }

  if (configValue.every((item) => typeof item === 'boolean')) {
    for (const item of configValue) {
      callLuaFunction(
        lua,
        luaState,
        LUA_CONTROLLER_APPEND_CONFIG_BOOLEAN_LIST_ITEM_HELPER_FUNCTION_NAME,
        [characterId, configKey, item ? 1 : 0]
      )
    }

    return
  }

  throw new Error(
    `Unsupported Lua controller config list "${configKey}": list items must all share one primitive type`
  )
}

const removeControllerConfigFromState = (
  lua: LuaModule,
  luaState: number,
  characterId: string
) => {
  callLuaFunction(lua, luaState, LUA_CONTROLLER_REMOVE_CONFIG_HELPER_FUNCTION_NAME, [
    characterId
  ])
}

const loadLuaModuleFactory = async (): Promise<LuaModuleFactory> => {
  const modulePath = getLuaModuleUrl()
  const luaModule = await import(/* @vite-ignore */ modulePath)

  return luaModule.default as LuaModuleFactory
}

const getLuaModuleUrl = (): string =>
  new URL(getBasePath(LUA_MODULE_PATH), window.location.href).href

const getLuaAssetBaseUrl = (): string =>
  new URL(getBasePath('/vendor/lua/'), window.location.href).href

const getBasePath = (path: string): string => {
  const normalizedBaseUrl = import.meta.env.BASE_URL.endsWith('/')
    ? import.meta.env.BASE_URL
    : `${import.meta.env.BASE_URL}/`

  return `${normalizedBaseUrl}${path.replace(/^\//, '')}`
}

// Validate controller source before it can replace the active runtime.
const validateLuaControllerScript = (
  lua: LuaModule,
  scriptId: string,
  script: LuaControllerScriptSource
) => {
  const luaState = createValidationLuaState(lua)

  try {
    compileLuaChunk(lua, luaState, script.source, `@${scriptId}.lua`)
    validateLuaControllerModule(lua, luaState, scriptId, script)
  } catch (error) {
    throw createLuaControllerScriptError(scriptId, script.source, error)
  } finally {
    lua._lua_close(luaState)
  }
}

const createValidationLuaState = (lua: LuaModule): number => {
  const luaState = lua._luaL_newstate()

  if (!luaState) {
    throw new Error('Failed to create a Lua validation state.')
  }

  lua._luaL_openlibs(luaState)
  runLuaChunk(lua, luaState, LUA_CONTROLLER_RUNTIME_HOST_API_SOURCE, '@engine-runtime.lua')

  return luaState
}

// Compile once without running so syntax errors fail early.
const compileLuaChunk = (
  lua: LuaModule,
  luaState: number,
  source: string,
  chunkName: string
) => {
  const baseTop = lua._lua_gettop(luaState)
  const sourceLength = lua.lengthBytesUTF8(source)
  const sourcePointer = lua._malloc(sourceLength + 1)
  const chunkNameLength = lua.lengthBytesUTF8(chunkName)
  const chunkNamePointer = lua._malloc(chunkNameLength + 1)

  lua.stringToUTF8(source, sourcePointer, sourceLength + 1)
  lua.stringToUTF8(chunkName, chunkNamePointer, chunkNameLength + 1)

  const loadStatus = lua._luaL_loadbufferx(
    luaState,
    sourcePointer,
    sourceLength,
    chunkNamePointer,
    0
  )

  lua._free(chunkNamePointer)
  lua._free(sourcePointer)

  if (loadStatus !== 0) {
    const errorMessage = readLuaError(lua, luaState)

    lua._lua_settop(luaState, baseTop)
    throw new Error(`Lua controller syntax error: ${errorMessage}`)
  }

  lua._lua_settop(luaState, baseTop)
}

// Run the module in an isolated env and verify the reserved method shape.
const validateLuaControllerModule = (
  lua: LuaModule,
  luaState: number,
  scriptId: string,
  script: LuaControllerScriptSource
) => {
  try {
    callLuaFunction(lua, luaState, LUA_CONTROLLER_VALIDATE_HELPER_FUNCTION_NAME, [
      scriptId,
      script.source,
      `@${scriptId}.lua`
    ])
  } catch (error) {
    throw normalizeLuaControllerValidationError(scriptId, error)
  }
}

const runLuaChunk = (
  lua: LuaModule,
  luaState: number,
  source: string,
  chunkName: string
) => {
  const sourceLength = lua.lengthBytesUTF8(source)
  const sourcePointer = lua._malloc(sourceLength + 1)
  const chunkNameLength = lua.lengthBytesUTF8(chunkName)
  const chunkNamePointer = lua._malloc(chunkNameLength + 1)

  lua.stringToUTF8(source, sourcePointer, sourceLength + 1)
  lua.stringToUTF8(chunkName, chunkNamePointer, chunkNameLength + 1)

  const loadStatus = lua._luaL_loadbufferx(
    luaState,
    sourcePointer,
    sourceLength,
    chunkNamePointer,
    0
  )

  lua._free(chunkNamePointer)
  lua._free(sourcePointer)

  if (loadStatus !== 0) {
    throw new Error(`Failed to compile Lua chunk: ${readLuaError(lua, luaState)}`)
  }

  const callStatus = lua._lua_pcallk(luaState, 0, 0, 0, 0, 0)

  if (callStatus !== 0) {
    throw new Error(`Failed to execute Lua chunk: ${readLuaError(lua, luaState)}`)
  }
}

const loadLuaControllerModule = (
  lua: LuaModule,
  luaState: number,
  scriptId: string,
  script: LuaControllerScriptSource
) => {
  callLuaFunction(
    lua,
    luaState,
    LUA_CONTROLLER_LOAD_HELPER_FUNCTION_NAME,
    [scriptId, script.source, `@${scriptId}.lua`]
  )
}

const callLuaFunction = (
  lua: LuaModule,
  luaState: number,
  functionName: string,
  arguments_: LuaFunctionArgument[],
  resultCount = 0
): number[] => {
  const baseTop = lua._lua_gettop(luaState)

  pushGlobalFunction(lua, luaState, functionName)

  for (const argument of arguments_) {
    if (typeof argument === 'number') {
      lua._lua_pushnumber(luaState, argument)
      continue
    }

    pushLuaString(lua, luaState, argument)
  }

  const callStatus = lua._lua_pcallk(
    luaState,
    arguments_.length,
    resultCount,
    0,
    0,
    0
  )

  if (callStatus !== 0) {
    const errorMessage = readLuaError(lua, luaState)

    lua._lua_settop(luaState, baseTop)
    throw new Error(`Lua function ${functionName} failed: ${errorMessage}`)
  }

  const results = Array.from({ length: resultCount }, (_, index) =>
    lua._lua_tonumberx(luaState, baseTop + 1 + index, 0)
  )

  lua._lua_settop(luaState, baseTop)

  return results
}

const hasLuaControllerMethod = (
  lua: LuaModule,
  luaState: number,
  scriptId: string,
  methodName: LuaControllerMethodName
): boolean =>
  callLuaFunction(
    lua,
    luaState,
    LUA_CONTROLLER_HAS_METHOD_HELPER_FUNCTION_NAME,
    [scriptId, methodName],
    1
  )[0] !== 0

const callLuaControllerMethod = (
  lua: LuaModule,
  luaState: number,
  scriptId: string,
  methodName: LuaControllerMethodName,
  context: LuaControllerCallContext,
  arguments_: LuaFunctionArgument[],
  resultCount = 0
): number[] =>
  callLuaFunction(
    lua,
    luaState,
    LUA_CONTROLLER_CALL_HELPER_FUNCTION_NAME,
    [
      scriptId,
      methodName,
      context.characterId,
      context.sourceCharacterId ?? '',
      ...arguments_
    ],
    resultCount
  )

const callLuaFunctionForString = (
  lua: LuaModule,
  luaState: number,
  functionName: string,
  arguments_: LuaFunctionArgument[]
): string | undefined => {
  const baseTop = lua._lua_gettop(luaState)

  pushGlobalFunction(lua, luaState, functionName)

  for (const argument of arguments_) {
    if (typeof argument === 'number') {
      lua._lua_pushnumber(luaState, argument)
      continue
    }

    pushLuaString(lua, luaState, argument)
  }

  const callStatus = lua._lua_pcallk(luaState, arguments_.length, 1, 0, 0, 0)

  if (callStatus !== 0) {
    const errorMessage = readLuaError(lua, luaState)

    lua._lua_settop(luaState, baseTop)
    throw new Error(`Lua function ${functionName} failed: ${errorMessage}`)
  }

  const result = readLuaString(lua, luaState, baseTop + 1)

  lua._lua_settop(luaState, baseTop)

  return result
}

const callLuaFunctionForStringAndNumber = (
  lua: LuaModule,
  luaState: number,
  functionName: string,
  arguments_: LuaFunctionArgument[]
): [string | undefined, number | undefined] => {
  const baseTop = lua._lua_gettop(luaState)

  pushGlobalFunction(lua, luaState, functionName)

  for (const argument of arguments_) {
    if (typeof argument === 'number') {
      lua._lua_pushnumber(luaState, argument)
      continue
    }

    pushLuaString(lua, luaState, argument)
  }

  const callStatus = lua._lua_pcallk(luaState, arguments_.length, 2, 0, 0, 0)

  if (callStatus !== 0) {
    const errorMessage = readLuaError(lua, luaState)

    lua._lua_settop(luaState, baseTop)
    throw new Error(`Lua function ${functionName} failed: ${errorMessage}`)
  }

  const message = readLuaString(lua, luaState, baseTop + 1)
  const durationSeconds = readLuaNumber(lua, luaState, baseTop + 2)

  lua._lua_settop(luaState, baseTop)

  return [message, durationSeconds]
}

const callLuaControllerMethodForStringAndNumber = (
  lua: LuaModule,
  luaState: number,
  scriptId: string,
  methodName: LuaControllerMethodName,
  context: LuaControllerCallContext,
  arguments_: LuaFunctionArgument[]
): [string | undefined, number | undefined] =>
  callLuaFunctionForStringAndNumber(
    lua,
    luaState,
    LUA_CONTROLLER_CALL_HELPER_FUNCTION_NAME,
    [
      scriptId,
      methodName,
      context.characterId,
      context.sourceCharacterId ?? '',
      ...arguments_
    ]
  )

const drainLuaControllerRuntimeEvents = (
  lua: LuaModule,
  luaState: number
): LuaControllerRuntimeEvent[] => {
  const serializedEvents = callLuaFunctionForString(
    lua,
    luaState,
    LUA_CONTROLLER_DRAIN_EVENTS_FUNCTION_NAME,
    []
  )

  if (!serializedEvents || serializedEvents === '[]') {
    return []
  }

  return parseLuaControllerRuntimeEvents(serializedEvents)
}

const pushGlobalFunction = (
  lua: LuaModule,
  luaState: number,
  functionName: string
) => {
  const functionNameLength = lua.lengthBytesUTF8(functionName)
  const functionNamePointer = lua._malloc(functionNameLength + 1)

  lua.stringToUTF8(functionName, functionNamePointer, functionNameLength + 1)
  lua._lua_getglobal(luaState, functionNamePointer)
  lua._free(functionNamePointer)
}

const pushLuaString = (
  lua: LuaModule,
  luaState: number,
  value: string
) => {
  const valueLength = lua.lengthBytesUTF8(value)
  const valuePointer = lua._malloc(valueLength + 1)

  lua.stringToUTF8(value, valuePointer, valueLength + 1)
  lua._lua_pushlstring(luaState, valuePointer, valueLength)
  lua._free(valuePointer)
}

const readLuaError = (lua: LuaModule, luaState: number): string => {
  const messagePointer = lua._lua_tolstring(luaState, -1, 0)

  return messagePointer === 0
    ? 'unknown Lua error'
    : lua.UTF8ToString(Number(messagePointer))
}

const readLuaString = (
  lua: LuaModule,
  luaState: number,
  index: number
): string | undefined => {
  const valuePointer = lua._lua_tolstring(luaState, index, 0)

  if (valuePointer === 0) {
    return undefined
  }

  const value = lua.UTF8ToString(Number(valuePointer))

  return value.length === 0 ? undefined : value
}

const readLuaNumber = (
  lua: LuaModule,
  luaState: number,
  index: number
): number | undefined => {
  const LUA_TYPE_NUMBER = 3

  if (lua._lua_type(luaState, index) !== LUA_TYPE_NUMBER) {
    return undefined
  }

  return lua._lua_tonumberx(luaState, index, 0)
}

const parseLuaControllerRuntimeEvents = (
  serializedEvents: string
): LuaControllerRuntimeEvent[] => {
  const parsedEvents = JSON.parse(serializedEvents) as unknown

  if (!Array.isArray(parsedEvents)) {
    throw new Error('Lua runtime event drain did not return an array.')
  }

  return parsedEvents.flatMap((parsedEvent) => {
    if (
      typeof parsedEvent !== 'object' ||
      parsedEvent === null ||
      parsedEvent.kind !== 'show-character-message' ||
      typeof parsedEvent.characterId !== 'string' ||
      typeof parsedEvent.message !== 'string' ||
      typeof parsedEvent.durationMilliseconds !== 'number' ||
      !Number.isFinite(parsedEvent.durationMilliseconds)
    ) {
      throw new Error('Lua runtime event drain returned an invalid event.')
    }

    return [
      {
        kind: 'show-character-message',
        characterId: parsedEvent.characterId,
        message: parsedEvent.message,
        durationMilliseconds: parsedEvent.durationMilliseconds
      }
    ]
  })
}

const getOptionalScriptSource = (
  scriptSources: Map<string, LuaControllerScriptSource>,
  scriptId: string
): string | undefined => scriptSources.get(scriptId)?.source

const normalizeLuaControllerBridgeError = (error: unknown): Error => {
  const baseError = error instanceof Error ? error : new Error(String(error))
  let message = baseError.message
  const helperFunctionNames = [
    LUA_CONTROLLER_LOAD_HELPER_FUNCTION_NAME,
    LUA_CONTROLLER_CALL_HELPER_FUNCTION_NAME,
    LUA_CONTROLLER_HAS_METHOD_HELPER_FUNCTION_NAME,
    LUA_CONTROLLER_VALIDATE_HELPER_FUNCTION_NAME,
    LUA_CONTROLLER_DRAIN_EVENTS_FUNCTION_NAME
  ]

  for (const helperName of helperFunctionNames) {
    const helperPrefix = `Lua function ${helperName} failed: `

    if (message.startsWith(helperPrefix)) {
      message = message.slice(helperPrefix.length)
      break
    }
  }

  message = message.replace(/^engine-runtime\.lua:\d+:\s*/u, '')

  if (message === baseError.message) {
    return baseError
  }

  const normalizedError = new Error(message)

  normalizedError.name = baseError.name

  return normalizedError
}

const createLuaControllerScriptError = (
  scriptId: string,
  source: string,
  error: unknown
): Error => {
  const baseError = normalizeLuaControllerBridgeError(error)
  const lineNumber = extractLuaLineNumber(baseError.message)

  if (!lineNumber) {
    return baseError
  }

  const sourceLine = source.split(/\r?\n/u)[lineNumber - 1]?.trimEnd()

  if (!sourceLine) {
    return baseError
  }

  const annotatedMessage = `${baseError.message}\nLua source ${scriptId}.lua:${lineNumber}\n> ${sourceLine}`

  if (annotatedMessage === baseError.message) {
    return baseError
  }

  const annotatedError = new Error(annotatedMessage)

  annotatedError.name = baseError.name

  return annotatedError
}

const extractLuaLineNumber = (message: string): number | undefined => {
  const match = message.match(
    /(?:\[string "?@?[^"\]]+"?\]|@?[^:\n]+\.lua):(\d+):/u
  )

  if (!match) {
    return undefined
  }

  const lineNumber = Number.parseInt(match[1], 10)

  return Number.isFinite(lineNumber) ? lineNumber : undefined
}

const normalizeLuaControllerValidationError = (
  scriptId: string,
  error: unknown
): Error => {
  const baseError = normalizeLuaControllerBridgeError(error)
  let message = baseError.message

  if (
    !message.startsWith('Lua controller validation failed')
    && !message.startsWith('Lua controller syntax error')
  ) {
    message = `Lua controller validation failed for "${scriptId}": ${message}`
  }

  if (message === baseError.message) {
    return baseError
  }

  const normalizedError = new Error(message)

  normalizedError.name = baseError.name

  return normalizedError
}

const reportLuaRuntimeError = (
  lastLoggedErrorByKey: Map<string, string>,
  errorKey: string,
  error: unknown,
  scriptSource?: {
    scriptId: string
    source: string
  }
) => {
  const normalizedError = scriptSource
    ? createLuaControllerScriptError(scriptSource.scriptId, scriptSource.source, error)
    : error instanceof Error
      ? error
      : new Error(String(error))
  const message = normalizedError.message

  if (lastLoggedErrorByKey.get(errorKey) === message) {
    return
  }

  lastLoggedErrorByKey.set(errorKey, message)
  console.error(`Lua controller error [${errorKey}]`, normalizedError)
}

const clearLoggedError = (
  lastLoggedErrorByKey: Map<string, string>,
  errorKey: string
) => {
  lastLoggedErrorByKey.delete(errorKey)
}
