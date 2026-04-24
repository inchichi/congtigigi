import type {
  CharacterState,
  LuaCharacterController
} from '../characterState'

type LuaControllerScriptSource = {
  registerFunctionName: string
  stepFunctionName: string
  source: string
}

type CreateLuaCharacterControllerRuntimeInput = {
  scriptsById: Record<string, LuaControllerScriptSource>
}

type LuaFunctionArgument = number | string

type LuaModule = {
  _luaL_newstate: () => number
  _luaL_openlibs: (luaState: number) => void
  _lua_close: (luaState: number) => void
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
  getControllerDirection: (
    character: CharacterState,
    controller: LuaCharacterController,
    deltaMilliseconds: number
  ) => { x: number; y: number } | undefined
  destroy: () => void
}

type LuaModuleFactory = (moduleArg?: {
  locateFile?: (path: string) => string
}) => Promise<LuaModule>

const LUA_MODULE_PATH = '/vendor/lua/lua-5.3.6.mjs'

export const createLuaCharacterControllerRuntime = async ({
  scriptsById
}: CreateLuaCharacterControllerRuntimeInput): Promise<LuaCharacterControllerRuntime> => {
  const createLuaModule = await loadLuaModuleFactory()
  const luaAssetBaseUrl = getLuaAssetBaseUrl()
  const lua = await createLuaModule({
    locateFile: (path: string) => new URL(path, luaAssetBaseUrl).href
  })
  const luaState = lua._luaL_newstate()
  const registeredControllerKeys = new Set<string>()

  if (!luaState) {
    throw new Error('Failed to create a Lua state.')
  }

  lua._luaL_openlibs(luaState)

  for (const [scriptId, script] of Object.entries(scriptsById)) {
    runLuaChunk(lua, luaState, script.source, `@${scriptId}.lua`)
  }

  return {
    getControllerDirection: (character, controller, deltaMilliseconds) => {
      const script = scriptsById[controller.scriptId]

      if (!script) {
        throw new Error(`Missing Lua controller script ${controller.scriptId}`)
      }

      const controllerKey = `${controller.scriptId}:${character.id}`

      if (!registeredControllerKeys.has(controllerKey)) {
        callLuaFunction(lua, luaState, script.registerFunctionName, [
          character.id,
          character.position.x,
          character.position.y,
          controller.radiusInTiles
        ])
        registeredControllerKeys.add(controllerKey)
      }

      const [x, y] = callLuaFunction(
        lua,
        luaState,
        script.stepFunctionName,
        [
          character.id,
          deltaMilliseconds / 1000,
          character.position.x,
          character.position.y
        ],
        2
      )

      if (x === 0 && y === 0) {
        return undefined
      }

      return {
        x,
        y
      }
    },
    destroy: () => {
      lua._lua_close(luaState)
    }
  }
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
