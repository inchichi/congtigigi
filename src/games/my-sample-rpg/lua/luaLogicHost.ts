// 범용 Lua 로직 호스트 — 순수 게임 로직(.lua)을 같은 Lua 5.3.6 WASM VM으로 실행한다.
// 캐릭터 컨트롤러 런타임(createLuaCharacterControllerRuntime)과 동일한 VM/마샬링 패턴을 쓰되,
// 그 런타임을 건드리지 않도록 독립 모듈로 둔다(컨트롤러 = 매 프레임 이동, 여기 = 턴/이벤트 로직).
//
// 모듈은 전역 함수들을 정의하고(예: function monster_rewards_gold(level) ... end), 호스트가
// 그 전역 함수를 원시값(number|string) 인자로 호출해 number/string 결과를 받는다(빠른 평탄 경로).
// 객체/배열 인자·반환이 필요한 모듈은 callJson(...)을 쓴다 — json-codec.lua가 부팅 시 로드돼
// __logic_call_json(fn, args_json)으로 디코드→호출→인코드한다.

import jsonCodecSource from '../assets/lua/json-codec.lua?raw'

type LuaModuleInitOptions = {
  locateFile?: (path: string) => string
  wasmBinary?: ArrayBuffer | Uint8Array
  print?: (value: string) => void
  printErr?: (value: string) => void
}

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

type LuaModuleFactory = (moduleArg?: LuaModuleInitOptions) => Promise<LuaModule>

type LuaFunctionArgument = number | string

export type CreateLuaLogicHostInput = {
  // 테스트(node)에서 디스크의 wasm/mjs를 주입할 때 사용. 브라우저에선 생략(기본 로더 사용).
  createLuaModuleFactory?: () => Promise<LuaModuleFactory>
  createLuaModuleOptions?: LuaModuleInitOptions
}

export type LuaLogicHost = {
  // 모듈 소스를 평가해 그 전역 함수들을 등록한다(여러 번 호출 가능 — 모듈을 누적).
  runModule: (source: string, chunkName: string) => void
  // 전역 함수를 호출하고 number 결과 배열을 돌려준다.
  callNumbers: (
    functionName: string,
    args: LuaFunctionArgument[],
    resultCount?: number
  ) => number[]
  // 단일 number 결과 헬퍼.
  callNumber: (functionName: string, ...args: LuaFunctionArgument[]) => number
  // 단일 string 결과 헬퍼.
  callString: (functionName: string, ...args: LuaFunctionArgument[]) => string
  // 구조적(객체/배열) 인자·반환 — JSON 마샬링. 전역 fn(args...) 호출, 단일 반환값을 받는다.
  callJson: <Result = unknown>(functionName: string, ...args: unknown[]) => Result
  close: () => void
}

const LUA_MODULE_PATH = '/vendor/lua/lua-5.3.6.mjs'

export const createLuaLogicHost = async ({
  createLuaModuleFactory,
  createLuaModuleOptions
}: CreateLuaLogicHostInput = {}): Promise<LuaLogicHost> => {
  const createLuaModule = createLuaModuleFactory
    ? await createLuaModuleFactory()
    : await loadLuaModuleFactory()
  const lua = await createLuaModule(
    createLuaModuleOptions ?? {
      locateFile: (path: string) => new URL(path, getLuaAssetBaseUrl()).href
    }
  )
  const luaState = lua._luaL_newstate()

  if (!luaState) {
    throw new Error('Failed to create a Lua state.')
  }

  lua._luaL_openlibs(luaState)
  // JSON 코덱 + 디스패처(__logic_call_json)를 먼저 로드 — callJson이 이를 쓴다.
  runLuaChunk(lua, luaState, jsonCodecSource, '@json-codec.lua')
  let isClosed = false

  return {
    runModule: (source: string, chunkName: string): void => {
      if (isClosed) {
        throw new Error('Lua logic host is closed.')
      }

      runLuaChunk(lua, luaState, source, chunkName)
    },
    callNumbers: (
      functionName: string,
      args: LuaFunctionArgument[],
      resultCount = 1
    ): number[] => {
      if (isClosed) {
        throw new Error('Lua logic host is closed.')
      }

      return callLuaFunction(lua, luaState, functionName, args, resultCount)
    },
    callNumber: (
      functionName: string,
      ...args: LuaFunctionArgument[]
    ): number => {
      if (isClosed) {
        throw new Error('Lua logic host is closed.')
      }

      return callLuaFunction(lua, luaState, functionName, args, 1)[0]
    },
    callString: (
      functionName: string,
      ...args: LuaFunctionArgument[]
    ): string => {
      if (isClosed) {
        throw new Error('Lua logic host is closed.')
      }

      return callLuaFunctionForString(lua, luaState, functionName, args)
    },
    callJson: <Result = unknown>(
      functionName: string,
      ...args: unknown[]
    ): Result => {
      if (isClosed) {
        throw new Error('Lua logic host is closed.')
      }

      const resultJson = callLuaFunctionForString(lua, luaState, '__logic_call_json', [
        functionName,
        JSON.stringify(args)
      ])
      const parsed = JSON.parse(resultJson) as { value: Result }

      return parsed.value
    },
    close: (): void => {
      if (isClosed) {
        return
      }

      lua._lua_close(luaState)
      isClosed = true
    }
  }
}

const runLuaChunk = (
  lua: LuaModule,
  luaState: number,
  source: string,
  chunkName: string
): void => {
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
  args: LuaFunctionArgument[],
  resultCount = 0
): number[] => {
  const baseTop = lua._lua_gettop(luaState)

  pushGlobalFunction(lua, luaState, functionName)

  for (const argument of args) {
    if (typeof argument === 'number') {
      lua._lua_pushnumber(luaState, argument)
      continue
    }

    pushLuaString(lua, luaState, argument)
  }

  const callStatus = lua._lua_pcallk(luaState, args.length, resultCount, 0, 0, 0)

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

const callLuaFunctionForString = (
  lua: LuaModule,
  luaState: number,
  functionName: string,
  args: LuaFunctionArgument[]
): string => {
  const baseTop = lua._lua_gettop(luaState)

  pushGlobalFunction(lua, luaState, functionName)

  for (const argument of args) {
    if (typeof argument === 'number') {
      lua._lua_pushnumber(luaState, argument)
      continue
    }

    pushLuaString(lua, luaState, argument)
  }

  const callStatus = lua._lua_pcallk(luaState, args.length, 1, 0, 0, 0)

  if (callStatus !== 0) {
    const errorMessage = readLuaError(lua, luaState)

    lua._lua_settop(luaState, baseTop)
    throw new Error(`Lua function ${functionName} failed: ${errorMessage}`)
  }

  const valuePointer = lua._lua_tolstring(luaState, baseTop + 1, 0)
  const value = valuePointer === 0 ? '' : lua.UTF8ToString(Number(valuePointer))

  lua._lua_settop(luaState, baseTop)

  return value
}

const pushGlobalFunction = (
  lua: LuaModule,
  luaState: number,
  functionName: string
): void => {
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
): void => {
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

const loadLuaModuleFactory = async (): Promise<LuaModuleFactory> => {
  const luaModule = await import(/* @vite-ignore */ getLuaModuleUrl())

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
