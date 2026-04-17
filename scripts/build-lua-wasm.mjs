import { access, mkdir } from 'node:fs/promises'
import { readdirSync } from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

import {
  luaBuildOutputDir,
  luaModuleJsPath,
  luaModuleWasmPath,
  luaReleaseName,
  luaSourceDir,
  luaSrcDir
} from './lua-config.mjs'

const exportedFunctions = [
  '_luaL_newstate',
  '_luaL_openlibs',
  '_luaL_loadstring',
  '_luaL_loadbufferx',
  '_luaL_loadfilex',
  '_lua_close',
  '_lua_pcallk',
  '_lua_settop',
  '_lua_gettop',
  '_lua_tolstring',
  '_lua_type',
  '_lua_typename',
  '_lua_getglobal',
  '_lua_setglobal',
  '_lua_pushlstring',
  '_lua_pushnumber',
  '_lua_pushinteger',
  '_lua_pushboolean',
  '_lua_toboolean',
  '_lua_tonumberx',
  '_lua_tointegerx',
  '_lua_rawlen',
  '_malloc',
  '_free'
]

const exportedRuntimeMethods = [
  'ccall',
  'cwrap',
  'FS',
  'UTF8ToString',
  'stringToUTF8',
  'lengthBytesUTF8'
]

const main = async () => {
  await assertPathExists(luaSrcDir, `Lua source was not found at ${luaSourceDir}`)

  const emccCheck = spawnSync('emcc', ['--version'], { stdio: 'ignore' })

  if (emccCheck.error || emccCheck.status !== 0) {
    throw new Error(
      'emcc was not found in PATH. Install Emscripten and run this script again.'
    )
  }

  await mkdir(luaBuildOutputDir, { recursive: true })

  const sourceFiles = readdirSync(luaSrcDir)
    .filter((fileName) => fileName.endsWith('.c'))
    .filter((fileName) => fileName !== 'lua.c' && fileName !== 'luac.c')
    .map((fileName) => path.join(luaSrcDir, fileName))
    .sort()

  const buildResult = spawnSync(
    'emcc',
    [
      ...sourceFiles,
      '-I',
      luaSrcDir,
      '-O3',
      '--no-entry',
      '-sWASM=1',
      '-sALLOW_MEMORY_GROWTH=1',
      '-sENVIRONMENT=web,worker',
      '-sMODULARIZE=1',
      '-sEXPORT_ES6=1',
      '-sEXPORT_NAME=createLuaModule',
      `-sEXPORTED_FUNCTIONS=${JSON.stringify(exportedFunctions)}`,
      `-sEXPORTED_RUNTIME_METHODS=${JSON.stringify(exportedRuntimeMethods)}`,
      '-o',
      luaModuleJsPath
    ],
    { stdio: 'inherit' }
  )

  if (buildResult.status !== 0) {
    throw new Error(`emcc exited with status ${buildResult.status ?? 'unknown'}`)
  }

  console.log(`Built ${luaReleaseName} wasm module:`)
  console.log(`- ${luaModuleJsPath}`)
  console.log(`- ${luaModuleWasmPath}`)
}

const assertPathExists = async (targetPath, errorMessage) => {
  try {
    await access(targetPath)
  } catch {
    throw new Error(errorMessage)
  }
}

await main()
