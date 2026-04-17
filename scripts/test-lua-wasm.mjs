import { access, readFile, readdir } from 'node:fs/promises'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

import {
  luaModuleJsPath,
  luaModuleWasmPath,
  luaOfficialTestsDir,
  luaOfficialTestsEntryFile
} from './lua-config.mjs'

const main = async () => {
  await assertPathExists(
    luaModuleJsPath,
    `Lua wasm module was not found at ${luaModuleJsPath}. Run npm run lua:build first.`
  )
  await assertPathExists(
    luaModuleWasmPath,
    `Lua wasm binary was not found at ${luaModuleWasmPath}. Run npm run lua:build first.`
  )
  await assertPathExists(
    luaOfficialTestsEntryFile,
    `Lua official tests were not found at ${luaOfficialTestsDir}. Run node scripts/fetch-lua-tests.mjs first.`
  )

  const { default: createLuaModule } = await import(pathToFileURL(luaModuleJsPath).href)
  const wasmBinary = await readFile(luaModuleWasmPath)
  const outputLines = []
  const errorLines = []
  const lua = await createLuaModule({
    wasmBinary,
    print: createLogHandler(outputLines),
    printErr: createLogHandler(errorLines)
  })
  const testFiles = await readLuaTestFiles(luaOfficialTestsDir)

  const luaState = lua._luaL_newstate()

  if (!luaState) {
    throw new Error('Failed to create a Lua state.')
  }

  try {
    lua._luaL_openlibs(luaState)
    runLuaChunk(lua, luaState, createLuaTestsBootstrap(testFiles))
  } finally {
    lua._lua_close(luaState)
  }

  if (!outputLines.some((line) => line.includes('final OK'))) {
    throw new Error('Lua official basic tests did not report final OK.')
  }

  console.log('Lua official basic tests passed.')
}

const createLogHandler = (lines) => (value) => {
  const line = String(value)
  lines.push(line)
  console.log(line)
}

const readLuaTestFiles = async (sourceDir, currentDir = sourceDir) => {
  const testFiles = {}
  const entries = await readdir(currentDir, { withFileTypes: true })

  for (const entry of entries) {
    const sourcePath = path.join(currentDir, entry.name)

    if (entry.isDirectory()) {
      Object.assign(testFiles, await readLuaTestFiles(sourceDir, sourcePath))
      continue
    }

    if (!entry.isFile() || !entry.name.endsWith('.lua')) {
      continue
    }

    const relativePath = path.relative(sourceDir, sourcePath).split(path.sep).join('/')
    testFiles[relativePath] = await readFile(sourcePath)
  }

  return testFiles
}

const runLuaChunk = (lua, luaState, source) => {
  const sourceLength = lua.lengthBytesUTF8(source)
  const sourcePointer = lua._malloc(sourceLength + 1)
  const chunkNamePointer = lua._malloc(18)

  lua.stringToUTF8(source, sourcePointer, sourceLength + 1)
  lua.stringToUTF8('@codex-lua-tests', chunkNamePointer, 18)

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
    throw new Error(`Lua tests failed: ${readLuaError(lua, luaState)}`)
  }
}

const readLuaError = (lua, luaState) => {
  const messagePointer = lua._lua_tolstring(luaState, -1, 0)
  const message =
    messagePointer === 0
      ? 'unknown Lua error'
      : lua.UTF8ToString(Number(messagePointer))

  lua._lua_settop(luaState, -2)
  return message
}

const assertPathExists = async (targetPath, errorMessage) => {
  try {
    await access(targetPath)
  } catch {
    throw new Error(errorMessage)
  }
}

const createLuaTestsBootstrap = (testFiles) => {
  const fileEntries = Object.entries(testFiles)
    .sort(([leftName], [rightName]) => leftName.localeCompare(rightName))
    .map(
      ([fileName, source]) =>
        `  ["${escapeLuaText(fileName)}"] = ${escapeLuaBytes(source)}`
    )
    .join(',\n')

  return `
local __codex_files = {
${fileEntries}
}

local __codex_original_loadfile = loadfile

local function __codex_normalize_source(source)
  if string.byte(source, 1) == 35 then
    return (string.gsub(source, "^[^\\n]*\\n", ""))
  end

  return source
end

function loadfile(filename, ...)
  local argumentCount = select("#", ...)
  local mode = select(1, ...)
  local env = select(2, ...)
  local source = __codex_files[filename]

  if source ~= nil then
    source = __codex_normalize_source(source)

    if argumentCount >= 2 then
      return load(source, "@" .. filename, mode or "bt", env)
    end

    return load(source, "@" .. filename, mode or "bt")
  end

  if argumentCount >= 2 then
    return __codex_original_loadfile(filename, mode, env)
  end

  if argumentCount >= 1 then
    return __codex_original_loadfile(filename, mode)
  end

  return __codex_original_loadfile(filename)
end

function dofile(filename)
  local chunk, message = loadfile(filename)

  if not chunk then
    error(message, 2)
  end

  return chunk()
end

if os and os.setlocale then
  local __codex_original_setlocale = os.setlocale

  os.setlocale = function(locale, category)
    if locale == nil or locale == "C" then
      return __codex_original_setlocale(locale, category)
    end

    return nil
  end
end

_U = true
dofile("all.lua")
`
}

const escapeLuaText = (value) =>
  value
    .replace(/\\/g, '\\\\')
    .replace(/\r/g, '\\r')
    .replace(/\n/g, '\\n')
    .replace(/\t/g, '\\t')
    .replace(/\f/g, '\\f')
    .replace(/\v/g, '\\v')
    .replace(/"/g, '\\"')

const escapeLuaBytes = (value) => {
  let escapedValue = '"'

  for (const byte of value) {
    if (byte === 34) {
      escapedValue += '\\"'
      continue
    }

    if (byte === 92) {
      escapedValue += '\\\\'
      continue
    }

    if (byte === 10) {
      escapedValue += '\\n'
      continue
    }

    if (byte === 13) {
      escapedValue += '\\r'
      continue
    }

    if (byte === 9) {
      escapedValue += '\\t'
      continue
    }

    if (byte >= 32 && byte <= 126) {
      escapedValue += String.fromCharCode(byte)
      continue
    }

    escapedValue += `\\x${byte.toString(16).padStart(2, '0')}`
  }

  escapedValue += '"'
  return escapedValue
}

await main()
