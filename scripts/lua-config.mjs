import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptsDir = path.dirname(fileURLToPath(import.meta.url))

export const repoRoot = path.resolve(scriptsDir, '..')
export const luaVersion = '5.3.6'
export const luaReleaseName = `lua-${luaVersion}`
export const luaTarballUrl = `https://www.lua.org/ftp/${luaReleaseName}.tar.gz`
export const luaOfficialTestsVersion = '5.3.4'
export const luaOfficialTestsReleaseName = `lua-${luaOfficialTestsVersion}-tests`
export const luaOfficialTestsTarballUrl =
  `https://www.lua.org/tests/${luaOfficialTestsReleaseName}.tar.gz`
export const thirdPartyDir = path.join(repoRoot, 'third_party')
export const luaSourceDir = path.join(thirdPartyDir, luaReleaseName)
export const luaSrcDir = path.join(luaSourceDir, 'src')
export const luaOfficialTestsDir = path.join(
  thirdPartyDir,
  luaOfficialTestsReleaseName
)
export const luaOfficialTestsEntryFile = path.join(luaOfficialTestsDir, 'all.lua')
export const luaBuildOutputDir = path.join(repoRoot, 'public', 'vendor', 'lua')
export const luaModuleBaseName = luaReleaseName
export const luaModuleJsPath = path.join(
  luaBuildOutputDir,
  `${luaModuleBaseName}.mjs`
)
export const luaModuleWasmPath = path.join(
  luaBuildOutputDir,
  `${luaModuleBaseName}.wasm`
)
