import { createWriteStream } from 'node:fs'
import { access, cp, mkdir, mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { spawnSync } from 'node:child_process'

import {
  luaReleaseName,
  luaSourceDir,
  luaSrcDir,
  luaTarballUrl,
  thirdPartyDir
} from './lua-config.mjs'

const main = async () => {
  if (await pathExists(luaSrcDir)) {
    console.log(`Lua ${luaReleaseName} already exists at ${luaSourceDir}`)
    return
  }

  await mkdir(thirdPartyDir, { recursive: true })

  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'lua-fetch-'))
  const tarballPath = path.join(tempDir, `${luaReleaseName}.tar.gz`)

  try {
    await downloadTarball(tarballPath)

    const extractResult = spawnSync(
      'tar',
      ['-xzf', tarballPath, '-C', tempDir],
      { stdio: 'inherit' }
    )

    if (extractResult.status !== 0) {
      throw new Error(`tar exited with status ${extractResult.status ?? 'unknown'}`)
    }

    await rm(luaSourceDir, { recursive: true, force: true })
    await cp(path.join(tempDir, luaReleaseName), luaSourceDir, { recursive: true })

    console.log(`Fetched Lua ${luaReleaseName} into ${luaSourceDir}`)
  } finally {
    await rm(tempDir, { recursive: true, force: true })
  }
}

const downloadTarball = async (tarballPath) => {
  const response = await fetch(luaTarballUrl)

  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${luaTarballUrl}`)
  }

  await pipeline(
    Readable.fromWeb(response.body),
    createWriteStream(tarballPath)
  )
}

const pathExists = async (targetPath) => {
  try {
    await access(targetPath)
    return true
  } catch {
    return false
  }
}

await main()
