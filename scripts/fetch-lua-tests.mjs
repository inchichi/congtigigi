import { createWriteStream } from 'node:fs'
import { access, cp, mkdir, mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { spawnSync } from 'node:child_process'

import {
  luaOfficialTestsDir,
  luaOfficialTestsEntryFile,
  luaOfficialTestsReleaseName,
  luaOfficialTestsTarballUrl,
  thirdPartyDir
} from './lua-config.mjs'

const main = async () => {
  if (await pathExists(luaOfficialTestsEntryFile)) {
    console.log(
      `Lua ${luaOfficialTestsReleaseName} already exists at ${luaOfficialTestsDir}`
    )
    return
  }

  await mkdir(thirdPartyDir, { recursive: true })

  const tempDir = await mkdtemp(path.join(os.tmpdir(), 'lua-tests-fetch-'))
  const tarballPath = path.join(tempDir, `${luaOfficialTestsReleaseName}.tar.gz`)

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

    await rm(luaOfficialTestsDir, { recursive: true, force: true })
    await cp(path.join(tempDir, luaOfficialTestsReleaseName), luaOfficialTestsDir, {
      recursive: true
    })

    console.log(
      `Fetched Lua ${luaOfficialTestsReleaseName} into ${luaOfficialTestsDir}`
    )
  } finally {
    await rm(tempDir, { recursive: true, force: true })
  }
}

const downloadTarball = async (tarballPath) => {
  const response = await fetch(luaOfficialTestsTarballUrl)

  if (!response.ok || !response.body) {
    throw new Error(`Failed to download ${luaOfficialTestsTarballUrl}`)
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
