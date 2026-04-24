import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['src/game/lua/createLuaCharacterControllerRuntime.bridge.spec.ts']
  }
})
