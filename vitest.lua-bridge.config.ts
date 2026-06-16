import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: [
      'src/games/my-sample-rpg/lua/createLuaCharacterControllerRuntime.bridge.spec.ts'
    ]
  }
})
