import { defineConfig } from 'vitest/config'

// 실제 Lua WASM VM을 node에서 로드하는 로직-호스트 브리지 테스트 전용 설정.
// (기본 `vitest run`은 *.test.ts만 포함하므로 *.bridge.spec.ts는 여기서 따로 돌린다.)
export default defineConfig({
  test: {
    include: [
      'src/games/my-sample-rpg/lua/*Lua.bridge.spec.ts',
      'src/games/my-sample-rpg/lua/luaGameLogic.bridge.spec.ts',
      'src/games/my-sample-rpg/lua/luaLogicHost.bridge.spec.ts'
    ]
  }
})
