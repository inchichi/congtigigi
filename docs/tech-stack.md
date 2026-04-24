# Tech Stack

- 클라이언트/서버: Node.js
- 게임 기반 및 서버 개발 언어: TypeScript
- 플랫폼: 웹브라우저 (데스크탑 / 모바일)
- 렌더링: PixiJS
- 타일맵 렌더링: `@pixi/tilemap`
- 맵 포맷: Tiled TMX/TSX
  - 타일 레이어와 object layer 기반 이벤트를 함께 사용한다.
  - `character` 이벤트는 `type` property 로 `tiny-dungeon-16` 외형 키를 참조한다.
- 게임 스크립팅: Lua 5.3.6
- Lua 웹 런타임 빌드: Emscripten 으로 WebAssembly 모듈을 만든다.
- Lua 호환성 확인: `npm run lua:test` 에서만 공식 Lua 5.3 basic tests 를 wasm 런타임에 대해 실행한다.
- 번들러: Vite
- 테스트: Vitest

## 리소스 패킹

- 정적 리소스는 `src/assets/` 아래에 둔다.
- 애플리케이션 코드에서는 Vite import 를 통해 필요한 리소스를 가져온다.
- Lua 소스는 `third_party/lua-5.3.6/` 아래에 vendoring 한다.
- Lua 공식 테스트 스위트는 `third_party/lua-5.3.4-tests/` 아래에 필요할 때만 내려받는다.
- Lua wasm 산출물은 `public/vendor/lua/` 아래에 생성한다.
