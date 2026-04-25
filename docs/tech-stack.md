# Tech Stack

- Client and server runtime: Node.js
- Main game and server language: TypeScript
- Platform: web browsers (desktop and mobile)
- Rendering: PixiJS
- Tilemap rendering: `@pixi/tilemap`
- Map format: Tiled TMX/TSX
  - Use tile layers together with object-layer events.
  - `character` events use the `type` property to reference a `tiny-dungeon-16` appearance key.
- Game scripting: Lua 5.3.6
- Lua web runtime build: build the WebAssembly module with Emscripten.
- Lua controller contract: each script returns one controller table with reserved runtime methods such as `register`, `unregister`, `step`, and optional `interact`.
- During development, Vite HMR applies Lua controller source changes immediately, and the runtime reloads controller modules and reattaches current characters to the new Lua state.
- Lua compatibility check: run the official Lua 5.3 basic tests against the wasm runtime only through `npm run lua:test`.
- Bundler: Vite
- Tests: Vitest

## Resource Layout

- Keep static assets under `src/assets/`.
- Application code should load needed assets through Vite imports.
- Keep project Lua controller scripts under `src/assets/lua/`.
- Vendor the Lua source under `third_party/lua-5.3.6/`.
- Download the official Lua test suite into `third_party/lua-5.3.4-tests/` only when needed.
- Generate Lua wasm output under `public/vendor/lua/`.
