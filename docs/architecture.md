# Architecture

This document is a short guide for module boundaries and code placement.

## Current Structure

- `src/main.ts`: web entry point. Compose the application and connect browser-specific code here.
- `src/assets/`: runtime assets that are imported by the web client.
- `src/assets/lua/`: project Lua controller scripts that are loaded by the web client at runtime.
- `src/game/`: pure game or engine logic that should stay easy to test with Vitest.
- `src/game/characterState.ts`: shared character state, controller decisions, and movement rules for both the player and NPCs.
- `src/game/createCharacterControllerRuntime.ts`: controller attachment lifecycle, shared movement dispatch, and Lua script hot-update coordination.
- `src/game/lua/`: Lua wasm bridge code that loads controller scripts and evaluates them for runtime characters.
- `src/game/lua/luaControllerApi.ts`: source of truth for the Lua-visible controller contract.
- `src/game/tiled/`: TMX/TSX parsing, tile metadata, and event-layer data extraction.
- `src/rendering/`: PixiJS rendering code and asset-to-view adaptation.
- `src/rendering/`: map tile rendering, depth sorting, and event character presentation.
- `scripts/`: project automation scripts such as third-party fetch/build steps.
- `third_party/`: vendored external source code kept in-repo for deterministic builds.
- `public/vendor/`: generated static artifacts served as-is by Vite.

## Boundary Rules

- Keep browser DOM code out of `src/game/` when possible.
- Prefer pure data and pure functions in game logic so tests stay small and stable.
- Keep controller definitions as plain data on the character state. Put attach, detach, and runtime-side script management in the controller runtime layer.
- Keep the Lua controller surface narrow. Change `src/game/lua/luaControllerApi.ts` and `docs/lua-controller-api.md` before exposing new engine data to Lua.
- When a rendering library is introduced later, keep rendering concerns separate from core game rules.
