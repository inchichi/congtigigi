# Architecture

This document is a short guide for module boundaries and code placement.

## Current Structure

- `src/main.ts`: web entry point. Compose the application and connect browser-specific code here.
- `src/assets/`: runtime assets that are imported by the web client.
- `src/game/`: pure game or engine logic that should stay easy to test with Vitest.
- `src/rendering/`: PixiJS rendering code and asset-to-view adaptation.
- `scripts/`: project automation scripts such as third-party fetch/build steps.
- `third_party/`: vendored external source code kept in-repo for deterministic builds.
- `public/vendor/`: generated static artifacts served as-is by Vite.

## Boundary Rules

- Keep browser DOM code out of `src/game/` when possible.
- Prefer pure data and pure functions in game logic so tests stay small and stable.
- When a rendering library is introduced later, keep rendering concerns separate from core game rules.
