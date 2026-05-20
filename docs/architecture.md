# Architecture

This document is a short guide for module boundaries and code placement.

## Current Structure

- `src/main.ts`: web entry point. Compose the application, bootstrap scenes, and route portal-driven scene transitions here.
- `src/assets/`: runtime assets that are imported by the web client.
- `src/assets/lua/`: project Lua controller scripts that are loaded by the web client at runtime.
- `src/game/`: pure game or engine logic that should stay easy to test with Vitest.
- `src/game/characterState.ts`: shared character state, optional monster level metadata, optional fixed sign text, controller decisions, and movement rules for both the player and NPCs.
- `src/game/createCharacterControllerRuntime.ts`: controller attachment lifecycle, shared movement dispatch, and Lua script hot-update coordination.
- `src/game/events/`: small central queue models for frame-level game events.
- `src/game/interaction/`: target resolution and event processing for interaction flow.
- `src/game/lua/`: Lua wasm bridge code that loads controller modules, exposes the public `engine.*` Lua API, and evaluates scripts for runtime characters.
- `src/game/lua/luaControllerApi.ts`: source of truth for the Lua-visible controller contract.
- `src/game/blacksmithShop.ts`: blacksmith merchant inventory, buy/sell trade rules, and stock initialization for the shop NPC.
- `src/game/monsterCombat.ts`: monster HP, contact damage, and defeat-state rules shared by combat scenes.
- `src/game/monsterRewards.ts`: monster reward amounts such as beginner gold and experience drops.
- `src/game/playerExperience.ts`: player experience gain, level-up application, and the level 100 cap.
- `src/game/playerEquipment.ts`: player-facing equipment state, starter gear data, blacksmith gear data, item price metadata, and item icon metadata used by the combined player panel.
- `src/game/playerProfile.ts`: player-facing name, starter beginner class, 10-level promotion check, level 100 cap, future job-to-primary-stat mapping, resource, exp, stat, and skill data used by the HUD.
- `src/game/playerInventory.ts`: shared gold-and-slot inventory shape, including the player backpack and NPC trade inventories, plus slot mutation helpers.
- `src/game/playerLoadout.ts`: pure equip and unequip transitions between the backpack and the starter gear slots.
- `src/game/playerProgression.ts`: level-up rewards, stat-point spending, and skill-point spending for the player profile.
- `src/game/playerStatEffects.ts`: derived player stat effects for physical attack, movement speed, and evade chance.
- `src/game/sceneIntro.ts`: scene-id to localized intro text mapping for the temporary map transition banner.
- `src/game/tiled/`: TMX/TSX parsing, tile metadata, and event-layer data extraction.
- `src/game/tiled/createNpcCharactersFromEventLayers.ts`: translate `character` object-layer events plus `controller.*`, `monster.level`, and optional `displayText` TMX properties into shared NPC character state.
- `src/game/tiled/createMapPortalsFromEventLayers.ts`: translate `portal` object-layer events into scene transition data for map exits and entrances.
- `src/rendering/`: PixiJS rendering code and asset-to-view adaptation.
- `src/rendering/`: map tile rendering, depth sorting, event character presentation, and fixed-screen HUD overlays.
- `src/rendering/loadMonsterSheetTextures.ts`: shared sheet slicing and background-keying helper for monster sprite sheets.
- `src/rendering/loadMonsterPigAnimationTextures.ts`, `src/rendering/loadMonsterSlimeAnimationTextures.ts`: sprite-sheet slicing for the beginner monster appearances.
- `src/rendering/getResponsiveUiScale.ts`: shared viewport-based UI scale used by all fixed-screen overlays so they scale together across viewport sizes, with the current default tuned to 1.2x.
- `src/rendering/createPixiTiledMapView.ts`: owns the live world scene, fixed 16:9 game viewport, character-follow camera zoom, character sprite placement, fixed sign posts and labels, the player weapon sprite attached to the player render node, the basic attack swing and afterimage trail state, the player hit recoil motion, the blacksmith shop open state, the temporary scene intro banner, the beginner monster appearance-specific animation hookup for pigs and slimes, the monster level text and HP bar rendering, the player stat effect application for damage, movement speed, and evade chance, the monster gold and experience reward flow, and portal-triggered scene transition requests.
- `src/rendering/createPlayerHudOverlay.ts`: compact fixed-screen character status bar anchored near the bottom center with bag icon, skill slots, and the experience bar. The bag icon opens the combined player panel.
- `src/rendering/createPlayerInventoryOverlay.ts`: combined fixed-screen player panel that shows the equipment layout, a centered player portrait preview, the backpack grid, and current gold, with click-to-equip and click-to-unequip behavior.
- `src/rendering/createPlayerStatOverlay.ts`: compact fixed-screen player stat side panel opened with `S` that spends stat points on strength, agility, intelligence, and luck, and highlights the matching job primary stat without covering the world center as much.
- `src/rendering/createPlayerSkillOverlay.ts`: fixed-screen player skill window opened with `K` that spends skill points on individual skill levels.
- `src/rendering/createBlacksmithShopOverlay.ts`: fixed-screen blacksmith service overlay that starts with a service menu and then shows the blacksmith trade modal with portrait headers, category tabs, and side-by-side purchase/sale lists for buy/sell clicks.
- `src/rendering/createPixiTiledMapView.ts`: also resolves basic player-versus-monster combat, including player attack hits, monster contact damage, player death and respawn timing, monster aggro, auto-aggro at close range, monster attack timing, hit recoil, respawn timing, defeat visibility, floating damage text, and gold drops.
- `scripts/`: project automation scripts such as third-party fetch/build steps.
- `third_party/`: vendored external source code kept in-repo for deterministic builds.
- `public/vendor/`: generated static artifacts served as-is by Vite.

## Lua Controller Interface

- `src/assets/lua/`: each Lua controller script should return one controller table.
- Reserved controller methods:
  - `register`: optional setup when the runtime attaches the script to one character
  - `unregister`: optional cleanup when the runtime detaches the script from one character
  - `step`: required movement-intent update
  - `interact`: optional response when another character interacts with this character
- `src/main.ts`: map only `scriptId -> source`. Do not repeat Lua method names in the app entry point.
- TMX owns which NPC uses which controller.
  Use `controller.kind`, `controller.scriptId`, `controller.radiusInTiles`, and `controller.moveSpeedTilesPerSecond` on `character` events.
  Treat other `controller.*` properties as controller config values.
- `src/game/lua/createLuaCharacterControllerRuntime.ts`: own Lua module loading, method dispatch, script reload, and wasm bridge details.
- `src/game/lua/luaControllerApi.ts`: own the public Lua-visible contract and reserved method names.
- `docs/lua-controller-api.md`: explain the same contract in human-readable form for AI and script authors.
- Validate Lua controller scripts before first load and hot reload. Use compile checking plus isolated contract validation so a broken script does not replace the active runtime.
- Keep the public Lua surface narrow. Expose new data or actions through `engine.*` only after updating both the TypeScript contract and the Lua API document.
- Let Lua return movement intent and request explicit `engine.*` actions only. Keep collision, target resolution, cooldown, event ordering, and final rendering effects outside Lua.
- Use `engine.self.get_controller_config()` when Lua needs TMX-authored per-character settings such as dialogue lists or small behavior flags.

## Boundary Rules

- Keep browser DOM code out of `src/game/` when possible.
- Prefer pure data and pure functions in game logic so tests stay small and stable.
- Keep controller definitions as plain data on the character state. Put attach, detach, and runtime-side script management in the controller runtime layer.
- When a rendering library is introduced later, keep rendering concerns separate from core game rules.
