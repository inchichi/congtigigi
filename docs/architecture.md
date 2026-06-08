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
- `src/game/monsterDisplayName.ts`: monster name formatting for on-map labels and HUD badges.
- `src/game/monsterRewards.ts`: monster reward amounts such as beginner gold, experience, and level-based skill-point drops.
- `src/game/questLog.ts`: pure quest definitions, quest progress state transitions, prerequisite checks, objective event matching, tracker visibility, NPC badge state, dialogue text formatting, and reward data for the 티르코네일 beginner quest arc.
- `src/game/playerExperience.ts`: player experience gain, level-up application, and the level 100 cap.
- `src/game/playerEquipment.ts`: player-facing equipment state, starter gear data, blacksmith gear data, item price metadata, and item icon metadata used by the combined player panel.
- `src/game/playerProfile.ts`: player-facing name, starter beginner class, 10-level promotion check, level 100 cap, future job-to-primary-stat mapping, resource, exp, stat, and skill data used by the HUD, plus available and total skill-point tracking for the skill window.
- `src/game/playerSkills.ts`: executable skill metadata, including skill-bar display lookup helpers, skill-level-based mana costs, smash damage, protect duration, skill unlock checks, and icon lookup data.
- `src/game/playerSkillSlots.ts`: Q/W/E/R skill bar slot state, slot assignment helpers, and keyboard slot lookup helpers used by the HUD and world input handler.
- `src/game/playerInventory.ts`: shared gold-and-slot inventory shape, including the player backpack and NPC trade inventories, plus slot mutation helpers.
- `src/game/playerQuickslots.ts`: quickslot bindings for inventory consumables, including assignment, clearing, and slot lookup helpers used by the HUD and inventory view.
- `src/game/playerControls.ts`: default player control bindings, key-code matching, display labels, and remap helpers used by the pause menu and world input handler.
- `src/game/playerLoadout.ts`: pure equip and unequip transitions between the backpack and the starter gear slots.
- `src/game/playerProgression.ts`: level-up rewards, stat-point spending, monster skill-point rewards, skill-user-level and mana-cap helpers, and level-based skill-point spending for the player profile.
- `src/game/playerStatEffects.ts`: derived player stat effects for physical attack, movement speed, and evade chance.
- `src/game/sceneIntro.ts`: scene-id to localized intro text mapping for the temporary map transition banner.
- `src/game/tiled/`: TMX/TSX parsing, tile metadata, and event-layer data extraction.
- `src/game/tiled/createNpcCharactersFromEventLayers.ts`: translate `character` object-layer events plus `controller.*`, `monster.level`, and optional `displayText` TMX properties into shared NPC character state.
- `src/game/tiled/createMapPortalsFromEventLayers.ts`: translate `portal` object-layer events into scene transition data for map exits and entrances.
- `src/editor/`: the LLM game-content editor, kept separate from the game runtime. It reads a game-structure profile, generates an event draft, validates it, and hands a `HolidayDialogueEventSpec` to the game through the scene renderer's `applyEventDraft` sink. It depends on the game only through that contract (`HolidayDialogueEventSpec`, `ApplyEventDraftResult`), not game internals.
- `src/editor/createLlmPanel.ts`: in-game DOM overlay opened with `L` that drives the editor flow — API-key gate, game-structure analysis and GUS readiness gate, natural-language to event JSON generation, field editing and validation, code preview, and apply-to-scene.
- `src/editor/openaiEventJsonGenerator.ts`: real OpenAI Responses API call that returns a validated `GeneratedEventJson`, injecting the current game-structure profile (existing map/npc/item ids) into the prompt so generated events reference real assets.
- `src/editor/gameStructureProfile.ts`, `src/editor/currentGameProjectSnapshot.ts`, `src/editor/gameStructureAnalyzer.ts`, `src/editor/gusCalculator.ts`: the game-structure model, the current (static) project snapshot, the analysis step, and the GUS readiness score used as a generation gate.
- `src/editor/eventJsonSchema.ts`, `src/editor/eventJsonGenerator.ts`, `src/editor/eventCodeGenerator.ts`: the generated-event JSON shape plus its deterministic validator, the rule-based mock generator (kept for tests and offline fallback), and the code/spec preview helpers.
- `src/rendering/`: PixiJS rendering code and asset-to-view adaptation.
- `src/rendering/`: map tile rendering, depth sorting, event character presentation, and fixed-screen HUD overlays.
- `src/rendering/loadMonsterSheetTextures.ts`: shared sheet slicing and background-keying helper for monster sprite sheets.
- `src/rendering/loadMonsterPigAnimationTextures.ts`, `src/rendering/loadMonsterSlimeAnimationTextures.ts`: sprite-sheet slicing for the beginner monster appearances.
- `src/rendering/getResponsiveUiScale.ts`: shared viewport-based UI scale used by all fixed-screen overlays so they scale together across viewport sizes, with the current default tuned to 1.2x.
- `src/rendering/createPixiTiledMapView.ts`: owns the live world scene, fixed 16:9 game viewport, character-follow camera zoom, top-left map visibility hotkey state, grass footstep loop state, character sprite placement, fixed sign posts and labels, the player weapon sprite attached to the player render node, the basic attack swing and afterimage trail state, Q/W/E/R skill-slot execution, the player hit recoil motion, the blacksmith shop open state, the pause menu open state, player control remapping state, the temporary scene intro banner, multi-NPC quest interaction hookup, the beginner monster appearance-specific animation hookup for pigs and slimes, the monster level text and HP bar rendering, the player stat effect application for damage, movement speed, and evade chance, the monster gold, experience, and skill-point reward flow, and F-key portal scene transition requests; skill execution checks unlock state and skill-level mana before firing.
- `src/rendering/createPlayerHudOverlay.ts`: compact fixed-screen character status bar anchored near the bottom center with bag icon, Q/W/E/R skill slots, and the experience bar. The bag icon opens the backpack panel, and the skill slots accept dragged skills from the skill window.
- `src/rendering/createPlayerInventoryOverlay.ts`: fixed-screen backpack window that uses compact Maple-style category tabs, shows the backpack grid and current gold, shows item details only in a hover tooltip, supports click-to-equip plus drag-to-equip for gear and quickslot dragging for consumables, and stays non-modal.
- `src/rendering/createPlayerEquipmentOverlay.ts`: fixed-screen equipment window opened with `U` that shows the current loadout, portrait preview, and accepts dragged gear from the backpack anywhere in the window in addition to click-to-unequip behavior, and it stays non-modal.
- `src/rendering/createPlayerStatOverlay.ts`: compact fixed-screen player stat side panel opened with `S` that spends stat points on strength, agility, intelligence, and luck, and stays non-modal so other player windows and gameplay can continue.
- `src/rendering/createPlayerSkillOverlay.ts`: fixed-screen player skill window opened with `K` that spends available skill points on individual skill levels while staying non-modal, shows the user level derived from total skill points earned, and exposes only unlocked skills as drag sources for the Q/W/E/R slots.
- `src/rendering/createBlacksmithShopOverlay.ts`: fixed-screen blacksmith service overlay that starts with a service menu and then shows the blacksmith trade view with portrait headers, category tabs, and side-by-side purchase/sale lists without blocking the rest of the game UI.
- `src/rendering/createPauseMenuOverlay.ts`: fixed-screen pause/audio menu overlay with resume, BGM volume, SFX volume, and a separate key-binding editor screen.
- `src/rendering/createQuestLogOverlay.ts`: fixed-screen quest window opened with `B`, showing accepted quest lists, quest details, objective progress, tracker toggles, and abandon confirmation.
- `src/rendering/createQuestTrackerOverlay.ts`: fixed-screen quest tracker panel that displays quests whose per-quest tracker visibility is enabled, with a close button that hides tracker entries without changing quest progress.
- `src/rendering/createPixiTiledMapView.ts`: also resolves basic player-versus-monster combat, including player attack hits, monster contact damage, player death and respawn timing, monster aggro, auto-aggro at close range, monster attack timing, hit recoil, respawn timing, defeat visibility, floating damage text, gold drops, and the player protect shield visual.
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
