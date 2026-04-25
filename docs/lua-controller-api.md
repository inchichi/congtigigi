# Lua Controller API

This document is the human-readable guide for Lua character controllers.

The code source of truth is:

- `src/game/lua/luaControllerApi.ts`

When you write or change a Lua controller, use this document and that TypeScript file first.
Do not assume hidden engine APIs.

## Purpose

Lua controllers may decide how one character wants to move.
Lua controllers may also decide how that character responds when another character interacts with it.
Lua controllers may request limited engine-side actions through the shared `engine.*` API.

Lua controllers may not directly change:

- the map
- other characters
- rendering state
- browser APIs
- network state

If a Lua controller needs more data, extend `src/game/lua/luaControllerApi.ts` and the runtime bridge first.
Do not invent extra globals or side channels inside Lua.

## Public Lua Namespace

Lua-visible helper functions live under the `engine` global namespace.
New APIs should follow the same pattern, such as `engine.ui.*` or `engine.map.*`.

### `engine.self.get_controller_config`

```lua
local config = engine.self.get_controller_config()
```

Rules:

- this returns the current character's TMX-authored controller config
- only properties with the `controller.` prefix are exposed here
- reserved setup keys such as `controller.kind` and `controller.scriptId` are not included
- `type="list"` properties become Lua string, number, or boolean lists
- treat the returned table as read-only input

### `engine.ui.show_message`

```lua
engine.ui.show_message(message, duration_seconds)
```

Rules:

- this uses the current controller callback context, so the bubble is attached to the character that is currently running Lua code
- `message` should be a non-empty string
- `duration_seconds` is optional
- Lua does not render the bubble directly
- the runtime converts this request into a shared game event, and TypeScript decides how to render it

## Controller Module Shape

Each Lua controller script must return one controller table.
Method names are reserved by the runtime and are not configured in TypeScript per script.

```lua
local controller = {}

function controller.register(id, home_x, home_y, radius_in_tiles)
end

function controller.unregister(id)
end

function controller.step(id, delta_seconds, x, y)
  return 0, 0
end

function controller.interact(id, source_character_id)
end

return controller
```

Lifecycle rules:

- `register`: optional setup for one character when the controller is attached
- `unregister`: optional cleanup for one character when the controller is detached
- `step`: required per-frame update that returns movement intent
- `interact`: optional callback when another character interacts with this character
- keep module top-level code simple and side-effect free
- the runtime validates Lua scripts on first load and hot reload before they replace the active controller
- validation errors should include the Lua line number and line text when that information is available

## TMX Controller Schema

For `character` object-layer events:

- `type`: appearance key from `tiny-dungeon-16`
- `blocksMovement`: optional bool
- `controller.kind`: optional, `idle` or `lua`
  - you can usually omit this because `controller.scriptId` already implies `lua`
- `controller.scriptId`: required when the controller kind is `lua`
- `controller.radiusInTiles`: optional number for Lua movement radius
- `controller.moveSpeedTilesPerSecond`: optional number
- any other `controller.*` property becomes one entry in `engine.self.get_controller_config()`
- use `type="list"` when a controller setting should become a Lua array

Example:

```xml
<object name="blacksmith" type="character" x="400" y="516">
  <properties>
    <property name="type" value="character_bearded_apron_man"/>
    <property name="controller.scriptId" value="reply-with-message"/>
    <property name="controller.dialogueLines" type="list">
      <item value="Need any tools?"/>
      <item value="Best steel in town."/>
    </property>
    <property name="controller.messageDurationSeconds" type="float" value="2.8"/>
  </properties>
</object>
```

## Public Data Model

Lua currently receives only a narrow snapshot of its own character state.

### Register Input

- `character.id`
- `character.position.x`
- `character.position.y`
- `radiusInTiles`

This position is the home position captured when the controller is attached.

### Unregister Input

- `characterId`

### Step Input

- `character.id`
- `deltaSeconds`
- `character.position.x`
- `character.position.y`

This position is the current character position for the current frame.

### Interact Input

- `character.id`
- `sourceCharacterId`

This callback runs only after TypeScript resolves the target character.
Lua does not choose the target by itself.

## Movement Result

The `controller.step` method must return two numbers:

- `moveX`
- `moveY`

Rules:

- return `0, 0` when the character should stay still
- return a direction or intent, not a final teleported position
- the engine applies movement speed, collision, and bounds outside Lua

## Interaction Result

The `controller.interact` method may still return:

- `message`
- `durationSeconds`

Rules:

- return `nil, nil` when there is no response
- this is kept for compatibility with older scripts
- prefer `engine.ui.show_message(...)` for new scripts
- do not mix `engine.ui.show_message(...)` and a direct message return in the same callback unless you intentionally want multiple results
- the engine decides target selection, cooldown, event ordering, and speech bubble rendering outside Lua

## Current Runtime Method Signatures

The current runtime bridge still calls the reserved methods with positional arguments.

### Register

```lua
controller.register(id, home_x, home_y, radius_in_tiles)
```

### Unregister

```lua
controller.unregister(id)
```

### Step

```lua
move_x, move_y = controller.step(id, delta_seconds, x, y)
```

### Interact

```lua
message, duration_seconds = controller.interact(id, source_character_id)
```

These positional arguments are a serialized form of the models from `src/game/lua/luaControllerApi.ts`.
The `engine` global namespace is also available inside these methods.

## AI Guidance

When you ask an AI assistant to write or modify a Lua controller:

1. Tell it to read `docs/lua-controller-api.md`.
2. Tell it to read `src/game/lua/luaControllerApi.ts`.
3. Tell it not to assume any Lua-visible API beyond that contract.
4. Tell it to request a contract change first if more game data is needed.
