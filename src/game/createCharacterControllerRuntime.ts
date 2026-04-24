import {
  getCharacterControllerDelta,
  type CharacterMoveDirection,
  type CharacterState
} from './characterState'
import type {
  LuaCharacterControllerRuntime,
  LuaControllerScriptSource
} from './lua/createLuaCharacterControllerRuntime'

type CharacterControllerAttachment = {
  attachCharacter: (character: CharacterState) => void
  detachCharacter: (character: CharacterState) => void
  getMovementDelta: (input: {
    character: CharacterState
    deltaMilliseconds: number
    pressedDirections?: ReadonlySet<CharacterMoveDirection>
  }) => { x: number; y: number } | undefined
  destroy?: () => void
}

type CreateCharacterControllerRuntimeInput = {
  luaControllerRuntime?: LuaCharacterControllerRuntime
}

export type CharacterControllerRuntime = {
  syncCharacters: (characters: CharacterState[]) => void
  getMovementDelta: (input: {
    character: CharacterState
    deltaMilliseconds: number
    pressedDirections?: ReadonlySet<CharacterMoveDirection>
  }) => { x: number; y: number } | undefined
  updateLuaControllerScript: (
    scriptId: string,
    script: LuaControllerScriptSource
  ) => void
  destroy: () => void
}

export const createCharacterControllerRuntime = ({
  luaControllerRuntime
}: CreateCharacterControllerRuntimeInput = {}): CharacterControllerRuntime => {
  const attachedCharacters = new Map<string, CharacterState>()
  const keyboardAttachment = createKeyboardControllerAttachment()
  const npcAttachment = createNpcControllerAttachment()
  const luaAttachment = createLuaControllerAttachment(luaControllerRuntime)
  let isDestroyed = false

  const resolveAttachment = (
    character: CharacterState
  ): CharacterControllerAttachment => {
    switch (character.controller.kind) {
      case 'keyboard':
        return keyboardAttachment
      case 'npc':
        return npcAttachment
      case 'lua':
        return luaAttachment
    }
  }

  const syncCharacters = (characters: CharacterState[]) => {
    const nextCharacters = new Map(
      characters.map((character) => [character.id, character] as const)
    )

    for (const [characterId, previousCharacter] of attachedCharacters) {
      const nextCharacter = nextCharacters.get(characterId)

      if (
        nextCharacter &&
        getControllerAttachmentKey(previousCharacter) ===
          getControllerAttachmentKey(nextCharacter)
      ) {
        attachedCharacters.set(characterId, nextCharacter)
        continue
      }

      resolveAttachment(previousCharacter).detachCharacter(previousCharacter)
      attachedCharacters.delete(characterId)
    }

    for (const character of characters) {
      const previousCharacter = attachedCharacters.get(character.id)

      if (
        previousCharacter &&
        getControllerAttachmentKey(previousCharacter) ===
          getControllerAttachmentKey(character)
      ) {
        continue
      }

      resolveAttachment(character).attachCharacter(character)
      attachedCharacters.set(character.id, character)
    }
  }

  return {
    syncCharacters,
    getMovementDelta: ({ character, deltaMilliseconds, pressedDirections }) =>
      resolveAttachment(character).getMovementDelta({
        character,
        deltaMilliseconds,
        pressedDirections
      }),
    updateLuaControllerScript: (scriptId, script) => {
      luaControllerRuntime?.updateScript(scriptId, script)
    },
    destroy: () => {
      if (isDestroyed) {
        return
      }

      for (const character of attachedCharacters.values()) {
        resolveAttachment(character).detachCharacter(character)
      }

      keyboardAttachment.destroy?.()
      npcAttachment.destroy?.()
      luaAttachment.destroy?.()
      attachedCharacters.clear()
      isDestroyed = true
    }
  }
}

const createKeyboardControllerAttachment = (): CharacterControllerAttachment => ({
  attachCharacter: () => {},
  detachCharacter: () => {},
  getMovementDelta: ({ character, deltaMilliseconds, pressedDirections }) =>
    getCharacterControllerDelta({
      character,
      deltaMilliseconds,
      pressedDirections
    })
})

const createNpcControllerAttachment = (): CharacterControllerAttachment => ({
  attachCharacter: () => {},
  detachCharacter: () => {},
  getMovementDelta: ({ character, deltaMilliseconds }) =>
    getCharacterControllerDelta({
      character,
      deltaMilliseconds
    })
})

const createLuaControllerAttachment = (
  luaControllerRuntime?: LuaCharacterControllerRuntime
): CharacterControllerAttachment => ({
  attachCharacter: (character) => {
    if (character.controller.kind !== 'lua') {
      return
    }

    if (!luaControllerRuntime) {
      throw new Error('Lua controller runtime is required for lua characters')
    }

    luaControllerRuntime.attachCharacter(character, character.controller)
  },
  detachCharacter: (character) => {
    if (character.controller.kind !== 'lua' || !luaControllerRuntime) {
      return
    }

    luaControllerRuntime.detachCharacter(character, character.controller)
  },
  getMovementDelta: ({ character, deltaMilliseconds }) => {
    if (character.controller.kind !== 'lua') {
      return undefined
    }

    if (!luaControllerRuntime) {
      throw new Error('Lua controller runtime is required for lua characters')
    }

    return luaControllerRuntime.getMovementDelta(
      character,
      character.controller,
      deltaMilliseconds
    )
  },
  destroy: () => {
    luaControllerRuntime?.destroy()
  }
})

const getControllerAttachmentKey = (character: CharacterState): string => {
  switch (character.controller.kind) {
    case 'keyboard':
      return `keyboard:${character.controller.moveSpeedTilesPerSecond}`
    case 'npc':
      return `npc:${character.controller.behavior}:${character.controller.moveSpeedTilesPerSecond}`
    case 'lua':
      return `lua:${character.controller.scriptId}:${character.controller.radiusInTiles}:${character.controller.moveSpeedTilesPerSecond}`
  }
}
