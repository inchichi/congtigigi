import {
  getCharacterControllerIntent,
  type CharacterAction,
  type CharacterControllerIntent,
  type CharacterMoveDirection,
  type CharacterState
} from './characterState'
import type {
  LuaCharacterControllerRuntime,
  LuaControllerScriptSource,
  LuaRuntimeSnapshot
} from './lua/createLuaCharacterControllerRuntime'
import type { LuaControllerRuntimeEvent } from './lua/luaControllerApi'

type CharacterControllerAttachment = {
  attachCharacter: (character: CharacterState) => void
  detachCharacter: (character: CharacterState) => void
  getIntent: (input: {
    character: CharacterState
    deltaMilliseconds: number
    pressedDirections?: ReadonlySet<CharacterMoveDirection>
    triggeredActions?: ReadonlySet<CharacterAction>
  }) => CharacterControllerIntent | undefined
  canReceiveInteraction: (character: CharacterState) => boolean
  handleInteraction: (input: {
    targetCharacter: CharacterState
    sourceCharacter: CharacterState
  }) => CharacterInteractionResponse | undefined
  destroy?: () => void
}

type CreateCharacterControllerRuntimeInput = {
  luaControllerRuntime?: LuaCharacterControllerRuntime
}

export type CharacterControllerRuntime = {
  syncCharacters: (characters: CharacterState[]) => void
  getIntent: (input: {
    character: CharacterState
    deltaMilliseconds: number
    pressedDirections?: ReadonlySet<CharacterMoveDirection>
    triggeredActions?: ReadonlySet<CharacterAction>
  }) => CharacterControllerIntent | undefined
  canReceiveInteraction: (character: CharacterState) => boolean
  handleInteraction: (input: {
    targetCharacter: CharacterState
    sourceCharacter: CharacterState
  }) => CharacterInteractionResponse | undefined
  drainEvents: () => LuaControllerRuntimeEvent[]
  getRuntimeWarnings: () => string[]
  updateLuaControllerScript: (
    scriptId: string,
    script: LuaControllerScriptSource
  ) => void
  pushSnapshot: (snapshot: LuaRuntimeSnapshot) => void
  // 게임 규칙을 Lua 로 실행: 테이블/값을 반환하는 Lua 데이터·로직 모듈을 마샬링해 돌려준다.
  loadDataModule: (source: string) => unknown
  destroy: () => void
}

export type CharacterInteractionResponse = {
  kind: 'message'
  message: string
  durationMilliseconds: number
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
    getIntent: ({
      character,
      deltaMilliseconds,
      pressedDirections,
      triggeredActions
    }) =>
      resolveAttachment(character).getIntent({
        character,
        deltaMilliseconds,
        pressedDirections,
        triggeredActions
      }),
    canReceiveInteraction: (character) =>
      resolveAttachment(character).canReceiveInteraction(character),
    handleInteraction: ({ targetCharacter, sourceCharacter }) =>
      resolveAttachment(targetCharacter).handleInteraction({
        targetCharacter,
        sourceCharacter
      }),
    drainEvents: () => luaControllerRuntime?.drainEvents() ?? [],
    getRuntimeWarnings: () => luaControllerRuntime?.getActiveErrorMessages() ?? [],
    updateLuaControllerScript: (scriptId, script) => {
      luaControllerRuntime?.updateScript(scriptId, script)
    },
    pushSnapshot: (snapshot) => {
      luaControllerRuntime?.pushSnapshot(snapshot)
    },
    loadDataModule: (source) => luaControllerRuntime?.loadDataModule(source) ?? null,
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
  getIntent: ({
    character,
    deltaMilliseconds,
    pressedDirections,
    triggeredActions
  }) =>
    getCharacterControllerIntent({
      character,
      deltaMilliseconds,
      pressedDirections,
      triggeredActions
    }),
  canReceiveInteraction: () => false,
  handleInteraction: () => undefined
})

const createNpcControllerAttachment = (): CharacterControllerAttachment => ({
  attachCharacter: () => {},
  detachCharacter: () => {},
  getIntent: ({ character, deltaMilliseconds }) =>
    getCharacterControllerIntent({
      character,
      deltaMilliseconds
    }),
  canReceiveInteraction: (character) =>
    isMonsterCharacter(character) && character.blocksMovement,
  handleInteraction: ({ targetCharacter }) =>
    isMonsterCharacter(targetCharacter) && targetCharacter.blocksMovement
      ? {
          kind: 'message',
          message: '!',
          durationMilliseconds: 600
        }
      : undefined
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
  getIntent: ({ character, deltaMilliseconds }) => {
    if (character.controller.kind !== 'lua') {
      return undefined
    }

    if (!luaControllerRuntime) {
      throw new Error('Lua controller runtime is required for lua characters')
    }

    const movement = luaControllerRuntime.getMovementDelta(
      character,
      character.controller,
      deltaMilliseconds
    )

    if (!movement) {
      return undefined
    }

    return {
      movement: scaleLuaMovementIntent(
        movement,
        character.controller.moveSpeedTilesPerSecond,
        deltaMilliseconds
      )
    }
  },
  canReceiveInteraction: (character) => {
    if (character.controller.kind !== 'lua' || !luaControllerRuntime) {
      return false
    }

    return luaControllerRuntime.canReceiveInteraction(
      character,
      character.controller
    )
  },
  handleInteraction: ({ targetCharacter, sourceCharacter }) => {
    if (targetCharacter.controller.kind !== 'lua' || !luaControllerRuntime) {
      return undefined
    }

    const response = luaControllerRuntime.handleInteraction(
      targetCharacter,
      targetCharacter.controller,
      sourceCharacter
    )

    if (!response) {
      return undefined
    }

    return {
      kind: 'message',
      message: response.message,
      durationMilliseconds: response.durationMilliseconds
    }
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
      return `lua:${character.controller.scriptId}:${character.controller.radiusInTiles}:${character.controller.moveSpeedTilesPerSecond}:${JSON.stringify(character.controller.config)}`
  }
}

const isMonsterCharacter = (character: CharacterState): boolean =>
  character.appearanceType.startsWith('monster_')

const scaleLuaMovementIntent = (
  movement: { x: number; y: number },
  moveSpeedTilesPerSecond: number,
  deltaMilliseconds: number
): { x: number; y: number } => {
  const magnitude = Math.hypot(movement.x, movement.y)

  if (magnitude === 0) {
    return movement
  }

  const distanceInTiles =
    (moveSpeedTilesPerSecond * deltaMilliseconds) / 1000
  const normalizedMovement =
    magnitude > 1
      ? {
          x: movement.x / magnitude,
          y: movement.y / magnitude
        }
      : movement

  return {
    x: normalizedMovement.x * distanceInTiles,
    y: normalizedMovement.y * distanceInTiles
  }
}
