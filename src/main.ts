import huntingGroundMapXml from './assets/maps/hunting-ground.tmx?raw'
import townMapXml from './assets/maps/town.tmx?raw'
import replyWithMessageControllerLua from './assets/lua/reply-with-message.lua?raw'
import wanderNearHomeControllerLua from './assets/lua/wander-near-home.lua?raw'
import huntingGroundMusicUrl from './assets/sounds/전투브금.mp3'
import townMusicUrl from './assets/sounds/브금5.mp3'
import townTilesetXml from './assets/tilesets/town-32.tsx?raw'
import townTilesetUrl from './assets/tilesets/town-32.png'
import tinyDungeonTilesetXml from './assets/tilesets/tiny-dungeon-16.tsx?raw'
import tinyDungeonTilesetUrl from './assets/tilesets/tiny-dungeon-16.png'

import {
  PLAYER_CHARACTER_ID,
  createInitialPlayerCharacter,
  type CharacterMoveDirection,
  type CharacterState
} from './game/characterState'
import { createCharacterControllerRuntime } from './game/createCharacterControllerRuntime'
import { createInitialBlacksmithInventory } from './game/blacksmithShop'
import { createLuaCharacterControllerRuntime } from './game/lua/createLuaCharacterControllerRuntime'
import { createNpcCharactersFromEventLayers } from './game/tiled/createNpcCharactersFromEventLayers'
import { parseTiledMap, parseTiledTileset } from './game/tiled/parseTiledMap'
import { createInitialPlayerEquipment } from './game/playerEquipment'
import { createInitialPlayerInventory } from './game/playerInventory'
import { createInitialPlayerProfile } from './game/playerProfile'
import { getSceneIntroMessage } from './game/sceneIntro'
import { createPixiTiledMapView } from './rendering/createPixiTiledMapView'
import type {
  SceneTransitionRequest
} from './rendering/createPixiTiledMapView'
import './styles.css'

type SceneId = 'town' | 'hunting-ground'

type SceneSpawn = {
  x: number
  y: number
  facing?: CharacterMoveDirection
}

type SceneRenderer = {
  destroy: () => void
}

const rootElement = document.querySelector<HTMLDivElement>('#app')

if (!rootElement) {
  throw new Error('Missing #app root element')
}

const parsedTownMap = parseTiledMap({
  mapXml: townMapXml,
  externalTilesets: {
    '../tilesets/town-32.tsx': townTilesetXml
  }
})
const parsedHuntingGroundMap = parseTiledMap({
  mapXml: huntingGroundMapXml,
  externalTilesets: {
    '../tilesets/town-32.tsx': townTilesetXml
  }
})
const tinyDungeonTileset = parseTiledTileset({
  firstGid: 1,
  source: '../tilesets/tiny-dungeon-16.tsx',
  tilesetXml: tinyDungeonTilesetXml
})
const characterSpriteScale = 2
const replyWithMessageScriptId = 'reply-with-message'
const wanderNearHomeScriptId = 'wander-near-home'
const availableLuaControllerScriptsById: Record<string, { source: string }> = {
  [replyWithMessageScriptId]: {
    source: replyWithMessageControllerLua
  },
  [wanderNearHomeScriptId]: {
    source: wanderNearHomeControllerLua
  }
}
const sceneMaps: Record<SceneId, typeof parsedTownMap> = {
  town: parsedTownMap,
  'hunting-ground': parsedHuntingGroundMap
}
const sceneMusicUrls: Record<SceneId, string> = {
  town: townMusicUrl,
  'hunting-ground': huntingGroundMusicUrl
}
const playerProfile = createInitialPlayerProfile()
let playerEquipment = createInitialPlayerEquipment()
let playerInventory = createInitialPlayerInventory()
let merchantInventory = createInitialBlacksmithInventory()
let activeControllerRuntime:
  | ReturnType<typeof createCharacterControllerRuntime>
  | undefined
let activeSceneRenderer: SceneRenderer | undefined
let activeSceneMusic: HTMLAudioElement | undefined
let activeSceneMusicUrl = ''
let isSceneMusicRetryQueued = false
let pendingSceneTransition: SceneTransitionRequest | undefined
let isSceneTransitionScheduled = false

rootElement.className = 'game-root'

const bootstrapScene = async (
  sceneId: SceneId,
  spawn?: SceneSpawn
): Promise<void> => {
  const sceneMap = sceneMaps[sceneId]

  if (!sceneMap) {
    throw new Error(`Unknown scene "${sceneId}"`)
  }

  destroyActiveScene()
  playSceneMusic(sceneId)

  const characters = createSceneCharacters({
    map: sceneMap,
    spawn
  })
  const hasLuaControlledCharacter = characters.some(
    (character) => character.controller.kind === 'lua'
  )
  const luaControllerRuntime = hasLuaControlledCharacter
    ? await createLuaCharacterControllerRuntime({
        scriptsById: collectLuaControllerScripts(characters)
      })
    : undefined
  const controllerRuntime = createCharacterControllerRuntime({
    luaControllerRuntime
  })

  activeControllerRuntime = controllerRuntime
  activeSceneRenderer = await createPixiTiledMapView({
    mountElement: rootElement,
    map: sceneMap,
    characters,
    playerProfile,
    playerEquipment,
    playerInventory,
    merchantInventory,
    sceneIntroMessage: getSceneIntroMessage(sceneId),
    cameraTargetCharacterId: PLAYER_CHARACTER_ID,
    characterSpriteSheet: {
      tileset: tinyDungeonTileset,
      scale: characterSpriteScale
    },
    imageUrls: {
      'town-32.png': townTilesetUrl,
      'tiny-dungeon-16.png': tinyDungeonTilesetUrl
    },
    controllerRuntime,
    onPlayerInventoryChange: (nextInventory) => {
      playerInventory = nextInventory
    },
    onPlayerEquipmentChange: (nextEquipment) => {
      playerEquipment = nextEquipment
    },
    onMerchantInventoryChange: (nextInventory) => {
      merchantInventory = nextInventory
    },
    onRequestSceneChange: scheduleSceneTransition
  })
}

const createSceneCharacters = ({
  map,
  spawn
}: {
  map: typeof parsedTownMap
  spawn?: SceneSpawn
}): CharacterState[] => {
  const playerCharacter = createInitialPlayerCharacter({
    mapWidth: map.width,
    mapHeight: map.height
  })

  if (spawn) {
    playerCharacter.position = {
      x: clampSpawnCoordinate(
        spawn.x,
        map.width - playerCharacter.collisionSize.width
      ),
      y: clampSpawnCoordinate(
        spawn.y,
        map.height - playerCharacter.collisionSize.height
      )
    }

    if (spawn.facing) {
      playerCharacter.facing = spawn.facing
    }
  }

  return [
    playerCharacter,
    ...createNpcCharactersFromEventLayers({
      map,
      defaultPixelWidth: tinyDungeonTileset.tileWidth * characterSpriteScale,
      defaultPixelHeight: tinyDungeonTileset.tileHeight * characterSpriteScale
    })
  ]
}

const collectLuaControllerScripts = (
  characters: CharacterState[]
): Record<string, { source: string }> =>
  Object.fromEntries(
    characters.flatMap((character) => {
      if (character.controller.kind !== 'lua') {
        return []
      }

      const script = availableLuaControllerScriptsById[character.controller.scriptId]

      if (!script) {
        throw new Error(
          `Missing Lua controller source for scriptId "${character.controller.scriptId}"`
        )
      }

      return [[character.controller.scriptId, script]]
    })
  )

const destroyActiveScene = () => {
  activeControllerRuntime = undefined
  activeSceneRenderer?.destroy()
  activeSceneRenderer = undefined
}

const playSceneMusic = (sceneId: SceneId) => {
  const musicUrl = sceneMusicUrls[sceneId]

  if (activeSceneMusic && activeSceneMusicUrl === musicUrl) {
    playActiveSceneMusic()
    return
  }

  activeSceneMusic?.pause()

  activeSceneMusic = new Audio(musicUrl)
  activeSceneMusic.loop = true
  activeSceneMusicUrl = musicUrl

  playActiveSceneMusic()
}

const playActiveSceneMusic = () => {
  if (!activeSceneMusic) {
    return
  }

  void activeSceneMusic.play().catch(queueSceneMusicRetry)
}

const queueSceneMusicRetry = () => {
  if (isSceneMusicRetryQueued) {
    return
  }

  isSceneMusicRetryQueued = true

  const retrySceneMusic = () => {
    isSceneMusicRetryQueued = false
    playActiveSceneMusic()
  }

  window.addEventListener('pointerdown', retrySceneMusic, { once: true })
  window.addEventListener('keydown', retrySceneMusic, { once: true })
}

const scheduleSceneTransition = (request: SceneTransitionRequest) => {
  pendingSceneTransition = request

  if (isSceneTransitionScheduled) {
    return
  }

  isSceneTransitionScheduled = true
  void Promise.resolve()
    .then(async () => {
      isSceneTransitionScheduled = false
      const nextRequest = pendingSceneTransition

      pendingSceneTransition = undefined

      if (!nextRequest) {
        return
      }

      await bootstrapScene(nextRequest.sceneId as SceneId, {
        x: nextRequest.spawn.x,
        y: nextRequest.spawn.y,
        facing: nextRequest.facing
      })
    })
    .catch(renderFatalError)
}

const clampSpawnCoordinate = (value: number, max: number): number =>
  Math.max(0, Math.min(value, Math.max(0, max)))

const renderFatalError = (error: unknown) => {
  const message = error instanceof Error ? error.message : String(error)

  rootElement.innerHTML = `
    <div class="error-panel">
      <h2>Renderer Failed</h2>
      <p>${message}</p>
    </div>
  `
}

if (import.meta.hot) {
  import.meta.hot.accept('./assets/lua/reply-with-message.lua?raw', (nextModule) => {
    if (!nextModule || !activeControllerRuntime) {
      return
    }

    activeControllerRuntime.updateLuaControllerScript(replyWithMessageScriptId, {
      source: nextModule.default
    })
  })

  import.meta.hot.accept('./assets/lua/wander-near-home.lua?raw', (nextModule) => {
    if (!nextModule || !activeControllerRuntime) {
      return
    }

    activeControllerRuntime.updateLuaControllerScript(wanderNearHomeScriptId, {
      source: nextModule.default
    })
  })
}

void bootstrapScene('town').catch(renderFatalError)
