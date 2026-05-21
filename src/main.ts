import huntingGroundMapXml from './assets/maps/hunting-ground.tmx?raw'
import caveMapXml from './assets/maps/cave.tmx?raw'
import townMapXml from './assets/maps/town.tmx?raw'
import replyWithMessageControllerLua from './assets/lua/reply-with-message.lua?raw'
import wanderNearHomeControllerLua from './assets/lua/wander-near-home.lua?raw'
import huntingGroundMusicUrl from './assets/sounds/전투브금.mp3'
import townMusicUrl from './assets/sounds/브금5.mp3'
import questFinUrl from './assets/tilesets/quest_fin.png'
import questNewUrl from './assets/tilesets/quest_new.png'
import caveEntranceVisibleUrl from './assets/tilesets/cave1-visible.png'
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
import { createInitialPotionInventory } from './game/potionShop'
import { createLuaCharacterControllerRuntime } from './game/lua/createLuaCharacterControllerRuntime'
import { createNpcCharactersFromEventLayers } from './game/tiled/createNpcCharactersFromEventLayers'
import { parseTiledMap, parseTiledTileset } from './game/tiled/parseTiledMap'
import { createInitialPlayerEquipment } from './game/playerEquipment'
import { createInitialPlayerInventory } from './game/playerInventory'
import { createInitialPlayerProfile } from './game/playerProfile'
import { createInitialPlayerQuickslots } from './game/playerQuickslots'
import {
  normalizeStoredPlayerControlBindings,
  PLAYER_CONTROL_BINDINGS_STORAGE_KEY,
  type PlayerControlBindings
} from './game/playerControls'
import { createInitialQuestLog } from './game/questLog'
import { getSceneIntroMessage } from './game/sceneIntro'
import { createPixiTiledMapView } from './rendering/createPixiTiledMapView'
import type { AudioSettings } from './rendering/createPauseMenuOverlay'
import type {
  SceneTransitionRequest
} from './rendering/createPixiTiledMapView'
import './styles.css'

type SceneId = 'town' | 'hunting-ground' | 'cave'

type SceneSpawn = {
  x: number
  y: number
  facing?: CharacterMoveDirection
}

type SceneRenderer = {
  destroy: () => void
}

const AUDIO_SETTINGS_STORAGE_KEY = 'my-sample-rpg:audio-settings'
const DEFAULT_AUDIO_SETTINGS: AudioSettings = {
  bgmVolume: 1,
  sfxVolume: 1
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
const parsedCaveMap = parseTiledMap({
  mapXml: caveMapXml,
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
  'hunting-ground': parsedHuntingGroundMap,
  cave: parsedCaveMap
}
const sceneMusicUrls: Record<SceneId, string> = {
  town: townMusicUrl,
  'hunting-ground': huntingGroundMusicUrl,
  cave: huntingGroundMusicUrl
}
const playerProfile = createInitialPlayerProfile()
let playerEquipment = createInitialPlayerEquipment()
let playerInventory = createInitialPlayerInventory()
let playerQuickslots = createInitialPlayerQuickslots()
let playerControlBindings = readStoredPlayerControlBindings()
let questLog = createInitialQuestLog()
let merchantInventory = createInitialBlacksmithInventory()
let potionMerchantInventory = createInitialPotionInventory()
let activeControllerRuntime:
  | ReturnType<typeof createCharacterControllerRuntime>
  | undefined
let activeSceneRenderer: SceneRenderer | undefined
let activeSceneMusic: HTMLAudioElement | undefined
let activeSceneMusicUrl = ''
let isSceneMusicRetryQueued = false
let pendingSceneTransition: SceneTransitionRequest | undefined
let isSceneTransitionScheduled = false
let audioSettings = readStoredAudioSettings()

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
    playerQuickslots,
    playerControlBindings,
    questLog,
    merchantInventory,
    potionMerchantInventory,
    sceneIntroMessage: getSceneIntroMessage(sceneId),
    cameraTargetCharacterId: PLAYER_CHARACTER_ID,
    characterSpriteSheet: {
      tileset: tinyDungeonTileset,
      scale: characterSpriteScale
    },
    imageUrls: {
      'cave1-visible.png': caveEntranceVisibleUrl,
      'quest_fin.png': questFinUrl,
      'quest_new.png': questNewUrl,
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
    onPlayerQuickslotsChange: (nextQuickslots) => {
      playerQuickslots = nextQuickslots
    },
    onPlayerControlBindingsChange: (nextControlBindings) => {
      playerControlBindings = nextControlBindings
      savePlayerControlBindings(nextControlBindings)
    },
    onQuestLogChange: (nextQuestLog) => {
      questLog = nextQuestLog
    },
    onMerchantInventoryChange: (nextInventory) => {
      merchantInventory = nextInventory
    },
    onPotionMerchantInventoryChange: (nextInventory) => {
      potionMerchantInventory = nextInventory
    },
    audioSettings,
    onAudioSettingsChange: updateAudioSettings,
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
    applyActiveSceneMusicVolume()
    playActiveSceneMusic()
    return
  }

  activeSceneMusic?.pause()

  activeSceneMusic = new Audio(musicUrl)
  activeSceneMusic.loop = true
  activeSceneMusicUrl = musicUrl
  applyActiveSceneMusicVolume()

  playActiveSceneMusic()
}

const playActiveSceneMusic = () => {
  if (!activeSceneMusic) {
    return
  }

  void activeSceneMusic.play().catch(queueSceneMusicRetry)
}

const applyActiveSceneMusicVolume = () => {
  if (activeSceneMusic) {
    activeSceneMusic.volume = audioSettings.bgmVolume
  }
}

const updateAudioSettings = (nextAudioSettings: AudioSettings) => {
  audioSettings = {
    bgmVolume: clampVolume(nextAudioSettings.bgmVolume),
    sfxVolume: clampVolume(nextAudioSettings.sfxVolume)
  }
  saveAudioSettings(audioSettings)
  applyActiveSceneMusicVolume()
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

const clampVolume = (value: number): number =>
  Number.isFinite(value) ? Math.max(0, Math.min(value, 1)) : 1

function readStoredAudioSettings(): AudioSettings {
  const storedAudioSettings = window.localStorage.getItem(AUDIO_SETTINGS_STORAGE_KEY)

  if (!storedAudioSettings) {
    return DEFAULT_AUDIO_SETTINGS
  }

  try {
    const parsedAudioSettings = JSON.parse(storedAudioSettings) as Partial<AudioSettings>

    return {
      bgmVolume: clampVolume(
        parsedAudioSettings.bgmVolume ?? DEFAULT_AUDIO_SETTINGS.bgmVolume
      ),
      sfxVolume: clampVolume(
        parsedAudioSettings.sfxVolume ?? DEFAULT_AUDIO_SETTINGS.sfxVolume
      )
    }
  } catch {
    return DEFAULT_AUDIO_SETTINGS
  }
}

function saveAudioSettings(nextAudioSettings: AudioSettings): void {
  window.localStorage.setItem(
    AUDIO_SETTINGS_STORAGE_KEY,
    JSON.stringify(nextAudioSettings)
  )
}

function readStoredPlayerControlBindings(): PlayerControlBindings {
  const storedControlBindings = window.localStorage.getItem(
    PLAYER_CONTROL_BINDINGS_STORAGE_KEY
  )

  if (!storedControlBindings) {
    return normalizeStoredPlayerControlBindings(undefined)
  }

  try {
    return normalizeStoredPlayerControlBindings(JSON.parse(storedControlBindings))
  } catch {
    return normalizeStoredPlayerControlBindings(undefined)
  }
}

function savePlayerControlBindings(
  nextControlBindings: PlayerControlBindings
): void {
  window.localStorage.setItem(
    PLAYER_CONTROL_BINDINGS_STORAGE_KEY,
    JSON.stringify(nextControlBindings)
  )
}

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
