import huntingGroundMapXml from './assets/maps/hunting-ground.tmx?raw'
import caveMapXml from './assets/maps/cave.tmx?raw'
import townMapXml from './assets/maps/town.tmx?raw'
import replyWithMessageControllerLua from './assets/lua/reply-with-message.lua?raw'
import wanderNearHomeControllerLua from './assets/lua/wander-near-home.lua?raw'
import vnDialogueControllerLua from './assets/lua/vn-dialogue.lua?raw'
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
  type CharacterMoveDirection,
  type CharacterState
} from './characterState'
import { createCharacterControllerRuntime } from './createCharacterControllerRuntime'
import { createLuaCharacterControllerRuntime } from './lua/createLuaCharacterControllerRuntime'
import {
  initLuaGameLogic,
  createInitialPlayerEquipment,
  createInitialPotionInventory,
  createInitialBlacksmithInventory,
  createInitialPlayerQuickslots,
  createInitialPlayerSkillSlots,
  createInitialPlayerCharacter,
  parseStoredPlayerSaveState,
  serializePlayerSaveState,
  createHolidayDialogueEventDraftFromText,
  createHolidayDialogueEventValidationErrors,
  createTiledNpcEventObject,
  normalizeStoredPlayerControlBindings
} from './lua/luaGameLogic'
import { createNpcCharactersFromEventLayers } from './tiled/createNpcCharactersFromEventLayers'
import { parseTiledMap, parseTiledTileset } from './tiled/parseTiledMap'
import { createInitialPlayerInventory } from './playerInventory'
import { createInitialPlayerProfile } from './playerProfile'
import {
  PLAYER_SAVE_STATE_STORAGE_KEY,
  type PlayerSaveState
} from './playerSaveState'
import {
  OPENAI_HOLIDAY_EVENT_DRAFT_MODEL,
  generateHolidayDialogueEventDraftWithOpenAi
} from './openaiHolidayEventDraft'
import {
  type HolidayDialogueEventSpec,
  HOLIDAY_DIALOGUE_CONTROLLER_SCRIPT_ID
} from './eventGeneration'
import {
  PLAYER_CONTROL_BINDINGS_STORAGE_KEY,
  type PlayerControlBindings
} from './playerControls'
// 퀘스트/씬 인트로는 develop-chich의 TS를 직접 쓴다(동적 퀘스트는 TS 모듈 상태라
// Lua 퍼사드 경유 시 안 보임 — 퀘스트 호출부는 TS 유지). 아래 facade import에서는 제외했다.
import {
  createInitialQuestLog,
  clearDynamicQuestDefinitions,
  ensureQuestProgressEntries,
  recordSceneEnterQuestProgress,
  registerDynamicQuestDefinitions,
  setQuestDefinitionVisibilityFilter
} from './questLog'
import {
  clearPendingQuests,
  loadPendingQuests,
  PENDING_QUESTS_STORAGE_KEY
} from '../../editor/pendingQuests'
import { getSceneIntroMessage } from './sceneIntro'
import { createPixiTiledMapView } from './rendering/createPixiTiledMapView'
import {
  loadPendingEvents,
  loadPendingLuaScripts,
  PENDING_EVENTS_STORAGE_KEY,
  PENDING_LUA_SCRIPTS_STORAGE_KEY,
  type PendingLuaScript
} from '../../editor/pendingEvents'
import {
  PENDING_PLACEMENTS_STORAGE_KEY,
  type PlacementTemplate
} from '../../editor/placementStore'
import {
  PENDING_NPCS_STORAGE_KEY,
  type NpcWireTemplate
} from '../../editor/npcStore'
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
  // 에디터의 음소거 토글 등 외부에서 오디오 설정을 갱신할 때 — SFX 마스터 볼륨까지 반영된다.
  updateAudioSettings: (nextAudioSettings: AudioSettings) => void
  applyEventDraft: (
    draft: HolidayDialogueEventSpec,
    input?: { targetCharacterId?: string }
  ) => { didApply: boolean; targetCharacterId?: string }
  applyLuaScript: (input: {
    targetCharacterId: string
    source: string
  }) => { didApply: boolean; targetCharacterId?: string }
  // 마우스 에셋 배치(에디터 배치 모드). NPC 템플릿(kind:'npc')도 같은 채널로 전달된다.
  setPlacementMode: (mode: 'off' | 'place' | 'erase') => void
  setPlacementTemplate: (
    template: PlacementTemplate | NpcWireTemplate | null
  ) => void
  refreshPlacements: () => void
  refreshNpcs: () => void
}

// 배경음악(BGM) 전역 사용 여부. false면 어떤 씬에서도 BGM을 재생하지 않는다(효과음은 그대로).
// 저장된 볼륨 설정과 무관하게 코드에서 끄는 스위치 — 다시 켜려면 true로 바꾸면 된다.
const BGM_ENABLED = false

const AUDIO_SETTINGS_STORAGE_KEY = 'my-sample-rpg:audio-settings'
const EVENT_DRAFT_MODE_STORAGE_KEY = 'my-sample-rpg:event-draft-mode'
const OPENAI_API_KEY_STORAGE_KEY = 'my-sample-rpg:openai-api-key'
const DEFAULT_AUDIO_SETTINGS: AudioSettings = {
  bgmVolume: 1,
  sfxVolume: 1,
  isMuted: false
}

// 에디터 미리보기는 게임을 iframe으로 띄운다. 에디터로 편집하는 동안에는 효과음·배경음이
// 들리지 않도록, 게임이 iframe(=에디터) 안에서 실행되면 강제로 음소거한다.
// 단독 실행(직접 플레이)에서는 window.self === window.top 이라 영향이 없다.
const IS_EMBEDDED_IN_EDITOR = ((): boolean => {
  try {
    return window.self !== window.top
  } catch {
    // top이 교차 출처라 접근이 막히면 임베드로 간주해 안전하게 음소거한다.
    return true
  }
})()

// 에디터 안에서 실행 중이면 어떤 오디오 설정이 와도 음소거를 강제한다.
const applyEditorMute = (settings: AudioSettings): AudioSettings =>
  IS_EMBEDDED_IN_EDITOR ? { ...settings, isMuted: true } : settings
type EventDraftMode = 'rule' | 'llm'
const DEFAULT_EVENT_DRAFT_MODE: EventDraftMode = 'rule'
const EVENT_DRAFT_TARGET_CHARACTER_ID = 'santa'
const DEFAULT_OPENAI_MODEL = OPENAI_HOLIDAY_EVENT_DRAFT_MODEL

const appRootElement = document.querySelector<HTMLDivElement>('#app')

if (!appRootElement) {
  throw new Error('Missing #app root element')
}

const gameRootElement = document.createElement('div')
const uiRootElement = document.createElement('div')

appRootElement.className = 'app-shell'
gameRootElement.className = 'game-root'
uiRootElement.className = 'ui-root'
appRootElement.replaceChildren(gameRootElement, uiRootElement)

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
const vnDialogueScriptId = 'vn-dialogue'
const availableLuaControllerScriptsById: Record<string, { source: string }> = {
  [replyWithMessageScriptId]: {
    source: replyWithMessageControllerLua
  },
  [wanderNearHomeScriptId]: {
    source: wanderNearHomeControllerLua
  },
  [vnDialogueScriptId]: {
    source: vnDialogueControllerLua
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
const storedPlayerSaveState = readStoredPlayerSaveState()
const playerProfile = storedPlayerSaveState?.profile ?? createInitialPlayerProfile()
// 저장 시점에 사망(hp 0) 상태였다면 로드 후 갇히지 않도록 체력을 회복해 부활시킨다.
if (playerProfile.hp.current <= 0) {
  playerProfile.hp.current = playerProfile.hp.max
}
let playerEquipment =
  storedPlayerSaveState?.equipment ?? createInitialPlayerEquipment()
let playerInventory =
  storedPlayerSaveState?.inventory ?? createInitialPlayerInventory()
let playerQuickslots =
  storedPlayerSaveState?.quickslots ?? createInitialPlayerQuickslots()
let playerSkillSlots =
  storedPlayerSaveState?.skillSlots ?? createInitialPlayerSkillSlots()
let playerControlBindings = readStoredPlayerControlBindings()
let questLog = createInitialQuestLog()
// 에디터가 생성·주입한 동적 퀘스트를 런타임 퀘스트 엔진에 등록하고, 진행도 항목을 채운다.
// 부팅 전에 questLog를 갱신해야 bootstrapScene이 그걸 렌더러로 넘긴다(배지·추적·완료 전부 작동).
const applyPendingQuests = (): void => {
  const pendingQuests = loadPendingQuests()
  clearDynamicQuestDefinitions()
  registerDynamicQuestDefinitions(pendingQuests)
  setQuestDefinitionVisibilityFilter(pendingQuests.map((quest) => quest.id))
  if (pendingQuests.length > 0) {
    clearPendingQuests()
  }
  questLog = ensureQuestProgressEntries(questLog)
}
applyPendingQuests()
let merchantInventory = createInitialBlacksmithInventory()
let potionMerchantInventory = createInitialPotionInventory()
let activeControllerRuntime:
  | ReturnType<typeof createCharacterControllerRuntime>
  | undefined
let activeSceneRenderer: SceneRenderer | undefined
// 라이브 퀘스트 주입 시 다시 부팅할 현재 씬. bootstrapScene이 매번 갱신한다.
let activeSceneId: SceneId = 'town'
let activeSceneMusic: HTMLAudioElement | undefined
let activeSceneMusicUrl = ''
let isSceneMusicRetryQueued = false
let pendingSceneTransition: SceneTransitionRequest | undefined
let isSceneTransitionScheduled = false
let audioSettings = applyEditorMute(readStoredAudioSettings())
let refreshEventDraftPreview: (() => void) | undefined

const bootstrapScene = async (
  sceneId: SceneId,
  spawn?: SceneSpawn
): Promise<void> => {
  const sceneMap = sceneMaps[sceneId]

  if (!sceneMap) {
    throw new Error(`Unknown scene "${sceneId}"`)
  }

  // 현재 씬을 기억한다 — 에디터가 퀘스트를 라이브로 주입하면 이 씬을 다시 부팅해 반영한다.
  activeSceneId = sceneId

  // 에디터 프리뷰(부모 창)에 현재 맵을 알린다 — 에디터가 그 맵의 편집 가능한 요소만 보여줄 수 있게.
  // 모든 맵 전환(초기 로드·포털 이동·에디터 맵 버튼)이 이 함수를 거치므로 여기 한 곳이면 전부 커버된다.
  // iframe이 아니면 window.parent === window라 자기 자신에게 가지만, 게임 핸들러는 editor:switch-scene
  // 만 처리하므로 무해하다(루프 없음). sceneId 는 에디터의 map.id(=tmx 파일명)와 동일하다.
  window.parent.postMessage({ type: 'game:scene-changed', sceneId }, '*')

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
    mountElement: gameRootElement,
    map: sceneMap,
    characters,
    playerProfile,
    playerEquipment,
    playerInventory,
    playerQuickslots,
    playerSkillSlots,
    playerControlBindings,
    questLog,
    merchantInventory,
    potionMerchantInventory,
    sceneId,
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
      savePlayerState()
    },
    onPlayerEquipmentChange: (nextEquipment) => {
      playerEquipment = nextEquipment
      savePlayerState()
    },
    onPlayerQuickslotsChange: (nextQuickslots) => {
      playerQuickslots = nextQuickslots
      savePlayerState()
    },
    onPlayerSkillSlotsChange: (nextSkillSlots) => {
      playerSkillSlots = nextSkillSlots
      savePlayerState()
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

  for (const pendingEvent of loadPendingEvents()) {
    activeSceneRenderer?.applyEventDraft(pendingEvent, {
      targetCharacterId: pendingEvent.npc.id
    })
  }

  for (const pendingLuaScript of loadPendingLuaScripts()) {
    applyPendingLuaScript(pendingLuaScript)
  }
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
): Record<string, { source: string }> => {
  characters.forEach((character) => {
    if (
      character.controller.kind === 'lua' &&
      !availableLuaControllerScriptsById[character.controller.scriptId]
    ) {
      throw new Error(
        `Missing Lua controller source for scriptId "${character.controller.scriptId}"`
      )
    }
  })

  // 맵이 지금 쓰는 스크립트만 등록하면, 에디터가 나중에 주입하는 대화 컨트롤러
  // (reply-with-message 등)가 미등록 상태로 남아 씬 진입 후 적용이 실패한다.
  // 그래서 검증은 캐릭터 기준으로 하되 등록은 사용 가능한 스크립트 전부로 한다.
  return { ...availableLuaControllerScriptsById }
}

const destroyActiveScene = () => {
  activeControllerRuntime = undefined
  activeSceneRenderer?.destroy()
  activeSceneRenderer = undefined
}

const playSceneMusic = (sceneId: SceneId) => {
  // BGM이 꺼져 있으면 오디오를 만들지도 재생하지도 않는다(파일 로드도 생략).
  if (!BGM_ENABLED) {
    activeSceneMusic?.pause()
    activeSceneMusic = undefined
    activeSceneMusicUrl = ''
    return
  }

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
    // 음소거는 볼륨값과 독립적으로 곱한다 — 해제하면 저장된 볼륨이 그대로 돌아온다.
    activeSceneMusic.volume = audioSettings.isMuted ? 0 : audioSettings.bgmVolume
  }
}

const updateAudioSettings = (nextAudioSettings: AudioSettings) => {
  const chosenIsMuted = nextAudioSettings.isMuted === true
  audioSettings = {
    bgmVolume: clampVolume(nextAudioSettings.bgmVolume),
    sfxVolume: clampVolume(nextAudioSettings.sfxVolume),
    // 에디터(iframe) 안에서는 항상 음소거 — 단독 실행 시에만 설정값을 따른다.
    isMuted: IS_EMBEDDED_IN_EDITOR ? true : chosenIsMuted
  }
  // 저장은 사용자의 실제 선택값으로 — 에디터·게임이 localStorage를 공유하므로, 에디터의 강제
  // 음소거가 단독 실행(플레이)의 음소거 설정을 덮어쓰지 않게 한다.
  saveAudioSettings({ ...audioSettings, isMuted: chosenIsMuted })
  applyActiveSceneMusicVolume()
}

const queueSceneMusicRetry = () => {
  if (isSceneMusicRetryQueued) {
    return
  }

  isSceneMusicRetryQueued = true

  const retrySceneMusic = () => {
    window.removeEventListener('pointerdown', retrySceneMusic)
    window.removeEventListener('keydown', retrySceneMusic)
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

      questLog = recordSceneEnterQuestProgress(questLog, nextRequest.sceneId)
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
      ),
      isMuted: parsedAudioSettings.isMuted === true
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

function readStoredEventDraftMode(): EventDraftMode {
  const storedEventDraftMode = window.localStorage.getItem(
    EVENT_DRAFT_MODE_STORAGE_KEY
  )

  if (storedEventDraftMode === 'llm' || storedEventDraftMode === 'rule') {
    return storedEventDraftMode
  }

  return DEFAULT_EVENT_DRAFT_MODE
}

function saveEventDraftMode(nextEventDraftMode: EventDraftMode): void {
  window.localStorage.setItem(EVENT_DRAFT_MODE_STORAGE_KEY, nextEventDraftMode)
}

function readStoredOpenAiApiKey(): string {
  return window.localStorage.getItem(OPENAI_API_KEY_STORAGE_KEY) ?? ''
}

function saveStoredOpenAiApiKey(nextApiKey: string): void {
  window.localStorage.setItem(OPENAI_API_KEY_STORAGE_KEY, nextApiKey)
}

export function createEventDraftPanel(mountElement: HTMLDivElement): () => void {
  const overlayRoot = document.createElement('section')
  const panel = document.createElement('div')
  const header = document.createElement('header')
  const title = document.createElement('h2')
  const hint = document.createElement('p')
  const modeRow = document.createElement('div')
  const ruleModeButton = document.createElement('button')
  const llmModeButton = document.createElement('button')
  const textarea = document.createElement('textarea')
  const apiKeySection = document.createElement('section')
  const apiKeyLabel = document.createElement('label')
  const apiKeyInput = document.createElement('input')
  const apiKeyHint = document.createElement('p')
  const modelHint = document.createElement('p')
  const actionRow = document.createElement('div')
  const generateButton = document.createElement('button')
  const status = document.createElement('p')
  const jsonLabel = document.createElement('p')
  const jsonOutput = document.createElement('pre')
  const tiledLabel = document.createElement('p')
  const tiledOutput = document.createElement('pre')

  let eventDraftMode = readStoredEventDraftMode()
  let openAiApiKey = readStoredOpenAiApiKey()
  let isGenerating = false

  overlayRoot.className = 'event-draft-panel'
  panel.className = 'event-draft-panel__window'
  header.className = 'event-draft-panel__header'
  title.className = 'event-draft-panel__title'
  hint.className = 'event-draft-panel__hint'
  modeRow.className = 'event-draft-panel__mode-row'
  ruleModeButton.className = 'event-draft-panel__mode-button'
  llmModeButton.className = 'event-draft-panel__mode-button'
  textarea.className = 'event-draft-panel__textarea'
  apiKeySection.className = 'event-draft-panel__api-key'
  apiKeyLabel.className = 'event-draft-panel__api-key-label'
  apiKeyInput.className = 'event-draft-panel__api-key-input'
  apiKeyHint.className = 'event-draft-panel__hint'
  modelHint.className = 'event-draft-panel__hint'
  actionRow.className = 'event-draft-panel__actions'
  generateButton.className = 'event-draft-panel__button'
  status.className = 'event-draft-panel__status'
  jsonLabel.className = 'event-draft-panel__section-title'
  jsonOutput.className = 'event-draft-panel__output'
  tiledLabel.className = 'event-draft-panel__section-title'
  tiledOutput.className = 'event-draft-panel__output'

  title.textContent = '이벤트 초안'
  hint.textContent =
    '자연어를 입력하면 규칙 모드에서는 즉시 초안을 만들고, LLM 모드에서는 OpenAI Responses API를 호출한다.'
  textarea.value =
    '크리스마스 이벤트를 만들어줘. 산타 NPC가 등장하고 대화하면 선물을 주도록 해줘.'
  textarea.rows = 5
  textarea.spellcheck = false
  ruleModeButton.type = 'button'
  llmModeButton.type = 'button'
  generateButton.type = 'button'
  apiKeyLabel.htmlFor = 'event-draft-openai-api-key'
  apiKeyInput.id = 'event-draft-openai-api-key'
  apiKeyInput.type = 'password'
  apiKeyInput.value = openAiApiKey
  apiKeyInput.placeholder = 'sk-...'
  apiKeyInput.autocomplete = 'off'
  apiKeyInput.spellcheck = false
  apiKeyHint.textContent = 'API 키는 브라우저 localStorage에 저장한다.'
  modelHint.textContent = `LLM 모드 모델: ${DEFAULT_OPENAI_MODEL} / structured outputs 사용`
  jsonLabel.textContent = 'JSON 초안'
  tiledLabel.textContent = 'Tiled 미리보기'

  const setStatus = (message: string) => {
    status.textContent = message
  }

  const syncModeUi = () => {
    const isRuleMode = eventDraftMode === 'rule'

    ruleModeButton.textContent = isRuleMode ? '규칙 모드' : '규칙 모드'
    llmModeButton.textContent = isRuleMode ? 'LLM 모드' : 'LLM 모드'
    ruleModeButton.dataset.active = isRuleMode ? 'true' : 'false'
    llmModeButton.dataset.active = isRuleMode ? 'false' : 'true'
    generateButton.textContent = isRuleMode ? 'JSON 생성' : 'LLM 생성'
    apiKeySection.hidden = isRuleMode
    modelHint.hidden = isRuleMode
    apiKeyHint.hidden = isRuleMode
    apiKeyInput.disabled = isGenerating || isRuleMode
    generateButton.disabled = isGenerating
    generateButton.textContent = isRuleMode ? 'JSON 생성' : 'LLM 생성'
    ruleModeButton.disabled = isGenerating
    llmModeButton.disabled = isGenerating
  }

  const applyDraftToPreview = (
    draft: HolidayDialogueEventSpec,
    sourceLabel: string
  ) => {
    const validationErrors = createHolidayDialogueEventValidationErrors(draft)
    const tiledNpcObject = createTiledNpcEventObject(draft)

    jsonOutput.textContent = JSON.stringify(draft, null, 2)
    tiledOutput.textContent = JSON.stringify(tiledNpcObject, null, 2)

    if (validationErrors.length > 0) {
      setStatus(`검증 실패: ${validationErrors.join(', ')}`)
      return
    }

    const applyResult = activeSceneRenderer
      ? activeSceneRenderer.applyEventDraft(draft, {
          targetCharacterId: EVENT_DRAFT_TARGET_CHARACTER_ID
        })
      : undefined
    const appliedMessage = activeSceneRenderer
      ? applyResult?.didApply
        ? `적용됨: ${applyResult.targetCharacterId ?? EVENT_DRAFT_TARGET_CHARACTER_ID}`
        : '적용 실패: 씬에서 대상 NPC를 찾지 못했습니다.'
      : '씬 준비 전'

    setStatus(
      `${sourceLabel} 생성 완료: ${draft.event_name} / controller.scriptId = ${HOLIDAY_DIALOGUE_CONTROLLER_SCRIPT_ID} / ${appliedMessage}`
    )
  }

  const updateRulePreview = () => {
    const draft = createHolidayDialogueEventDraftFromText(textarea.value)

    if (!draft) {
      setStatus(
        '지원하지 않는 요청입니다. 크리스마스 또는 할로윈 이벤트를 입력해 주세요.'
      )
      jsonOutput.textContent = ''
      tiledOutput.textContent = ''
      return
    }

    applyDraftToPreview(draft, '규칙 모드')
  }

  const runLlmPreview = async () => {
    const trimmedApiKey = openAiApiKey.trim()

    if (trimmedApiKey.length === 0) {
      setStatus('LLM 모드에서는 OpenAI API 키가 필요합니다.')
      return
    }

    isGenerating = true
    syncModeUi()
    setStatus('OpenAI Responses API로 JSON 초안을 생성 중이다.')

    try {
      const draft = await generateHolidayDialogueEventDraftWithOpenAi({
        apiKey: trimmedApiKey,
        userPrompt: textarea.value
      })

      applyDraftToPreview(draft, 'LLM 모드')
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setStatus(`LLM 생성 실패: ${message}`)
    } finally {
      isGenerating = false
      syncModeUi()
    }
  }

  const updatePreview = () => {
    if (eventDraftMode === 'rule') {
      updateRulePreview()
      return
    }

    setStatus(
      'LLM 모드입니다. API 키를 입력한 뒤 JSON 생성 버튼을 눌러 OpenAI를 호출한다.'
    )
  }

  const switchMode = (nextMode: EventDraftMode) => {
    eventDraftMode = nextMode
    saveEventDraftMode(nextMode)
    syncModeUi()

    if (nextMode === 'rule') {
      updateRulePreview()
      return
    }

    setStatus(
      'LLM 모드입니다. API 키를 입력한 뒤 JSON 생성 버튼을 눌러 OpenAI를 호출한다.'
    )
  }

  refreshEventDraftPreview = updatePreview

  ruleModeButton.addEventListener('click', () => {
    switchMode('rule')
  })
  llmModeButton.addEventListener('click', () => {
    switchMode('llm')
  })
  generateButton.addEventListener('click', () => {
    if (eventDraftMode === 'rule') {
      updateRulePreview()
      return
    }

    void runLlmPreview()
  })
  textarea.addEventListener('input', () => {
    if (eventDraftMode === 'rule') {
      updateRulePreview()
      return
    }

    setStatus(
      'LLM 모드입니다. 변경한 문장을 반영하려면 JSON 생성 버튼을 다시 눌러야 한다.'
    )
  })
  textarea.addEventListener('keydown', (event) => {
    event.stopPropagation()

    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
      if (eventDraftMode === 'rule') {
        updateRulePreview()
      } else {
        void runLlmPreview()
      }
    }
  })
  textarea.addEventListener('keyup', (event) => {
    event.stopPropagation()
  })
  apiKeyInput.addEventListener('input', () => {
    openAiApiKey = apiKeyInput.value
    saveStoredOpenAiApiKey(openAiApiKey)
  })

  header.append(title, hint)
  modeRow.append(ruleModeButton, llmModeButton)
  apiKeyLabel.append('OpenAI API 키')
  apiKeySection.append(apiKeyLabel, apiKeyInput, apiKeyHint, modelHint)
  actionRow.append(generateButton)
  panel.append(
    header,
    modeRow,
    textarea,
    apiKeySection,
    actionRow,
    status,
    jsonLabel,
    jsonOutput,
    tiledLabel,
    tiledOutput
  )
  overlayRoot.append(panel)
  mountElement.append(overlayRoot)

  syncModeUi()
  updatePreview()

  return () => {
    if (refreshEventDraftPreview === updatePreview) {
      refreshEventDraftPreview = undefined
    }

    overlayRoot.remove()
  }
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

function readStoredPlayerSaveState(): PlayerSaveState | undefined {
  try {
    return parseStoredPlayerSaveState(
      window.localStorage.getItem(PLAYER_SAVE_STATE_STORAGE_KEY)
    )
  } catch {
    return undefined
  }
}

// 현재 플레이어 상태를 통째로 저장한다. playerProfile 은 렌더러가 in-place로 갱신하는
// 동일 객체라, 호출 시점의 최신 레벨·경험치·HP/MP·스탯이 그대로 담긴다.
function savePlayerState(): void {
  try {
    window.localStorage.setItem(
      PLAYER_SAVE_STATE_STORAGE_KEY,
      serializePlayerSaveState({
        profile: playerProfile,
        equipment: playerEquipment,
        inventory: playerInventory,
        quickslots: playerQuickslots,
        skillSlots: playerSkillSlots
      })
    )
  } catch {
    // localStorage 사용 불가(프라이빗 모드·용량 초과 등)면 저장을 조용히 건너뛴다.
  }
}

const renderFatalError = (error: unknown) => {
  const message = error instanceof Error ? error.message : String(error)

  appRootElement.innerHTML = `
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

  import.meta.hot.accept('./assets/lua/vn-dialogue.lua?raw', (nextModule) => {
    if (!nextModule || !activeControllerRuntime) {
      return
    }

    activeControllerRuntime.updateLuaControllerScript(vnDialogueScriptId, {
      source: nextModule.default
    })
  })
}

// 에디터(별도 page/iframe)가 이벤트를 저장하면 같은 origin의 다른 문서에서 storage 이벤트가
// 발생한다. 게임 프리뷰가 열려 있으면 새로고침 없이 즉시 적용한다.
// 에디터가 생성한 Lua 컨트롤러를 대상 NPC에 핫 적용한다. 런타임이 Lua를 검증·재빌드하므로
// 잘못된 코드면 throw 하고, 여기서 잡아 게임이 죽지 않게 한다(권위 있는 검증은 게임 측).
function applyPendingLuaScript(pendingLuaScript: PendingLuaScript): void {
  try {
    activeSceneRenderer?.applyLuaScript({
      targetCharacterId: pendingLuaScript.target_character_id,
      source: pendingLuaScript.source
    })
  } catch (error) {
    console.error(
      `Failed to apply pending Lua script for "${pendingLuaScript.target_character_id}".`,
      error
    )
  }
}

// 에디터(별도 page/iframe)가 이벤트/Lua를 저장하면 같은 origin의 다른 문서에서 storage 이벤트가
// 발생한다. 게임 프리뷰가 열려 있으면 새로고침 없이 즉시 적용한다.
window.addEventListener('storage', (event) => {
  // 에디터가 퀘스트를 주입하면 동적 퀘스트를 등록하고 현재 씬을 다시 부팅해 반영한다(렌더러는
  // 부팅 시 갱신된 questLog를 받으므로 배지·추적·완료가 모두 새 퀘스트에 작동한다).
  if (event.key === PENDING_QUESTS_STORAGE_KEY) {
    applyPendingQuests()
    void bootstrapScene(activeSceneId).catch(renderFatalError)
    return
  }

  // 배치 변경(에디터의 전체 지우기 등)은 게임이 즉시 다시 그린다.
  if (event.key === PENDING_PLACEMENTS_STORAGE_KEY) {
    activeSceneRenderer?.refreshPlacements()
    return
  }

  // NPC 변경(에디터의 전체 지우기 등)도 즉시 스폰/디스폰으로 반영한다.
  if (event.key === PENDING_NPCS_STORAGE_KEY) {
    activeSceneRenderer?.refreshNpcs()
    return
  }

  if (event.key === PENDING_EVENTS_STORAGE_KEY) {
    for (const pendingEvent of loadPendingEvents()) {
      activeSceneRenderer?.applyEventDraft(pendingEvent, {
        targetCharacterId: pendingEvent.npc.id
      })
    }
    return
  }

  if (event.key === PENDING_LUA_SCRIPTS_STORAGE_KEY) {
    for (const pendingLuaScript of loadPendingLuaScripts()) {
      applyPendingLuaScript(pendingLuaScript)
    }
  }
})

// 레벨·경험치·HP/MP·스탯 같은 프로필 변경은 렌더러가 콜백 없이 in-place로 바꾸므로,
// 페이지를 떠나기 직전에 현재 상태를 저장해 진행을 보존한다. visibilitychange 는
// beforeunload 가 신뢰도 낮은 모바일/탭 전환까지 잡아주는 보강 신호다.
window.addEventListener('beforeunload', savePlayerState)
window.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    savePlayerState()
  }
})

// 에디터 프리뷰(부모 창)의 메시지 처리: 씬 전환 + 음소거 토글.
window.addEventListener('message', (event) => {
  const data = event.data as {
    type?: unknown
    sceneId?: unknown
    isMuted?: unknown
    mode?: unknown
    template?: unknown
  } | null

  if (!data) {
    return
  }

  // 마우스 배치: 모드/놓을 항목을 게임 렌더러에 전달.
  if (data.type === 'editor:placement-mode') {
    const mode = data.mode
    if (mode === 'off' || mode === 'place' || mode === 'erase') {
      activeSceneRenderer?.setPlacementMode(mode)
    }
    return
  }
  if (data.type === 'editor:placement-template') {
    // 타일/오브젝트 배치 템플릿 또는 NPC 와이어 템플릿(kind:'npc') — 렌더러가 kind로 분기한다.
    activeSceneRenderer?.setPlacementTemplate(
      (data.template as PlacementTemplate | NpcWireTemplate | null) ?? null
    )
    return
  }

  if (data.type === 'editor:set-mute') {
    const nextSettings = { ...audioSettings, isMuted: data.isMuted === true }
    // 렌더러를 거치면 SFX 마스터 볼륨까지 갱신되고, 렌더러가 onAudioSettingsChange로
    // updateAudioSettings(저장 + BGM 반영)를 되부른다. 씬 로딩 중이면 직접 갱신만 한다.
    if (activeSceneRenderer) {
      activeSceneRenderer.updateAudioSettings(nextSettings)
    } else {
      updateAudioSettings(nextSettings)
    }
    return
  }

  if (data.type !== 'editor:switch-scene') {
    return
  }

  const sceneId = data.sceneId

  if (sceneId === 'town' || sceneId === 'hunting-ground' || sceneId === 'cave') {
    questLog = recordSceneEnterQuestProgress(questLog, sceneId)
    void bootstrapScene(sceneId).catch(renderFatalError)
  }
})

// 첫 씬을 띄우기 전에 Lua 게임 로직(전투/보상/드롭/표시/경험치)을 초기화한다.
// 실패해도(WASM 로드 불가 등) 퍼사드가 TS로 폴백하므로 게임은 계속 실행된다.
void initLuaGameLogic()
  .catch((error) => {
    console.warn('[lua] 게임 로직 Lua 초기화 실패 — TS 폴백으로 계속합니다.', error)
  })
  .finally(() => {
    void bootstrapScene('town').catch(renderFatalError)
  })
