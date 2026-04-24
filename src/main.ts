import townMapXml from './assets/maps/town.tmx?raw'
import wanderNearHomeControllerLua from './assets/lua/wander-near-home.lua?raw'
import townTilesetXml from './assets/tilesets/town-32.tsx?raw'
import townTilesetUrl from './assets/tilesets/town-32.png'
import tinyDungeonTilesetXml from './assets/tilesets/tiny-dungeon-16.tsx?raw'
import tinyDungeonTilesetUrl from './assets/tilesets/tiny-dungeon-16.png'

import {
  PLAYER_CHARACTER_ID,
  createLuaCharacterController,
  createInitialPlayerCharacter
} from './game/characterState'
import { createLuaCharacterControllerRuntime } from './game/lua/createLuaCharacterControllerRuntime'
import { createNpcCharactersFromEventLayers } from './game/tiled/createNpcCharactersFromEventLayers'
import { parseTiledMap, parseTiledTileset } from './game/tiled/parseTiledMap'
import { createPixiTiledMapView } from './rendering/createPixiTiledMapView'
import './styles.css'

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
const tinyDungeonTileset = parseTiledTileset({
  firstGid: 1,
  source: '../tilesets/tiny-dungeon-16.tsx',
  tilesetXml: tinyDungeonTilesetXml
})
const characterSpriteScale = 2
const initialCharacters = [
  createInitialPlayerCharacter({
    mapWidth: parsedTownMap.width,
    mapHeight: parsedTownMap.height
  }),
  ...createNpcCharactersFromEventLayers({
    map: parsedTownMap,
    defaultPixelWidth: tinyDungeonTileset.tileWidth * characterSpriteScale,
    defaultPixelHeight: tinyDungeonTileset.tileHeight * characterSpriteScale
  })
].map((character) =>
  character.id !== 'villager_1'
    ? character
    : {
        ...character,
        controller: createLuaCharacterController({
          scriptId: 'wander-near-home',
          radiusInTiles: 2,
          moveSpeedTilesPerSecond: 1.5
        })
      }
)

rootElement.className = 'game-root'

const bootstrap = async () => {
  const hasLuaControlledCharacter = initialCharacters.some(
    (character) => character.controller.kind === 'lua'
  )
  const luaControllerRuntime = hasLuaControlledCharacter
    ? await createLuaCharacterControllerRuntime({
        scriptsById: {
          'wander-near-home': {
            registerFunctionName: 'register_wander_controller',
            stepFunctionName: 'step_wander_controller',
            source: wanderNearHomeControllerLua
          }
        }
      })
    : undefined

  await createPixiTiledMapView({
    mountElement: rootElement,
    map: parsedTownMap,
    characters: initialCharacters,
    cameraTargetCharacterId: PLAYER_CHARACTER_ID,
    characterSpriteSheet: {
      tileset: tinyDungeonTileset,
      scale: characterSpriteScale
    },
    imageUrls: {
      'town-32.png': townTilesetUrl,
      'tiny-dungeon-16.png': tinyDungeonTilesetUrl
    },
    luaControllerRuntime
  })
}

void bootstrap().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error)

  rootElement.innerHTML = `
    <div class="error-panel">
      <h2>Renderer Failed</h2>
      <p>${message}</p>
    </div>
  `
})
