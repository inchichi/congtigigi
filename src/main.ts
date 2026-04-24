import townMapXml from './assets/maps/town.tmx?raw'
import townTilesetXml from './assets/tilesets/town-32.tsx?raw'
import townTilesetUrl from './assets/tilesets/town-32.png'
import tinyDungeonTilesetXml from './assets/tilesets/tiny-dungeon-16.tsx?raw'
import tinyDungeonTilesetUrl from './assets/tilesets/tiny-dungeon-16.png'

import { createInitialPlayerState } from './game/playerState'
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
const initialPlayerState = createInitialPlayerState({
  mapWidth: parsedTownMap.width,
  mapHeight: parsedTownMap.height
})
const tinyDungeonTileset = parseTiledTileset({
  firstGid: 1,
  source: '../tilesets/tiny-dungeon-16.tsx',
  tilesetXml: tinyDungeonTilesetXml
})

rootElement.className = 'game-root'

const bootstrap = async () => {
  await createPixiTiledMapView({
    mountElement: rootElement,
    map: parsedTownMap,
    player: initialPlayerState,
    playerSpriteSheet: {
      imageUrl: tinyDungeonTilesetUrl,
      tileWidth: 16,
      tileHeight: 16,
      columns: 12,
      localId: initialPlayerState.tileLocalId,
      scale: 2
    },
    eventSpriteSheet: {
      tileset: tinyDungeonTileset,
      scale: 2
    },
    imageUrls: {
      'town-32.png': townTilesetUrl,
      'tiny-dungeon-16.png': tinyDungeonTilesetUrl
    }
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
