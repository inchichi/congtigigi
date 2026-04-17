import sampleMapXml from './assets/maps/sample-map.tmx?raw'
import tinyDungeonTilesetXml from './assets/tilesets/tiny-dungeon-16.tsx?raw'
import tinyDungeonTilesetUrl from './assets/tilesets/tiny-dungeon-16.png'

import { createInitialPlayerState } from './game/playerState'
import { parseTiledMap } from './game/tiled/parseTiledMap'
import { createPixiTiledMapView } from './rendering/createPixiTiledMapView'
import './styles.css'

const rootElement = document.querySelector<HTMLDivElement>('#app')

if (!rootElement) {
  throw new Error('Missing #app root element')
}

const parsedSampleMap = parseTiledMap({
  mapXml: sampleMapXml,
  externalTilesets: {
    '../tilesets/tiny-dungeon-16.tsx': tinyDungeonTilesetXml
  }
})
const initialPlayerState = createInitialPlayerState({
  mapWidth: parsedSampleMap.width,
  mapHeight: parsedSampleMap.height
})

rootElement.className = 'game-root'

const bootstrap = async () => {
  await createPixiTiledMapView({
    mountElement: rootElement,
    map: parsedSampleMap,
    player: initialPlayerState,
    imageUrls: {
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
