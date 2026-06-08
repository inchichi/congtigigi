import './styles.css'
import townMapXml from './assets/maps/town.tmx?raw'
import huntingGroundMapXml from './assets/maps/hunting-ground.tmx?raw'
import caveMapXml from './assets/maps/cave.tmx?raw'
import townTilesetXml from './assets/tilesets/town-32.tsx?raw'
import { parseTiledMap } from './game/tiled/parseTiledMap'
import { createLlmPanel } from './editor/createLlmPanel'
import { savePendingEvent } from './editor/pendingEvents'
import type { EditorMapSource } from './editor/buildGameStructureProfile'

// 독립 에디터 페이지. 게임(/)과 분리된 별도 page(/editor.html)에서 에디터만 실행한다.
// 게임 런타임이 없으므로, .tmx를 직접 파싱해 구조 프로필을 만들고,
// '게임 적용'은 localStorage 아티팩트로 저장한다(게임이 다음 로드 때 읽어 적용).

const externalTilesets = { '../tilesets/town-32.tsx': townTilesetXml }

const mapSources: EditorMapSource[] = [
  {
    id: 'town',
    name: 'Town',
    file: 'src/assets/maps/town.tmx',
    map: parseTiledMap({ mapXml: townMapXml, externalTilesets })
  },
  {
    id: 'hunting-ground',
    name: 'Hunting Ground',
    file: 'src/assets/maps/hunting-ground.tmx',
    map: parseTiledMap({ mapXml: huntingGroundMapXml, externalTilesets })
  },
  {
    id: 'cave',
    name: 'Cave',
    file: 'src/assets/maps/cave.tmx',
    map: parseTiledMap({ mapXml: caveMapXml, externalTilesets })
  }
]

document.body.style.margin = '0'
document.body.style.minHeight = '100vh'
document.body.style.background =
  'radial-gradient(1200px 600px at 20% -10%, #1b1f2b 0%, #0b0d12 60%)'

const root = document.getElementById('editor-root')

if (!root) {
  throw new Error('editor-root element not found')
}

const panel = createLlmPanel({
  mountElement: root,
  getSceneRenderer: () => ({
    applyEventDraft: (spec) => {
      savePendingEvent(spec)
      return { didApply: true, targetCharacterId: spec.npc.id }
    }
  }),
  mapSources
})

panel.open()
