import './editor.css'
import townMapXml from '../games/my-sample-rpg/assets/maps/town.tmx?raw'
import huntingGroundMapXml from '../games/my-sample-rpg/assets/maps/hunting-ground.tmx?raw'
import caveMapXml from '../games/my-sample-rpg/assets/maps/cave.tmx?raw'
import townTilesetXml from '../games/my-sample-rpg/assets/tilesets/town-32.tsx?raw'
import { createEditorApp } from './createEditorApp'
import type { GameFile } from './loadGame'

// 독립 에디터 페이지(/editor.html). 시작 시 내 게임(my-sample-rpg)의 맵을 기본 로드하고,
// "게임 폴더 열기"로 다른 게임 폴더(예: legend-of-lua)를 열 수 있다. 어느 게임인지는
// loadGame이 어댑터로 자동 판별한다.

const initialFiles: GameFile[] = [
  { name: 'town.tmx', path: 'src/games/my-sample-rpg/assets/maps/town.tmx', text: townMapXml },
  {
    name: 'hunting-ground.tmx',
    path: 'src/games/my-sample-rpg/assets/maps/hunting-ground.tmx',
    text: huntingGroundMapXml
  },
  { name: 'cave.tmx', path: 'src/games/my-sample-rpg/assets/maps/cave.tmx', text: caveMapXml },
  // 타일셋도 같이 넣는다 — 타일 군집 인식(tmxTileEntities)이 타일별 type 주석을 여기서 읽는다.
  {
    name: 'town-32.tsx',
    path: 'src/games/my-sample-rpg/assets/tilesets/town-32.tsx',
    text: townTilesetXml
  }
]

document.body.style.margin = '0'

const root = document.getElementById('editor-root')

if (!root) {
  throw new Error('editor-root element not found')
}

createEditorApp({ mountElement: root, initialFiles, gamePreviewUrl: '/' })
