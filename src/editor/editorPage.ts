import './editor.css'
import townMapXml from '../assets/maps/town.tmx?raw'
import huntingGroundMapXml from '../assets/maps/hunting-ground.tmx?raw'
import caveMapXml from '../assets/maps/cave.tmx?raw'
import { createEditorApp } from './createEditorApp'
import type { GameFile } from './loadGame'

// 독립 에디터 페이지(/editor.html). 시작 시 내 게임(my-sample-rpg)의 맵을 기본 로드하고,
// "게임 폴더 열기"로 다른 게임 폴더(예: legend-of-lua)를 열 수 있다. 어느 게임인지는
// loadGame이 어댑터로 자동 판별한다.

const initialFiles: GameFile[] = [
  { name: 'town.tmx', path: 'src/assets/maps/town.tmx', text: townMapXml },
  {
    name: 'hunting-ground.tmx',
    path: 'src/assets/maps/hunting-ground.tmx',
    text: huntingGroundMapXml
  },
  { name: 'cave.tmx', path: 'src/assets/maps/cave.tmx', text: caveMapXml }
]

document.body.style.margin = '0'

const root = document.getElementById('editor-root')

if (!root) {
  throw new Error('editor-root element not found')
}

createEditorApp({ mountElement: root, initialFiles, gamePreviewUrl: '/' })
