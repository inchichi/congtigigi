import './editor.css'
import { createEditorApp } from './createEditorApp'
import { createThemeWorkflowPage } from './createThemeWorkflowPage'
import { SAMPLE_GAME_FILES } from './sampleGameFiles'

// 독립 에디터 페이지(/editor.html). 시작 시 내 게임(my-sample-rpg)의 맵을 기본 로드하고,
// "게임 폴더 열기"로 다른 게임 폴더(예: legend-of-lua)를 열 수 있다. 어느 게임인지는
// loadGame이 어댑터로 자동 판별한다.

document.body.style.margin = '0'

const root = document.getElementById('editor-root')

if (!root) {
  throw new Error('editor-root element not found')
}

const params = new URLSearchParams(window.location.search)
const workspace = params.get('workspace')

if (workspace === 'theme') {
  createThemeWorkflowPage({
    mountElement: root,
    initialFiles: SAMPLE_GAME_FILES,
    initialPrompt: params.get('prompt') ?? ''
  })
} else {
  createEditorApp({ mountElement: root, initialFiles: SAMPLE_GAME_FILES, gamePreviewUrl: '/' })
}
