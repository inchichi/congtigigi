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

// FLUX 스타일 서비스가 원격 GPU 서버의 에셋을 덮어쓰는 구성일 때, 서버 쪽 변경분을
// 로컬 게임 파일로 미러링한다(dev 전용 엔드포인트). 실패해도 에디터 동작에는 지장 없다.
void fetch('/__sync-styled-assets', { method: 'POST' }).catch(() => {})

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
