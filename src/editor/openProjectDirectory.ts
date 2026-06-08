import type { GameFile } from './loadGame'

// File System Access API. 표준 lib.dom 타입에 일부가 없을 수 있어 최소 선언.
declare global {
  interface Window {
    showDirectoryPicker?: (options?: {
      mode?: 'read' | 'readwrite'
    }) => Promise<FileSystemDirectoryHandle>
  }
  interface FileSystemDirectoryHandle {
    entries(): AsyncIterableIterator<[string, FileSystemHandle]>
  }
}

export const isDirectoryPickerSupported = (): boolean =>
  typeof window.showDirectoryPicker === 'function'

const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', '.vite', 'coverage'])

// 엔티티 추출용 맵(.tmx)과 게임 판별용 시그니처(conf.lua/main.lua)만 모은다.
const isRelevantFile = (name: string): boolean =>
  name.endsWith('.tmx') || name === 'conf.lua' || name === 'main.lua'

const collectFiles = async (
  dir: FileSystemDirectoryHandle,
  prefix: string,
  out: GameFile[]
): Promise<void> => {
  for await (const [name, handle] of dir.entries()) {
    const path = prefix ? `${prefix}/${name}` : name

    if (handle.kind === 'file') {
      if (isRelevantFile(name)) {
        const file = await (handle as FileSystemFileHandle).getFile()
        out.push({ name, path, text: await file.text() })
      }
    } else if (handle.kind === 'directory' && !SKIP_DIRS.has(name)) {
      await collectFiles(handle as FileSystemDirectoryHandle, path, out)
    }
  }
}

// 사용자가 고른 게임 폴더에서 맵/시그니처 파일을 모아 raw 텍스트로 돌려준다.
// 파싱/어댑터 판별은 loadGame이 담당한다.
export const openProjectDirectory = async (): Promise<GameFile[]> => {
  if (!window.showDirectoryPicker) {
    throw new Error('이 브라우저는 폴더 열기를 지원하지 않습니다. Chrome 또는 Edge를 사용하세요.')
  }

  const directoryHandle = await window.showDirectoryPicker()
  const files: GameFile[] = []
  await collectFiles(directoryHandle, '', files)
  return files
}
