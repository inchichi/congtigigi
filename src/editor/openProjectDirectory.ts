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

const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']

const isImageFile = (name: string): boolean =>
  IMAGE_EXTENSIONS.some((extension) => name.toLowerCase().endsWith(extension))

// 라이브 맵 프리뷰를 그리려면 타일셋 정의(.tsx)와 그 이미지(.png 등)도 필요하다.
const isAssetFile = (name: string): boolean =>
  name.endsWith('.tsx') || isImageFile(name)

// 엔티티 추출용 맵(.tmx)·게임 판별용 시그니처(conf.lua/main.lua) + 프리뷰용 타일셋 자산.
const isRelevantFile = (name: string): boolean =>
  name.endsWith('.tmx') ||
  name === 'conf.lua' ||
  name === 'main.lua' ||
  isAssetFile(name)

const readGameFile = async (
  name: string,
  path: string,
  handle: FileSystemFileHandle
): Promise<GameFile> => {
  const file = await handle.getFile()

  // 이미지는 텍스트로 못 읽으므로 object URL로 들고 다닌다(Pixi가 이걸로 텍스처를 만든다).
  if (isImageFile(name)) {
    return { name, path, text: '', url: URL.createObjectURL(file) }
  }

  return { name, path, text: await file.text() }
}

const collectFiles = async (
  dir: FileSystemDirectoryHandle,
  prefix: string,
  out: GameFile[],
  // _old 같은 백업 폴더에 든 옛 .tmx가 엔티티 트리를 오염시키지 않게, _ 폴더는 자산만 수집한다.
  assetsOnly: boolean
): Promise<void> => {
  for await (const [name, handle] of dir.entries()) {
    const path = prefix ? `${prefix}/${name}` : name

    if (handle.kind === 'file') {
      const include = assetsOnly ? isAssetFile(name) : isRelevantFile(name)

      if (include) {
        out.push(await readGameFile(name, path, handle as FileSystemFileHandle))
      }
    } else if (handle.kind === 'directory' && !SKIP_DIRS.has(name)) {
      await collectFiles(
        handle as FileSystemDirectoryHandle,
        path,
        out,
        assetsOnly || name.startsWith('_')
      )
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
  await collectFiles(directoryHandle, '', files, false)
  return files
}
