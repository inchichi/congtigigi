import { parseTiledMap, type ParsedTiledMap } from '../game/tiled/parseTiledMap'
import type { GameFile } from './loadGame'

// 열린 폴더의 raw 파일들에서 한 맵(.tmx)을 라이브 프리뷰로 그릴 재료를 만든다:
// 외부 타일셋(.tsx) 텍스트와 타일셋 이미지 URL을 TMX의 상대경로대로 이어 맞춘 뒤 parseTiledMap을
// 통과시킨다. CSV·orthogonal·외부 타일셋 같은 파서 제약에 걸리면 throw 대신 친절한 에러로 돌려준다.

export type MapPreviewInputs = {
  map: ParsedTiledMap
  imageUrls: Record<string, string>
}

export type MapPreviewResult =
  | { ok: true; inputs: MapPreviewInputs }
  | { ok: false; error: string }

const dirname = (path: string): string => {
  const lastSlash = path.lastIndexOf('/')
  return lastSlash === -1 ? '' : path.slice(0, lastSlash)
}

const basename = (path: string): string => {
  const lastSlash = path.lastIndexOf('/')
  return lastSlash === -1 ? path : path.slice(lastSlash + 1)
}

// posix 스타일 상대경로 해석(./, ../ 정리). 수집된 path가 모두 '/' 구분자라 단순 처리로 충분하다.
const resolvePath = (baseDir: string, relativePath: string): string => {
  const segments = (baseDir ? baseDir.split('/') : []).concat(
    relativePath.split('/')
  )
  const stack: string[] = []

  for (const segment of segments) {
    if (segment === '' || segment === '.') {
      continue
    }

    if (segment === '..') {
      stack.pop()
      continue
    }

    stack.push(segment)
  }

  return stack.join('/')
}

// 정확한 상대경로로 먼저 찾고, 못 찾으면 파일명만으로 매칭한다(폴더 구조가 살짝 달라도 잇게).
const findFile = (
  files: GameFile[],
  resolvedPath: string
): GameFile | undefined => {
  const byPath = files.find((file) => file.path === resolvedPath)

  if (byPath) {
    return byPath
  }

  const wantedName = basename(resolvedPath)
  return files.find((file) => file.name === wantedName)
}

export const buildMapPreviewInputs = (
  files: GameFile[],
  mapPath: string
): MapPreviewResult => {
  const mapFile = files.find((file) => file.path === mapPath)

  if (!mapFile) {
    return { ok: false, error: `맵 파일을 찾지 못했습니다: ${mapPath}` }
  }

  const mapDir = dirname(mapPath)

  // TMX의 각 <tileset> 참조에서 source를 뽑는다. source가 있으면 외부 .tsx, 없으면 내장
  // (embedded) 타일셋이다. 외부 것만 .tsx 텍스트로 이어 주고, 내장 타일셋은 parseTiledMap이
  // TMX 본문에서 바로 읽으므로 여기서는 둘 필요가 없다.
  const tilesetTags = mapFile.text.match(/<tileset\b[^>]*>/gu) ?? []
  const externalTilesets: Record<string, string> = {}

  for (const tag of tilesetTags) {
    const sourceMatch = tag.match(/\bsource="([^"]+)"/u)

    if (!sourceMatch) {
      continue
    }

    const source = sourceMatch[1]
    const tsxFile = findFile(files, resolvePath(mapDir, source))

    if (!tsxFile) {
      return { ok: false, error: `타일셋(.tsx)을 찾지 못했습니다: ${source}` }
    }

    externalTilesets[source] = tsxFile.text
  }

  let map: ParsedTiledMap

  try {
    map = parseTiledMap({ mapXml: mapFile.text, externalTilesets })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return { ok: false, error: `맵을 해석하지 못했습니다: ${message}` }
  }

  // 타일셋 이미지 경로의 기준이 다르다: 외부 .tsx 타일셋은 그 .tsx 파일 기준, 내장 타일셋은
  // .tmx 파일 기준 상대경로다(parseTiledMap이 내장 타일셋의 source를 'embedded:'로 표시한다).
  const imageUrls: Record<string, string> = {}

  for (const tileset of map.tilesets) {
    const baseDir = tileset.source.startsWith('embedded:')
      ? mapDir
      : dirname(resolvePath(mapDir, tileset.source))
    const imageFile = findFile(files, resolvePath(baseDir, tileset.image.source))

    if (!imageFile?.url) {
      return {
        ok: false,
        error: `타일셋 이미지를 찾지 못했습니다: ${tileset.image.source}`
      }
    }

    imageUrls[tileset.image.source] = imageFile.url
  }

  return { ok: true, inputs: { map, imageUrls } }
}
