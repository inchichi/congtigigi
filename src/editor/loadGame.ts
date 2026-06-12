import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { detectAdapter, type GameAdapter, type GameEntity } from './gameAdapter'
import { extractTmxLayerNames, extractTmxObjects, type TmxObject } from './tmxObjects'
import {
  extractTmxObjectCellsBatch,
  extractTmxObjectCellsInRect,
  extractTmxTileClusterDetails,
  extractTmxTileClusters,
  type TmxTileCluster,
  type TmxTileClusterCell,
  type TmxTileClusterDetail,
  type TmxTileExcludeRect
} from './tmxTileEntities'
import type { GameStructureProfile } from './gameStructureProfile'

export type GameFile = {
  name: string
  path: string
  // 텍스트 자산(.tmx/.tsx/.lua)의 원문. 바이너리(이미지)면 빈 문자열.
  text: string
  // 바이너리 자산(타일셋 이미지)의 로드 가능한 URL(blob/object URL). 텍스트 자산이면 undefined.
  url?: string
}

export type LoadedGameMap = {
  id: string
  name: string
  file: string
  entities: GameEntity[]
  // 타일/이미지 레이어 이름들("ground" 등). 객체가 아니라 "맵 자체"의 구성으로, 에디터에 보기 전용 표시.
  layers: string[]
}

export type LoadedGame = {
  adapter: GameAdapter
  maps: LoadedGameMap[]
  // rpg(내 게임)만: 생성 파이프라인(openaiEventJsonGenerator)이 쓰는 구조 프로필.
  profile?: GameStructureProfile
  // 파싱에 실패한 맵 파일 경로들. loadGame 자체는 throw하지 않고(에디터가 통째로 안 뜨는 일 방지)
  // 여기에 모아, 호출부가 status로 알릴 수 있게 한다.
  parseErrors: string[]
}

export const loadGame = (files: GameFile[]): LoadedGame => {
  const adapter = detectAdapter(files.map((file) => file.name))
  const parseErrors: string[] = []

  const maps: LoadedGameMap[] = files
    .filter((file) => file.name.endsWith('.tmx'))
    .map((file) => {
      const id = file.name.replace(/\.tmx$/u, '')
      // 한 맵이 깨졌다고 전체 로드를 죽이지 않는다. 실패한 맵은 엔티티 0개로 두고 경로만 모은다.
      let entities: GameEntity[] = []
      try {
        const objects = extractTmxObjects(file.text)
        entities = adapter.extractEntities(id, objects).concat(
          // object layer에 없는 "타일로 그려진" 구조물(나무·분수·가로등 등)도 종류별로 인식해
          // 트리에 보여준다(보기 전용). 이미 object로 등록된 영역(건물 등)과 겹치는 군집은 중복 제외.
          buildTileClusterEntities(id, file, files, objects)
        )
      } catch {
        parseErrors.push(file.path)
      }
      // 레이어 이름은 표시용 부가 정보라 파싱 실패해도 throw하지 않고 []를 돌려준다(엔티티와 별개).
      return { id, name: id, file: file.path, entities, layers: extractTmxLayerNames(file.text) }
    })

  const profile: GameStructureProfile | undefined =
    adapter.id === 'my-sample-rpg'
      ? {
          ...CURRENT_GAME_PROJECT_PROFILE,
          maps: maps.map((map) => ({ id: map.id, name: map.name, file: map.file })),
          // profile.npcs는 의도적으로 "편집 가능한(대화) NPC" 집합으로만 큐레이션한다(kind==='npc').
          // 트리는 몬스터·표지판·포털까지 다 보여주지만(rpgAdapter.extractEntities), 대화 이벤트 생성은
          // NPC만 대상이라, LLM 프롬프트(허용 NPC 목록)·검증·생성이 이 좁은 집합으로 일관돼야 한다.
          // 여기서 NPC만 거르지 않으면 몬스터가 "유효한 대화 대상"으로 새어 들어가 검증이 오염된다.
          npcs: maps.flatMap((map) =>
            map.entities
              .filter((entity) => entity.kind === 'npc')
              .map((entity) => ({
                id: entity.id,
                name: entity.name,
                map: map.id,
                file: map.file
              }))
          )
        }
      : undefined

  return { adapter, maps, profile, parseErrors }
}

// 파일 기준 상대 참조("../tilesets/town-32.tsx")를 그 파일의 경로 기준으로 푼다.
export const resolveRelativePath = (fromFilePath: string, source: string): string => {
  const segments = fromFilePath.split('/').slice(0, -1)
  for (const part of source.split('/')) {
    if (part === '..') {
      segments.pop()
    } else if (part !== '.' && part !== '') {
      segments.push(part)
    }
  }
  return segments.join('/')
}

// 맵의 타일셋 참조를 맵 파일 경로 기준으로 풀어, 열린 파일들에서 찾는다.
// 폴더 업로드마다 path 형태가 달라서(절대/상대 혼재) 정규화한 suffix 일치 → 파일명 일치 순으로 찾는다.
export const findFileByRelativeSource = (
  files: GameFile[],
  fromFilePath: string,
  source: string
): GameFile | undefined => {
  const resolved = resolveRelativePath(fromFilePath, source)
  const baseName = source.split('/').pop() ?? source
  const bySuffix = files.find(
    (file) => file.path === resolved || file.path.endsWith(`/${resolved}`)
  )
  return bySuffix ?? files.find((file) => file.name === baseName)
}

const createTilesetResolver =
  (files: GameFile[], mapPath: string) =>
  (source: string): string | undefined =>
    findFileByRelativeSource(files, mapPath, source)?.text

// 타일 군집에서 만들어진 보기 전용 엔티티인지. 트리에는 보여주되, 생성 대상 선택·LLM 분석
// 폴백 판단("어댑터가 아무것도 못 찾은 게임인가")에서는 제외해야 한다.
export const isTileClusterEntity = (entity: GameEntity): boolean =>
  entity.id.startsWith('tile:')

export const buildTileClusterEntities = (
  mapId: string,
  mapFile: GameFile,
  files: GameFile[],
  objects: TmxObject[]
): GameEntity[] => {
  // 같은 종류의 두 군집이 bbox 좌상단을 공유할 수 있어(ㄱ자 군집 옆 1타일, 흡수로 늘어난 bbox)
  // 좌표 기반 id에 일련번호를 붙여 맵 안에서 유일하게 만든다.
  const seen = new Map<string, number>()
  return extractTmxTileClusters(mapFile.text, {
    resolveTilesetText: createTilesetResolver(files, mapFile.path),
    // 면적이 있는 object(건물·분수 등)와 절반 이상 겹치는 군집은 같은 사물의 중복 — 제외.
    // object의 type을 kind로 넘겨, 독립 사물(분수·나무 등)은 같은 종류 object에만 억제되게 한다.
    excludeRects: objects
      .filter((object) => object.width > 0 && object.height > 0)
      .map((object) => ({
        x: object.x,
        y: object.y,
        width: object.width,
        height: object.height,
        kind: object.type
      }))
  }).map((cluster) => {
    const base = `tile:${cluster.kind}:${cluster.tileX},${cluster.tileY}`
    const duplicates = seen.get(base) ?? 0
    seen.set(base, duplicates + 1)
    return {
      id: duplicates === 0 ? base : `${base}#${duplicates}`,
      name: tileClusterName(cluster),
      kind: cluster.kind,
      mapId
    }
  })
}

// 타일 군집에는 이름이 없어서 위치(타일 좌표)와 크기로 구분한다. 종류는 트리 그룹 헤더가 보여준다.
const tileClusterName = (cluster: TmxTileCluster): string =>
  cluster.widthTiles > 1 || cluster.heightTiles > 1
    ? `(${cluster.tileX}, ${cluster.tileY}) · ${cluster.widthTiles}×${cluster.heightTiles}`
    : `(${cluster.tileX}, ${cluster.tileY})`

// 부분 스타일 변환용 셀 묶음 — 군집 엔티티와 영역 오브젝트 엔티티가 같은 형태로 돌려준다.
export type StyleTargetCells = {
  cells: TmxTileClusterCell[]
  tilesetSource?: string
  sharedOutsideCells: number
}

// 영역 오브젝트(이름 있는 사각형 — 건물·분수·나무 장식 등) 엔티티의 부분 변환 셀을 모은다.
// 엔티티 id는 rpgAdapter 규칙(object.name, 없으면 `${kind}-${object.id}`)을 따라 역매칭한다.
export const findObjectKindCells = (
  mapFile: GameFile,
  files: GameFile[],
  objects: TmxObject[],
  entity: GameEntity
): StyleTargetCells | undefined => {
  const object =
    objects.find((candidate) => candidate.name === entity.id) ??
    objects.find((candidate) => `${entity.kind}-${candidate.id}` === entity.id)
  if (!object || object.width <= 0 || object.height <= 0) {
    return undefined
  }
  // objectKind/objectRects: 사각형이 이웃 구조물과 겹칠 때 종류가 다른 공유 타일(예: 나무
  // 사각형에 걸친 건물 창문 — 마을 전체 창문과 공유)이 수집되는 오염을 막기 위한 문맥.
  return extractTmxObjectCellsInRect(
    mapFile.text,
    object,
    { resolveTilesetText: createTilesetResolver(files, mapFile.path) },
    {
      objectKind: object.type || entity.kind,
      objectRects: objects
        .filter((candidate) => candidate.width > 0 && candidate.height > 0)
        .map((candidate) => ({
          x: candidate.x,
          y: candidate.y,
          width: candidate.width,
          height: candidate.height,
          kind: candidate.type
        }))
    }
  )
}

// 맵 인식 시점의 일괄 수집: 엔티티마다 TMX를 재파싱하면(엔티티 40개 × xmldom 파싱) 메인
// 스레드가 수백 ms 멈춘다. 군집 상세 1회 + 영역 오브젝트 배치 1회 — 파싱 2회로 끝낸다.
export const findAllStyleTargetCells = (
  mapFile: GameFile,
  files: GameFile[],
  objects: TmxObject[],
  entities: GameEntity[]
): Map<string, StyleTargetCells> => {
  const out = new Map<string, StyleTargetCells>()
  const options = { resolveTilesetText: createTilesetResolver(files, mapFile.path) }
  const areaRects: TmxTileExcludeRect[] = objects
    .filter((object) => object.width > 0 && object.height > 0)
    .map((object) => ({
      x: object.x,
      y: object.y,
      width: object.width,
      height: object.height,
      kind: object.type
    }))

  // 군집 엔티티: 한 번의 상세 추출 + buildTileClusterEntities와 동일한 id 재현.
  const seen = new Map<string, number>()
  for (const cluster of extractTmxTileClusterDetails(mapFile.text, {
    ...options,
    excludeRects: areaRects
  })) {
    const base = `tile:${cluster.kind}:${cluster.tileX},${cluster.tileY}`
    const duplicates = seen.get(base) ?? 0
    seen.set(base, duplicates + 1)
    out.set(duplicates === 0 ? base : `${base}#${duplicates}`, cluster)
  }

  // 영역 오브젝트 엔티티: 사각형들을 모아 배치로 수집.
  const matched = entities
    .filter((entity) => !isTileClusterEntity(entity))
    .map((entity) => {
      const object =
        objects.find((candidate) => candidate.name === entity.id) ??
        objects.find((candidate) => `${entity.kind}-${candidate.id}` === entity.id)
      return object && object.width > 0 && object.height > 0 ? { entity, object } : undefined
    })
    .filter(
      (pair): pair is { entity: GameEntity; object: TmxObject } => pair !== undefined
    )
  const results = extractTmxObjectCellsBatch(
    mapFile.text,
    matched.map(({ entity, object }) => ({
      rect: object,
      objectKind: object.type || entity.kind
    })),
    options,
    areaRects
  )
  matched.forEach(({ entity }, index) => {
    const result = results[index]
    if (result) {
      out.set(entity.id, result)
    }
  })
  return out
}

// 부분 스타일 변환용: 군집 엔티티 id로 상세 군집(셀·타일 id 목록)을 되찾는다.
// buildTileClusterEntities와 같은 입력·같은 옵션으로 다시 추출하면 군집 순서가 결정적으로
// 일치하므로, id 생성 규칙(좌표 base + 일련번호)을 그대로 재현해 짝을 맞춘다.
export const findTileClusterDetail = (
  mapFile: GameFile,
  files: GameFile[],
  objects: TmxObject[],
  entityId: string
): TmxTileClusterDetail | undefined => {
  const seen = new Map<string, number>()
  for (const cluster of extractTmxTileClusterDetails(mapFile.text, {
    resolveTilesetText: createTilesetResolver(files, mapFile.path),
    excludeRects: objects
      .filter((object) => object.width > 0 && object.height > 0)
      .map((object) => ({
        x: object.x,
        y: object.y,
        width: object.width,
        height: object.height,
        kind: object.type
      }))
  })) {
    const base = `tile:${cluster.kind}:${cluster.tileX},${cluster.tileY}`
    const duplicates = seen.get(base) ?? 0
    seen.set(base, duplicates + 1)
    const id = duplicates === 0 ? base : `${base}#${duplicates}`
    if (id === entityId) {
      return cluster
    }
  }
  return undefined
}
