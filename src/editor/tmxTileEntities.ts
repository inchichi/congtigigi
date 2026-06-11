import { DOMParser } from '@xmldom/xmldom'

// 타일 레이어에 "그려진" 구조물(나무·분수·가로등·울타리 등)을 종류별 군집으로 인식한다.
// TMX의 object layer에는 이름 있는 개체만 있고, 맵의 대부분은 타일 번호 격자라 그대로는
// "무엇이 어디 있는지"를 알 수 없다. 대신 타일셋(.tsx)의 타일별 type 주석(예: tree_large_canopy,
// fountain_basin_left)을 읽어 타일 → 종류로 분류하고, 같은 종류가 상하좌우로 이어진 영역을
// 하나의 군집(=트리에 보여줄 한 항목)으로 묶는다.
export type TmxTileClusterRect = {
  x: number
  y: number
  width: number
  height: number
}

export type TmxTileCluster = {
  kind: string
  tileX: number
  tileY: number
  widthTiles: number
  heightTiles: number
  // 픽셀 단위 바운딩 박스(object layer의 사각형과 같은 좌표계) — 중복 판정·표시용.
  rect: TmxTileClusterRect
}

export type ExtractTmxTileClustersOptions = {
  // 외부 타일셋(.tsx) 텍스트를 source 경로("../tilesets/town-32.tsx")로 찾아준다.
  // 없으면(미제공/못 찾음) 그 타일셋의 타일들은 분류 불가로 조용히 무시된다.
  resolveTilesetText?: (source: string) => string | undefined
  // 이 사각형들(픽셀)과 절반 이상 겹치는 군집은 버린다 — 이미 object로 등록된 건물 등과의
  // 중복 방지(예: town_hall 객체가 있는데 지붕 타일 군집이 또 "건물"로 잡히는 것).
  excludeRects?: TmxTileClusterRect[]
}

// 타일 type의 첫 단어 → 에디터 종류. 표에 없는 첫 단어(ground/grass/cobble/garden/cliff 등
// 지형·바닥·절벽 채움)는 "사물"이 아니라 맵 표면이라 군집으로 잡지 않는다. 이름에 shadow가
// 들어간 타일(그림자)도 제외. wall은 building에 합치지 않는다 — 동굴처럼 벽만으로 이루어진
// 맵에서 벽 전체가 "건물"로 잡히면 틀린 말이 된다(건물 지붕·창문은 어차피 wall과 붙어 있어
// 같은 군집으로 안 묶여도 object 등록 건물에 흡수/중복 제거된다).
const KIND_BY_TYPE_PREFIX: Record<string, string> = {
  roof: 'building',
  chimney: 'building',
  gable: 'building',
  ladder: 'building',
  window: 'window',
  wall: 'wall',
  market: 'tent',
  clocktower: 'clocktower',
  fountain: 'fountain',
  streetlamp: 'lamp',
  banner: 'banner',
  tree: 'tree',
  stump: 'tree',
  bush: 'tree',
  hedge: 'hedge',
  planter: 'flower',
  flower: 'flower',
  barrel: 'prop',
  crate: 'prop',
  bottle: 'prop',
  basket: 'prop',
  bucket: 'prop',
  shovel: 'prop',
  metal: 'prop',
  town: 'prop',
  post: 'prop',
  rocks: 'rock',
  stairs: 'stairs',
  sign: 'sign'
}

// 이름은 사물 같지만 실제로는 바닥/지형/외장 트림으로 쓰이는 채움 타일들 — 정확한 이름 단위로
// 제외한다. (wall_cobble_fill: 마을 광장 바닥 포장 텍스처. clocktower_roof_top_02: 성벽 아래
// 장식 띠에 단독으로 박히는 지붕 마감 조각. facade_segment_*는 prefix 표에서 통째로 빠져 있다 —
// 성벽 외장 트림이라 독립 사물이 아니다.)
const EXCLUDED_TILE_TYPES = new Set(['wall_cobble_fill', 'clocktower_roof_top_02'])

// 중복 제거(excludeRects)를 적용할 종류들: object로 등록된 사각형(건물 등)과 "같은 구조물"일 수
// 있는 건물 구성 요소만. 분수·나무·소품처럼 독립적인 사물은 건물 영역 안에 있어도(입구 옆 분수,
// 문 앞 통) 별개의 것이라 남긴다.
const STRUCTURAL_KINDS = new Set(['building', 'wall', 'window', 'tent', 'clocktower', 'stairs'])

// 한 사물이 여러 종류 타일로 그려져 군집이 쪼개지는 알려진 경우들. 이런 군집이 대상 종류와
// 맞닿아 있으면 그쪽(배열 앞 우선)으로 흡수한다. 맞닿은 게 없으면 제 종류로 남는다.
// - clocktower: 가로등 돌기둥·성문 기둥이 시계탑 기둥 타일(clocktower_face_*)을 재사용한다.
// - window: 창문은 건물 벽이나 성벽에 박힌 부속이지 독립 사물이 아니다.
const ABSORB_KIND_INTO: Record<string, string[]> = {
  clocktower: ['lamp', 'wall'],
  window: ['building', 'wall']
}

// Tiled가 회전/반전 타일에 켜는 상위 비트 — 타일 id를 읽으려면 벗겨야 한다.
const GID_TRANSFORM_FLAGS_MASK = 0x0fffffff

const parseLenientXml = (text: string): Document | undefined => {
  try {
    return new DOMParser({
      errorHandler: { warning: () => {}, error: () => {}, fatalError: () => {} }
    }).parseFromString(text, 'text/xml') as Document | undefined
  } catch {
    return undefined
  }
}

const directChildren = (parent: Element, tagName: string): Element[] => {
  const out: Element[] = []
  for (let i = 0; i < parent.childNodes.length; i += 1) {
    const node = parent.childNodes[i]
    if (node.nodeType === 1 && (node as Element).tagName === tagName) {
      out.push(node as Element)
    }
  }
  return out
}

const numberAttribute = (element: Element, name: string): number | undefined => {
  const raw = element.getAttribute(name)
  if (raw === null || raw === '') {
    return undefined
  }
  const value = Number(raw)
  return Number.isFinite(value) ? value : undefined
}

// <tileset> 한 개의 타일 id → type 표. Tiled 신버전은 class=, 구버전은 type= 를 쓴다.
const collectTileTypes = (tilesetElement: Element): Map<number, string> => {
  const types = new Map<number, string>()
  for (const tile of directChildren(tilesetElement, 'tile')) {
    const id = numberAttribute(tile, 'id')
    // ??가 아니라 || 인 이유: @xmldom/xmldom은 없는 속성에 null이 아니라 ''를 돌려줘서,
    // nullish 폴백으로는 class 부재 시 type으로 넘어가지 못한다.
    const type = tile.getAttribute('class') || tile.getAttribute('type')
    if (id !== undefined && type) {
      types.set(id, type)
    }
  }
  return types
}

const kindForTileType = (type: string): string | undefined => {
  // 그림자 타일은 사물 본체가 아니라 옆 칸에 드리운 효과라, 군집 모양만 부풀린다 — 제외.
  if (type.includes('shadow') || EXCLUDED_TILE_TYPES.has(type)) {
    return undefined
  }
  return KIND_BY_TYPE_PREFIX[type.split('_')[0]]
}

// Tiled의 <group> 폴더 안에 중첩된 타일 레이어도 문서 순서대로 모은다(나중 레이어가 이긴다는
// 병합 규칙이 유지되도록).
const collectTileLayers = (parent: Element): Element[] => {
  const out: Element[] = []
  for (let i = 0; i < parent.childNodes.length; i += 1) {
    const node = parent.childNodes[i]
    if (node.nodeType !== 1) {
      continue
    }
    const element = node as Element
    if (element.tagName === 'layer') {
      out.push(element)
    } else if (element.tagName === 'group') {
      out.push(...collectTileLayers(element))
    }
  }
  return out
}

// 표시용 부가 정보라 extractTmxLayerNames처럼 어떤 입력에도 throw하지 않는다 — 못 읽으면 [].
export const extractTmxTileClusters = (
  mapText: string,
  options: ExtractTmxTileClustersOptions = {}
): TmxTileCluster[] => {
  const doc = parseLenientXml(mapText)
  if (!doc || !doc.documentElement || doc.documentElement.tagName !== 'map') {
    return []
  }
  const mapElement = doc.documentElement as unknown as Element

  const width = numberAttribute(mapElement, 'width')
  const height = numberAttribute(mapElement, 'height')
  const tileWidth = numberAttribute(mapElement, 'tilewidth')
  const tileHeight = numberAttribute(mapElement, 'tileheight')
  if (!width || !height || !tileWidth || !tileHeight) {
    return []
  }

  // 타일셋들: firstgid 오름차순으로 모아 gid → (타일셋, 로컬 id) 역산에 쓴다.
  const tilesets: Array<{ firstGid: number; types: Map<number, string> }> = []
  for (const tilesetElement of directChildren(mapElement, 'tileset')) {
    const firstGid = numberAttribute(tilesetElement, 'firstgid')
    if (firstGid === undefined) {
      continue
    }
    const source = tilesetElement.getAttribute('source')
    if (source) {
      const tilesetText = options.resolveTilesetText?.(source)
      const tilesetDoc = tilesetText ? parseLenientXml(tilesetText) : undefined
      const tilesetRoot =
        tilesetDoc?.documentElement?.tagName === 'tileset'
          ? (tilesetDoc.documentElement as unknown as Element)
          : undefined
      // 해석에 실패해도 빈 표로라도 gid 범위를 점유해야 한다 — 빼버리면 이 범위의 gid가
      // 앞 타일셋의 것으로 오인돼(마지막 firstGid<=gid 탐색) 엉뚱한 종류로 분류된다.
      tilesets.push({
        firstGid,
        types: tilesetRoot ? collectTileTypes(tilesetRoot) : new Map()
      })
    } else {
      // 외부 .tsx 없이 맵에 내장된 타일셋(legend-of-lua 방식).
      tilesets.push({ firstGid, types: collectTileTypes(tilesetElement) })
    }
  }
  tilesets.sort((a, b) => a.firstGid - b.firstGid)
  if (tilesets.length === 0) {
    return []
  }

  const kindForGid = (rawGid: number): string | undefined => {
    const gid = (rawGid & GID_TRANSFORM_FLAGS_MASK) >>> 0
    if (gid === 0) {
      return undefined
    }
    let owner: { firstGid: number; types: Map<number, string> } | undefined
    for (const tileset of tilesets) {
      if (tileset.firstGid <= gid) {
        owner = tileset
      } else {
        break
      }
    }
    const type = owner?.types.get(gid - owner.firstGid)
    return type ? kindForTileType(type) : undefined
  }

  // 모든 타일 레이어를 한 격자에 겹쳐 셀별 종류를 만든다(지붕 위 굴뚝처럼 레이어가 쌓여도
  // 같은 사물이면 한 군집이 되도록). 종류가 다른 게 겹치면 나중 레이어가 이긴다.
  const kindGrid: Array<string | undefined> = new Array(width * height).fill(undefined)
  for (const layerElement of collectTileLayers(mapElement)) {
    const dataElement = directChildren(layerElement, 'data')[0]
    if (!dataElement) {
      continue
    }
    const encoding = dataElement.getAttribute('encoding')
    if (encoding !== null && encoding !== 'csv') {
      // base64 등은 이 에디터 표시용 파서의 범위 밖 — 그 레이어만 건너뛴다.
      continue
    }
    const values = (dataElement.textContent ?? '')
      .split(',')
      .map((token) => Number.parseInt(token.trim(), 10))
      .filter((value) => Number.isFinite(value))
    if (values.length !== width * height) {
      continue
    }
    for (let i = 0; i < values.length; i += 1) {
      const kind = kindForGid(values[i])
      if (kind !== undefined) {
        kindGrid[i] = kind
      }
    }
  }

  // 같은 종류가 상하좌우로 이어진 영역 = 군집 하나. 대각선은 잇지 않는다(이웃한 다른 사물이
  // 모서리만 닿았을 때 합쳐지는 것을 막는다).
  type WorkingCluster = {
    kind: string
    minX: number
    minY: number
    maxX: number
    maxY: number
    cells: number[]
    absorbed: boolean
  }
  const working: WorkingCluster[] = []
  // 셀 → 군집 번호. 흡수 단계에서 "이웃 칸이 어느 군집인지"를 바로 찾는 데 쓴다.
  const clusterIdGrid = new Array<number>(width * height).fill(-1)
  const neighborsOf = (index: number): number[] => {
    const x = index % width
    const y = Math.floor(index / width)
    return [
      x > 0 ? index - 1 : -1,
      x < width - 1 ? index + 1 : -1,
      y > 0 ? index - width : -1,
      y < height - 1 ? index + width : -1
    ].filter((neighbor) => neighbor >= 0)
  }
  for (let start = 0; start < kindGrid.length; start += 1) {
    const kind = kindGrid[start]
    if (kind === undefined || clusterIdGrid[start] !== -1) {
      continue
    }
    const clusterId = working.length
    const cluster: WorkingCluster = {
      kind,
      minX: width,
      minY: height,
      maxX: -1,
      maxY: -1,
      cells: [],
      absorbed: false
    }
    clusterIdGrid[start] = clusterId
    const stack = [start]
    while (stack.length > 0) {
      const index = stack.pop() as number
      const x = index % width
      const y = Math.floor(index / width)
      cluster.minX = Math.min(cluster.minX, x)
      cluster.minY = Math.min(cluster.minY, y)
      cluster.maxX = Math.max(cluster.maxX, x)
      cluster.maxY = Math.max(cluster.maxY, y)
      cluster.cells.push(index)
      for (const neighbor of neighborsOf(index)) {
        if (clusterIdGrid[neighbor] === -1 && kindGrid[neighbor] === kind) {
          clusterIdGrid[neighbor] = clusterId
          stack.push(neighbor)
        }
      }
    }
    working.push(cluster)
  }

  // 흡수 단계: 타일 공유로 쪼개진 파편(ABSORB_KIND_INTO)을 맞닿은 대상 군집에 합친다.
  for (const cluster of working) {
    const targetKinds = ABSORB_KIND_INTO[cluster.kind]
    if (targetKinds === undefined) {
      continue
    }
    // 파편이 본체로 흡수되는 방향만 허용한다(작은 쪽 → 크거나 같은 쪽). 반대를 허용하면
    // 진짜 시계탑(큰 군집)이 옆에 붙은 1타일 가로등으로 빨려 들어가는 식의 오인이 생긴다.
    const neighborClusters = cluster.cells
      .flatMap(neighborsOf)
      .map((neighbor) => clusterIdGrid[neighbor])
      .filter((id) => id >= 0)
      .map((id) => working[id])
      .filter((candidate) => candidate.cells.length >= cluster.cells.length)
    const target = targetKinds
      .map((kind) => neighborClusters.find((candidate) => candidate.kind === kind))
      .find((candidate) => candidate !== undefined)
    if (target) {
      target.minX = Math.min(target.minX, cluster.minX)
      target.minY = Math.min(target.minY, cluster.minY)
      target.maxX = Math.max(target.maxX, cluster.maxX)
      target.maxY = Math.max(target.maxY, cluster.maxY)
      // 셀도 합친다 — 아래 중복 제거가 셀 단위 커버리지로 판정하기 때문.
      target.cells.push(...cluster.cells)
      cluster.absorbed = true
    }
  }

  // 중복 제거는 셀 단위 커버리지로 판정한다. bbox 면적으로 재면 ㄱ자/ㅁ자 군집(성벽 등)의
  // 빈 안쪽에 object 사각형이 있을 때 실제로는 안 겹치는데도 지워진다.
  const excludeRects = options.excludeRects ?? []
  const isSuppressed = (cluster: WorkingCluster): boolean => {
    if (excludeRects.length === 0 || !STRUCTURAL_KINDS.has(cluster.kind)) {
      return false
    }
    let covered = 0
    for (const index of cluster.cells) {
      const centerX = ((index % width) + 0.5) * tileWidth
      const centerY = (Math.floor(index / width) + 0.5) * tileHeight
      const inside = excludeRects.some(
        (rect) =>
          centerX >= rect.x &&
          centerX < rect.x + rect.width &&
          centerY >= rect.y &&
          centerY < rect.y + rect.height
      )
      if (inside) {
        covered += 1
      }
    }
    return covered >= cluster.cells.length * 0.5
  }

  return working
    .filter((cluster) => !cluster.absorbed && !isSuppressed(cluster))
    .map((cluster) => ({
      kind: cluster.kind,
      tileX: cluster.minX,
      tileY: cluster.minY,
      widthTiles: cluster.maxX - cluster.minX + 1,
      heightTiles: cluster.maxY - cluster.minY + 1,
      rect: {
        x: cluster.minX * tileWidth,
        y: cluster.minY * tileHeight,
        width: (cluster.maxX - cluster.minX + 1) * tileWidth,
        height: (cluster.maxY - cluster.minY + 1) * tileHeight
      }
    }))
}
