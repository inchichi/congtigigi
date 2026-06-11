import { describe, expect, it } from 'vitest'

import { extractTmxTileClusters } from './tmxTileEntities'

// 테스트용 타일셋: gid = id + 1 (firstgid=1).
// gid 1=나무, 2=지형(무시), 3=가로등, 4=시계탑, 5=분수, 6=지붕(건물), 7=그림자(무시), 8=바닥 채움(무시)
const EMBEDDED_TILESET = `<tileset firstgid="1" name="t" tilewidth="16" tileheight="16" tilecount="8" columns="4">
  <tile id="0" type="tree_large_canopy"/>
  <tile id="1" type="ground_grass_fill"/>
  <tile id="2" type="streetlamp_lit_top"/>
  <tile id="3" type="clocktower_face_mid"/>
  <tile id="4" type="fountain_basin_left"/>
  <tile id="5" type="roof_blue_shingle"/>
  <tile id="6" type="tree_canopy_shadow"/>
  <tile id="7" type="wall_cobble_fill"/>
</tileset>`

const mapWith = (width: number, height: number, csv: string): string => `<?xml version="1.0"?>
<map width="${width}" height="${height}" tilewidth="16" tileheight="16">
 ${EMBEDDED_TILESET}
 <layer id="1" name="ground" width="${width}" height="${height}">
  <data encoding="csv">${csv}</data>
 </layer>
</map>`

describe('extractTmxTileClusters', () => {
  it('groups adjacent same-kind tiles into one cluster and ignores terrain/shadow/excluded tiles', () => {
    // 나무(1)가 ㄱ자로 붙어 있고, 지형(2)·그림자(7)·바닥 채움(8)은 사물이 아니라 무시된다.
    const clusters = extractTmxTileClusters(mapWith(4, 2, '1,1,2,8, 1,7,0,0'))

    expect(clusters).toEqual([
      {
        kind: 'tree',
        tileX: 0,
        tileY: 0,
        widthTiles: 2,
        heightTiles: 2,
        rect: { x: 0, y: 0, width: 32, height: 32 }
      }
    ])
  })

  it('separates same-kind tiles that only touch diagonally (4-connectivity)', () => {
    const clusters = extractTmxTileClusters(mapWith(2, 2, '1,0, 0,1'))

    expect(clusters).toHaveLength(2)
    expect(clusters.map((cluster) => cluster.kind)).toEqual(['tree', 'tree'])
  })

  it('strips Tiled flip bits from gids', () => {
    // 0x80000001 = 수평 반전된 gid 1(나무).
    const clusters = extractTmxTileClusters(mapWith(1, 1, '2147483649'))

    expect(clusters.map((cluster) => cluster.kind)).toEqual(['tree'])
  })

  it('absorbs a small clocktower fragment into the adjacent larger lamp (shared pillar tiles)', () => {
    // 가로등 4타일 + 시계탑 기둥 1타일 — 파편이 본체로 흡수돼 가로등 하나가 된다.
    const clusters = extractTmxTileClusters(mapWith(3, 2, '3,3,4, 3,3,0'))

    expect(clusters).toEqual([
      {
        kind: 'lamp',
        tileX: 0,
        tileY: 0,
        widthTiles: 3,
        heightTiles: 2,
        rect: { x: 0, y: 0, width: 48, height: 32 }
      }
    ])
  })

  it('does not absorb a larger clocktower into a smaller adjacent lamp', () => {
    // 진짜 시계탑(4타일)이 옆의 1타일 가로등으로 빨려 들어가면 안 된다.
    const clusters = extractTmxTileClusters(mapWith(3, 2, '4,4,3, 4,4,0'))

    expect(clusters.map((cluster) => cluster.kind).sort()).toEqual(['clocktower', 'lamp'])
  })

  it('drops structural clusters that mostly overlap an exclude rect but keeps freestanding kinds', () => {
    // 지붕(건물)과 분수가 같은 영역에 있을 때: 건물은 object 등록분과의 중복이라 빠지고,
    // 분수는 독립 사물이라 남는다(건물 입구 옆 분수 같은 경우).
    const map = mapWith(2, 2, '6,6, 5,5')
    const excludeRects = [{ x: 0, y: 0, width: 32, height: 32 }]

    const clusters = extractTmxTileClusters(map, { excludeRects })

    expect(clusters.map((cluster) => cluster.kind)).toEqual(['fountain'])
  })

  it('drops a freestanding cluster only when the exclude rect declares the same kind', () => {
    // 분수가 object로도 등록되면(kind: 'fountain') 그 군집은 중복이라 빠진다 —
    // 종류가 다른 object(건물 등)에는 영향받지 않는다.
    const map = mapWith(2, 1, '5,5')
    const fountainRect = [{ x: 0, y: 0, width: 32, height: 16, kind: 'fountain' }]
    const buildingRect = [{ x: 0, y: 0, width: 32, height: 16, kind: 'building' }]

    expect(extractTmxTileClusters(map, { excludeRects: fountainRect })).toEqual([])
    expect(
      extractTmxTileClusters(map, { excludeRects: buildingRect }).map((c) => c.kind)
    ).toEqual(['fountain'])
  })

  it('resolves external tilesets through the provided resolver', () => {
    const sources: string[] = []
    const map = `<?xml version="1.0"?>
<map width="1" height="1" tilewidth="16" tileheight="16">
 <tileset firstgid="1" source="../tilesets/props.tsx"/>
 <layer id="1" name="ground" width="1" height="1"><data encoding="csv">1</data></layer>
</map>`
    const tileset = `<?xml version="1.0"?>
<tileset name="props" tilewidth="16" tileheight="16" tilecount="1" columns="1">
  <tile id="0" type="fountain_basin_left"/>
</tileset>`

    const clusters = extractTmxTileClusters(map, {
      resolveTilesetText: (source) => {
        sources.push(source)
        return tileset
      }
    })

    expect(sources).toEqual(['../tilesets/props.tsx'])
    expect(clusters.map((cluster) => cluster.kind)).toEqual(['fountain'])
  })

  it('does not let gids of an unresolved tileset bleed into the previous tileset', () => {
    // 두 번째 타일셋(firstgid=3)을 못 찾아도 그 범위의 gid(여기선 6)가 첫 타일셋의
    // 타일(6-1=5 → roof)로 오인되면 안 된다 — 빈 표로라도 범위를 점유해야 한다.
    const map = `<?xml version="1.0"?>
<map width="1" height="1" tilewidth="16" tileheight="16">
 <tileset firstgid="1" name="a" tilewidth="16" tileheight="16" tilecount="2" columns="2">
  <tile id="5" type="roof_blue_shingle"/>
 </tileset>
 <tileset firstgid="3" source="missing.tsx"/>
 <layer id="1" name="ground" width="1" height="1"><data encoding="csv">6</data></layer>
</map>`

    expect(extractTmxTileClusters(map)).toEqual([])
  })

  it('finds tile layers nested inside Tiled <group> folders', () => {
    const map = `<?xml version="1.0"?>
<map width="1" height="1" tilewidth="16" tileheight="16">
 ${EMBEDDED_TILESET}
 <group id="2" name="folder">
  <layer id="1" name="ground" width="1" height="1"><data encoding="csv">1</data></layer>
 </group>
</map>`

    expect(extractTmxTileClusters(map).map((cluster) => cluster.kind)).toEqual(['tree'])
  })

  it('judges exclude-rect overlap by occupied cells, not by the bounding box', () => {
    // ㄱ자 지붕(윗줄 3칸 + 왼쪽 기둥 1칸)의 bbox는 2×2 영역을 덮지만, 오른쪽 아래 칸에는
    // 셀이 없다. 그 빈 칸만 덮는 사각형으로는 군집이 지워지면 안 된다.
    const map = mapWith(2, 2, '6,6, 6,0')
    const clusters = extractTmxTileClusters(map, {
      excludeRects: [{ x: 16, y: 16, width: 16, height: 16 }]
    })

    expect(clusters.map((cluster) => cluster.kind)).toEqual(['building'])
  })

  it('returns [] when the tileset cannot be resolved or the input is not a TMX map', () => {
    const mapWithMissingTileset = `<?xml version="1.0"?>
<map width="1" height="1" tilewidth="16" tileheight="16">
 <tileset firstgid="1" source="missing.tsx"/>
 <layer id="1" name="ground" width="1" height="1"><data encoding="csv">1</data></layer>
</map>`

    expect(extractTmxTileClusters(mapWithMissingTileset)).toEqual([])
    expect(extractTmxTileClusters('not xml at all')).toEqual([])
    expect(extractTmxTileClusters('<html></html>')).toEqual([])
  })
})
