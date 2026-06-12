import { describe, expect, it } from 'vitest'

import { buildMapPreviewInputs } from './buildMapPreviewInputs'
import type { GameFile } from './loadGame'

const mapTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map orientation="orthogonal" width="2" height="2" tilewidth="16" tileheight="16">
  <tileset firstgid="1" source="../tilesets/overworld.tsx"/>
  <layer id="1" name="ground" width="2" height="2">
    <data encoding="csv">
1,2,
3,4
</data>
  </layer>
</map>`

const tilesetTsx = `<?xml version="1.0" encoding="UTF-8"?>
<tileset name="overworld" tilewidth="16" tileheight="16" tilecount="100" columns="10">
  <image source="overworld.png" width="160" height="160"/>
</tileset>`

const textFile = (path: string, text: string): GameFile => ({
  name: path.slice(path.lastIndexOf('/') + 1),
  path,
  text
})

const imageFile = (path: string, url: string): GameFile => ({
  name: path.slice(path.lastIndexOf('/') + 1),
  path,
  text: '',
  url
})

const completeFiles = (): GameFile[] => [
  textFile('maps/test.tmx', mapTmx),
  textFile('tilesets/overworld.tsx', tilesetTsx),
  imageFile('tilesets/overworld.png', 'blob:overworld')
]

describe('buildMapPreviewInputs', () => {
  it('이어 맞춘 외부 타일셋·이미지로 맵을 파싱하고 image source로 URL을 매핑한다', () => {
    const result = buildMapPreviewInputs(completeFiles(), 'maps/test.tmx')

    expect(result.ok).toBe(true)

    if (!result.ok) {
      return
    }

    expect(result.inputs.map.tilesets).toHaveLength(1)
    expect(result.inputs.map.layers[0].tiles).toHaveLength(4)
    // 렌더러가 tileset.image.source로 조회하므로 그 키로 URL이 잡혀야 한다.
    expect(result.inputs.imageUrls).toEqual({ 'overworld.png': 'blob:overworld' })
  })

  it('타일셋이 TMX에 내장(embedded)된 맵은 이미지를 .tmx 기준으로 이어 렌더한다', () => {
    const embeddedTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map orientation="orthogonal" width="1" height="1" tilewidth="16" tileheight="16">
  <tileset firstgid="1" name="inline" tilewidth="16" tileheight="16" tilecount="4" columns="2">
    <image source="inline.png" width="32" height="32"/>
  </tileset>
  <layer id="1" name="ground" width="1" height="1">
    <data encoding="csv">1</data>
  </layer>
</map>`
    const result = buildMapPreviewInputs(
      [
        textFile('maps/test.tmx', embeddedTmx),
        // 내장 타일셋의 이미지 source는 .tmx 파일 기준 상대경로다.
        imageFile('maps/inline.png', 'blob:inline')
      ],
      'maps/test.tmx'
    )

    expect(result.ok).toBe(true)

    if (!result.ok) {
      return
    }

    expect(result.inputs.map.tilesets[0].source).toBe('embedded:1')
    expect(result.inputs.imageUrls).toEqual({ 'inline.png': 'blob:inline' })
  })

  it('타일셋(.tsx)을 못 찾으면 어떤 source인지 알려준다', () => {
    const result = buildMapPreviewInputs(
      [textFile('maps/test.tmx', mapTmx)],
      'maps/test.tmx'
    )

    expect(result.ok).toBe(false)

    if (result.ok) {
      return
    }

    expect(result.error).toContain('overworld.tsx')
  })

  it('타일셋 이미지를 못 찾으면(또는 URL 없음) 안내를 돌려준다', () => {
    const result = buildMapPreviewInputs(
      [
        textFile('maps/test.tmx', mapTmx),
        textFile('tilesets/overworld.tsx', tilesetTsx)
        // overworld.png 누락
      ],
      'maps/test.tmx'
    )

    expect(result.ok).toBe(false)

    if (result.ok) {
      return
    }

    expect(result.error).toContain('overworld.png')
  })

  it('파일명만으로도(폴더 구조가 살짝 달라도) 타일셋·이미지를 이어 맞춘다', () => {
    // .tsx/이미지가 TMX가 가리킨 정확한 경로가 아니라 평평하게 놓여 있어도 basename으로 잇는다.
    const result = buildMapPreviewInputs(
      [
        textFile('maps/test.tmx', mapTmx),
        textFile('overworld.tsx', tilesetTsx),
        imageFile('overworld.png', 'blob:flat')
      ],
      'maps/test.tmx'
    )

    expect(result.ok).toBe(true)

    if (!result.ok) {
      return
    }

    expect(result.inputs.imageUrls).toEqual({ 'overworld.png': 'blob:flat' })
  })
})
