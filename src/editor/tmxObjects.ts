import { DOMParser } from '@xmldom/xmldom'

// 타일셋 해석 없이 TMX의 object layer만 읽는 관용 파서. parseTiledMap과 달리 외부 .tsx를
// 요구하지 않아서(legend-of-lua처럼 임베드 타일셋이어도) 어느 게임의 맵이든 엔티티를 뽑을 수 있다.
export type TmxObject = {
  id: string
  name: string
  type: string
  group: string
  properties: Record<string, string>
  // 픽셀 단위 위치/크기. 점 객체(캐릭터 등)는 width/height가 0이다.
  x: number
  y: number
  width: number
  height: number
}

export const extractTmxObjects = (tmxText: string): TmxObject[] => {
  // @xmldom/xmldom는 깨진 XML에도 throw하지 않고 빈/부분 문서를 돌려준다. 그러면 "파싱 실패"와
  // "엔티티가 원래 없는 맵"이 구분되지 않으므로(둘 다 0개), 진짜 실패만 throw해 호출부가 알 수 있게 한다.
  //
  // 판별은 error "레벨"만으로 하면 안 된다: xmldom의 error 레벨은 'entity not found:&foo;'(이름 속
  // 미정의 엔티티 참조 — 문서는 멀쩡히 나오고 objects도 정상 추출)와 'Hierarchy request error /
  // unexpected end of input'(잘림·이중 루트 — 콘텐츠가 통째로 날아감)을 한데 섞기 때문이다. 그래서
  //  (1) 복구 가능한 'entity not found'는 통과시키고 그 외 error/fatalError는 구조적 실패로 보며,
  //  (2) 루트가 없거나(<map> 시작 전 비XML) 루트가 TMX 루트(<map>)가 아니면 실패로 본다.
  let structuralError = false
  const doc = new DOMParser({
    errorHandler: {
      warning: () => {},
      error: (message) => {
        if (!message.includes('entity not found')) {
          structuralError = true
        }
      },
      fatalError: () => {
        structuralError = true
      }
    }
  }).parseFromString(tmxText, 'text/xml') as Document | undefined

  if (!doc || !doc.documentElement || doc.documentElement.tagName !== 'map' || structuralError) {
    throw new Error('TMX 파싱 실패: 유효한 TMX(.tmx) 문서가 아닙니다.')
  }

  return collectObjects(doc)
}

// TMX의 타일/이미지 레이어 이름만 읽는다("ground" 같은 지형·바닥, 즉 "맵 자체"를 에디터에 보여주려고).
// 객체가 아니라 표시용 정보라서, 파싱이 어긋나도 throw하지 않고 빈 배열을 돌려준다(에디터를 죽이지 않음).
export const extractTmxLayerNames = (tmxText: string): string[] => {
  let doc: Document | undefined
  try {
    doc = new DOMParser({
      errorHandler: { warning: () => {}, error: () => {}, fatalError: () => {} }
    }).parseFromString(tmxText, 'text/xml') as Document | undefined
  } catch {
    return []
  }

  if (!doc || !doc.documentElement || doc.documentElement.tagName !== 'map') {
    return []
  }

  const names: string[] = []
  for (const tag of ['layer', 'imagelayer'] as const) {
    const nodes = doc.getElementsByTagName(tag)
    for (let i = 0; i < nodes.length; i += 1) {
      const name = nodes[i].getAttribute('name')
      if (name && name.length > 0) {
        names.push(name)
      }
    }
  }

  return names
}

const numberAttribute = (element: Element, name: string): number => {
  const value = Number(element.getAttribute(name) ?? '')
  return Number.isFinite(value) ? value : 0
}

const collectObjects = (doc: Document): TmxObject[] => {
  const objects: TmxObject[] = []
  const groups = doc.getElementsByTagName('objectgroup')

  for (let g = 0; g < groups.length; g += 1) {
    const group = groups[g]
    const groupName = group.getAttribute('name') ?? ''
    const objectNodes = group.getElementsByTagName('object')

    for (let i = 0; i < objectNodes.length; i += 1) {
      const node = objectNodes[i]
      const properties: Record<string, string> = {}
      const propertyNodes = node.getElementsByTagName('property')

      for (let p = 0; p < propertyNodes.length; p += 1) {
        const propertyName = propertyNodes[p].getAttribute('name')
        if (propertyName) {
          properties[propertyName] = propertyNodes[p].getAttribute('value') ?? ''
        }
      }

      objects.push({
        id: node.getAttribute('id') ?? '',
        name: node.getAttribute('name') ?? '',
        // Tiled 신버전은 class=, 구버전은 type= 를 쓴다 — Tiled 자체 규칙대로 class 우선
        // (tmxTileEntities의 타일 분류와 같은 순서). ??가 아니라 || 인 이유: @xmldom/xmldom은
        // 없는 속성에 null이 아니라 ''를 돌려줘서 nullish 폴백이 동작하지 않는다.
        type: node.getAttribute('class') || node.getAttribute('type') || '',
        group: groupName,
        properties,
        x: numberAttribute(node, 'x'),
        y: numberAttribute(node, 'y'),
        width: numberAttribute(node, 'width'),
        height: numberAttribute(node, 'height')
      })
    }
  }

  return objects
}
