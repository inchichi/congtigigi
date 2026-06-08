import { DOMParser } from '@xmldom/xmldom'

// 타일셋 해석 없이 TMX의 object layer만 읽는 관용 파서. parseTiledMap과 달리 외부 .tsx를
// 요구하지 않아서(legend-of-lua처럼 임베드 타일셋이어도) 어느 게임의 맵이든 엔티티를 뽑을 수 있다.
export type TmxObject = {
  id: string
  name: string
  type: string
  group: string
  properties: Record<string, string>
}

export const extractTmxObjects = (tmxText: string): TmxObject[] => {
  const doc = new DOMParser().parseFromString(tmxText, 'text/xml')
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
        // Tiled 신버전은 class=, 구버전은 type= 를 쓴다.
        type: node.getAttribute('type') ?? node.getAttribute('class') ?? '',
        group: groupName,
        properties
      })
    }
  }

  return objects
}
