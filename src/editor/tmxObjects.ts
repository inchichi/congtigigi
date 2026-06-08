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
  // @xmldom/xmldom는 깨진 XML에도 throw하지 않고 빈/부분 문서를 돌려준다. 그러면 "파싱 실패"와
  // "엔티티가 원래 없는 맵"이 구분되지 않으므로(둘 다 0개), 완전 실패만 throw해 호출부가 알 수 있게 한다.
  //
  // 판별은 error/fatalError "레벨"이 아니라 결과 문서로 한다: xmldom의 error 레벨은 &foo; 같은
  // '미정의 엔티티 참조'처럼 문서가 멀쩡히 나오는 복구 가능한 경우에도 불리기 때문에, 레벨로
  // throw하면 이름에 &가 들어간 멀쩡한 맵까지 버린다. 루트 엘리먼트조차 못 만든 경우(빈 문자열·비XML)만
  // 진짜 실패로 보고 throw한다. errorHandler는 콘솔 노이즈만 죽이고 판별엔 쓰지 않는다.
  const doc = new DOMParser({
    errorHandler: { warning: () => {}, error: () => {}, fatalError: () => {} }
  }).parseFromString(tmxText, 'text/xml') as Document | undefined

  // 완전 실패는 doc 자체가 undefined이거나(빈 문자열·비XML) 루트 엘리먼트를 못 만든 경우다.
  if (!doc || !doc.documentElement) {
    throw new Error('TMX 파싱 실패: 유효한 XML 문서가 아닙니다.')
  }

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
