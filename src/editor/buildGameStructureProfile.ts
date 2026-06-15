import type { ParsedTiledMap } from '../games/my-sample-rpg/tiled/parseTiledMap'
import type {
  GameStructureNpcInfo,
  GameStructureProfile
} from './gameStructureProfile'

// 게임이 이미 파싱한 맵(ParsedTiledMap)을 받아 에디터의 구조 프로필을 만든다.
// 맵·NPC는 실제 .tmx에서 추출하고, 아이템·시스템 등 코드 정의 정보는 base에서 가져온다.
// NPC 필터는 게임 런타임(createNpcCharactersFromEventLayers)과 동일해서, 추출 결과가
// 실제로 씬에 존재하는 NPC와 일치한다.

export type EditorMapSource = {
  id: string
  name: string
  file: string
  map: ParsedTiledMap
}

export const extractNpcsFromParsedMap = ({
  id: mapId,
  file,
  map
}: EditorMapSource): GameStructureNpcInfo[] =>
  map.eventLayers.flatMap((eventLayer) =>
    eventLayer.events.flatMap((event) => {
      if (!event.visible || event.className !== 'character' || !event.appearanceType) {
        return []
      }

      const id = event.name || `character-${event.id}`
      const displayText = event.properties.displayText
      const name =
        typeof displayText === 'string' && displayText.trim().length > 0
          ? displayText
          : id

      return [{ id, name, map: mapId, file }]
    })
  )

export const buildGameStructureProfile = (
  sources: EditorMapSource[],
  base: GameStructureProfile
): GameStructureProfile => ({
  ...base,
  maps: sources.map((source) => ({
    id: source.id,
    name: source.name,
    file: source.file
  })),
  npcs: sources.flatMap(extractNpcsFromParsedMap)
})
