import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { detectAdapter, type GameAdapter, type GameEntity } from './gameAdapter'
import { extractTmxObjects } from './tmxObjects'
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
        entities = adapter.extractEntities(id, extractTmxObjects(file.text))
      } catch {
        parseErrors.push(file.path)
      }
      return { id, name: id, file: file.path, entities }
    })

  const profile: GameStructureProfile | undefined =
    adapter.id === 'my-sample-rpg'
      ? {
          ...CURRENT_GAME_PROJECT_PROFILE,
          maps: maps.map((map) => ({ id: map.id, name: map.name, file: map.file })),
          // profile.npcs는 의도적으로 "편집 가능한(대화) NPC" 집합 = 선택 트리와 동일하다(rpgAdapter가
          // 표지판 sign_*·몬스터 monster_*를 제외한 character_* 외형만 NPC로 추출). 런타임이 만드는
          // 전체 캐릭터 집합보다 좁지만, 에디터는 대화 이벤트를 이 큐레이션된 대상에만 생성하므로
          // LLM 프롬프트(허용 NPC 목록)·검증·트리가 모두 같은 집합으로 일관된다.
          npcs: maps.flatMap((map) =>
            map.entities.map((entity) => ({
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
