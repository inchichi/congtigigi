import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { detectAdapter, type GameAdapter, type GameEntity } from './gameAdapter'
import { extractTmxObjects } from './tmxObjects'
import type { GameStructureProfile } from './gameStructureProfile'

export type GameFile = {
  name: string
  path: string
  text: string
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
}

export const loadGame = (files: GameFile[]): LoadedGame => {
  const adapter = detectAdapter(files.map((file) => file.name))

  const maps: LoadedGameMap[] = files
    .filter((file) => file.name.endsWith('.tmx'))
    .map((file) => {
      const id = file.name.replace(/\.tmx$/u, '')
      const entities = adapter.extractEntities(id, extractTmxObjects(file.text))
      return { id, name: id, file: file.path, entities }
    })

  const profile: GameStructureProfile | undefined =
    adapter.id === 'my-sample-rpg'
      ? {
          ...CURRENT_GAME_PROJECT_PROFILE,
          maps: maps.map((map) => ({ id: map.id, name: map.name, file: map.file })),
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

  return { adapter, maps, profile }
}
