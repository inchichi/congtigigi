import type { TmxObject } from './tmxObjects'

// 게임마다 맵/엔티티 규칙이 달라서, "이 게임의 객체를 에디터 엔티티로 어떻게 읽나"를
// 어댑터로 분리한다. 새 게임 지원 = 새 어댑터 추가.
export type GameEntity = {
  id: string
  name: string
  kind: string
  mapId: string
}

export type GameAdapter = {
  id: string
  name: string
  // 열린 폴더의 파일 이름들로 어느 게임인지 판별한다.
  detect: (fileNames: string[]) => boolean
  // 한 맵의 TMX 객체들을 이 게임의 엔티티로 변환한다.
  extractEntities: (mapId: string, objects: TmxObject[]) => GameEntity[]
  // 이 게임에 대해 에디터가 생성→적용까지 지원하는지.
  supportsApply: boolean
}

export const rpgAdapter: GameAdapter = {
  id: 'my-sample-rpg',
  name: 'My Sample RPG (TS/Pixi)',
  detect: (fileNames) =>
    fileNames.includes('town.tmx') ||
    fileNames.some((name) => name === 'createPixiTiledMapView.ts'),
  extractEntities: (mapId, objects) =>
    objects
      .filter(
        (object) =>
          object.type === 'character' &&
          (object.properties.type !== undefined ||
            object.properties.appearanceType !== undefined)
      )
      .map((object) => ({
        id: object.name || `character-${object.id}`,
        name: object.properties.displayText || object.name || `character-${object.id}`,
        kind: 'npc',
        mapId
      })),
  supportsApply: true
}

const LEGEND_KIND_BY_GROUP: Record<string, string> = {
  Enemies: 'enemy',
  NPCs: 'npc',
  Chests: 'chest',
  Loot: 'loot'
}

export const legendOfLuaAdapter: GameAdapter = {
  id: 'legend-of-lua',
  name: 'Legend of Lua (Love2D)',
  detect: (fileNames) =>
    fileNames.includes('conf.lua') || fileNames.includes('main.lua'),
  extractEntities: (mapId, objects) =>
    objects
      .filter(
        (object) =>
          LEGEND_KIND_BY_GROUP[object.group] !== undefined && object.name.length > 0
      )
      .map((object) => ({
        id: `${object.group}-${object.id}`,
        name: object.name,
        kind: LEGEND_KIND_BY_GROUP[object.group],
        mapId
      })),
  // Love2D 런타임에 라이브 적용은 아직 미구현(Stage 3). 지금은 엔티티 브라우징·생성까지.
  supportsApply: false
}

export const GAME_ADAPTERS: GameAdapter[] = [legendOfLuaAdapter, rpgAdapter]

export const detectAdapter = (fileNames: string[]): GameAdapter =>
  GAME_ADAPTERS.find((adapter) => adapter.detect(fileNames)) ?? rpgAdapter
