import type { GameEntity } from './gameAdapter'
import type { LoadedGame } from './loadGame'

// legend-of-lua처럼 폴더로 연 외부 게임은 퀘스트 카탈로그(몬스터/아이템 의미)를 모른다.
// 우리가 아는 유일한 근거는 그 게임의 TMX에 "배치된 엔티티"(legendOfLuaAdapter.extractEntities 결과)다.
// 그래서 퀘스트 목표 grounding/검증을 이 배치 엔티티로만 한다 — 없는 대상은 게임에서 추적 불가.

export type LuaQuestCatalogEntity = {
  id: string
  name: string
  mapId: string
}

export type LuaQuestCatalog = {
  enemies: LuaQuestCatalogEntity[]
  npcs: LuaQuestCatalogEntity[]
  acquirables: LuaQuestCatalogEntity[]
  scenes: string[]
}

const toCatalogEntity = (entity: GameEntity): LuaQuestCatalogEntity => ({
  id: entity.id,
  name: entity.name,
  mapId: entity.mapId
})

export const buildLuaQuestCatalog = (game: LoadedGame): LuaQuestCatalog => {
  const enemies: LuaQuestCatalogEntity[] = []
  const npcs: LuaQuestCatalogEntity[] = []
  const acquirables: LuaQuestCatalogEntity[] = []
  const scenes: string[] = []

  for (const map of game.maps) {
    scenes.push(map.id)
    for (const entity of map.entities) {
      if (entity.kind === 'enemy') {
        enemies.push(toCatalogEntity(entity))
      } else if (entity.kind === 'npc') {
        npcs.push(toCatalogEntity(entity))
      } else if (entity.kind === 'chest' || entity.kind === 'loot') {
        acquirables.push(toCatalogEntity(entity))
      }
    }
  }

  return { enemies, npcs, acquirables, scenes }
}

const hasEntityIn = (
  entities: LuaQuestCatalogEntity[],
  entityId: string
): boolean => entities.some((entity) => entity.id === entityId)

export const isLuaQuestEnemy = (catalog: LuaQuestCatalog, id: string): boolean =>
  hasEntityIn(catalog.enemies, id)
export const isLuaQuestNpc = (catalog: LuaQuestCatalog, id: string): boolean =>
  hasEntityIn(catalog.npcs, id)
export const isLuaQuestAcquirable = (
  catalog: LuaQuestCatalog,
  id: string
): boolean => hasEntityIn(catalog.acquirables, id)
export const isLuaQuestScene = (
  catalog: LuaQuestCatalog,
  mapId: string
): boolean => catalog.scenes.includes(mapId)

// LLM 프롬프트에 넣을 사람이 읽는 카탈로그 요약(맵별 enemy/npc/chest id+이름).
export const buildLuaQuestCatalogText = (catalog: LuaQuestCatalog): string => {
  const fmt = (entities: LuaQuestCatalogEntity[]): string =>
    entities.length > 0
      ? entities
          .map((entity) => `${entity.id}(${entity.name}, map=${entity.mapId})`)
          .join(', ')
      : '(없음)'
  return [
    `처치 대상 적(defeat target.entityId): ${fmt(catalog.enemies)}`,
    `대화/기버 NPC(talk·giver target.entityId): ${fmt(catalog.npcs)}`,
    `획득 대상(acquire target.entityId): ${fmt(catalog.acquirables)}`,
    `이동 목표 씬(reach target.mapId): ${catalog.scenes.join(', ') || '(없음)'}`
  ].join('\n')
}
