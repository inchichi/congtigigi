import type { LuaQuestCatalog } from './luaQuestCatalog'

// legend-of-lua용으로 새로 "생성"하는 NPC의 구조화 출력. 퀘스트(luaQuestSchema)와 평행하지만,
// 기존 배치 엔티티를 가리키는 게 아니라 게임에 없던 NPC를 만든다 — 그래서 타깃 실존 검증 대신
// "맵 실존 + 외형은 빌려온 에셋 목록 안 + id 충돌 없음(=정말 새 것)"을 검증한다.
//
// 이번 범위: 에디터 검증 → 카탈로그 인식 → Lua 출력까지. 실제 게임 런타임 스폰은 포함하지 않는다.

// my-sample-rpg에서 빌려오는 캐릭터 스프라이트(외형) 목록. 이 id가 곧 생성 NPC의 appearance가 된다.
// (출처: src/games/my-sample-rpg/assets/tilesets/tiny-dungeon-16.tsx 의 character_* 정의)
export const LUA_NPC_APPEARANCES = [
  'character_villager_brown_tunic',
  'character_villager_flower_dress',
  'character_commoner_tan_tunic',
  'character_bearded_apron_man',
  'character_elder_gray_hair',
  'character_wizard_purple',
  'character_ranger_green',
  'character_knight_open_helmet',
  'character_knight_closed_helmet',
  'character_knight_gray_helmet',
  'character_adventurer_brown_hair'
] as const
export type LuaNpcAppearance = (typeof LUA_NPC_APPEARANCES)[number]

// my-sample-rpg의 Lua 컨트롤러 행동(wander-near-home / reply-with-message)에서 가져온 개념.
// 'wander'면 home 기준 radius 안을 배회, 'stationary'면 제자리. 대사는 dialogue_lines로 응답한다.
export type LuaNpcBehaviorType = 'wander' | 'stationary'
export const LUA_NPC_BEHAVIOR_TYPES: LuaNpcBehaviorType[] = ['wander', 'stationary']

export type GeneratedLuaNpcJson = {
  npc_id: string // 영문 snake_case, 기존 엔티티와 겹치지 않는 새 id
  name: string
  map_id: string // 이 게임에 실재하는 맵(tmx 파일명)
  appearance: string // LUA_NPC_APPEARANCES 중 하나
  position: { x: number; y: number } // 타일 좌표
  dialogue_lines: string[] // 상호작용 대사(최소 1줄)
  behavior: { type: LuaNpcBehaviorType; radius: number } // wander면 radius > 0
}

export type GeneratedLuaNpcValidationIssue = { path: string; message: string }

const isSnakeCase = (value: string): boolean => /^[a-z0-9_]+$/u.test(value)

// 생성 NPC가 게임 엔티티로서 가질 id — 배치 NPC와 같은 "{group}-{id}" 규칙(group=NPCs)을 따라,
// 카탈로그·퀘스트 기버 검증이 이 NPC를 "배치된 NPC"처럼 인식하게 한다.
export const luaNpcEntityId = (npc: Pick<GeneratedLuaNpcJson, 'npc_id'>): string =>
  `NPCs-${npc.npc_id}`

// 생성 NPC가 게임에 더해질 수 있는지 검증. 핵심: 맵 실존 + 외형 실존 + id가 "새 것"(충돌 없음).
export const createGeneratedLuaNpcValidationIssues = (
  npc: GeneratedLuaNpcJson,
  catalog: LuaQuestCatalog
): GeneratedLuaNpcValidationIssue[] => {
  const issues: GeneratedLuaNpcValidationIssue[] = []

  if (!isSnakeCase(npc.npc_id)) {
    issues.push({
      path: 'npc_id',
      message: 'npc_id는 영문 소문자·숫자·밑줄(snake_case)이어야 한다.'
    })
  }
  if (npc.name.trim().length === 0) {
    issues.push({ path: 'name', message: 'name은 비어 있을 수 없다.' })
  }
  if (!catalog.scenes.includes(npc.map_id)) {
    issues.push({ path: 'map_id', message: `이 게임의 맵이 아니다: ${npc.map_id}` })
  }
  if (!LUA_NPC_APPEARANCES.includes(npc.appearance as LuaNpcAppearance)) {
    issues.push({
      path: 'appearance',
      message: `빌려올 수 있는 외형이 아니다: ${npc.appearance}`
    })
  }
  if (!Number.isFinite(npc.position.x) || npc.position.x < 0) {
    issues.push({ path: 'position.x', message: 'position.x는 0 이상의 숫자여야 한다.' })
  }
  if (!Number.isFinite(npc.position.y) || npc.position.y < 0) {
    issues.push({ path: 'position.y', message: 'position.y는 0 이상의 숫자여야 한다.' })
  }
  if (npc.dialogue_lines.length === 0) {
    issues.push({ path: 'dialogue_lines', message: '대사가 최소 1줄 필요하다.' })
  }
  if (!LUA_NPC_BEHAVIOR_TYPES.includes(npc.behavior.type)) {
    issues.push({
      path: 'behavior.type',
      message: `알 수 없는 행동: ${npc.behavior.type}`
    })
  }
  if (!Number.isFinite(npc.behavior.radius) || npc.behavior.radius < 0) {
    issues.push({ path: 'behavior.radius', message: 'radius는 0 이상의 숫자여야 한다.' })
  }
  if (npc.behavior.type === 'wander' && npc.behavior.radius <= 0) {
    issues.push({
      path: 'behavior.radius',
      message: 'wander 행동은 radius가 0보다 커야 한다.'
    })
  }

  // 생성 NPC는 "새 것"이어야 한다 — 이미 배치된 엔티티 id와 겹치면 추가가 아니라 충돌이다.
  const entityId = luaNpcEntityId(npc)
  const collides =
    catalog.npcs.some((entity) => entity.id === entityId) ||
    catalog.enemies.some((entity) => entity.id === entityId) ||
    catalog.acquirables.some((entity) => entity.id === entityId)
  if (collides) {
    issues.push({
      path: 'npc_id',
      message: `이미 존재하는 엔티티 id와 겹친다: ${entityId}`
    })
  }

  return issues
}

export const isGeneratedLuaNpcValid = (
  npc: GeneratedLuaNpcJson,
  catalog: LuaQuestCatalog
): boolean => createGeneratedLuaNpcValidationIssues(npc, catalog).length === 0
