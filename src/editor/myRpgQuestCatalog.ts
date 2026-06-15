// My Sample RPG의 퀘스트 목표 grounding/검증 카탈로그.
// 프로필(gameStructureProfile)에는 맵·NPC·아이템만 있고 몬스터/상점/씬-목표 정보가 없어서,
// 런타임이 실제로 추적하는 값들(questLog.ts의 정적 퀘스트가 쓰는 target 값과 동일)을 여기 모은다.
// 이 값들이 틀리면 게임에서 목표가 영원히 진행되지 않으므로, 생성·검증 양쪽에서 이 카탈로그로 접지한다.

export type QuestObjectiveType =
  | 'monster-defeat'
  | 'item-use'
  | 'shop-open'
  | 'scene-enter'
  | 'talk'

export const QUEST_OBJECTIVE_TYPES: QuestObjectiveType[] = [
  'monster-defeat',
  'item-use',
  'shop-open',
  'scene-enter',
  'talk'
]

// 처치 추적이 되는 몬스터 외형 타입(런타임 appearanceType). 현재 슬라임/돼지 2종뿐.
export const QUEST_MONSTERS: { appearanceType: string; label: string }[] = [
  { appearanceType: 'monster_slime', label: '슬라임(말캉이)' },
  { appearanceType: 'monster_pig', label: '돼지' }
]

// 몬스터가 등장하는 씬(monster-defeat의 sceneId로 써야 진행됨).
export const QUEST_MONSTER_SCENES = ['hunting-ground', 'cave']

// scene-enter 목표로 쓸 수 있는 씬. town은 시작 지점이라 즉시 완료돼 버리므로 제외한다.
export const QUEST_SCENE_ENTER_SCENES = ['hunting-ground', 'cave']

// 상점 열기 목표의 shopId(런타임이 실제로 발생시키는 값: blacksmith, potion).
export const QUEST_SHOPS: { shopId: string; label: string }[] = [
  { shopId: 'blacksmith', label: '대장간 상점' },
  { shopId: 'potion', label: '포션 상점' }
]

// 퀘스트를 줄 수 있는 NPC(시작/진행/완료 대화 흐름이 보장되는 wired NPC).
export const QUEST_GIVER_NPCS: { npcId: string; label: string }[] = [
  { npcId: 'wizard', label: '마법사' },
  { npcId: 'potion_merchant', label: '포션 상인' },
  { npcId: 'blacksmith', label: '대장장이' }
]

export const isQuestMonsterAppearance = (appearanceType: string): boolean =>
  QUEST_MONSTERS.some((monster) => monster.appearanceType === appearanceType)

export const isQuestMonsterScene = (sceneId: string): boolean =>
  QUEST_MONSTER_SCENES.includes(sceneId)

export const isQuestSceneEnterScene = (sceneId: string): boolean =>
  QUEST_SCENE_ENTER_SCENES.includes(sceneId)

export const isQuestShop = (shopId: string): boolean =>
  QUEST_SHOPS.some((shop) => shop.shopId === shopId)

export const isQuestGiverNpc = (npcId: string): boolean =>
  QUEST_GIVER_NPCS.some((giver) => giver.npcId === npcId)

// LLM 프롬프트에 넣을 사람이 읽는 카탈로그 요약.
export const buildQuestCatalogText = (): string => {
  const monsters = QUEST_MONSTERS.map(
    (monster) => `${monster.appearanceType}(${monster.label})`
  ).join(', ')
  const shops = QUEST_SHOPS.map((shop) => `${shop.shopId}(${shop.label})`).join(', ')
  const givers = QUEST_GIVER_NPCS.map(
    (giver) => `${giver.npcId}(${giver.label})`
  ).join(', ')
  return [
    `처치 가능 몬스터(appearanceType): ${monsters}`,
    `몬스터 등장 씬(sceneId): ${QUEST_MONSTER_SCENES.join(', ')}`,
    `이동 목표 씬(scene-enter sceneId): ${QUEST_SCENE_ENTER_SCENES.join(', ')}`,
    `상점(shopId): ${shops}`,
    `퀘스트 기버 NPC(giver_npc_id): ${givers}`
  ].join('\n')
}
