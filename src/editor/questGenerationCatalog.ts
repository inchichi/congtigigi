export type QuestObjectiveType =
  | 'monster-defeat'
  | 'item-use'
  | 'item-acquire'
  | 'shop-open'
  | 'scene-enter'
  | 'talk'

export const QUEST_OBJECTIVE_TYPES: QuestObjectiveType[] = [
  'monster-defeat',
  'item-use',
  'item-acquire',
  'shop-open',
  'scene-enter',
  'talk'
]

// Items that can exist in the project inventory or shops.
export const ACQUIRABLE_ITEMS: { itemId: string; label: string }[] = [
  { itemId: 'iron-sword', label: '강철 검' },
  { itemId: 'bronze-sword', label: '청동 검' },
  { itemId: 'battle-axe', label: '전투 도끼' },
  { itemId: 'long-spear', label: '장창' },
  { itemId: 'quick-dagger', label: '단검' },
  { itemId: 'spiked-mace', label: '철퇴' },
  { itemId: 'magic-staff', label: '마법 지팡이' },
  { itemId: 'iron-armor', label: '철 갑옷' },
  { itemId: 'Leather_Armor', label: '가죽 갑옷' },
  { itemId: 'Leather_Helmet', label: '가죽 투구' },
  { itemId: 'Chain_Armor', label: '사슬 갑옷' },
  { itemId: 'Chain_Helmet', label: '사슬 투구' },
  { itemId: 'Iron_Armor', label: '철 갑옷' },
  { itemId: 'Iron_Helmet', label: '철 투구' },
  { itemId: 'leather-boots', label: '가죽 장화' },
  { itemId: 'smith-charm', label: '수호 부적' },
  { itemId: 'health-potion', label: '체력 회복 포션' },
  { itemId: 'mana-potion', label: '마나 회복 포션' }
]

// Items that can be validated as monster drops in the current game.
export const QUEST_MONSTER_DROP_ITEMS: { itemId: string; label: string }[] = [
  { itemId: 'iron-sword', label: '강철 검' },
  { itemId: 'battle-axe', label: '전투 도끼' },
  { itemId: 'long-spear', label: '장창' },
  { itemId: 'quick-dagger', label: '단검' },
  { itemId: 'spiked-mace', label: '철퇴' },
  { itemId: 'magic-staff', label: '마법 지팡이' },
  { itemId: 'Leather_Armor', label: '가죽 갑옷' },
  { itemId: 'Leather_Helmet', label: '가죽 투구' },
  { itemId: 'Chain_Armor', label: '사슬 갑옷' },
  { itemId: 'Chain_Helmet', label: '사슬 투구' },
  { itemId: 'Iron_Armor', label: '철 갑옷' },
  { itemId: 'Iron_Helmet', label: '철 투구' }
]

export const QUEST_MONSTERS: { appearanceType: string; label: string }[] = [
  { appearanceType: 'monster_slime', label: '슬라임' },
  { appearanceType: 'monster_pig', label: '돼지' }
]

export const QUEST_MONSTER_SCENES = ['hunting-ground', 'cave']

// town is the start scene and would complete immediately, so it is excluded.
export const QUEST_SCENE_ENTER_SCENES = ['hunting-ground', 'cave']

export const QUEST_SHOPS: { shopId: string; label: string }[] = [
  { shopId: 'blacksmith', label: '대장간 상점' },
  { shopId: 'potion', label: '포션 상점' }
]

// All NPCs that exist in the current project snapshot and can plausibly give quests.
export const QUEST_GIVER_NPCS: { npcId: string; label: string }[] = [
  { npcId: 'wizard', label: '마법사' },
  { npcId: 'potion_merchant', label: '포션 상인' },
  { npcId: 'blacksmith', label: '대장장이' },
  { npcId: 'santa', label: '산타' },
  { npcId: 'villager_1', label: '마을 사람' }
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

export const isAcquirableItem = (itemId: string): boolean =>
  ACQUIRABLE_ITEMS.some((item) => item.itemId === itemId)

export const isQuestMonsterDropItem = (itemId: string): boolean =>
  QUEST_MONSTER_DROP_ITEMS.some((item) => item.itemId === itemId)

export const buildQuestCatalogText = (): string => {
  const monsters = QUEST_MONSTERS.map(
    (monster) => `${monster.appearanceType}(${monster.label})`
  ).join(', ')
  const shops = QUEST_SHOPS.map((shop) => `${shop.shopId}(${shop.label})`).join(', ')
  const givers = QUEST_GIVER_NPCS.map(
    (giver) => `${giver.npcId}(${giver.label})`
  ).join(', ')
  const monsterDrops = QUEST_MONSTER_DROP_ITEMS.map(
    (item) => `${item.itemId}(${item.label})`
  ).join(', ')

  return [
    `몬스터 appearanceType: ${monsters}`,
    `몬스터 sceneId: ${QUEST_MONSTER_SCENES.join(', ')}`,
    `이동 목표 scene-enter sceneId: ${QUEST_SCENE_ENTER_SCENES.join(', ')}`,
    `상점 shopId: ${shops}`,
    `획득 목표 item-acquire itemId (monster drop only): ${monsterDrops}`,
    `퀘스트 기버 NPC givers: ${givers}`
  ].join('\n')
}
