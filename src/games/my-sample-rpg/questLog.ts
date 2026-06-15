export const FIRST_SLIME_HUNT_QUEST_ID = 'q001-first-slime-hunt'
export const POTION_SURVIVAL_BASICS_QUEST_ID =
  'q002-potion-survival-basics'
export const PIG_TROUBLE_QUEST_ID = 'q003-pig-trouble'
export const BLACKSMITH_PREPARATION_QUEST_ID = 'q004-before-cave'
export const CAVE_ENTRANCE_INVESTIGATION_QUEST_ID =
  'q005-investigate-cave-entrance'
export const SLIME_BOSS_SHADOW_QUEST_ID = 'q006-slime-boss-shadow'
export const FINAL_SUPPLIES_QUEST_ID = 'q007-final-supplies'
export const PIG_BOSS_THREAT_QUEST_ID = 'q008-pig-boss-threat'

export const FIRST_SLIME_HUNT_OBJECTIVE_ID = 'defeat-slimes'
export const FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS = 3
export const FIRST_SLIME_HUNT_REWARD_GOLD = 100
export const FIRST_SLIME_HUNT_REWARD_EXPERIENCE = 216

export const WIZARD_NPC_ID = 'wizard'
export const POTION_MERCHANT_NPC_ID = 'potion_merchant'
export const BLACKSMITH_NPC_ID = 'blacksmith'
export const QUEST_GIVER_NPC_IDS = [
  WIZARD_NPC_ID,
  POTION_MERCHANT_NPC_ID,
  BLACKSMITH_NPC_ID
] as const

export type QuestStatus =
  | 'not-started'
  | 'active'
  | 'ready-to-turn-in'
  | 'completed'

export type QuestObjectiveType =
  | 'monster-defeat'
  | 'item-use'
  | 'shop-open'
  | 'scene-enter'
  | 'talk'

export type QuestObjectiveTarget = {
  sceneId?: string
  appearanceType?: string
  itemId?: string
  shopId?: string
  npcId?: string
}

export type QuestObjectiveDefinition = {
  id: string
  label: string
  required: number
  type: QuestObjectiveType
  target: QuestObjectiveTarget
}

export type QuestItemReward = {
  id: string
  label: string
  quantity: number
}

export type QuestRewards = {
  gold: number
  experience: number
  items: QuestItemReward[]
}

export type QuestDefinition = {
  id: string
  regionName: string
  giverNpcId: string
  giverName: string
  title: string
  trackerLabel: string
  turnInTrackerText?: string
  prerequisiteQuestIds: string[]
  requestText: string
  guideText: string
  startDialogueLines: string[]
  activeDialogueLines: string[]
  completionDialogueLines: string[]
  arcCompletionMessage?: string
  objectives: QuestObjectiveDefinition[]
  rewards: QuestRewards
}

export type QuestProgress = {
  id: string
  status: QuestStatus
  objectives: Record<string, number>
  trackerVisible: boolean
}

export type QuestLogState = {
  progressByQuestId: Record<string, QuestProgress>
}

export type QuestNpcBadgeKind = 'new' | 'finish'

export type QuestTrackerItem = {
  questId: string
  text: string
}

export type NpcQuestInteractionAction = 'start' | 'active' | 'complete'

export type NpcQuestInteraction = {
  questId: string
  action: NpcQuestInteractionAction
  definition: QuestDefinition
  progress: QuestProgress
}

export type CompleteQuestResult = {
  nextQuestLog: QuestLogState
  didComplete: boolean
  goldReward: number
  experienceReward: number
  itemRewards: QuestItemReward[]
}

export type QuestTextFormatContext = {
  playerName: string
}

const REGION_TIR_CHONAIL = '티르코네일 마을'
const SCENE_HUNTING_GROUND = 'hunting-ground'
const SCENE_CAVE = 'cave'
const MONSTER_SLIME_APPEARANCE_TYPE = 'monster_slime'
const MONSTER_PIG_APPEARANCE_TYPE = 'monster_pig'
const HEALTH_POTION_REWARD: QuestItemReward = {
  id: 'health-potion',
  label: '체력 회복 포션',
  quantity: 1
}
const MANA_POTION_REWARD: QuestItemReward = {
  id: 'mana-potion',
  label: '마나 회복 포션',
  quantity: 1
}

export const QUEST_DEFINITIONS: QuestDefinition[] = [
  {
    id: FIRST_SLIME_HUNT_QUEST_ID,
    regionName: REGION_TIR_CHONAIL,
    giverNpcId: WIZARD_NPC_ID,
    giverName: '마법사',
    title: '첫 사냥: 말캉이 처치',
    trackerLabel: '첫 사냥: 말캉이 처치',
    turnInTrackerText: '첫 사냥: 마법사에게 돌아가기',
    prerequisiteQuestIds: [],
    requestText: '마을의 마법사가 퀘스트를 의뢰했다.',
    guideText:
      '마을 오른쪽 "사냥터로 가는 길"로 포탈을 타 말캉이 3마리를 잡고 오자.',
    startDialogueLines: [
      '요즘 마을 밖 사냥터에 말캉이들이 자주 나타나고 있단다.',
      '아직 위험한 수준은 아니지만, 초보자인 네가 전투에 익숙해지기에는 딱 좋겠구나.',
      '마을 오른쪽의 "사냥터로 가는 길"을 따라가서 말캉이 3마리를 처치하고 돌아오너라.'
    ],
    activeDialogueLines: [
      '아직 말캉이 기운이 남아 있구나.',
      '사냥터에서 말캉이 3마리를 처치하고 돌아오너라.'
    ],
    completionDialogueLines: [
      '잘했다, {playerName}.',
      '생각보다 훨씬 침착하게 싸우는구나.',
      '하지만 사냥터의 기운이 조금 이상하다. 이건 단순한 말캉이 소동이 아닐지도 모르겠구나.'
    ],
    objectives: [
      {
        id: FIRST_SLIME_HUNT_OBJECTIVE_ID,
        label: '말캉이',
        required: FIRST_SLIME_HUNT_REQUIRED_SLIME_DEFEATS,
        type: 'monster-defeat',
        target: {
          sceneId: SCENE_HUNTING_GROUND,
          appearanceType: MONSTER_SLIME_APPEARANCE_TYPE
        }
      }
    ],
    rewards: {
      gold: FIRST_SLIME_HUNT_REWARD_GOLD,
      experience: FIRST_SLIME_HUNT_REWARD_EXPERIENCE,
      items: []
    }
  },
  {
    id: POTION_SURVIVAL_BASICS_QUEST_ID,
    regionName: REGION_TIR_CHONAIL,
    giverNpcId: POTION_MERCHANT_NPC_ID,
    giverName: '물약상인',
    title: '사냥 준비는 생명줄',
    trackerLabel: '사냥 준비는 생명줄',
    prerequisiteQuestIds: [FIRST_SLIME_HUNT_QUEST_ID],
    requestText:
      '물약상인이 사냥을 계속하려면 회복 물약을 제대로 쓰는 법을 알아야 한다고 했다.',
    guideText: '체력 회복 포션을 사용해보고 다시 말을 걸자.',
    startDialogueLines: [
      '물약이 필요하면 여기로 와.',
      '체력 물약과 마나 물약을 준비해뒀어.',
      '사냥터에 계속 나갈 생각이면 물약을 아끼기만 해서는 안 돼.',
      '위험할 때 바로 쓰는 것도 실력이야. 체력 회복 포션을 한 번 사용해보고 와.'
    ],
    activeDialogueLines: [
      '체력 회복 포션을 한 번 사용해봐.',
      '위험할 때 바로 쓰는 것도 실력이야.'
    ],
    completionDialogueLines: [
      '좋아. 이제 최소한 쓰러지기 전에 물약을 누를 줄은 알겠네.',
      '이건 내가 챙겨주는 보급품이야. 동굴 같은 곳에 들어갈 때는 꼭 여유 있게 챙겨.'
    ],
    objectives: [
      {
        id: 'use-health-potion',
        label: '체력 회복 포션 사용',
        required: 1,
        type: 'item-use',
        target: {
          itemId: 'health-potion'
        }
      }
    ],
    rewards: {
      gold: 0,
      experience: 108,
      items: [
        {
          ...HEALTH_POTION_REWARD,
          quantity: 5
        },
        {
          ...MANA_POTION_REWARD,
          quantity: 3
        }
      ]
    }
  },
  {
    id: PIG_TROUBLE_QUEST_ID,
    regionName: REGION_TIR_CHONAIL,
    giverNpcId: WIZARD_NPC_ID,
    giverName: '마법사',
    title: '사냥터의 꿀꿀한 소동',
    trackerLabel: '사냥터의 꿀꿀한 소동',
    prerequisiteQuestIds: [
      FIRST_SLIME_HUNT_QUEST_ID,
      POTION_SURVIVAL_BASICS_QUEST_ID
    ],
    requestText:
      '마법사는 사냥터에 말캉이보다 더 거친 꿀꿀이들이 나타났다고 했다.',
    guideText: '사냥터로 가서 꿀꿀이 2마리를 처치하자.',
    startDialogueLines: [
      '말캉이만 있는 줄 알았더니, 꿀꿀이들도 점점 사나워지고 있구나.',
      '꿀꿀이는 말캉이보다 조금 더 강하다. 방심하면 안 된다.',
      '이번에는 사냥터에서 꿀꿀이 2마리를 처치하고 돌아오너라.'
    ],
    activeDialogueLines: [
      '꿀꿀이는 말캉이보다 조금 더 강하다.',
      '사냥터에서 꿀꿀이 2마리를 처치하고 돌아오너라.'
    ],
    completionDialogueLines: [
      '역시 이상하군.',
      '말캉이와 꿀꿀이가 동시에 사나워지는 건 흔한 일이 아니다.',
      '사냥터 깊은 곳에 있는 동굴 쪽에서 이상한 기운이 흘러나오는 것 같구나.'
    ],
    objectives: [
      {
        id: 'defeat-pigs',
        label: '꿀꿀이',
        required: 2,
        type: 'monster-defeat',
        target: {
          sceneId: SCENE_HUNTING_GROUND,
          appearanceType: MONSTER_PIG_APPEARANCE_TYPE
        }
      }
    ],
    rewards: {
      gold: 120,
      experience: 288,
      items: []
    }
  },
  {
    id: BLACKSMITH_PREPARATION_QUEST_ID,
    regionName: REGION_TIR_CHONAIL,
    giverNpcId: BLACKSMITH_NPC_ID,
    giverName: '대장장이',
    title: '동굴에 들어가기 전에',
    trackerLabel: '동굴에 들어가기 전에',
    prerequisiteQuestIds: [PIG_TROUBLE_QUEST_ID],
    requestText:
      '마법사는 동굴에 들어가기 전 대장장이에게 장비를 점검받으라고 했다.',
    guideText: '대장장이에게 가서 동굴 탐사 준비를 하자.',
    startDialogueLines: [
      '힘세고 강한 아침, 만일 내게 물어보면 I AM 대장장이.',
      '동굴에 들어간다고? 기본 무기만 들고 가기엔 좀 불안한데.',
      '나중에는 YOU 에게 무기도 팔게 될 거야.',
      '일단 이 수호 부적을 가져가. 초보자에게는 작은 방어력도 큰 차이를 만든다.'
    ],
    activeDialogueLines: [
      '장비 상점을 한번 둘러봐.',
      '청동 검, 철 옷, 가죽 신발 같은 장비도 확인해두면 좋다.'
    ],
    completionDialogueLines: [
      '장비는 목숨이다.',
      '청동 검, 철 옷, 가죽 신발 같은 장비도 둘러봐.',
      '돈이 모이면 꼭 바꿔 끼고 가라. 맨몸으로 용감한 척하다가 누우면 아무 의미 없다.'
    ],
    objectives: [
      {
        id: 'open-blacksmith-shop',
        label: '장비 상점 확인',
        required: 1,
        type: 'shop-open',
        target: {
          shopId: 'blacksmith'
        }
      }
    ],
    rewards: {
      gold: 0,
      experience: 180,
      items: [
        {
          id: 'smith-charm',
          label: '수호 부적',
          quantity: 1
        }
      ]
    }
  },
  {
    id: CAVE_ENTRANCE_INVESTIGATION_QUEST_ID,
    regionName: REGION_TIR_CHONAIL,
    giverNpcId: WIZARD_NPC_ID,
    giverName: '마법사',
    title: '동굴입구 조사',
    trackerLabel: '동굴입구 조사',
    prerequisiteQuestIds: [BLACKSMITH_PREPARATION_QUEST_ID],
    requestText:
      '사냥터 깊은 곳의 동굴 안쪽에서 이상한 기운이 느껴진다고 한다.',
    guideText: '사냥터의 "동굴입구" 표지판을 따라가 포탈에서 F를 눌러 동굴에 들어가 보자.',
    startDialogueLines: [
      '이제 사냥터의 원인을 확인할 때가 되었구나.',
      '사냥터에 있는 "동굴입구" 표지판을 찾아라.',
      '그 안쪽에서 이상한 마력이 흘러나오고 있다.',
      '포탈 앞에서 F를 눌러 동굴 안으로 들어가 보거라.'
    ],
    activeDialogueLines: [
      '사냥터에 있는 "동굴입구" 표지판을 찾아라.',
      '포탈 앞에서 F를 눌러 동굴 안으로 들어가 보거라.'
    ],
    completionDialogueLines: [
      '동굴 안으로 들어갔었구나.',
      '네 몸에 어두운 기운이 조금 묻어 있다.',
      '안쪽에 분명 평범한 몬스터보다 강한 존재가 있을 것이다.'
    ],
    objectives: [
      {
        id: 'enter-cave',
        label: '동굴 입장',
        required: 1,
        type: 'scene-enter',
        target: {
          sceneId: SCENE_CAVE
        }
      }
    ],
    rewards: {
      gold: 100,
      experience: 252,
      items: []
    }
  },
  {
    id: SLIME_BOSS_SHADOW_QUEST_ID,
    regionName: REGION_TIR_CHONAIL,
    giverNpcId: WIZARD_NPC_ID,
    giverName: '마법사',
    title: '동굴 속 말캉한 그림자',
    trackerLabel: '동굴 속 말캉한 그림자',
    prerequisiteQuestIds: [CAVE_ENTRANCE_INVESTIGATION_QUEST_ID],
    requestText:
      '동굴 안에는 평범한 말캉이보다 훨씬 큰 말캉이-보스가 있다고 한다.',
    guideText: '동굴로 들어가 말캉이-보스를 처치하자.',
    startDialogueLines: [
      '동굴 안의 기운을 살펴보니, 말캉이의 형태를 한 큰 마력 덩어리가 느껴진다.',
      '평범한 말캉이라고 생각하면 다칠 수 있다.',
      '그건 말캉이-보스다. 체력도 높고 공격도 훨씬 강할 것이다.',
      '준비가 되었다면 동굴로 들어가 말캉이-보스를 처치하고 돌아오너라.'
    ],
    activeDialogueLines: [
      '말캉이-보스는 동굴 안쪽에 있다.',
      '평범한 말캉이라고 생각하면 다칠 수 있다.'
    ],
    completionDialogueLines: [
      '말캉이-보스를 쓰러뜨렸구나.',
      '훌륭하다. 하지만 동굴의 기운이 아직 사라지지 않았다.',
      '오히려 더 깊은 곳에서 무언가가 화가 난 것처럼 움직이고 있구나.'
    ],
    objectives: [
      {
        id: 'defeat-slime-boss',
        label: '말캉이-보스',
        required: 1,
        type: 'monster-defeat',
        target: {
          sceneId: SCENE_CAVE,
          appearanceType: MONSTER_SLIME_APPEARANCE_TYPE
        }
      }
    ],
    rewards: {
      gold: 180,
      experience: 432,
      items: []
    }
  },
  {
    id: FINAL_SUPPLIES_QUEST_ID,
    regionName: REGION_TIR_CHONAIL,
    giverNpcId: POTION_MERCHANT_NPC_ID,
    giverName: '물약상인',
    title: '마지막 보급품',
    trackerLabel: '마지막 보급품',
    prerequisiteQuestIds: [SLIME_BOSS_SHADOW_QUEST_ID],
    requestText:
      '물약상인이 동굴 깊은 곳으로 들어가기 전에 마지막 보급품을 챙겨주겠다고 했다.',
    guideText: '마을의 물약상인에게 가자.',
    startDialogueLines: [
      '동굴 안쪽까지 들어갈 생각이야?',
      '그럼 이번엔 장난이 아니야. 보스 몬스터는 평범한 몬스터보다 훨씬 오래 버틴다고.',
      '여기, 마지막 보급품이야. 아끼지 말고 써.',
      '살아 돌아오는 게 제일 중요하니까.'
    ],
    activeDialogueLines: [
      '준비는 끝났어.',
      '무리하지 말고, 체력이 위험하면 바로 물약을 써.'
    ],
    completionDialogueLines: [
      '준비는 끝났어.',
      '무리하지 말고, 체력이 위험하면 바로 물약을 써.',
      '그리고 돌아오면 꼭 들러. 다친 곳 없는지 봐줄게.'
    ],
    objectives: [
      {
        id: 'talk-to-potion-merchant',
        label: '물약상인과 대화',
        required: 1,
        type: 'talk',
        target: {
          npcId: POTION_MERCHANT_NPC_ID
        }
      }
    ],
    rewards: {
      gold: 0,
      experience: 144,
      items: [
        {
          ...HEALTH_POTION_REWARD,
          quantity: 8
        },
        {
          ...MANA_POTION_REWARD,
          quantity: 5
        }
      ]
    }
  },
  {
    id: PIG_BOSS_THREAT_QUEST_ID,
    regionName: REGION_TIR_CHONAIL,
    giverNpcId: WIZARD_NPC_ID,
    giverName: '마법사',
    title: '티르코네일을 위협하는 꿀꿀이-보스',
    trackerLabel: '꿀꿀이-보스 처치',
    prerequisiteQuestIds: [FINAL_SUPPLIES_QUEST_ID],
    requestText:
      '동굴 깊은 곳에서 꿀꿀이-보스가 마을 주변 몬스터들을 사납게 만들고 있었다.',
    guideText:
      '동굴로 들어가 꿀꿀이-보스를 처치하고 티르코네일 마을로 돌아오자.',
    startDialogueLines: [
      '이제 원인을 알겠다.',
      '동굴 깊은 곳의 꿀꿀이-보스가 주변 몬스터들을 자극하고 있었던 것이다.',
      '지금 막지 않으면 사냥터를 넘어 마을까지 위험해질 수 있다.',
      '{playerName}, 네가 동굴로 들어가 꿀꿀이-보스를 처치해다오.'
    ],
    activeDialogueLines: [
      '동굴 깊은 곳의 꿀꿀이-보스를 처치해야 한다.',
      '지금 막지 않으면 사냥터를 넘어 마을까지 위험해질 수 있다.'
    ],
    completionDialogueLines: [
      '돌아왔구나, {playerName}.',
      '동굴의 어두운 기운이 사라졌다. 네가 해낸 것이다.',
      '이제 티르코네일 마을은 당분간 안전할 것이다.',
      '하지만 이것은 네 모험의 시작일 뿐이다.',
      '레벨 10이 되면 너는 전사, 궁수, 마법사, 도적 중 하나의 길을 선택할 수 있을 것이다.'
    ],
    arcCompletionMessage:
      '티르코네일의 이상한 기운을 해결했다.\n마을 사람들은 {playerName}을 진짜 모험가로 인정하기 시작했다.\n레벨 10이 되면 전직을 통해 새로운 길을 선택할 수 있다.',
    objectives: [
      {
        id: 'defeat-pig-boss',
        label: '꿀꿀이-보스',
        required: 1,
        type: 'monster-defeat',
        target: {
          sceneId: SCENE_CAVE,
          appearanceType: MONSTER_PIG_APPEARANCE_TYPE
        }
      }
    ],
    rewards: {
      gold: 300,
      experience: 720,
      items: []
    }
  }
]

const QUEST_DEFINITION_BY_ID = Object.fromEntries(
  QUEST_DEFINITIONS.map((definition) => [definition.id, definition])
) as Record<string, QuestDefinition>

// 에디터에서 생성·주입한 동적 퀘스트. 정적 QUEST_DEFINITIONS와 합쳐 런타임에서 함께 동작한다.
// 모듈 전역이라 테스트에서는 clearDynamicQuestDefinitions로 비워야 한다(정적 아크 테스트 보호).
const dynamicQuestDefinitionById: Record<string, QuestDefinition> = {}

// 정적 + 동적 전체 정의. 트래커/배지/목표매칭 등 "모든 퀘스트"를 도는 곳은 이걸 써야 한다.
export const getAllQuestDefinitions = (): QuestDefinition[] => [
  ...QUEST_DEFINITIONS,
  ...Object.values(dynamicQuestDefinitionById)
]

const findQuestDefinition = (questId: string): QuestDefinition | undefined =>
  QUEST_DEFINITION_BY_ID[questId] ?? dynamicQuestDefinitionById[questId]

// 진행도 항목을 throw 없이 조회(동적 퀘스트가 아직 진행도에 없을 수 있어 순회 시 방어용).
const questProgressOrUndefined = (
  questLog: QuestLogState,
  questId: string
): QuestProgress | undefined => questLog.progressByQuestId[questId]

// 동적 퀘스트 등록(멱등): 정적 id와 충돌하거나 이미 등록된 id는 무시한다. 씬 로드마다 재주입돼도
// 안전하다. 등록 후에는 ensureQuestProgressEntries로 진행도 항목을 채워야 한다.
export const registerDynamicQuestDefinitions = (
  definitions: QuestDefinition[]
): void => {
  for (const definition of definitions) {
    if (QUEST_DEFINITION_BY_ID[definition.id]) {
      continue
    }
    dynamicQuestDefinitionById[definition.id] = definition
  }
}

// 등록된 퀘스트 중 진행도 항목이 없는 것만 채운다(기존 진행도는 절대 덮어쓰지 않음). 동적 퀘스트는
// createInitialQuestLog 이후에 등록되므로, 등록 직후 이 함수로 진행도를 보강해야 추적이 시작된다.
export const ensureQuestProgressEntries = (
  questLog: QuestLogState
): QuestLogState => {
  let next = questLog
  for (const definition of getAllQuestDefinitions()) {
    if (!next.progressByQuestId[definition.id]) {
      next = {
        progressByQuestId: {
          ...next.progressByQuestId,
          [definition.id]: createInitialQuestProgress(definition)
        }
      }
    }
  }
  return next
}

// 테스트 격리용 — 동적 레지스트리를 비운다.
export const clearDynamicQuestDefinitions = (): void => {
  for (const id of Object.keys(dynamicQuestDefinitionById)) {
    delete dynamicQuestDefinitionById[id]
  }
}

export const createInitialQuestLog = (): QuestLogState => ({
  progressByQuestId: Object.fromEntries(
    QUEST_DEFINITIONS.map((definition) => [
      definition.id,
      createInitialQuestProgress(definition)
    ])
  ) as Record<string, QuestProgress>
})

export const getQuestDefinition = (questId: string): QuestDefinition => {
  const definition = findQuestDefinition(questId)

  if (!definition) {
    throw new Error(`Unknown quest id ${questId}`)
  }

  return definition
}

export const getQuestProgress = (
  questLog: QuestLogState,
  questId: string
): QuestProgress => {
  const quest = questLog.progressByQuestId[questId]

  if (!quest) {
    throw new Error(`Unknown quest id ${questId}`)
  }

  return quest
}

export const isQuestUnlocked = (
  questLog: QuestLogState,
  questId: string
): boolean =>
  getQuestDefinition(questId).prerequisiteQuestIds.every(
    (prerequisiteQuestId) =>
      getQuestProgress(questLog, prerequisiteQuestId).status === 'completed'
  )

export const startQuest = (
  questLog: QuestLogState,
  questId: string
): QuestLogState => {
  const quest = getQuestProgress(questLog, questId)

  if (quest.status !== 'not-started' || !isQuestUnlocked(questLog, questId)) {
    return questLog
  }

  return updateQuestProgress(questLog, {
    ...quest,
    status: 'active',
    trackerVisible: true
  })
}

export const recordQuestObjectiveProgress = (
  questLog: QuestLogState,
  questId: string,
  objectiveId: string,
  amount = 1
): QuestLogState => {
  const quest = getQuestProgress(questLog, questId)

  if (quest.status !== 'active') {
    return questLog
  }

  const definition = getQuestDefinition(questId)
  const objective = definition.objectives.find(
    (candidate) => candidate.id === objectiveId
  )

  if (!objective) {
    return questLog
  }

  const current = quest.objectives[objectiveId] ?? 0
  const nextCurrent = Math.min(
    objective.required,
    current + Math.max(0, Math.floor(amount))
  )
  const nextQuest = {
    ...quest,
    objectives: {
      ...quest.objectives,
      [objectiveId]: nextCurrent
    }
  }

  return updateQuestProgress(questLog, {
    ...nextQuest,
    status: areQuestObjectivesComplete(definition, nextQuest)
      ? 'ready-to-turn-in'
      : 'active'
  })
}

export const recordMonsterDefeatQuestProgress = (
  questLog: QuestLogState,
  target: {
    sceneId: string
    appearanceType: string
  }
): QuestLogState =>
  recordMatchingQuestObjectiveProgress(questLog, (objective) =>
    objective.type === 'monster-defeat' &&
    objective.target.sceneId === target.sceneId &&
    objective.target.appearanceType === target.appearanceType
  )

export const recordItemUseQuestProgress = (
  questLog: QuestLogState,
  itemId: string
): QuestLogState =>
  recordMatchingQuestObjectiveProgress(questLog, (objective) =>
    objective.type === 'item-use' && objective.target.itemId === itemId
  )

export const recordShopOpenQuestProgress = (
  questLog: QuestLogState,
  shopId: string
): QuestLogState =>
  recordMatchingQuestObjectiveProgress(questLog, (objective) =>
    objective.type === 'shop-open' && objective.target.shopId === shopId
  )

export const recordSceneEnterQuestProgress = (
  questLog: QuestLogState,
  sceneId: string
): QuestLogState =>
  recordMatchingQuestObjectiveProgress(questLog, (objective) =>
    objective.type === 'scene-enter' && objective.target.sceneId === sceneId
  )

export const recordTalkQuestProgress = (
  questLog: QuestLogState,
  npcId: string
): QuestLogState =>
  recordMatchingQuestObjectiveProgress(questLog, (objective) =>
    objective.type === 'talk' && objective.target.npcId === npcId
  )

export const setQuestTrackerVisible = (
  questLog: QuestLogState,
  questId: string,
  trackerVisible: boolean
): QuestLogState => {
  const quest = getQuestProgress(questLog, questId)

  if (
    trackerVisible &&
    quest.status !== 'active' &&
    quest.status !== 'ready-to-turn-in'
  ) {
    return questLog
  }

  if (quest.trackerVisible === trackerVisible) {
    return questLog
  }

  return updateQuestProgress(questLog, {
    ...quest,
    trackerVisible
  })
}

export const hideVisibleQuestTrackers = (
  questLog: QuestLogState
): QuestLogState => ({
  progressByQuestId: Object.fromEntries(
    Object.entries(questLog.progressByQuestId).map(([questId, quest]) => [
      questId,
      {
        ...quest,
        trackerVisible: false
      }
    ])
  ) as Record<string, QuestProgress>
})

export const abandonQuest = (
  questLog: QuestLogState,
  questId: string
): QuestLogState =>
  updateQuestProgress(
    questLog,
    createInitialQuestProgress(getQuestDefinition(questId))
  )

export const completeQuest = (
  questLog: QuestLogState,
  questId: string
): CompleteQuestResult => {
  const quest = getQuestProgress(questLog, questId)
  const definition = getQuestDefinition(questId)

  if (quest.status !== 'ready-to-turn-in') {
    return {
      nextQuestLog: questLog,
      didComplete: false,
      goldReward: 0,
      experienceReward: 0,
      itemRewards: []
    }
  }

  return {
    nextQuestLog: updateQuestProgress(questLog, {
      ...quest,
      status: 'completed',
      trackerVisible: false
    }),
    didComplete: true,
    goldReward: definition.rewards.gold,
    experienceReward: definition.rewards.experience,
    itemRewards: definition.rewards.items.map(cloneQuestItemReward)
  }
}

export const getVisibleQuestTrackers = (
  questLog: QuestLogState
): QuestTrackerItem[] =>
  getAllQuestDefinitions().flatMap((definition) => {
    const quest = questProgressOrUndefined(questLog, definition.id)

    if (!quest || !quest.trackerVisible) {
      return []
    }

    switch (quest.status) {
      case 'active':
        const objective = definition.objectives[0]

        return [
          {
            questId: definition.id,
            text: `${definition.trackerLabel} ${quest.objectives[objective.id]}/${objective.required}`
          }
        ]
      case 'ready-to-turn-in':
        return [
          {
            questId: definition.id,
            text:
              definition.turnInTrackerText ??
              `${definition.title}: ${definition.giverName}에게 돌아가기`
          }
        ]
      case 'not-started':
      case 'completed':
        return []
    }
  })

export const getQuestNpcBadgeKind = (
  questLog: QuestLogState,
  questId: string
): QuestNpcBadgeKind | undefined => {
  const quest = getQuestProgress(questLog, questId)

  switch (quest.status) {
    case 'not-started':
      return isQuestUnlocked(questLog, questId) ? 'new' : undefined
    case 'ready-to-turn-in':
      return 'finish'
    case 'active':
    case 'completed':
      return undefined
  }
}

export const getQuestNpcBadgeKindForNpc = (
  questLog: QuestLogState,
  npcId: string
): QuestNpcBadgeKind | undefined => {
  const definitions = getQuestDefinitionsForNpc(npcId)

  if (
    definitions.some(
      (definition) =>
        questProgressOrUndefined(questLog, definition.id)?.status ===
        'ready-to-turn-in'
    )
  ) {
    return 'finish'
  }

  return definitions.some((definition) => {
    const quest = questProgressOrUndefined(questLog, definition.id)

    return (
      quest?.status === 'not-started' && isQuestUnlocked(questLog, definition.id)
    )
  })
    ? 'new'
    : undefined
}

export const getNextQuestInteractionForNpc = (
  questLog: QuestLogState,
  npcId: string
): NpcQuestInteraction | undefined => {
  const definitions = getQuestDefinitionsForNpc(npcId)
  const readyDefinition = definitions.find(
    (definition) =>
      questProgressOrUndefined(questLog, definition.id)?.status ===
      'ready-to-turn-in'
  )

  if (readyDefinition) {
    return createNpcQuestInteraction(questLog, readyDefinition, 'complete')
  }

  const activeDefinition = definitions.find(
    (definition) =>
      questProgressOrUndefined(questLog, definition.id)?.status === 'active'
  )

  if (activeDefinition) {
    return createNpcQuestInteraction(questLog, activeDefinition, 'active')
  }

  const unlockedDefinition = definitions.find((definition) => {
    const quest = questProgressOrUndefined(questLog, definition.id)

    return (
      quest?.status === 'not-started' && isQuestUnlocked(questLog, definition.id)
    )
  })

  return unlockedDefinition
    ? createNpcQuestInteraction(questLog, unlockedDefinition, 'start')
    : undefined
}

export const formatQuestText = (
  text: string,
  context: QuestTextFormatContext
): string => text.replace(/\{playerName\}/g, context.playerName)

export const formatQuestTextLines = (
  lines: string[],
  context: QuestTextFormatContext
): string[] => lines.map((line) => formatQuestText(line, context))

const getQuestDefinitionsForNpc = (npcId: string): QuestDefinition[] =>
  getAllQuestDefinitions().filter((definition) => definition.giverNpcId === npcId)

const createNpcQuestInteraction = (
  questLog: QuestLogState,
  definition: QuestDefinition,
  action: NpcQuestInteractionAction
): NpcQuestInteraction => ({
  questId: definition.id,
  action,
  definition,
  progress: getQuestProgress(questLog, definition.id)
})

const recordMatchingQuestObjectiveProgress = (
  questLog: QuestLogState,
  matchesObjective: (objective: QuestObjectiveDefinition) => boolean
): QuestLogState => {
  let nextQuestLog = questLog

  for (const definition of getAllQuestDefinitions()) {
    const quest = questProgressOrUndefined(nextQuestLog, definition.id)

    if (!quest || quest.status !== 'active') {
      continue
    }

    for (const objective of definition.objectives) {
      if (matchesObjective(objective)) {
        nextQuestLog = recordQuestObjectiveProgress(
          nextQuestLog,
          definition.id,
          objective.id
        )
      }
    }
  }

  return nextQuestLog
}

const createInitialQuestProgress = (
  definition: QuestDefinition
): QuestProgress => ({
  id: definition.id,
  status: 'not-started',
  objectives: Object.fromEntries(
    definition.objectives.map((objective) => [objective.id, 0])
  ) as Record<string, number>,
  trackerVisible: false
})

const updateQuestProgress = (
  questLog: QuestLogState,
  quest: QuestProgress
): QuestLogState => ({
  progressByQuestId: {
    ...questLog.progressByQuestId,
    [quest.id]: quest
  }
})

const areQuestObjectivesComplete = (
  definition: QuestDefinition,
  quest: QuestProgress
): boolean =>
  definition.objectives.every(
    (objective) => (quest.objectives[objective.id] ?? 0) >= objective.required
  )

const cloneQuestItemReward = (itemReward: QuestItemReward): QuestItemReward => ({
  id: itemReward.id,
  label: itemReward.label,
  quantity: itemReward.quantity
})
