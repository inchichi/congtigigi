import type { GameStructureProfile } from './gameStructureProfile'
import {
  QUEST_OBJECTIVE_TYPES,
  isQuestGiverNpc,
  isQuestMonsterAppearance,
  isQuestMonsterScene,
  isQuestSceneEnterScene,
  isQuestShop,
  type QuestObjectiveType
} from './myRpgQuestCatalog'

// LLM이 생성하는 "진짜 퀘스트"의 구조화 출력. 런타임 QuestDefinition(questLog.ts)으로 변환된다.
export type GeneratedQuestObjective = {
  type: QuestObjectiveType
  label: string
  required: number
  target: {
    sceneId?: string
    appearanceType?: string
    itemId?: string
    shopId?: string
    npcId?: string
  }
}

export type GeneratedQuestRewardItem = {
  item_id: string
  quantity: number
}

export type GeneratedQuestJson = {
  quest_id: string
  title: string
  giver_npc_id: string
  region: string
  request_text: string
  guide_text: string
  start_dialogue_lines: string[]
  active_dialogue_lines: string[]
  completion_dialogue_lines: string[]
  objectives: GeneratedQuestObjective[]
  rewards: {
    gold: number
    experience: number
    items: GeneratedQuestRewardItem[]
  }
}

export type GeneratedQuestValidationIssue = { path: string; message: string }

const isSnakeCase = (value: string): boolean => /^[a-z0-9_]+$/u.test(value)

// 생성과 분리된 결정적 검증: 목표 타깃이 런타임이 실제로 추적하는 값인지(카탈로그/프로필) 확인한다.
// 여기서 통과해야 게임에서 목표가 진행된다(없는 몬스터/씬/상점/NPC/아이템이면 영원히 안 깨짐).
export const createGeneratedQuestValidationIssues = (
  quest: GeneratedQuestJson,
  profile: GameStructureProfile
): GeneratedQuestValidationIssue[] => {
  const issues: GeneratedQuestValidationIssue[] = []
  const npcIds = new Set(profile.npcs.map((npc) => npc.id))
  const itemIds = new Set(profile.items.map((item) => item.id))

  if (!isSnakeCase(quest.quest_id)) {
    issues.push({
      path: 'quest_id',
      message: 'quest_id는 영문 소문자·숫자·밑줄(snake_case)이어야 한다.'
    })
  }
  if (quest.title.trim().length === 0) {
    issues.push({ path: 'title', message: 'title은 비어 있을 수 없다.' })
  }
  if (!isQuestGiverNpc(quest.giver_npc_id)) {
    issues.push({
      path: 'giver_npc_id',
      message: `퀘스트를 줄 수 있는 NPC가 아니다: ${quest.giver_npc_id}`
    })
  }
  if (quest.start_dialogue_lines.length === 0) {
    issues.push({
      path: 'start_dialogue_lines',
      message: '시작 대사가 최소 1줄 필요하다.'
    })
  }
  if (quest.completion_dialogue_lines.length === 0) {
    issues.push({
      path: 'completion_dialogue_lines',
      message: '완료 대사가 최소 1줄 필요하다.'
    })
  }
  if (quest.objectives.length === 0) {
    issues.push({ path: 'objectives', message: '목표가 최소 1개 필요하다.' })
  }

  quest.objectives.forEach((objective, index) => {
    const path = `objectives[${index}]`
    if (!QUEST_OBJECTIVE_TYPES.includes(objective.type)) {
      issues.push({ path: `${path}.type`, message: `알 수 없는 목표 타입: ${objective.type}` })
      return
    }
    if (!Number.isInteger(objective.required) || objective.required < 1) {
      issues.push({ path: `${path}.required`, message: 'required는 1 이상의 정수여야 한다.' })
    }
    const target = objective.target
    switch (objective.type) {
      case 'monster-defeat':
        if (!target.appearanceType || !isQuestMonsterAppearance(target.appearanceType)) {
          issues.push({
            path: `${path}.target.appearanceType`,
            message: `처치 가능한 몬스터가 아니다: ${target.appearanceType ?? '(없음)'}`
          })
        }
        if (!target.sceneId || !isQuestMonsterScene(target.sceneId)) {
          issues.push({
            path: `${path}.target.sceneId`,
            message: `몬스터가 등장하는 씬이 아니다: ${target.sceneId ?? '(없음)'}`
          })
        }
        break
      case 'item-use':
        if (!target.itemId || !itemIds.has(target.itemId)) {
          issues.push({
            path: `${path}.target.itemId`,
            message: `존재하는 아이템이 아니다: ${target.itemId ?? '(없음)'}`
          })
        }
        break
      case 'shop-open':
        if (!target.shopId || !isQuestShop(target.shopId)) {
          issues.push({
            path: `${path}.target.shopId`,
            message: `존재하는 상점이 아니다: ${target.shopId ?? '(없음)'}`
          })
        }
        break
      case 'scene-enter':
        if (!target.sceneId || !isQuestSceneEnterScene(target.sceneId)) {
          issues.push({
            path: `${path}.target.sceneId`,
            message: `이동 목표로 쓸 수 없는 씬이다(town 제외): ${target.sceneId ?? '(없음)'}`
          })
        }
        break
      case 'talk':
        if (!target.npcId || !npcIds.has(target.npcId)) {
          issues.push({
            path: `${path}.target.npcId`,
            message: `존재하는 NPC가 아니다: ${target.npcId ?? '(없음)'}`
          })
        }
        break
    }
  })

  // 보상 아이템 id 미존재는 막지 않는다(런타임이 라벨 폴백으로 관용 — 드라이런에서 warn으로만 표시).
  // 수량만 검증한다.
  quest.rewards.items.forEach((item, index) => {
    const path = `rewards.items[${index}]`
    if (!Number.isInteger(item.quantity) || item.quantity < 1) {
      issues.push({ path: `${path}.quantity`, message: 'quantity는 1 이상의 정수여야 한다.' })
    }
  })
  if (!Number.isFinite(quest.rewards.gold) || quest.rewards.gold < 0) {
    issues.push({ path: 'rewards.gold', message: 'gold는 0 이상의 숫자여야 한다.' })
  }
  if (!Number.isFinite(quest.rewards.experience) || quest.rewards.experience < 0) {
    issues.push({ path: 'rewards.experience', message: 'experience는 0 이상의 숫자여야 한다.' })
  }

  return issues
}

export const isGeneratedQuestValid = (
  quest: GeneratedQuestJson,
  profile: GameStructureProfile
): boolean => createGeneratedQuestValidationIssues(quest, profile).length === 0
