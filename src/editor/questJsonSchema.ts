import type { GameStructureProfile } from './gameStructureProfile'
import {
  QUEST_OBJECTIVE_TYPES,
  isQuestGiverNpc,
  isQuestMonsterAppearance,
  isQuestMonsterDropItem,
  isQuestMonsterScene,
  isQuestSceneEnterScene,
  isQuestShop,
  type QuestObjectiveType
} from './questGenerationCatalog'

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

export type GeneratedQuestValidationContext = {
  selectedEntityId?: string
}

const isSnakeCase = (value: string): boolean => /^[a-z0-9_]+$/u.test(value)

export const createGeneratedQuestValidationIssues = (
  quest: GeneratedQuestJson,
  profile: GameStructureProfile,
  context: GeneratedQuestValidationContext = {}
): GeneratedQuestValidationIssue[] => {
  const issues: GeneratedQuestValidationIssue[] = []
  const npcIds = new Set(profile.npcs.map((npc) => npc.id))
  const itemIds = new Set(profile.items.map((item) => item.id))

  if (!isSnakeCase(quest.quest_id)) {
    issues.push({
      path: 'quest_id',
      message: 'quest_id는 소문자와 밑줄만 써야 합니다.'
    })
  }
  if (quest.title.trim().length === 0) {
    issues.push({ path: 'title', message: 'title은 비어 있을 수 없습니다.' })
  }
  if (
    (context.selectedEntityId && quest.giver_npc_id !== context.selectedEntityId) ||
    !isQuestGiverNpc(quest.giver_npc_id)
  ) {
    issues.push({
      path: 'giver_npc_id',
      message:
        context.selectedEntityId && quest.giver_npc_id !== context.selectedEntityId
          ? `선택한 NPC가 아닌 giver_npc_id입니다: ${quest.giver_npc_id}`
          : `존재하는 NPC가 아닙니다: ${quest.giver_npc_id}`
    })
  }
  if (quest.start_dialogue_lines.length === 0) {
    issues.push({
      path: 'start_dialogue_lines',
      message: '시작 대사는 최소 1줄이 필요합니다.'
    })
  }
  if (quest.completion_dialogue_lines.length === 0) {
    issues.push({
      path: 'completion_dialogue_lines',
      message: '완료 대사는 최소 1줄이 필요합니다.'
    })
  }
  if (quest.objectives.length === 0) {
    issues.push({ path: 'objectives', message: '목표가 최소 1개 필요합니다.' })
  }

  quest.objectives.forEach((objective, index) => {
    const path = `objectives[${index}]`
    if (!QUEST_OBJECTIVE_TYPES.includes(objective.type)) {
      issues.push({ path: `${path}.type`, message: `지원하지 않는 목표 타입입니다: ${objective.type}` })
      return
    }
    if (!Number.isInteger(objective.required) || objective.required < 1) {
      issues.push({ path: `${path}.required`, message: 'required는 1 이상의 정수여야 합니다.' })
    }
    const target = objective.target
    switch (objective.type) {
      case 'monster-defeat':
        if (!target.appearanceType || !isQuestMonsterAppearance(target.appearanceType)) {
          issues.push({
            path: `${path}.target.appearanceType`,
            message: `처치 가능한 몬스터가 아닙니다: ${target.appearanceType ?? '(없음)'}`
          })
        }
        if (!target.sceneId || !isQuestMonsterScene(target.sceneId)) {
          issues.push({
            path: `${path}.target.sceneId`,
            message: `몬스터가 등장하지 않는 씬입니다: ${target.sceneId ?? '(없음)'}`
          })
        }
        break
      case 'item-use':
        if (!target.itemId || !itemIds.has(target.itemId)) {
          issues.push({
            path: `${path}.target.itemId`,
            message: `존재하는 아이템이 아닙니다: ${target.itemId ?? '(없음)'}`
          })
        }
        break
      case 'item-acquire':
        if (!target.itemId || !isQuestMonsterDropItem(target.itemId)) {
          issues.push({
            path: `${path}.target.itemId`,
            message: `몬스터 드롭으로 획득 가능한 아이템이 아닙니다: ${target.itemId ?? '(없음)'}`
          })
        }
        break
      case 'shop-open':
        if (!target.shopId || !isQuestShop(target.shopId)) {
          issues.push({
            path: `${path}.target.shopId`,
            message: `존재하는 상점이 아닙니다: ${target.shopId ?? '(없음)'}`
          })
        }
        break
      case 'scene-enter':
        if (!target.sceneId || !isQuestSceneEnterScene(target.sceneId)) {
          issues.push({
            path: `${path}.target.sceneId`,
            message: `이동 목표로 쓸 수 없는 씬입니다(town 제외): ${target.sceneId ?? '(없음)'}`
          })
          break
        }
        break
      case 'talk':
        if (!target.npcId || !npcIds.has(target.npcId)) {
          issues.push({
            path: `${path}.target.npcId`,
            message: `존재하는 NPC가 아닙니다: ${target.npcId ?? '(없음)'}`
          })
        }
        break
    }
  })

  quest.rewards.items.forEach((item, index) => {
    const path = `rewards.items[${index}]`
    if (!Number.isInteger(item.quantity) || item.quantity < 1) {
      issues.push({ path: `${path}.quantity`, message: 'quantity는 1 이상의 정수여야 합니다.' })
    }
  })
  if (!Number.isFinite(quest.rewards.gold) || quest.rewards.gold < 0) {
    issues.push({ path: 'rewards.gold', message: 'gold는 0 이상의 숫자여야 합니다.' })
  }
  if (!Number.isFinite(quest.rewards.experience) || quest.rewards.experience < 0) {
    issues.push({
      path: 'rewards.experience',
      message: 'experience는 0 이상의 숫자여야 합니다.'
    })
  }

  return issues
}

export const isGeneratedQuestValid = (
  quest: GeneratedQuestJson,
  profile: GameStructureProfile,
  context?: GeneratedQuestValidationContext
): boolean => createGeneratedQuestValidationIssues(quest, profile, context).length === 0
