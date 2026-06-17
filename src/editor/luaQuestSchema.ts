import {
  isLuaQuestAcquirable,
  isLuaQuestEnemy,
  isLuaQuestNpc,
  isLuaQuestScene,
  type LuaQuestCatalog
} from './luaQuestCatalog'

// legend-of-lua용 퀘스트의 구조화 출력. my-sample-rpg의 GeneratedQuestJson과 평행하지만, 목표 타깃이
// (카탈로그/프로필이 아니라) 열린 게임에 "배치된 엔티티 id" + mapId를 가리킨다.
export type LuaQuestObjectiveType = 'defeat' | 'talk' | 'acquire' | 'reach'

export const LUA_QUEST_OBJECTIVE_TYPES: LuaQuestObjectiveType[] = [
  'defeat',
  'talk',
  'acquire',
  'reach'
]

export type GeneratedLuaQuestObjective = {
  type: LuaQuestObjectiveType
  label: string
  required: number
  target: {
    entityId?: string
    mapId?: string
  }
}

export type GeneratedLuaQuestRewardItem = {
  label: string
  quantity: number
}

export type GeneratedLuaQuestJson = {
  quest_id: string
  title: string
  giver_npc_entity_id: string
  request_text: string
  guide_text: string
  start_dialogue_lines: string[]
  active_dialogue_lines: string[]
  completion_dialogue_lines: string[]
  objectives: GeneratedLuaQuestObjective[]
  rewards: {
    gold: number
    experience: number
    items: GeneratedLuaQuestRewardItem[]
  }
}

export type GeneratedLuaQuestValidationIssue = { path: string; message: string }

const isSnakeCase = (value: string): boolean => /^[a-z0-9_]+$/u.test(value)

// 목표 타깃이 열린 게임에 실재하는지 검증한다 — 없는 적/NPC/씬을 가리키면 게임이 추적할 수 없다.
export const createGeneratedLuaQuestValidationIssues = (
  quest: GeneratedLuaQuestJson,
  catalog: LuaQuestCatalog
): GeneratedLuaQuestValidationIssue[] => {
  const issues: GeneratedLuaQuestValidationIssue[] = []

  if (!isSnakeCase(quest.quest_id)) {
    issues.push({
      path: 'quest_id',
      message: 'quest_id는 영문 소문자·숫자·밑줄(snake_case)이어야 한다.'
    })
  }
  if (quest.title.trim().length === 0) {
    issues.push({ path: 'title', message: 'title은 비어 있을 수 없다.' })
  }
  if (!isLuaQuestNpc(catalog, quest.giver_npc_entity_id)) {
    issues.push({
      path: 'giver_npc_entity_id',
      message: `이 게임의 NPC 엔티티가 아니다: ${quest.giver_npc_entity_id}`
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
    if (!LUA_QUEST_OBJECTIVE_TYPES.includes(objective.type)) {
      issues.push({ path: `${path}.type`, message: `알 수 없는 목표 타입: ${objective.type}` })
      return
    }
    if (!Number.isInteger(objective.required) || objective.required < 1) {
      issues.push({ path: `${path}.required`, message: 'required는 1 이상의 정수여야 한다.' })
    }
    const entityId = objective.target.entityId
    const mapId = objective.target.mapId
    switch (objective.type) {
      case 'defeat':
        if (!entityId || !isLuaQuestEnemy(catalog, entityId)) {
          issues.push({
            path: `${path}.target.entityId`,
            message: `이 게임의 적 엔티티가 아니다: ${entityId ?? '(없음)'}`
          })
        }
        break
      case 'talk':
        if (!entityId || !isLuaQuestNpc(catalog, entityId)) {
          issues.push({
            path: `${path}.target.entityId`,
            message: `이 게임의 NPC 엔티티가 아니다: ${entityId ?? '(없음)'}`
          })
        }
        break
      case 'acquire':
        if (!entityId || !isLuaQuestAcquirable(catalog, entityId)) {
          issues.push({
            path: `${path}.target.entityId`,
            message: `이 게임의 획득 대상이 아니다: ${entityId ?? '(없음)'}`
          })
        }
        break
      case 'reach':
        if (!mapId || !isLuaQuestScene(catalog, mapId)) {
          issues.push({
            path: `${path}.target.mapId`,
            message: `이 게임의 맵이 아니다: ${mapId ?? '(없음)'}`
          })
        }
        break
    }
  })

  quest.rewards.items.forEach((item, index) => {
    if (!Number.isInteger(item.quantity) || item.quantity < 1) {
      issues.push({
        path: `rewards.items[${index}].quantity`,
        message: 'quantity는 1 이상의 정수여야 한다.'
      })
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

export const isGeneratedLuaQuestValid = (
  quest: GeneratedLuaQuestJson,
  catalog: LuaQuestCatalog
): boolean => createGeneratedLuaQuestValidationIssues(quest, catalog).length === 0
