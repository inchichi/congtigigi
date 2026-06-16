import type { GameStructureProfile } from './gameStructureProfile'
import { convertGeneratedQuestToDefinition } from './questCodeGenerator'
import {
  createGeneratedQuestValidationIssues,
  type GeneratedQuestJson,
  type GeneratedQuestValidationContext
} from './questJsonSchema'
import {
  isQuestGiverNpc,
  isQuestMonsterAppearance,
  isQuestMonsterDropItem,
  isQuestMonsterScene,
  isQuestSceneEnterScene,
  isQuestShop
} from './questGenerationCatalog'
import type { DryRunReport, DryRunStep, DryRunStepStatus } from './dryRunEventApply'

const objectiveTargetDetail = (
  objective: GeneratedQuestJson['objectives'][number],
  itemIds: Set<string>,
  npcIds: Set<string>
): { status: DryRunStepStatus; detail: string } => {
  const target = objective.target

  switch (objective.type) {
    case 'monster-defeat':
      if (!target.appearanceType || !isQuestMonsterAppearance(target.appearanceType)) {
        return {
          status: 'fail',
          detail: `처치 가능한 몬스터가 아닙니다: ${target.appearanceType ?? '(없음)'}`
        }
      }
      if (!target.sceneId || !isQuestMonsterScene(target.sceneId)) {
        return {
          status: 'fail',
          detail: `몬스터가 없는 장면입니다: ${target.sceneId ?? '(없음)'}`
        }
      }
      return {
        status: 'ok',
        detail: `${target.sceneId}에서 ${target.appearanceType} ${objective.required}마리`
      }

    case 'item-use':
      if (!target.itemId || !itemIds.has(target.itemId)) {
        return {
          status: 'fail',
          detail: `존재하지 않는 아이템입니다: ${target.itemId ?? '(없음)'}`
        }
      }
      return { status: 'ok', detail: `${target.itemId} ${objective.required}개 사용` }

    case 'item-acquire':
      if (!target.itemId || !isQuestMonsterDropItem(target.itemId)) {
        return {
          status: 'fail',
          detail: `몬스터 드롭이 아닌 획득 아이템입니다: ${target.itemId ?? '(없음)'}`
        }
      }
      return { status: 'ok', detail: `${target.itemId} ${objective.required}개 획득` }

    case 'shop-open':
      if (!target.shopId || !isQuestShop(target.shopId)) {
        return {
          status: 'fail',
          detail: `존재하지 않는 상점입니다: ${target.shopId ?? '(없음)'}`
        }
      }
      return { status: 'ok', detail: `${target.shopId} 상점 열기` }

    case 'scene-enter':
      if (!target.sceneId || !isQuestSceneEnterScene(target.sceneId)) {
        return {
          status: 'fail',
          detail: `이동 목표로 쓸 수 없는 씬입니다: ${target.sceneId ?? '(없음)'}`
        }
      }
      return { status: 'ok', detail: `${target.sceneId} 진입` }

    case 'talk':
      if (!target.npcId || !npcIds.has(target.npcId)) {
        return {
          status: 'fail',
          detail: `존재하지 않는 NPC입니다: ${target.npcId ?? '(없음)'}`
        }
      }
      return { status: 'ok', detail: `${target.npcId}와 대화` }

    default:
      return { status: 'fail', detail: `지원하지 않는 목표 타입입니다: ${objective.type}` }
  }
}

export const dryRunQuestApply = (
  quest: GeneratedQuestJson,
  profile: GameStructureProfile,
  context: GeneratedQuestValidationContext = {}
): DryRunReport => {
  const steps: DryRunStep[] = []
  const add = (
    id: string,
    label: string,
    status: DryRunStepStatus,
    detail: string
  ): void => {
    steps.push({ id, label, status, detail })
  }
  const itemIds = new Set(profile.items.map((item) => item.id))
  const npcIds = new Set(profile.npcs.map((npc) => npc.id))

  try {
    convertGeneratedQuestToDefinition(quest, profile)
    add('convert', '퀘스트 변환', 'ok', '퀘스트를 런타임 정의로 변환했습니다.')
  } catch (error) {
    add(
      'convert',
      '퀘스트 변환',
      'fail',
      error instanceof Error ? error.message : String(error)
    )
  }

  if (context.selectedEntityId) {
    if (quest.giver_npc_id === context.selectedEntityId && isQuestGiverNpc(quest.giver_npc_id)) {
      add('resolve_giver', '기버 NPC', 'ok', `선택한 NPC "${quest.giver_npc_id}" 확인`)
    } else {
      add(
        'resolve_giver',
        '기버 NPC',
        'fail',
        `선택한 NPC와 다릅니다: ${quest.giver_npc_id} (선택: ${context.selectedEntityId})`
      )
    }
  } else if (isQuestGiverNpc(quest.giver_npc_id)) {
    add('resolve_giver', '기버 NPC', 'ok', `기버 "${quest.giver_npc_id}" 확인`)
  } else {
    add('resolve_giver', '기버 NPC', 'fail', `존재하지 않는 NPC입니다: ${quest.giver_npc_id}`)
  }

  if (quest.objectives.length === 0) {
    add('objectives', '목표', 'fail', '목표가 없습니다.')
  } else {
    quest.objectives.forEach((objective, index) => {
      const { status, detail } = objectiveTargetDetail(objective, itemIds, npcIds)
      add(`objective_${index + 1}`, `목표 ${index + 1}: ${objective.label}`, status, detail)
    })
  }

  if (quest.rewards.items.length === 0) {
    add('reward_items', '보상 아이템', 'ok', '아이템 보상 없음(골드/경험치만)')
  } else {
    const unknown = quest.rewards.items.filter((item) => !itemIds.has(item.item_id))
    if (unknown.length === 0) {
      add('reward_items', '보상 아이템', 'ok', `보상 ${quest.rewards.items.length}종 확인`)
    } else {
      add(
        'reward_items',
        '보상 아이템',
        'warn',
        `목록에 없는 아이템: ${unknown.map((item) => item.item_id).join(', ')}`
      )
    }
  }

  const issues = createGeneratedQuestValidationIssues(quest, profile, context)
  if (issues.length === 0) {
    add('spec_validate', '사양 검증', 'ok', '검증 통과')
  } else {
    add('spec_validate', '사양 검증', 'fail', `검증 ${issues.length}건 실패`)
  }

  const jsonIssues = issues.map((issue) => `${issue.path} - ${issue.message}`)
  const failedStep = steps.find((step) => step.status === 'fail')

  return {
    ok: failedStep === undefined,
    steps,
    failedStepId: failedStep?.id,
    jsonIssues
  }
}
