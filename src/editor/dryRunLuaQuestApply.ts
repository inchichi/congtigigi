import type { LuaQuestCatalog } from './luaQuestCatalog'
import {
  isLuaQuestAcquirable,
  isLuaQuestEnemy,
  isLuaQuestNpc,
  isLuaQuestScene
} from './luaQuestCatalog'
import { renderGeneratedLuaQuestModule } from './luaQuestCodeGenerator'
import {
  createGeneratedLuaQuestValidationIssues,
  type GeneratedLuaQuestJson
} from './luaQuestSchema'
import type { DryRunReport, DryRunStep, DryRunStepStatus } from './dryRunEventApply'

const objectiveTargetDetail = (
  objective: GeneratedLuaQuestJson['objectives'][number],
  catalog: LuaQuestCatalog
): { status: DryRunStepStatus; detail: string } => {
  const target = objective.target

  switch (objective.type) {
    case 'defeat':
      if (!target.entityId || !isLuaQuestEnemy(catalog, target.entityId)) {
        return {
          status: 'fail',
          detail: `적 엔티티를 찾을 수 없다: ${target.entityId ?? '(없음)'}`
        }
      }
      return {
        status: 'ok',
        detail: `${target.entityId} ${objective.required}회 처치`
      }

    case 'talk':
      if (!target.entityId || !isLuaQuestNpc(catalog, target.entityId)) {
        return {
          status: 'fail',
          detail: `NPC 엔티티를 찾을 수 없다: ${target.entityId ?? '(없음)'}`
        }
      }
      return {
        status: 'ok',
        detail: `${target.entityId}와 대화`
      }

    case 'acquire':
      if (!target.entityId || !isLuaQuestAcquirable(catalog, target.entityId)) {
        return {
          status: 'fail',
          detail: `획득 대상을 찾을 수 없다: ${target.entityId ?? '(없음)'}`
        }
      }
      return {
        status: 'ok',
        detail: `${target.entityId} ${objective.required}개 획득`
      }

    case 'reach':
      if (!target.mapId || !isLuaQuestScene(catalog, target.mapId)) {
        return {
          status: 'fail',
          detail: `이동할 씬을 찾을 수 없다: ${target.mapId ?? '(없음)'}`
        }
      }
      return {
        status: 'ok',
        detail: `${target.mapId} 이동`
      }
  }
}

export const dryRunLuaQuestApply = (
  quest: GeneratedLuaQuestJson,
  catalog: LuaQuestCatalog,
  context: { selectedEntityId?: string } = {}
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

  try {
    renderGeneratedLuaQuestModule(quest)
    add('convert', 'Lua 변환', 'ok', 'Lua 모듈로 렌더링됨')
  } catch (error) {
    add(
      'convert',
      'Lua 변환',
      'fail',
      error instanceof Error ? error.message : String(error)
    )
  }

  if (context.selectedEntityId) {
    if (quest.giver_npc_entity_id === context.selectedEntityId && isLuaQuestNpc(catalog, quest.giver_npc_entity_id)) {
      add('resolve_giver', '기버 NPC', 'ok', `선택한 NPC "${quest.giver_npc_entity_id}" 확인`)
    } else {
      add(
        'resolve_giver',
        '기버 NPC',
        'fail',
        `선택한 NPC와 다르다: ${quest.giver_npc_entity_id} (선택: ${context.selectedEntityId})`
      )
    }
  } else if (isLuaQuestNpc(catalog, quest.giver_npc_entity_id)) {
    add('resolve_giver', '기버 NPC', 'ok', `기버 "${quest.giver_npc_entity_id}" 확인`)
  } else {
    add('resolve_giver', '기버 NPC', 'fail', `존재하지 않는 NPC: ${quest.giver_npc_entity_id}`)
  }

  if (quest.objectives.length === 0) {
    add('objectives', '목표', 'fail', '목표가 없다')
  } else {
    quest.objectives.forEach((objective, index) => {
      const { status, detail } = objectiveTargetDetail(objective, catalog)
      add(`objective_${index + 1}`, `목표 ${index + 1}: ${objective.label}`, status, detail)
    })
  }

  if (quest.rewards.items.length === 0) {
    add('reward_items', '보상 아이템', 'ok', '아이템 보상이 없다')
  } else {
    add(
      'reward_items',
      '보상 아이템',
      'ok',
      `보상 ${quest.rewards.items.length}종 확인`
    )
  }

  const issues = createGeneratedLuaQuestValidationIssues(quest, catalog)
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
