import type { GameStructureProfile } from './gameStructureProfile'
import { convertGeneratedQuestToDefinition } from './questCodeGenerator'
import {
  createGeneratedQuestValidationIssues,
  type GeneratedQuestJson
} from './questJsonSchema'
import {
  isQuestGiverNpc,
  isQuestMonsterAppearance,
  isQuestMonsterScene,
  isQuestSceneEnterScene,
  isQuestShop
} from './myRpgQuestCatalog'
import type { DryRunReport, DryRunStep, DryRunStepStatus } from './dryRunEventApply'

// 무결성 검증(드라이런) — 퀘스트 버전. 이벤트 드라이런과 같은 DryRunReport를 써서 UI가 그대로
// 동작한다. 런타임 QuestDefinition으로 변환 → 기버/목표/보상 타깃이 실제 추적되는 값인지 단계별
// 점검 → 통과해야만 적용을 허용한다. profile은 읽기만 한다(불변).

// 목표 한 개의 타깃이 런타임에서 추적되는지(=게임에서 진행되는지) 판정한다.
const objectiveTargetDetail = (
  objective: GeneratedQuestJson['objectives'][number],
  itemIds: Set<string>,
  npcIds: Set<string>
): { status: DryRunStepStatus; detail: string } => {
  const target = objective.target
  switch (objective.type) {
    case 'monster-defeat':
      if (!target.appearanceType || !isQuestMonsterAppearance(target.appearanceType)) {
        return { status: 'fail', detail: `처치 가능한 몬스터가 아님: ${target.appearanceType ?? '(없음)'}` }
      }
      if (!target.sceneId || !isQuestMonsterScene(target.sceneId)) {
        return { status: 'fail', detail: `몬스터가 없는 씬: ${target.sceneId ?? '(없음)'}` }
      }
      return { status: 'ok', detail: `${target.sceneId}에서 ${target.appearanceType} ${objective.required}마리` }
    case 'item-use':
      if (!target.itemId || !itemIds.has(target.itemId)) {
        return { status: 'fail', detail: `존재하지 않는 아이템: ${target.itemId ?? '(없음)'}` }
      }
      return { status: 'ok', detail: `${target.itemId} ${objective.required}회 사용` }
    case 'shop-open':
      if (!target.shopId || !isQuestShop(target.shopId)) {
        return { status: 'fail', detail: `존재하지 않는 상점: ${target.shopId ?? '(없음)'}` }
      }
      return { status: 'ok', detail: `${target.shopId} 상점 열기` }
    case 'scene-enter':
      if (!target.sceneId || !isQuestSceneEnterScene(target.sceneId)) {
        return { status: 'fail', detail: `이동 목표로 쓸 수 없는 씬: ${target.sceneId ?? '(없음)'}` }
      }
      return { status: 'ok', detail: `${target.sceneId} 진입` }
    case 'talk':
      if (!target.npcId || !npcIds.has(target.npcId)) {
        return { status: 'fail', detail: `존재하지 않는 NPC: ${target.npcId ?? '(없음)'}` }
      }
      return { status: 'ok', detail: `${target.npcId}와 대화` }
    default:
      return { status: 'fail', detail: `알 수 없는 목표 타입: ${objective.type}` }
  }
}

export const dryRunQuestApply = (
  quest: GeneratedQuestJson,
  profile: GameStructureProfile
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

  // 1) convert — 런타임 QuestDefinition으로 변환되는지(스냅샷; profile은 불변).
  try {
    convertGeneratedQuestToDefinition(quest, profile)
    add('convert', '퀘스트 변환', 'ok', '런타임 퀘스트로 변환됨')
  } catch (error) {
    add('convert', '퀘스트 변환', 'fail', error instanceof Error ? error.message : String(error))
  }

  // 2) resolve_giver — 시작/진행/완료 흐름이 보장되는 기버 NPC인지.
  if (isQuestGiverNpc(quest.giver_npc_id)) {
    add('resolve_giver', '기버 NPC', 'ok', `기버 "${quest.giver_npc_id}" 확인`)
  } else {
    add('resolve_giver', '기버 NPC', 'fail', `퀘스트를 줄 수 없는 NPC: ${quest.giver_npc_id}`)
  }

  // 3) 목표별 — 각 목표 타깃이 런타임에서 추적되는지.
  if (quest.objectives.length === 0) {
    add('objectives', '목표', 'fail', '목표가 없습니다.')
  } else {
    quest.objectives.forEach((objective, index) => {
      const { status, detail } = objectiveTargetDetail(objective, itemIds, npcIds)
      add(`objective_${index + 1}`, `목표 ${index + 1}: ${objective.label}`, status, detail)
    })
  }

  // 4) reward_items — 보상 아이템 실존(미존재는 런타임이 라벨 폴백으로 관용 → warn).
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

  // 5) spec_validate — 전체 결정적 검증(이게 실패면 적용 차단).
  const issues = createGeneratedQuestValidationIssues(quest, profile)
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
