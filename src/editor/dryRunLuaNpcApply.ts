import type { LuaQuestCatalog } from './luaQuestCatalog'
import { renderGeneratedLuaNpcModule } from './luaNpcCodeGenerator'
import {
  LUA_NPC_APPEARANCES,
  createGeneratedLuaNpcValidationIssues,
  luaNpcEntityId,
  type GeneratedLuaNpcJson,
  type LuaNpcAppearance
} from './luaNpcSchema'
import type { DryRunReport, DryRunStep, DryRunStepStatus } from './dryRunEventApply'

// 생성 NPC가 게임에 더해질 수 있는지 결정적으로 점검한다(생성과 분리된 검증). 퀘스트 드라이런과
// 평행하지만, 타깃 실존 대신 "맵 실존 · 외형 실존 · id 충돌 없음 · 사양 검증"을 단계로 보여준다.
export const dryRunLuaNpcApply = (
  npc: GeneratedLuaNpcJson,
  catalog: LuaQuestCatalog
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
    renderGeneratedLuaNpcModule(npc)
    add('convert', 'Lua 변환', 'ok', 'Lua 모듈로 렌더링됨')
  } catch (error) {
    add(
      'convert',
      'Lua 변환',
      'fail',
      error instanceof Error ? error.message : String(error)
    )
  }

  if (catalog.scenes.includes(npc.map_id)) {
    add('resolve_map', '배치 맵', 'ok', `맵 "${npc.map_id}" 확인`)
  } else {
    add('resolve_map', '배치 맵', 'fail', `존재하지 않는 맵: ${npc.map_id}`)
  }

  if (LUA_NPC_APPEARANCES.includes(npc.appearance as LuaNpcAppearance)) {
    add('resolve_appearance', '외형 에셋', 'ok', `외형 "${npc.appearance}" 확인`)
  } else {
    add(
      'resolve_appearance',
      '외형 에셋',
      'fail',
      `빌려올 수 없는 외형: ${npc.appearance}`
    )
  }

  const entityId = luaNpcEntityId(npc)
  const collides =
    catalog.npcs.some((entity) => entity.id === entityId) ||
    catalog.enemies.some((entity) => entity.id === entityId) ||
    catalog.acquirables.some((entity) => entity.id === entityId)
  if (collides) {
    add('id_collision', '새 id', 'fail', `이미 존재하는 엔티티: ${entityId}`)
  } else {
    add('id_collision', '새 id', 'ok', `새 엔티티 id "${entityId}"`)
  }

  const issues = createGeneratedLuaNpcValidationIssues(npc, catalog)
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
