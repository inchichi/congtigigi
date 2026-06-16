import { createHolidayDialogueEventSpecFromGeneratedEventJson } from './eventCodeGenerator'
import { createHolidayDialogueEventValidationErrors } from '../games/my-sample-rpg/eventGeneration'
import {
  createGeneratedEventJsonValidationIssues,
  type GeneratedEventJson
} from './eventJsonSchema'
import type { GameStructureProfile } from './gameStructureProfile'

// 무결성 검증(드라이런): 생성된 이벤트 JSON을 "복제 스냅샷"에 시험 적용하며 실제 런타임
// applyEventDraft(createPixiTiledMapView.ts)와 같은 순서로 단계별 점검한다. 복제본은 점검 후
// 폐기 = 롤백(실제 profile은 절대 변경하지 않는다). 게임 런타임은 건드리지 않는다.

export type DryRunStepStatus = 'ok' | 'fail' | 'warn'

// 단계 id는 표시용 키(UI는 label/detail만 렌더). 이벤트·퀘스트 드라이런이 DryRunReport를
// 공유하도록 string으로 둔다.
export type DryRunStepId = string

export type DryRunStep = {
  id: DryRunStepId
  label: string
  status: DryRunStepStatus
  // 실패/경고 위치·사유. status='ok'면 간단한 성공 메모.
  detail: string
}

export type DryRunReport = {
  // fail 단계가 하나도 없으면 통과(warn은 허용). 적용 버튼은 ok일 때만 활성한다.
  ok: boolean
  steps: DryRunStep[]
  failedStepId?: DryRunStepId
  // 기존 JSON 검증 결과(대체 아님, 보강). "path - message" 형태.
  jsonIssues: string[]
}

// 게임 런타임(characterStates)은 에디터에서 접근 못 하므로 profile로 복제 스냅샷을 만든다.
// 읽기 전용으로만 쓰고 점검 후 버리므로, 실제 profile은 불변이다(= 트랜잭션 롤백과 동일 효과).
type DryRunSnapshot = {
  npcs: { id: string; map: string }[]
  items: { id: string }[]
  maps: { id: string }[]
}

const cloneSnapshot = (profile: GameStructureProfile): DryRunSnapshot => {
  const source = { npcs: profile.npcs, items: profile.items, maps: profile.maps }
  // node 테스트 환경 등 structuredClone이 없을 수 있어 JSON 폴백(profile은 JSON-safe 데이터).
  const clone: typeof source =
    typeof structuredClone === 'function'
      ? structuredClone(source)
      : JSON.parse(JSON.stringify(source))
  return {
    npcs: clone.npcs.map((npc) => ({ id: npc.id, map: npc.map })),
    items: clone.items.map((item) => ({ id: item.id })),
    maps: clone.maps.map((map) => ({ id: map.id }))
  }
}

export const dryRunEventApply = (
  eventJson: GeneratedEventJson,
  profile: GameStructureProfile
): DryRunReport => {
  const snapshot = cloneSnapshot(profile) // transaction begin (복제본)
  const steps: DryRunStep[] = []
  const add = (
    id: DryRunStepId,
    label: string,
    status: DryRunStepStatus,
    detail: string
  ): void => {
    steps.push({ id, label, status, detail })
  }

  // 1) convert — 실제 적용이 쓰는 spec으로 변환되는지. dialogue가 비면 undefined를 돌려준다.
  let spec: ReturnType<typeof createHolidayDialogueEventSpecFromGeneratedEventJson>
  try {
    spec = createHolidayDialogueEventSpecFromGeneratedEventJson(eventJson)
    if (spec) {
      add('convert', '이벤트 변환', 'ok', '이벤트 사양으로 변환됨')
    } else {
      add('convert', '이벤트 변환', 'fail', '대사가 비어 있어 이벤트로 변환할 수 없습니다.')
    }
  } catch (error) {
    spec = undefined
    add(
      'convert',
      '이벤트 변환',
      'fail',
      error instanceof Error ? error.message : String(error)
    )
  }

  // 2) resolve_npc — 대상 NPC 해결. 정확히 일치하면 ok, 없으면 런타임이 폴백(산타/첫 NPC)하므로
  //    warn, NPC가 아예 없으면 fail.
  const matchedNpc = snapshot.npcs.find((npc) => npc.id === eventJson.npc.id)
  if (matchedNpc) {
    add('resolve_npc', '대상 NPC 해결', 'ok', `NPC "${eventJson.npc.id}" 확인`)
  } else if (snapshot.npcs.length > 0) {
    add(
      'resolve_npc',
      '대상 NPC 해결',
      'warn',
      `NPC "${eventJson.npc.id}"가 없어 실행 시 다른 NPC로 대체될 수 있습니다.`
    )
  } else {
    add('resolve_npc', '대상 NPC 해결', 'fail', '맵에 대상이 될 NPC가 없습니다.')
  }

  // 3) resolve_map — 대상 맵 실존 + (NPC를 찾았다면) 그 NPC의 맵과 일치(eventJsonSchema와 동일 규칙).
  const mapExists = snapshot.maps.some((map) => map.id === eventJson.location.map_id)
  if (!mapExists) {
    add('resolve_map', '대상 맵 해결', 'fail', `존재하지 않는 맵: ${eventJson.location.map_id}`)
  } else if (matchedNpc && matchedNpc.map !== eventJson.location.map_id) {
    add(
      'resolve_map',
      '대상 맵 해결',
      'fail',
      `NPC "${matchedNpc.id}"의 맵(${matchedNpc.map})과 이벤트 맵(${eventJson.location.map_id})이 다릅니다.`
    )
  } else {
    add('resolve_map', '대상 맵 해결', 'ok', `맵 "${eventJson.location.map_id}" 확인`)
  }

  // 4) dialogue — 런타임은 opening_lines[0]을 쓰므로 비어 있지 않은 대사 1줄 이상 필요.
  const dialogueLines = eventJson.dialogue
    .map((line) => line.text.trim())
    .filter((line) => line.length > 0)
  if (dialogueLines.length > 0) {
    add('dialogue', '대사 확인', 'ok', `대사 ${dialogueLines.length}줄`)
  } else {
    add('dialogue', '대사 확인', 'fail', '적용 가능한 대사가 한 줄도 없습니다.')
  }

  // 5) reward_item — id 빈값은 fail(런타임이 즉시 무시), 미존재는 warn(런타임이 라벨 폴백으로 관용).
  const rewardItemId = eventJson.reward.item_id.trim()
  if (rewardItemId.length === 0) {
    add('reward_item', '보상 아이템', 'fail', '보상 아이템 ID가 비어 있습니다.')
  } else if (!snapshot.items.some((item) => item.id === rewardItemId)) {
    add(
      'reward_item',
      '보상 아이템',
      'warn',
      `아이템 "${rewardItemId}"가 목록에 없습니다(실행 시 이름 없이 지급될 수 있음).`
    )
  } else {
    add('reward_item', '보상 아이템', 'ok', `아이템 "${rewardItemId}" 확인`)
  }

  // 6) spec_validate — 변환된 spec을 런타임과 같은 검증기로 최종 점검.
  if (spec) {
    const specErrors = createHolidayDialogueEventValidationErrors(spec)
    if (specErrors.length === 0) {
      add('spec_validate', '사양 검증', 'ok', '런타임 사양 검증 통과')
    } else {
      add('spec_validate', '사양 검증', 'fail', specErrors.join('; '))
    }
  } else {
    add('spec_validate', '사양 검증', 'fail', '변환 실패로 사양을 검증할 수 없습니다.')
  }

  // 복제본 폐기 = 롤백. profile은 읽기만 했으므로 그대로 보존된다.
  const jsonIssues = createGeneratedEventJsonValidationIssues(eventJson, profile).map(
    (issue) => `${issue.path} - ${issue.message}`
  )
  const failedStep = steps.find((step) => step.status === 'fail')

  return {
    ok: failedStep === undefined,
    steps,
    failedStepId: failedStep?.id,
    jsonIssues
  }
}
