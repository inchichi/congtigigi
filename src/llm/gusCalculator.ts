import {
  GAME_UNDERSTANDING_SCORE_THRESHOLD,
  type GameStructureProfile,
  type GameUnderstandingScoreDetails,
  type GameUnderstandingScoreReport
} from './gameStructureProfile'

export const calculateGameUnderstandingScore = (
  profile: GameStructureProfile
): GameUnderstandingScoreReport => {
  const details = createGameUnderstandingScoreDetails(profile)
  const gusScore = roundToOneDecimal(
    details.map_coverage * 0.2 +
      details.npc_coverage * 0.2 +
      details.item_coverage * 0.15 +
      details.event_system_accuracy * 0.2 +
      details.file_localization_accuracy * 0.15 +
      details.relationship_accuracy * 0.1
  )

  return {
    gus_score: gusScore,
    threshold: GAME_UNDERSTANDING_SCORE_THRESHOLD,
    status: gusScore >= GAME_UNDERSTANDING_SCORE_THRESHOLD ? 'passed' : 'failed',
    details,
    missing_items: createMissingItems(profile, details)
  }
}

const createGameUnderstandingScoreDetails = (
  profile: GameStructureProfile
): GameUnderstandingScoreDetails => {
  const hasDynamicEventRegistry = profile.events.some(
    (event) => event.id === 'dynamic_event_registry'
  )
  const hasMapFiles = profile.maps.every((map) => map.file.trim().length > 0)
  const hasNpcMapMappings = profile.npcs.every((npc) =>
    profile.maps.some((map) => map.id === npc.map)
  )
  const hasItemFiles = profile.items.every((item) => item.file.trim().length > 0)
  const hasDialogueBridge =
    profile.dialogue_system.file.trim().length > 0 &&
    profile.dialogue_system.format.trim().length > 0

  return {
    map_coverage: percent(profile.maps.length, 3),
    npc_coverage: percent(profile.npcs.length, 6),
    item_coverage: percent(profile.items.length, 5),
    event_system_accuracy: percent(
      [
        hasDynamicEventRegistry,
        hasDialogueBridge,
        profile.event_system.file.trim().length > 0,
        profile.event_system.register_function.trim().length > 0
      ].filter(Boolean).length,
      4
    ),
    file_localization_accuracy: percent(profile.modifiable_files.length, 8),
    relationship_accuracy: percent(
      [
        hasMapFiles,
        hasNpcMapMappings,
        hasItemFiles,
        hasDynamicEventRegistry
      ].filter(Boolean).length,
      4
    )
  }
}

const createMissingItems = (
  profile: GameStructureProfile,
  details: GameUnderstandingScoreDetails
): string[] => {
  const missingItems: string[] = []

  if (details.map_coverage < 100) {
    missingItems.push('맵 전체를 100% 분석하지 못했다.')
  }

  if (details.npc_coverage < 100) {
    missingItems.push('모든 NPC 관계를 완전히 정리하지 못했다.')
  }

  if (details.item_coverage < 100) {
    missingItems.push('모든 아이템 정의를 완전하게 열거하지 못했다.')
  }

  if (
    !profile.event_system.file.includes('DynamicEventManager') ||
    profile.event_system.register_function.length === 0
  ) {
    missingItems.push('이벤트 등록 경로를 확실히 고정하지 못했다.')
  }

  if (profile.modifiable_files.length < 8) {
    missingItems.push('수정 가능한 파일 목록을 더 확장할 수 있다.')
  }

  return missingItems
}

const percent = (value: number, total: number): number =>
  total === 0 ? 0 : roundToOneDecimal(Math.max(0, Math.min(100, (value / total) * 100)))

const roundToOneDecimal = (value: number): number => Math.round(value * 10) / 10
