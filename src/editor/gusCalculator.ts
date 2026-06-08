import {
  GAME_UNDERSTANDING_SCORE_THRESHOLD,
  type GameStructureProfile,
  type GameUnderstandingScoreDetails,
  type GameUnderstandingScoreReport
} from './gameStructureProfile'

type ScoreComponent = {
  score: number
  evidence: string[]
}

export const calculateGameUnderstandingScore = (
  profile: GameStructureProfile
): GameUnderstandingScoreReport => {
  const qaCorrectness = createQaCorrectness(profile)
  const qaCompleteness = createQaCompleteness(profile)
  const dependencyF1 = createDependencyF1(profile)
  const localizationAccuracy = createLocalizationAccuracy(profile)
  const traceCorrectness = createTraceCorrectness(profile)
  const groundingScore = createGroundingScore(profile)

  const details: GameUnderstandingScoreDetails = {
    qa_correctness: qaCorrectness.score,
    qa_completeness: qaCompleteness.score,
    dependency_f1: dependencyF1.score,
    localization_accuracy: localizationAccuracy.score,
    trace_correctness: traceCorrectness.score,
    grounding_score: groundingScore.score
  }

  const gusScore = roundToOneDecimal(
    details.qa_correctness * 0.25 +
      details.qa_completeness * 0.2 +
      details.dependency_f1 * 0.2 +
      details.localization_accuracy * 0.15 +
      details.trace_correctness * 0.1 +
      details.grounding_score * 0.1
  )

  return {
    gus_score: gusScore,
    threshold: GAME_UNDERSTANDING_SCORE_THRESHOLD,
    status: gusScore >= GAME_UNDERSTANDING_SCORE_THRESHOLD ? 'passed' : 'failed',
    details,
    missing_items: createMissingItems(profile, {
      qaCorrectness,
      qaCompleteness,
      dependencyF1,
      localizationAccuracy,
      traceCorrectness,
      groundingScore
    })
  }
}

const createQaCorrectness = (profile: GameStructureProfile): ScoreComponent => {
  const structureIntegrity = weightedScore([
    { score: booleanScore(profile.game_title.trim().length > 0), weight: 0.12 },
    { score: booleanScore(profile.engine.trim().length > 0), weight: 0.12 },
    { score: ratioScore(profile.maps.length, 5), weight: 0.2 },
    { score: ratioScore(profile.npcs.length, 8), weight: 0.2 },
    { score: ratioScore(profile.items.length, 6), weight: 0.16 },
    { score: ratioScore(profile.events.length, 5), weight: 0.12 },
    {
      score:
        booleanScore(profile.dialogue_system.file.trim().length > 0) *
        booleanScore(profile.dialogue_system.format.trim().length > 0),
      weight: 0.04
    },
    {
      score:
        booleanScore(profile.event_system.file.trim().length > 0) *
        booleanScore(profile.event_system.register_function.trim().length > 0),
      weight: 0.04
    }
  ])

  return {
    score: structureIntegrity,
    evidence: [
      `maps=${profile.maps.length}/5`,
      `npcs=${profile.npcs.length}/8`,
      `items=${profile.items.length}/6`,
      `events=${profile.events.length}/5`
    ]
  }
}

const createQaCompleteness = (profile: GameStructureProfile): ScoreComponent => {
  const completeness = weightedScore([
    { score: ratioScore(profile.maps.length, 4), weight: 0.16 },
    { score: ratioScore(profile.npcs.length, 6), weight: 0.2 },
    { score: ratioScore(profile.items.length, 5), weight: 0.16 },
    { score: ratioScore(profile.events.length, 5), weight: 0.16 },
    {
      score:
        booleanScore(profile.dialogue_system.file.trim().length > 0) *
        booleanScore(profile.dialogue_system.format.trim().length > 0),
      weight: 0.16
    },
    {
      score:
        booleanScore(profile.event_system.file.trim().length > 0) *
        booleanScore(profile.event_system.register_function.trim().length > 0),
      weight: 0.16
    }
  ])

  return {
    score: completeness,
    evidence: [
      `sections=${[
        profile.maps.length > 0,
        profile.npcs.length > 0,
        profile.items.length > 0,
        profile.events.length > 0
      ].filter(Boolean).length}/4`,
      `dialogue_system=${profile.dialogue_system.file.trim().length > 0 ? 'yes' : 'no'}`,
      `event_system=${profile.event_system.file.trim().length > 0 ? 'yes' : 'no'}`
    ]
  }
}

const createDependencyF1 = (profile: GameStructureProfile): ScoreComponent => {
  const relationChecks = [
    {
      label: 'npc_to_map',
      passed: profile.npcs.every((npc) => profile.maps.some((map) => map.id === npc.map))
    },
    {
      label: 'map_files',
      passed: profile.maps.every((map) => map.file.trim().length > 0)
    },
    {
      label: 'npc_files',
      passed: profile.npcs.every((npc) => npc.file.trim().length > 0)
    },
    {
      label: 'item_files',
      passed: profile.items.every((item) => item.file.trim().length > 0)
    },
    {
      label: 'event_files',
      passed: profile.events.every((event) => event.file.trim().length > 0)
    },
    {
      label: 'dialogue_bridge',
      passed:
        profile.dialogue_system.file.trim().length > 0 &&
        profile.dialogue_system.format.trim().length > 0
    },
    {
      label: 'event_bridge',
      passed:
        profile.event_system.file.trim().length > 0 &&
        profile.event_system.register_function.trim().length > 0
    }
  ]

  const baseScore = aggregateBinaryChecks(
    relationChecks.map((check) => ({
      passed: check.passed,
      evidence: check.label
    }))
  ).score

  const densityScore = weightedScore([
    { score: ratioScore(new Set(profile.maps.map((map) => map.file)).size, 4), weight: 0.35 },
    { score: ratioScore(new Set(profile.npcs.map((npc) => npc.file)).size, 4), weight: 0.25 },
    { score: ratioScore(new Set(profile.items.map((item) => item.file)).size, 3), weight: 0.2 },
    { score: ratioScore(new Set(profile.events.map((event) => event.file)).size, 3), weight: 0.2 }
  ])

  return {
    score: roundToOneDecimal(baseScore * 0.7 + densityScore * 0.3),
    evidence: relationChecks.filter((check) => check.passed).map((check) => check.label)
  }
}

const createLocalizationAccuracy = (profile: GameStructureProfile): ScoreComponent => {
  const keyFiles = new Set(
    [
      ...profile.maps.map((map) => map.file),
      ...profile.npcs.map((npc) => npc.file),
      ...profile.items.map((item) => item.file),
      ...profile.events.map((event) => event.file),
      profile.dialogue_system.file,
      profile.event_system.file
    ].filter((file) => file.trim().length > 0)
  )

  const fileBreadthScore = weightedScore([
    { score: ratioScore(keyFiles.size, 12), weight: 0.6 },
    { score: ratioScore(profile.modifiable_files.length, 8), weight: 0.4 }
  ])

  return {
    score: fileBreadthScore,
    evidence: [`key_files=${keyFiles.size}`, `modifiable_files=${profile.modifiable_files.length}`]
  }
}

const createTraceCorrectness = (profile: GameStructureProfile): ScoreComponent => {
  const traceAnchors = [
    profile.npcs.every((npc) => profile.maps.some((map) => map.id === npc.map)),
    profile.maps.every((map) => map.file.trim().length > 0),
    profile.npcs.every((npc) => npc.file.trim().length > 0),
    profile.items.every((item) => item.file.trim().length > 0),
    profile.events.every((event) => event.file.trim().length > 0),
    profile.dialogue_system.file.trim().length > 0 && profile.dialogue_system.format.trim().length > 0,
    profile.event_system.file.trim().length > 0 && profile.event_system.register_function.trim().length > 0,
    profile.modifiable_files.some((file) => file.includes('questLog'))
  ]

  const baseScore = aggregateBinaryChecks(
    traceAnchors.map((passed, index) => ({
      passed,
      evidence: `trace_anchor_${index + 1}`
    }))
  ).score

  const coverageScore = weightedScore([
    { score: ratioScore(new Set(profile.maps.map((map) => map.file)).size, 4), weight: 0.3 },
    { score: ratioScore(new Set(profile.npcs.map((npc) => npc.file)).size, 4), weight: 0.25 },
    { score: ratioScore(new Set(profile.items.map((item) => item.file)).size, 3), weight: 0.2 },
    { score: ratioScore(new Set(profile.events.map((event) => event.file)).size, 3), weight: 0.25 }
  ])

  return {
    score: roundToOneDecimal(baseScore * 0.65 + coverageScore * 0.35),
    evidence: traceAnchors
      .map((passed, index) => ({ passed, label: `trace_anchor_${index + 1}` }))
      .filter((entry) => entry.passed)
      .map((entry) => entry.label)
  }
}

const createGroundingScore = (profile: GameStructureProfile): ScoreComponent => {
  const grounding = weightedScore([
    { score: booleanScore(profile.game_title.trim().length > 0), weight: 0.1 },
    { score: booleanScore(profile.engine.trim().length > 0), weight: 0.1 },
    { score: ratioScore(profile.maps.length, 4), weight: 0.2 },
    { score: ratioScore(profile.npcs.length, 6), weight: 0.2 },
    { score: ratioScore(profile.items.length, 5), weight: 0.15 },
    { score: ratioScore(profile.events.length, 5), weight: 0.15 },
    { score: booleanScore(profile.dialogue_system.file.trim().length > 0), weight: 0.05 },
    { score: booleanScore(profile.event_system.file.trim().length > 0), weight: 0.05 }
  ])

  return {
    score: grounding,
    evidence: [
      `game_title=${profile.game_title}`,
      `engine=${profile.engine}`,
      `maps=${profile.maps.length}`,
      `npcs=${profile.npcs.length}`
    ]
  }
}

const createMissingItems = (
  profile: GameStructureProfile,
  components: Record<string, ScoreComponent>
): string[] => {
  const missingItems: string[] = []

  if (components.qaCorrectness.score < 80) {
    missingItems.push('프로젝트 구조의 핵심 항목 일부가 충분히 근거화되지 않았다.')
  }

  if (components.qaCompleteness.score < 80) {
    missingItems.push('구조 프로필의 필수 섹션 일부가 비어 있다.')
  }

  if (components.dependencyF1.score < 80) {
    missingItems.push('의존성 관계를 더 정확하게 정리할 수 있다.')
  }

  if (components.localizationAccuracy.score < 80) {
    missingItems.push('파일 위치를 더 폭넓게 찾을 수 있다.')
  }

  if (components.traceCorrectness.score < 80) {
    missingItems.push('코드 흐름 추적 근거를 더 보강할 수 있다.')
  }

  if (components.groundingScore.score < 80) {
    missingItems.push('프로필의 근거 파일과 설명을 더 촘촘히 연결할 수 있다.')
  }

  if (profile.modifiable_files.length < 7) {
    missingItems.push('수정 가능한 파일 목록을 더 확장할 수 있다.')
  }

  return missingItems
}

const aggregateBinaryChecks = (checks: Array<{ passed: boolean; evidence: string }>): ScoreComponent => {
  const total = checks.length
  const passed = checks.filter((check) => check.passed).length

  return {
    score: percent(passed, total),
    evidence: checks.filter((check) => check.passed).map((check) => check.evidence)
  }
}

const percent = (value: number, total: number): number =>
  total === 0 ? 0 : roundToOneDecimal(clampPercent((value / total) * 100))

const clampPercent = (value: number): number => Math.max(0, Math.min(100, value))

const ratioScore = (value: number, target: number): number =>
  target <= 0 ? 0 : roundToOneDecimal(clampPercent((value / target) * 100))

const booleanScore = (passed: boolean): number => (passed ? 100 : 0)

const weightedScore = (entries: Array<{ score: number; weight: number }>): number => {
  const totalWeight = entries.reduce((sum, entry) => sum + entry.weight, 0)

  if (totalWeight === 0) {
    return 0
  }

  const score = entries.reduce((sum, entry) => sum + entry.score * entry.weight, 0) / totalWeight
  return roundToOneDecimal(clampPercent(score))
}

const roundToOneDecimal = (value: number): number => Math.round(value * 10) / 10
