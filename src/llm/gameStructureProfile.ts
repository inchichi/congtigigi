export const GAME_UNDERSTANDING_SCORE_THRESHOLD = 75

export type GameStructureMapInfo = {
  id: string
  name: string
  file: string
}

export type GameStructureNpcInfo = {
  id: string
  name: string
  map: string
  file: string
}

export type GameStructureItemInfo = {
  id: string
  name: string
  file: string
}

export type GameStructureEventInfo = {
  id: string
  file: string
}

export type GameStructureDialogueSystemInfo = {
  file: string
  format: string
}

export type GameStructureEventSystemInfo = {
  file: string
  register_function: string
}

export type GameStructureProfile = {
  game_title: string
  engine: string
  maps: GameStructureMapInfo[]
  npcs: GameStructureNpcInfo[]
  items: GameStructureItemInfo[]
  events: GameStructureEventInfo[]
  dialogue_system: GameStructureDialogueSystemInfo
  event_system: GameStructureEventSystemInfo
  modifiable_files: string[]
}

export type GameUnderstandingScoreDetails = {
  map_coverage: number
  npc_coverage: number
  item_coverage: number
  event_system_accuracy: number
  file_localization_accuracy: number
  relationship_accuracy: number
}

export type GameUnderstandingScoreReport = {
  gus_score: number
  threshold: number
  status: 'passed' | 'failed'
  details: GameUnderstandingScoreDetails
  missing_items: string[]
}

export type GameStructureAnalysisResult = {
  profile: GameStructureProfile
  gus: GameUnderstandingScoreReport
  analysis_notes: string[]
}
