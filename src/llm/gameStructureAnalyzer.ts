import { CURRENT_GAME_PROJECT_PROFILE } from './currentGameProjectSnapshot'
import { calculateGameUnderstandingScore } from './gusCalculator'
import type {
  GameStructureAnalysisResult,
  GameStructureProfile
} from './gameStructureProfile'

export type GameStructureAnalysisProgress = {
  step: string
  progress: number
  detail: string
}

export type AnalyzeGameStructureOptions = {
  onProgress?: (progress: GameStructureAnalysisProgress) => void
}

export const analyzeCurrentGameStructure = async ({
  onProgress
}: AnalyzeGameStructureOptions = {}): Promise<GameStructureAnalysisResult> => {
  const steps: Array<Omit<GameStructureAnalysisProgress, 'progress'>> = [
    {
      step: 'maps',
      detail: '맵 파일과 Tiled 이벤트 레이어를 분석하는 중이다.'
    },
    {
      step: 'npcs',
      detail: 'NPC 배치, 대화, Lua 컨트롤러 연결을 분석하는 중이다.'
    },
    {
      step: 'items',
      detail: '포션과 장비 아이템 정의를 분석하는 중이다.'
    },
    {
      step: 'events',
      detail: '퀘스트와 이벤트 등록 경로를 분석하는 중이다.'
    },
    {
      step: 'relationships',
      detail: 'NPC-맵, 아이템-보상, 이벤트-트리거 관계를 평가하는 중이다.'
    }
  ]

  for (const [index, step] of steps.entries()) {
    onProgress?.({
      ...step,
      progress: Math.round(((index + 1) / steps.length) * 100)
    })

    await wait(160)
  }

  const profile = cloneProfile(CURRENT_GAME_PROJECT_PROFILE)
  const gus = calculateGameUnderstandingScore(profile)

  return {
    profile,
    gus,
    analysis_notes: [
      '현재 프로젝트는 Tiled 기반 맵과 Lua 대화 컨트롤러가 이미 연결되어 있다.',
      '전용 EventManager는 아직 없어서 runtime registry를 우선 경로로 둔다.',
      '이 단계에서는 실제 파일 자동 스캔 대신 코드베이스에 고정된 구조 정보를 모의 분석한다.'
    ]
  }
}

const cloneProfile = (profile: GameStructureProfile): GameStructureProfile =>
  structuredClone(profile)

const wait = async (milliseconds: number): Promise<void> =>
  new Promise((resolve) => {
    window.setTimeout(resolve, milliseconds)
  })
