import { extractTmxObjects } from './tmxObjects'
import { generateJsonWithClaude } from './anthropicGenerate'
import type { GameFile } from './loadGame'

// LLM이 임의의 게임 폴더를 이해하게 하는 분석기. 손으로 짠 어댑터 대신, 결정적으로 뽑은
// 맵 객체 구조를 "증거"로 LLM에 주고, 어느 그룹이 편집 엔티티인지 등을 분석하게 한다.
// (결정적 파싱으로 ground-truth를 주므로 LLM이 없는 엔티티를 지어내지 않는다.)

export type AnalyzedEntityGroup = {
  group: string
  kind: string
  editable: boolean
  sample_names: string[]
}

export type GameAnalysis = {
  game_name: string
  engine: string
  entity_groups: AnalyzedEntityGroup[]
  content_model: string
  apply_strategy: string
}

const MAX_MAPS = 8
const MAX_SAMPLE_NAMES = 6

// LLM에 넘길 증거 텍스트. 전체 파일 대신 맵별 object group/타입/이름 샘플만 요약한다.
export const buildAnalysisEvidence = (files: GameFile[]): string => {
  const lines: string[] = []
  lines.push(`파일 목록: ${files.map((file) => file.name).join(', ')}`)

  const tmxFiles = files.filter((file) => file.name.endsWith('.tmx')).slice(0, MAX_MAPS)

  for (const file of tmxFiles) {
    // 한 맵이 깨졌다고 전체 분석을 죽이지 않는다(loadGame과 동일한 맵 단위 격리). 건너뛰되 표시한다.
    let objects
    try {
      objects = extractTmxObjects(file.text)
    } catch {
      lines.push(`맵 ${file.name}: (파싱 실패 — 건너뜀)`)
      continue
    }
    const byGroup = new Map<string, { types: Set<string>; names: string[] }>()

    for (const object of objects) {
      const entry = byGroup.get(object.group) ?? { types: new Set(), names: [] }
      if (object.type) {
        entry.types.add(object.type)
      }
      if (object.name && entry.names.length < MAX_SAMPLE_NAMES) {
        entry.names.push(object.name)
      }
      byGroup.set(object.group, entry)
    }

    lines.push(`맵 ${file.name}:`)
    for (const [group, entry] of byGroup) {
      lines.push(
        `  - object group "${group}": types=[${[...entry.types].join(', ')}], names=[${entry.names.join(', ')}]`
      )
    }
  }

  return lines.join('\n')
}

const GAME_ANALYSIS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    game_name: { type: 'string' },
    engine: { type: 'string' },
    entity_groups: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          group: { type: 'string' },
          kind: { type: 'string' },
          editable: { type: 'boolean' },
          sample_names: { type: 'array', items: { type: 'string' } }
        },
        required: ['group', 'kind', 'editable', 'sample_names']
      }
    },
    content_model: { type: 'string' },
    apply_strategy: { type: 'string' }
  },
  required: ['game_name', 'engine', 'entity_groups', 'content_model', 'apply_strategy']
}

export const analyzeGame = async ({
  apiKey,
  files
}: {
  apiKey: string
  files: GameFile[]
}): Promise<GameAnalysis> =>
  generateJsonWithClaude<GameAnalysis>({
    apiKey,
    instructions: [
      '너는 게임 프로젝트 분석기다.',
      '주어진 맵 객체 구조 증거를 보고 다음을 분석한다:',
      '이 게임이 무엇인지(game_name), 엔진(engine),',
      '어느 object group이 편집 대상 엔티티인지(entity_groups: group/kind/editable/sample_names),',
      '대사·콘텐츠가 어떻게 표현되는지(content_model),',
      '변경을 어떻게 적용할지(apply_strategy).',
      '증거에 없는 그룹·이름은 지어내지 않는다. 응답에는 JSON만 포함한다.'
    ].join(' '),
    input: buildAnalysisEvidence(files),
    schemaName: 'game_analysis',
    schema: GAME_ANALYSIS_SCHEMA
  })
