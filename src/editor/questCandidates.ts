import { generateJson } from './llmProvider'
import {
  buildFeedbackInstruction,
  type GameEntity,
  type GenerationFeedback
} from './gameAdapter'
import type { GameStructureProfile } from './gameStructureProfile'

// 퀘스트 생성 1단계(피드백 루프 표면): 전체 이벤트 JSON 대신 짧은 자연어 후보를 N개 만든다.
// 유저가 가볍게 고르고/재생성하고 나서, 고른 후보 하나만 2단계에서 실제 이벤트 JSON으로 만든다.
export type QuestCandidate = {
  // 한 줄 제목(카드 헤더).
  title: string
  // 2~3문장 자연어 요약(무슨 퀘스트인지). JSON 스키마 없음 — 사람이 읽고 고르는 용도.
  summary: string
  // 이 퀘스트가 붙을 NPC id(있으면). 없으면 빈 문자열. profile.npcs의 id 중 하나여야 한다.
  target_hint: string
}

export type GenerateQuestCandidatesInput = {
  apiKey: string
  userPrompt: string
  profile: GameStructureProfile
  entity?: GameEntity
  gameContext?: string
  feedback?: GenerationFeedback
}

export const QUEST_CANDIDATE_COUNT = 3

// profile의 실재 id 목록을 프롬프트에 넣어 후보가 실재 NPC/맵/아이템에 근거하게 한다
// (claudeEventJsonGenerator 시스템 프롬프트와 같은 접지 전략).
const buildGroundingContext = (profile: GameStructureProfile): string => {
  const npcIds =
    profile.npcs.map((npc) => `${npc.id}(map=${npc.map})`).join(', ') || '(없음)'
  const mapIds = profile.maps.map((map) => map.id).join(', ') || '(없음)'
  const itemIds = profile.items.map((item) => item.id).join(', ') || '(없음)'
  return `\n\n사용 가능한 NPC: ${npcIds}\n사용 가능한 맵: ${mapIds}\n사용 가능한 아이템: ${itemIds}`
}

const normalizeCandidate = (candidate: Partial<QuestCandidate>): QuestCandidate => ({
  title: (candidate.title ?? '').trim(),
  summary: (candidate.summary ?? '').trim(),
  target_hint: (candidate.target_hint ?? '').trim()
})

const createQuestCandidatesSchema = (profile: GameStructureProfile): object => {
  const targetHintEnum = ['', ...profile.npcs.map((npc) => npc.id)]

  return {
    type: 'object',
    additionalProperties: false,
    properties: {
      candidates: {
        type: 'array',
        items: {
          type: 'object',
          additionalProperties: false,
          properties: {
            title: { type: 'string' },
            summary: { type: 'string' },
            target_hint: {
              type: 'string',
              enum: targetHintEnum
            }
          },
          required: ['title', 'summary', 'target_hint']
        }
      }
    },
    required: ['candidates']
  }
}

export const generateQuestCandidates = async ({
  apiKey,
  userPrompt,
  profile,
  entity,
  gameContext,
  feedback
}: GenerateQuestCandidatesInput): Promise<QuestCandidate[]> => {
  const targetLine = entity
    ? `\n\n이 퀘스트의 대상은 NPC id="${entity.id}"(${entity.name})로 한다.`
    : ''
  const contextLine = gameContext ? `\n\n게임 정보: ${gameContext}` : ''
  // 재생성(피드백 루프): 단일 흐름과 같은 buildFeedbackInstruction을 재사용한다.
  const feedbackLine = feedback ? buildFeedbackInstruction(feedback) : ''

  const generated = await generateJson<{ candidates: QuestCandidate[] }>({
    apiKey,
    instructions:
      `'${profile.game_title}' 게임에 어울리는 서로 다른 퀘스트 아이디어를 정확히 ${QUEST_CANDIDATE_COUNT}개 제안한다. ` +
      '각 후보는 전체 코드가 아니라 짧은 자연어 요약(title 1줄 + summary 2~3문장)이다. ' +
      'target_hint는 주어진 NPC id 중 하나이거나 빈 문자열이다. target_hint는 퀘스트의 대상 NPC 힌트이지 기버가 아니다. ' +
      'title/summary는 한국어로, 서로 충분히 다른 방향으로 작성한다.',
    input: `시나리오: ${userPrompt}${targetLine}${contextLine}${buildGroundingContext(profile)}${feedbackLine}`,
    schemaName: 'quest_candidates',
    schema: createQuestCandidatesSchema(profile)
  })

  // LLM이 개수를 안 지켜도(2개·0개) throw하지 않는다 — 호출부가 받은 만큼 보여주고 재생성하게 둔다.
  // 완전히 빈 후보(제목·요약 모두 공백)는 거른다.
  return (generated.candidates ?? [])
    .map(normalizeCandidate)
    .filter((candidate) => candidate.title.length > 0 || candidate.summary.length > 0)
}
