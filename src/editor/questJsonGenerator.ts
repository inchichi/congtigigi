import type { GameStructureProfile } from './gameStructureProfile'
import { generateJson } from './llmProvider'
import { buildFeedbackInstruction, type GenerationFeedback } from './gameAdapter'
import { buildQuestCatalogText } from './myRpgQuestCatalog'
import type { GeneratedQuestJson } from './questJsonSchema'
import type { QuestCandidate } from './questCandidates'

// 런타임 questLog가 추적하는 목표/타깃 구조 그대로의 JSON 스키마(구조화 출력 강제용).
const QUEST_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    quest_id: { type: 'string' },
    title: { type: 'string' },
    giver_npc_id: { type: 'string' },
    region: { type: 'string' },
    request_text: { type: 'string' },
    guide_text: { type: 'string' },
    start_dialogue_lines: { type: 'array', items: { type: 'string' } },
    active_dialogue_lines: { type: 'array', items: { type: 'string' } },
    completion_dialogue_lines: { type: 'array', items: { type: 'string' } },
    objectives: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          type: {
            type: 'string',
            enum: ['monster-defeat', 'item-use', 'shop-open', 'scene-enter', 'talk']
          },
          label: { type: 'string' },
          required: { type: 'integer' },
          target: {
            type: 'object',
            additionalProperties: false,
            properties: {
              sceneId: { type: 'string' },
              appearanceType: { type: 'string' },
              itemId: { type: 'string' },
              shopId: { type: 'string' },
              npcId: { type: 'string' }
            }
          }
        },
        required: ['type', 'label', 'required', 'target']
      }
    },
    rewards: {
      type: 'object',
      additionalProperties: false,
      properties: {
        gold: { type: 'integer' },
        experience: { type: 'integer' },
        items: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              item_id: { type: 'string' },
              quantity: { type: 'integer' }
            },
            required: ['item_id', 'quantity']
          }
        }
      },
      required: ['gold', 'experience', 'items']
    }
  },
  required: [
    'quest_id',
    'title',
    'giver_npc_id',
    'region',
    'request_text',
    'guide_text',
    'start_dialogue_lines',
    'active_dialogue_lines',
    'completion_dialogue_lines',
    'objectives',
    'rewards'
  ]
}

const createQuestSystemPrompt = (profile: GameStructureProfile): string => {
  const npcList = profile.npcs
    .map((npc) => `${npc.id} (${npc.name}, map=${npc.map})`)
    .join(', ')
  const itemList = profile.items.map((item) => `${item.id} (${item.name})`).join(', ')

  return [
    '너는 2D RPG 게임의 "단계별 퀘스트" 기획 도우미다.',
    '플레이어가 사냥/아이템 사용/상점 방문/이동/대화 같은 목표를 단계별로 수행해 깨는 진짜 퀘스트를 1개 만든다.',
    '제공된 도구 스키마를 정확히 따른다.',
    'quest_id는 반드시 영문 소문자·숫자·밑줄(snake_case)로 짓는다(예: hunt_forest_slimes). 시나리오가 한국어여도 quest_id는 영문이다.',
    'title·request_text·guide_text·각종 dialogue_lines·objective.label은 자연스러운 한국어로 작성한다.',
    'objectives는 1~3개. 각 목표의 type/target은 아래 "사용 가능한 값"에서만 고른다 — 그래야 게임에서 실제로 진행된다.',
    'monster-defeat → target.appearanceType + target.sceneId. item-use → target.itemId. shop-open → target.shopId. scene-enter → target.sceneId. talk → target.npcId.',
    'required는 1 이상의 정수(사냥은 2~5 권장).',
    'giver_npc_id는 아래 기버 NPC 중 하나. start_dialogue_lines(수락)·active_dialogue_lines(진행 중)·completion_dialogue_lines(완료)를 각각 1줄 이상 쓴다.',
    'rewards.items의 item_id와 item-use 목표의 itemId는 아래 "사용 가능한 아이템" id 중에서만 고른다. rewards.gold/experience는 0 이상 정수.',
    '\n[사용 가능한 값]',
    buildQuestCatalogText(),
    `사용 가능한 아이템(item_id): ${itemList}`,
    `게임 내 NPC(talk 목표 npcId용): ${npcList}`
  ].join('\n')
}

export const generateQuestJson = ({
  apiKey,
  userPrompt,
  profile,
  candidate,
  feedback
}: {
  apiKey: string
  userPrompt: string
  profile: GameStructureProfile
  candidate?: QuestCandidate
  feedback?: GenerationFeedback
}): Promise<GeneratedQuestJson> => {
  const candidateHint = candidate
    ? `\n\n선택된 퀘스트 후보를 그대로 구현한다:\n- 제목: ${candidate.title}\n- 내용: ${candidate.summary}` +
      (candidate.target_hint ? `\n- 대상 NPC: ${candidate.target_hint}` : '')
    : ''
  const feedbackHint = feedback ? buildFeedbackInstruction(feedback) : ''

  return generateJson<GeneratedQuestJson>({
    apiKey,
    instructions: createQuestSystemPrompt(profile),
    input: `${userPrompt}${candidateHint}${feedbackHint}`,
    schemaName: 'generated_quest_json',
    schema: QUEST_JSON_SCHEMA
  })
}
