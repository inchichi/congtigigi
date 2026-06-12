import type { GameStructureProfile } from './gameStructureProfile'
import type { GeneratedEventJson } from './eventJsonSchema'

// 실제 OpenAI 호출에 쓰는 모델/엔드포인트. 모델 id가 실존하지 않으면 라이브 호출이 전부 실패하므로
// 사용 가능한 OpenAI 모델 id로 유지한다. 엔드포인트는 vite.config.ts의 /api/openai 프록시를 거쳐
// 브라우저 CORS를 피한다.
export const OPENAI_EVENT_JSON_MODEL = 'gpt-4o-mini'
export const OPENAI_EVENT_JSON_ENDPOINT = '/api/openai/v1/responses'

type OpenAiResponsesApiError = {
  error?: {
    message?: string
    type?: string
    code?: string
  }
}

type OpenAiResponsesApiOutputContent = {
  type?: string
  text?: string
}

type OpenAiResponsesApiOutputItem = {
  type?: string
  text?: string
  content?: OpenAiResponsesApiOutputContent[]
}

type OpenAiResponsesApiPayload = {
  output_text?: string
  output?: OpenAiResponsesApiOutputItem[]
}

const OPENAI_EVENT_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    event_id: { type: 'string' },
    event_name: { type: 'string' },
    description: { type: 'string' },
    trigger: {
      type: 'object',
      additionalProperties: false,
      properties: {
        type: { type: 'string' },
        target: { type: 'string' },
        condition: { type: 'string' }
      },
      required: ['type', 'target', 'condition']
    },
    location: {
      type: 'object',
      additionalProperties: false,
      properties: {
        map_id: { type: 'string' },
        x: { type: 'number' },
        y: { type: 'number' }
      },
      required: ['map_id', 'x', 'y']
    },
    npc: {
      type: 'object',
      additionalProperties: false,
      properties: {
        id: { type: 'string' },
        name: { type: 'string' },
        dialogue_id: { type: 'string' }
      },
      required: ['id', 'name', 'dialogue_id']
    },
    dialogue: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          speaker: { type: 'string' },
          text: { type: 'string' }
        },
        required: ['speaker', 'text']
      }
    },
    reward: {
      type: 'object',
      additionalProperties: false,
      properties: {
        item_id: { type: 'string' },
        amount: { type: 'integer' }
      },
      required: ['item_id', 'amount']
    },
    duration: {
      type: 'object',
      additionalProperties: false,
      properties: {
        start: { type: 'string' },
        end: { type: 'string' }
      },
      required: ['start', 'end']
    },
    requires_new_asset: { type: 'boolean' },
    requires_new_asset_reasons: {
      type: 'array',
      items: { type: 'string' }
    }
  },
  required: [
    'event_id',
    'event_name',
    'description',
    'trigger',
    'location',
    'npc',
    'dialogue',
    'reward',
    'duration',
    'requires_new_asset',
    'requires_new_asset_reasons'
  ]
} as const

const createSystemPrompt = (profile: GameStructureProfile): string => {
  const mapList = profile.maps.map((map) => `${map.id} (${map.name})`).join(', ')
  const npcList = profile.npcs
    .map((npc) => `${npc.id} (${npc.name}, map=${npc.map})`)
    .join(', ')
  const itemList = profile.items.map((item) => `${item.id} (${item.name})`).join(', ')

  return [
    '너는 2D RPG 게임의 이벤트 기획 도우미다.',
    '사용자의 자연어 설명을 바탕으로 반드시 하나의 이벤트 JSON만 생성한다.',
    'JSON은 제공된 스키마를 정확히 따른다.',
    'event_id와 event_name은 snake_case로 작성한다.',
    `location.map_id는 반드시 다음 맵 id 중 하나여야 한다: ${mapList}.`,
    `npc.id는 반드시 다음 NPC id 중 하나여야 한다: ${npcList}.`,
    'location.map_id는 선택한 npc.id가 속한 맵(위 목록의 map=...)과 반드시 같아야 한다.',
    '기존 NPC의 대사를 교체하는 이벤트이므로, 화면에서 바로 보이도록 town 맵의 기존 NPC를 우선 사용한다.',
    `reward.item_id는 반드시 다음 아이템 id 중 하나여야 한다: ${itemList}.`,
    'dialogue는 1개 이상이며, speaker는 npc.name과 동일하게, text는 짧고 자연스러운 한국어 문장으로 작성한다.',
    'npc.dialogue_id는 "<npc.id>_dialogue" 형식으로 작성한다.',
    'reward.amount는 1 이상의 정수다.',
    'duration.start와 duration.end는 ISO 8601 문자열로 작성한다.',
    '기존 게임 자산만으로 구성하므로 requires_new_asset는 false, requires_new_asset_reasons는 빈 배열로 둔다.',
    '응답에는 설명 문장 없이 JSON만 포함한다.'
  ].join(' ')
}

export const generateEventJsonDraftWithOpenAi = async ({
  apiKey,
  userPrompt,
  profile
}: {
  apiKey: string
  userPrompt: string
  profile: GameStructureProfile
}): Promise<GeneratedEventJson> => {
  const response = await fetch(OPENAI_EVENT_JSON_ENDPOINT, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: OPENAI_EVENT_JSON_MODEL,
      instructions: createSystemPrompt(profile),
      input: userPrompt,
      text: {
        format: {
          type: 'json_schema',
          name: 'generated_event_json',
          strict: true,
          schema: OPENAI_EVENT_JSON_SCHEMA
        }
      }
    })
  })

  const payload = (await response.json()) as OpenAiResponsesApiPayload &
    OpenAiResponsesApiError

  if (!response.ok) {
    throw new Error(
      payload.error?.message ??
        `OpenAI API request failed with status ${response.status}`
    )
  }

  const responseText = extractOpenAiResponseText(payload)

  if (!responseText) {
    throw new Error('OpenAI response did not include structured output text')
  }

  try {
    return JSON.parse(responseText) as GeneratedEventJson
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`OpenAI response JSON parse failed: ${message}`)
  }
}

const extractOpenAiResponseText = (
  payload: OpenAiResponsesApiPayload
): string | undefined => {
  if (typeof payload.output_text === 'string' && payload.output_text.trim().length > 0) {
    return payload.output_text
  }

  if (!Array.isArray(payload.output)) {
    return undefined
  }

  for (const outputItem of payload.output) {
    if (typeof outputItem.text === 'string' && outputItem.text.trim().length > 0) {
      return outputItem.text
    }

    if (!Array.isArray(outputItem.content)) {
      continue
    }

    for (const contentItem of outputItem.content) {
      if (typeof contentItem.text === 'string' && contentItem.text.trim().length > 0) {
        return contentItem.text
      }
    }
  }

  return undefined
}
