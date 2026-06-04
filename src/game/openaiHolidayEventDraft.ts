import {
  createHolidayDialogueEventValidationErrors,
  HOLIDAY_DIALOGUE_EVENT_TYPE,
  TALK_EVENT_TRIGGER_TYPE,
  type HolidayDialogueEventSpec
} from './eventGeneration'

export const OPENAI_HOLIDAY_EVENT_DRAFT_MODEL = 'gpt-5.4-mini'
export const OPENAI_HOLIDAY_EVENT_DRAFT_ENDPOINT = '/api/openai/v1/responses'

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

const OPENAI_HOLIDAY_EVENT_DRAFT_SYSTEM_PROMPT = [
  '너는 게임 이벤트 기획 도우미다.',
  '사용자의 자연어 설명을 바탕으로 반드시 하나의 이벤트 JSON만 생성한다.',
  'JSON은 아래 스키마에 맞아야 한다.',
  'event_name은 snake_case로 작성한다.',
  'event_type은 holiday_dialogue로 고정한다.',
  'dialogue.opening_lines는 1개 이상, 짧고 자연스러운 문장으로 작성한다.',
  '할로윈이나 크리스마스처럼 현재 데모에서 바로 보이도록 할 때는 기존 town 씬의 NPC 설정을 생각해서 작성한다.',
  '응답에는 설명 문장 없이 JSON만 포함한다.'
].join(' ')

const OPENAI_HOLIDAY_EVENT_DRAFT_JSON_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    event_name: { type: 'string' },
    event_type: { type: 'string', const: HOLIDAY_DIALOGUE_EVENT_TYPE },
    npc: {
      type: 'object',
      additionalProperties: false,
      properties: {
        id: { type: 'string' },
        display_name: { type: 'string' },
        appearance_type: { type: 'string' }
      },
      required: ['id', 'display_name', 'appearance_type']
    },
    trigger: {
      type: 'object',
      additionalProperties: false,
      properties: {
        type: {
          type: 'string',
          enum: [TALK_EVENT_TRIGGER_TYPE, 'scene_enter', 'quest_start', 'quest_complete']
        },
        target_scene: { type: 'string' }
      },
      required: ['type', 'target_scene']
    },
    dialogue: {
      type: 'object',
      additionalProperties: false,
      properties: {
        opening_lines: {
          type: 'array',
          items: { type: 'string' }
        },
        active_lines: {
          type: 'array',
          items: { type: 'string' }
        },
        completion_lines: {
          type: 'array',
          items: { type: 'string' }
        }
      },
      required: ['opening_lines', 'active_lines', 'completion_lines']
    },
    reward: {
      anyOf: [
        {
          type: 'object',
          additionalProperties: false,
          properties: {
            type: { type: 'string', const: 'none' }
          },
          required: ['type']
        },
        {
          type: 'object',
          additionalProperties: false,
          properties: {
            type: { type: 'string', const: 'item' },
            id: { type: 'string' },
            count: { type: 'integer', minimum: 1 }
          },
          required: ['type', 'id', 'count']
        },
        {
          type: 'object',
          additionalProperties: false,
          properties: {
            type: { type: 'string', const: 'gold' },
            count: { type: 'integer', minimum: 1 }
          },
          required: ['type', 'count']
        },
        {
          type: 'object',
          additionalProperties: false,
          properties: {
            type: { type: 'string', const: 'experience' },
            count: { type: 'integer', minimum: 1 }
          },
          required: ['type', 'count']
        }
      ]
    },
    duration: { type: 'number', minimum: 0.1 }
  },
  required: ['event_name', 'event_type', 'npc', 'trigger', 'dialogue', 'reward', 'duration']
} as const

export const generateHolidayDialogueEventDraftWithOpenAi = async ({
  apiKey,
  userPrompt
}: {
  apiKey: string
  userPrompt: string
}): Promise<HolidayDialogueEventSpec> => {
  const response = await fetch(OPENAI_HOLIDAY_EVENT_DRAFT_ENDPOINT, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
      body: JSON.stringify({
        model: OPENAI_HOLIDAY_EVENT_DRAFT_MODEL,
        instructions: OPENAI_HOLIDAY_EVENT_DRAFT_SYSTEM_PROMPT,
        input: userPrompt,
        reasoning: {
          effort: 'low'
        },
        text: {
          format: {
            type: 'json_schema',
            name: 'holiday_dialogue_event',
          strict: true,
          schema: OPENAI_HOLIDAY_EVENT_DRAFT_JSON_SCHEMA
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

  let parsedDraft: HolidayDialogueEventSpec

  try {
    parsedDraft = JSON.parse(responseText) as HolidayDialogueEventSpec
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`OpenAI response JSON parse failed: ${message}`)
  }

  const validationErrors = createHolidayDialogueEventValidationErrors(parsedDraft)

  if (validationErrors.length > 0) {
    throw new Error(`OpenAI response failed validation: ${validationErrors.join(', ')}`)
  }

  return parsedDraft
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
