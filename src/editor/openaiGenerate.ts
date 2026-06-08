// 게임 무관 범용 OpenAI 구조화 생성 호출. 어댑터들이 각자 schema/프롬프트만 다르게 넣어 쓴다.
export const OPENAI_GENERATE_MODEL = 'gpt-4o-mini'
export const OPENAI_GENERATE_ENDPOINT = '/api/openai/v1/responses'

type OpenAiResponsesApiError = {
  error?: { message?: string }
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

export const generateJsonWithOpenAi = async <T>({
  apiKey,
  instructions,
  input,
  schemaName,
  schema
}: {
  apiKey: string
  instructions: string
  input: string
  schemaName: string
  schema: object
}): Promise<T> => {
  const response = await fetch(OPENAI_GENERATE_ENDPOINT, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: OPENAI_GENERATE_MODEL,
      instructions,
      input,
      text: {
        format: {
          type: 'json_schema',
          name: schemaName,
          strict: true,
          schema
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

  return JSON.parse(responseText) as T
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
