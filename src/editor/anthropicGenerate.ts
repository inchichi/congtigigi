// 게임 무관 범용 Claude(Anthropic Messages API) 구조화 생성 호출.
// 구조화 출력은 tool 정의(input_schema)를 주고 tool_choice로 그 도구를 강제 호출하게 해서 얻는다.
// 응답의 tool_use 블록 input이 스키마를 따르는 결과다.
export const ANTHROPIC_MODEL = 'claude-opus-4-8'
export const ANTHROPIC_ENDPOINT = '/api/anthropic/v1/messages'
export const ANTHROPIC_VERSION = '2023-06-01'

type AnthropicContentBlock = {
  type: string
  input?: unknown
  text?: string
}

type AnthropicMessageResponse = {
  content?: AnthropicContentBlock[]
  stop_reason?: string
}

type AnthropicErrorResponse = {
  error?: { message?: string; type?: string }
}

export const generateJsonWithClaude = async <T>({
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
  const response = await fetch(ANTHROPIC_ENDPOINT, {
    method: 'POST',
    headers: {
      'x-api-key': apiKey,
      'anthropic-version': ANTHROPIC_VERSION,
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      model: ANTHROPIC_MODEL,
      max_tokens: 4096,
      system: instructions,
      messages: [{ role: 'user', content: input }],
      tools: [
        {
          name: schemaName,
          description: '구조화된 결과를 반드시 이 도구로 반환한다.',
          input_schema: schema
        }
      ],
      tool_choice: { type: 'tool', name: schemaName }
    })
  })

  const payload = (await response.json()) as AnthropicMessageResponse &
    AnthropicErrorResponse

  if (!response.ok) {
    throw new Error(
      payload.error?.message ??
        `Anthropic API request failed with status ${response.status}`
    )
  }

  const toolUse = payload.content?.find((block) => block.type === 'tool_use')

  if (!toolUse || toolUse.input === undefined) {
    throw new Error('Claude 응답에 구조화된 tool_use 결과가 없습니다.')
  }

  return toolUse.input as T
}
