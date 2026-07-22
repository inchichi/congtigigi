export const QWEN_MODEL = 'Qwen3.6-27B'
export const QWEN_LOCAL_TOKEN = 'qwen-local'
export const QWEN_ENDPOINT = '/api/qwen/v1/chat/completions'

type QwenToolCall = {
  function?: { name?: string; arguments?: string }
}

type QwenMessage = {
  content?: string | null
  tool_calls?: QwenToolCall[]
}

type QwenResponse = {
  choices?: Array<{
    finish_reason?: string
    message?: QwenMessage
  }>
  error?: { message?: string }
}

const isLocalToken = (apiKey: string): boolean => apiKey.trim() === QWEN_LOCAL_TOKEN

const parseJsonContent = (content: string): unknown => {
  const withoutFence = content.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim()
  try {
    return JSON.parse(withoutFence)
  } catch {
    const start = withoutFence.indexOf('{')
    const end = withoutFence.lastIndexOf('}')
    if (start < 0 || end <= start) {
      throw new Error('Qwen 응답에서 JSON을 찾을 수 없습니다.')
    }
    try {
      return JSON.parse(withoutFence.slice(start, end + 1))
    } catch {
      throw new Error('Qwen JSON 응답 파싱에 실패했습니다.')
    }
  }
}

export const generateJsonWithQwen = async <T>({
  apiKey,
  model = QWEN_MODEL,
  instructions,
  input,
  schemaName,
  schema
}: {
  apiKey: string
  model?: string
  instructions: string
  input: string
  schemaName: string
  schema: object
}): Promise<T> => {
  const headers: Record<string, string> = { 'content-type': 'application/json' }
  if (!isLocalToken(apiKey)) {
    headers.authorization = `Bearer ${apiKey}`
  }

  const response = await fetch(QWEN_ENDPOINT, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: instructions },
        { role: 'user', content: input }
      ],
      // Qwen 서버는 현재 8192 토큰 컨텍스트로 실행되며, 게임 카탈로그와
      // 함수 스키마가 함께 들어가므로 출력은 1536 토큰 안에서 완료시킨다.
      max_tokens: 1536,
      temperature: 0.2,
      top_p: 0.8,
      presence_penalty: 1.5,
      chat_template_kwargs: { enable_thinking: false },
      tools: [
        {
          type: 'function',
          function: {
            name: schemaName,
            description: '구조화된 결과를 반드시 이 함수로 반환한다.',
            parameters: schema
          }
        }
      ],
      tool_choice: { type: 'function', function: { name: schemaName } },
      response_format: { type: 'json_object' }
    })
  })

  const rawText = await response.text()
  let payload: QwenResponse = {}
  try {
    payload = JSON.parse(rawText) as QwenResponse
  } catch {
    payload = {}
  }

  if (!response.ok) {
    throw new Error(payload.error?.message || rawText || `Qwen API request failed with status ${response.status}`)
  }

  const choice = payload.choices?.[0]
  if (choice?.finish_reason === 'length') {
    throw new Error('Qwen 응답이 길이 한도로 잘렸습니다. 더 짧게 요청하세요.')
  }

  const message = choice?.message
  const toolArguments = message?.tool_calls?.[0]?.function?.arguments
  const result = typeof toolArguments === 'string'
    ? parseJsonContent(toolArguments)
    : typeof message?.content === 'string'
      ? parseJsonContent(message.content)
      : undefined

  if (result === null || typeof result !== 'object' || result === undefined) {
    throw new Error('Qwen 응답에 구조화된 JSON 결과가 없습니다.')
  }

  const required = (schema as { required?: unknown }).required
  if (Array.isArray(required)) {
    const present = result as Record<string, unknown>
    const missing = required.filter((key) => typeof key === 'string' && !(key in present))
    if (missing.length > 0) {
      throw new Error(`Qwen 응답에 필수 필드가 없습니다: ${missing.join(', ')}`)
    }
  }

  return result as T
}
