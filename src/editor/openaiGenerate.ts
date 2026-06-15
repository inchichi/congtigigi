// 게임 무관 범용 OpenAI(Chat Completions) 구조화 생성 호출.
// 구조화 출력은 function tool 정의(parameters=JSON Schema)를 주고 tool_choice로 그 함수를
// 강제 호출하게 해서 얻는다. 응답 tool_calls[0].function.arguments(JSON 문자열)가 결과다.
// (anthropicGenerate.ts의 generateJsonWithClaude와 동일한 계약 — provider만 다름)
// 기본 OpenAI 모델. 설정에서 사용자가 바꿀 수 있다(예: gpt-4o). 키는 모델을 지정하지 않으므로
// 어떤 모델을 호출할지는 이 값/설정으로 정한다.
export const OPENAI_MODEL = 'gpt-5.4-mini'
export const OPENAI_ENDPOINT = '/api/openai/v1/chat/completions'

type OpenAiToolCall = {
  function?: { name?: string; arguments?: string }
}

type OpenAiChoice = {
  message?: { tool_calls?: OpenAiToolCall[] }
  finish_reason?: string
}

type OpenAiResponse = {
  choices?: OpenAiChoice[]
}

type OpenAiErrorResponse = {
  error?: { message?: string; type?: string }
}

export const generateJsonWithOpenAi = async <T>({
  apiKey,
  model = OPENAI_MODEL,
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
  const response = await fetch(OPENAI_ENDPOINT, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${apiKey}`,
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: instructions },
        { role: 'user', content: input }
      ],
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
      tool_choice: { type: 'function', function: { name: schemaName } }
    })
  })

  // ok 확인 전에 JSON 파싱하면 비-JSON 에러 바디에서 SyntaxError가 나 실제 에러를 삼킨다.
  const rawText = await response.text()
  let payload: OpenAiResponse & OpenAiErrorResponse = {}
  try {
    payload = JSON.parse(rawText) as OpenAiResponse & OpenAiErrorResponse
  } catch {
    payload = {}
  }

  if (!response.ok) {
    throw new Error(
      payload.error?.message ||
        rawText ||
        `OpenAI API request failed with status ${response.status}`
    )
  }

  const choice = payload.choices?.[0]

  if (choice?.finish_reason === 'length') {
    throw new Error(
      'GPT 응답이 길이 한도로 잘렸습니다. 더 짧게 요청하세요.'
    )
  }

  const toolArguments = choice?.message?.tool_calls?.[0]?.function?.arguments

  if (typeof toolArguments !== 'string') {
    throw new Error('GPT 응답에 구조화된 함수 호출 결과가 없습니다.')
  }

  let result: unknown
  try {
    result = JSON.parse(toolArguments)
  } catch {
    throw new Error('GPT 함수 인자 JSON 파싱에 실패했습니다.')
  }

  // 모델 출력이라 스키마를 어길 수 있다. 최소한 객체 + top-level required 필드 검사로
  // 읽을 수 있는 에러로 바꾼다(generateJsonWithClaude와 동일한 방어).
  if (result === null || typeof result !== 'object') {
    throw new Error('GPT 응답이 객체 형식이 아닙니다.')
  }

  const required = (schema as { required?: unknown }).required
  if (Array.isArray(required)) {
    const present = result as Record<string, unknown>
    const missing = required.filter(
      (key) => typeof key === 'string' && !(key in present)
    )
    if (missing.length > 0) {
      throw new Error(`GPT 응답에 필수 필드가 없습니다: ${missing.join(', ')}`)
    }
  }

  return result as T
}
