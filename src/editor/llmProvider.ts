import { ANTHROPIC_MODEL, generateJsonWithClaude } from './anthropicGenerate'
import { OPENAI_MODEL, generateJsonWithOpenAi } from './openaiGenerate'
import { generateJsonWithQwen, QWEN_MODEL } from './qwenGenerate'

// 키 하나만 붙이면 Claude/GPT를 자동 판별해 라우팅한다. (사용자: "gpt도 claude도 될 수 있어야")
export type LlmProvider = 'anthropic' | 'openai' | 'qwen'

export const PROVIDER_LABEL: Record<LlmProvider, string> = {
  anthropic: 'Claude',
  openai: 'GPT',
  qwen: 'Qwen (local)'
}

// provider별 선택 가능한 모델 목록(드롭다운용). OpenAI는 2026-06 기준 현행 모델
// (gpt-4o/4.1 계열은 2026-04 은퇴). 기본 선택값은 getProviderModel(=각 provider 기본 모델)이 정한다.
export const PROVIDER_MODELS: Record<LlmProvider, string[]> = {
  anthropic: ['claude-opus-4-8', 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001'],
  openai: [
    'gpt-5.5',
    'gpt-5.5-pro',
    'gpt-5.4',
    'gpt-5.4-pro',
    'gpt-5.4-mini',
    'gpt-5.4-nano'
  ],
  qwen: [QWEN_MODEL]
}

const DEFAULT_MODEL: Record<LlmProvider, string> = {
  anthropic: ANTHROPIC_MODEL,
  openai: OPENAI_MODEL,
  qwen: QWEN_MODEL
}

// 사용자가 설정에서 모델을 바꾸면 여기에 저장된다(provider별). generateJson이 이 값을 읽어 호출하므로
// analyzeGame/gameAdapter 등 호출부를 건드리지 않고도 모델이 반영된다.
const modelOverride: Partial<Record<LlmProvider, string>> = {}

export const setProviderModel = (provider: LlmProvider, model: string): void => {
  const trimmed = model.trim()
  if (trimmed) {
    modelOverride[provider] = trimmed
  } else {
    delete modelOverride[provider]
  }
}

export const getProviderModel = (provider: LlmProvider): string =>
  modelOverride[provider] ?? DEFAULT_MODEL[provider]

// 키 접두로 provider를 판별한다. Anthropic은 sk-ant-, OpenAI는 sk-(sk-proj- 포함).
export const detectProvider = (apiKey: string): LlmProvider | undefined => {
  const key = apiKey.trim()

  if (/^(qwen|local-qwen)(-|:|$)/iu.test(key) || key === 'local') {
    return 'qwen'
  }

  if (key.startsWith('sk-ant-')) {
    return 'anthropic'
  }

  if (key.startsWith('sk-')) {
    return 'openai'
  }

  return undefined
}

type GenerateJsonInput = {
  apiKey: string
  instructions: string
  input: string
  schemaName: string
  schema: object
}

// 구조화 JSON 생성의 단일 진입점. 키로 provider를 판별해 해당 구현으로 라우팅한다.
export const generateJson = async <T>(args: GenerateJsonInput): Promise<T> => {
  const provider = detectProvider(args.apiKey)

  if (provider === 'openai') {
    return generateJsonWithOpenAi<T>({ ...args, model: getProviderModel('openai') })
  }

  if (provider === 'qwen') {
    return generateJsonWithQwen<T>({ ...args, model: getProviderModel('qwen') })
  }

  // 기본값 anthropic (기존 동작 유지). 알 수 없는 형식이면 Claude 경로가 인증 에러로 알려준다.
  return generateJsonWithClaude<T>({ ...args, model: getProviderModel('anthropic') })
}

export type ApiKeyCheckStatus = 'empty' | 'unknown' | 'checking' | 'valid' | 'invalid'

export type ApiKeyCheck = {
  status: ApiKeyCheckStatus
  provider?: LlmProvider
  message: string
}

// provider별 /v1/models 엔드포인트로 가벼운(토큰 비용 없는) liveness 체크를 한다.
export const validateApiKey = async (apiKey: string): Promise<ApiKeyCheck> => {
  const key = apiKey.trim()

  if (key.length === 0) {
    return { status: 'empty', message: '키를 입력하세요.' }
  }

  const provider = detectProvider(key)

  if (!provider) {
    return {
      status: 'unknown',
      message: '알 수 없는 키 형식입니다 (Claude=sk-ant-…, GPT=sk-…, Qwen=qwen-local).'
    }
  }

  try {
    const response =
      provider === 'anthropic'
        ? await fetch('/api/anthropic/v1/models', {
            headers: {
              'x-api-key': key,
              'anthropic-version': '2023-06-01'
            }
          })
        : provider === 'openai'
          ? await fetch('/api/openai/v1/models', {
              headers: { authorization: `Bearer ${key}` }
            })
          : await fetch('/api/qwen/v1/models', {
              headers: key === 'qwen-local' ? {} : { authorization: `Bearer ${key}` }
            })

    if (response.ok) {
      return {
        status: 'valid',
        provider,
        message: `유효한 ${PROVIDER_LABEL[provider]} 키 (${getProviderModel(provider)})`
      }
    }

    if (response.status === 401 || response.status === 403) {
      return {
        status: 'invalid',
        provider,
        message: `유효하지 않은 ${PROVIDER_LABEL[provider]} 키 (${response.status})`
      }
    }

    return {
      status: 'invalid',
      provider,
      message: `키 확인 실패 (HTTP ${response.status})`
    }
  } catch (error) {
    return {
      status: 'invalid',
      provider,
      message: `확인 중 네트워크 오류: ${error instanceof Error ? error.message : String(error)}`
    }
  }
}
