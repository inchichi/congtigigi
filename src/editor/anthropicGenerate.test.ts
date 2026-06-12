import { afterEach, describe, expect, it, vi } from 'vitest'

import { generateJsonWithClaude } from './anthropicGenerate'

const schema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    name: { type: 'string' },
    value: { type: 'string' }
  },
  required: ['name', 'value']
}

const mockClaudeResponse = (toolInput: unknown): void => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => ({
      ok: true,
      status: 200,
      text: async () =>
        JSON.stringify({
          stop_reason: 'tool_use',
          content: [{ type: 'tool_use', input: toolInput }]
        })
    }))
  )
}

const call = (): Promise<{ name: string; value: string }> =>
  generateJsonWithClaude({
    apiKey: 'test',
    instructions: 'x',
    input: 'y',
    schemaName: 'thing',
    schema
  })

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('generateJsonWithClaude structured-output validation', () => {
  it('returns the tool input when it satisfies the schema required fields', async () => {
    mockClaudeResponse({ name: 'a', value: 'b' })
    await expect(call()).resolves.toEqual({ name: 'a', value: 'b' })
  })

  it('throws a readable error when a required top-level field is missing', async () => {
    mockClaudeResponse({ name: 'a' })
    await expect(call()).rejects.toThrow(/필수 필드가 없습니다.*value/u)
  })

  it('throws when the tool input is not an object', async () => {
    mockClaudeResponse('not-an-object')
    await expect(call()).rejects.toThrow(/객체 형식이 아닙니다/u)
  })
})
