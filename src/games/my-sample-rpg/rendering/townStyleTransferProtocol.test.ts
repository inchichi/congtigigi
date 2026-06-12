import { describe, expect, it } from 'vitest'

import {
  buildTownStyleTransferPrompt,
  extractTownStyleTransferResult
} from './townStyleTransferProtocol'

describe('town style transfer protocol', () => {
  it('builds a prompt that preserves the scenario and the tile grid', () => {
    const prompt = buildTownStyleTransferPrompt(
      '  눈 덮인 겨울 마을, 모든 것이 얼어붙었다.  '
    )

    expect(prompt).toContain('눈 덮인 겨울 마을, 모든 것이 얼어붙었다.')
    expect(prompt).toContain('natural-language art direction prompt')
    expect(prompt).toContain('uploaded tile sheet')
    expect(prompt).toContain('Preserve the exact tile layout')
    expect(prompt).toContain('Do not add, remove, or move tiles.')
    // Scenario is trimmed before being embedded in the prompt.
    expect(prompt).not.toContain('  눈 덮인')
  })

  it('extracts an image generation call from a responses-api output payload', () => {
    expect(
      extractTownStyleTransferResult({
        output: [
          {
            type: 'image_generation_call',
            result: 'AAAA'
          }
        ]
      })
    ).toEqual({ imageDataUrl: 'data:image/png;base64,AAAA' })
  })

  it('extracts a base64 image into a data URL', () => {
    expect(
      extractTownStyleTransferResult({
        data: [{ b64_json: 'AAAA' }]
      })
    ).toEqual({ imageDataUrl: 'data:image/png;base64,AAAA' })
  })

  it('falls back to a hosted image url when no base64 is present', () => {
    expect(
      extractTownStyleTransferResult({
        data: [{ url: 'https://example.com/styled.png' }]
      })
    ).toEqual({ imageDataUrl: 'https://example.com/styled.png' })
  })

  it('returns undefined for response shapes without an image', () => {
    expect(extractTownStyleTransferResult({})).toBeUndefined()
    expect(extractTownStyleTransferResult({ data: [] })).toBeUndefined()
    expect(extractTownStyleTransferResult({ data: [{}] })).toBeUndefined()
    expect(extractTownStyleTransferResult(null)).toBeUndefined()
  })
})
