export const TOWN_STYLE_TRANSFER_RESPONSE_ENDPOINT =
  'https://api.openai.com/v1/responses'
export const TOWN_STYLE_TRANSFER_ENDPOINT = TOWN_STYLE_TRANSFER_RESPONSE_ENDPOINT
export const TOWN_STYLE_TRANSFER_HOST_MODEL = 'gpt-5'
export const TOWN_STYLE_TRANSFER_MODEL = TOWN_STYLE_TRANSFER_HOST_MODEL

// town-32.png is a tall portrait tile sheet (256x2240), so 1024x1536 is the
// closest supported portrait edit size.
export const TOWN_STYLE_TRANSFER_SIZE = '1024x1536'

export const TOWN_TILESET_SOURCE_URL = new URL(
  '../assets/tilesets/town-32.png',
  import.meta.url
).href

export type TownStyleTransferResult = {
  /** Data URL (data:image/png;base64,...) ready to drop into <img src>. */
  imageDataUrl: string
}

export const buildTownStyleTransferPrompt = (scenario: string): string => {
  const trimmedScenario = scenario.trim()

  const prompt = [
    'You are editing an existing 2D top-down RPG town tileset.',
    'Treat the scenario below as a natural-language art direction prompt.',
    'Infer the dominant theme, mood, weather, palette, and architecture from the scenario, then repaint the uploaded tile sheet to match it.',
    'Preserve the exact tile layout, pixel-art scale, grid alignment, silhouettes, and transparent areas.',
    'Do not add, remove, or move tiles. Only change surface styling, lighting, and color treatment.',
    '',
    'Scenario:',
    trimmedScenario
  ]

  return prompt.join('\n')
}

export const extractTownStyleTransferResult = (
  response: unknown
): TownStyleTransferResult | undefined => {
  if (!isRecord(response)) {
    return undefined
  }

  const responsesApiImage = extractResponsesApiImage(response.output)

  if (responsesApiImage) {
    return { imageDataUrl: responsesApiImage }
  }

  const imageApiData = extractLegacyImageApiData(response.data)

  if (imageApiData) {
    return { imageDataUrl: imageApiData }
  }

  return undefined
}

const extractResponsesApiImage = (output: unknown): string | undefined => {
  if (!Array.isArray(output)) {
    return undefined
  }

  for (const item of output) {
    if (!isRecord(item) || item.type !== 'image_generation_call') {
      continue
    }

    const result = item.result
    const base64 =
      typeof result === 'string' && result.trim().length > 0
        ? result.trim()
        : typeof item.b64_json === 'string' && item.b64_json.trim().length > 0
          ? item.b64_json.trim()
          : typeof item.url === 'string' && item.url.trim().length > 0
            ? item.url.trim()
            : undefined

    if (!base64) {
      continue
    }

    return normalizeImageDataUrl(base64)
  }

  return undefined
}

const extractLegacyImageApiData = (data: unknown): string | undefined => {
  if (!Array.isArray(data)) {
    return undefined
  }

  for (const item of data) {
    if (!isRecord(item)) {
      continue
    }

    const base64 = item.b64_json

    if (typeof base64 === 'string' && base64.trim().length > 0) {
      return normalizeImageDataUrl(base64.trim())
    }

    const url = item.url

    if (typeof url === 'string' && url.trim().length > 0) {
      return url.trim()
    }
  }

  return undefined
}

const normalizeImageDataUrl = (value: string): string => {
  if (value.startsWith('data:')) {
    return value
  }

  return `data:image/png;base64,${value}`
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null
}
