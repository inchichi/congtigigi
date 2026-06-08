export const TOWN_STYLE_TRANSFER_ENDPOINT =
  'https://api.openai.com/v1/images/edits'
export const TOWN_STYLE_TRANSFER_MODEL = 'gpt-image-1'

// town-32.png is a tall portrait tile sheet (256x2240), so the closest
// supported gpt-image-1 output ratio is the 1024x1536 portrait size.
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
    'This image is a 32x32 grid tile sheet for a 2D top-down web RPG town.',
    'Repaint the tile art so its color palette, mood, and theme match the scenario below.',
    'Keep the exact same tile layout, grid alignment, silhouettes, and pixel-art style.',
    'Only restyle surfaces, lighting, and color — do not move, add, or remove tiles.',
    'Keep transparent areas fully transparent.',
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

  const data = response.data

  if (!Array.isArray(data)) {
    return undefined
  }

  for (const item of data) {
    if (!isRecord(item)) {
      continue
    }

    const base64 = item.b64_json

    if (typeof base64 === 'string' && base64.trim().length > 0) {
      return { imageDataUrl: `data:image/png;base64,${base64.trim()}` }
    }

    const url = item.url

    if (typeof url === 'string' && url.trim().length > 0) {
      return { imageDataUrl: url.trim() }
    }
  }

  return undefined
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null
}
