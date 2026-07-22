import type { GameEntity } from './gameAdapter'
import type { GameStructureProfile } from './gameStructureProfile'
import { generateJson } from './llmProvider'
import { QWEN_LOCAL_TOKEN } from './qwenGenerate'
import {
  createQuestJsonSchema,
  createQuestSystemPrompt
} from './questJsonGenerator'
import {
  createGeneratedQuestValidationIssues,
  type GeneratedQuestJson,
  type GeneratedQuestValidationIssue
} from './questJsonSchema'

const STYLE_SERVICE_BASE = '/api/style'
const MAX_STYLE_TARGETS = 2

const fluxPromptTranslationSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    prompt: { type: 'string', minLength: 1 }
  },
  required: ['prompt']
}

const containsNonAscii = (value: string): boolean => /[^\x00-\x7F]/u.test(value)

/**
 * Qwen handles the user's Korean request, but FLUX receives a final English
 * image-editing prompt. Keep this conversion at the last possible boundary so
 * quest text and the editor's Korean UI remain untouched.
 */
const translatePromptForFlux = async (prompt: string): Promise<string> => {
  const trimmed = prompt.trim()
  if (!containsNonAscii(trimmed)) {
    return trimmed
  }

  const translated = await generateJson<{ prompt: string }>({
    apiKey: QWEN_LOCAL_TOKEN,
    instructions: [
      'Translate the input into concise, natural English for the FLUX image-editing model.',
      'The input may be Korean or mixed Korean and English. If a phrase is already English, preserve its meaning and polish it only when needed.',
      'Preserve all technical constraints exactly: pixel-art identity, transparent background, canvas dimensions, sprite frame layout, tile grid, silhouette, and subject identity.',
      'Do not add new creative ideas, explanations, markdown, or UI text. Return only the JSON object.'
    ].join('\n'),
    input: trimmed,
    schemaName: 'flux_english_prompt',
    schema: fluxPromptTranslationSchema
  })

  const result = translated.prompt.trim()
  if (result.length === 0) {
    throw new Error('FLUX용 영어 프롬프트가 비어 있습니다.')
  }
  return result
}

// 현재 단계에서는 통합 테마 JSON을 My Sample RPG 런타임 계약으로 고정한다.
// 다른 게임을 지원할 때는 이 계약을 복제하지 않고 별도 어댑터/스키마를 추가한다.
export const MY_SAMPLE_RPG_GAME_ID = 'my-sample-rpg' as const
export const MY_SAMPLE_RPG_THEME_SCHEMA_VERSION = 1 as const

export type StyleCatalogEntry = {
  id: string
  label: string
}

export type StyleCatalog = {
  assets: StyleCatalogEntry[]
  objects: StyleCatalogEntry[]
}

export type UnifiedThemeStyleTarget = {
  target_ref: string
  prompt: string
  alpha: number
}

export type UnifiedThemePlan = {
  schema_version: typeof MY_SAMPLE_RPG_THEME_SCHEMA_VERSION
  game_id: typeof MY_SAMPLE_RPG_GAME_ID
  theme: string
  art_direction: {
    style: string
    mood: string
    palette: string
  }
  quest: GeneratedQuestJson
  style_targets: UnifiedThemeStyleTarget[]
}

export type UnifiedThemeDirection = Pick<
  UnifiedThemePlan,
  'theme' | 'art_direction'
>

export type UnifiedThemeStyleTargetDraft = Pick<
  UnifiedThemePlan,
  'style_targets'
>

export type UnifiedThemeValidationIssue = GeneratedQuestValidationIssue | {
  path: string
  message: string
}

export type UnifiedThemeStyleApplyResult = {
  applied: string[]
  failed: Array<{ targetRef: string; error: string }>
}

const PROTECTED_RUNTIME_ASSET_NAMES = new Set([
  'monster-pig-sheet.png',
  'pig-motion.png',
  '몬스터-말캉이.png',
  'town-32.png',
  'tiny-dungeon-16.png'
])

const isProtectedRuntimeAsset = (path: string): boolean => {
  const fileName = path.split('/').pop()?.toLowerCase()
  return fileName !== undefined && PROTECTED_RUNTIME_ASSET_NAMES.has(fileName)
}

const isRuntimeSheet = (path: string): boolean => {
  const fileName = path.split('/').pop()?.toLowerCase() ?? ''
  return /(?:^|[-_])(sheet|motion|spritesheet)(?:[-_.]|$)/u.test(fileName)
}

// My Sample RPG에서 FLUX가 직접 덮어쓸 수 있는 것은 정적 PNG뿐이다.
// 타일셋은 맵 전체와 공유되고, sheet/motion 파일은 런타임 애니메이션이므로
// 통합 테마 카탈로그에서 처음부터 제외해 적용 실패를 예방한다.
export const isStylableAsset = (path: string): boolean =>
  path.toLowerCase().endsWith('.png') &&
  !isProtectedRuntimeAsset(path) &&
  !isRuntimeSheet(path) &&
  ['/monsters/', '/portraits/', '/weapons/', '/armor/'].some((part) =>
    path.toLowerCase().includes(part)
  )

const requestJson = async <T>(url: string, errorMessage: string): Promise<T> => {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`${errorMessage} (HTTP ${response.status})`)
  }
  return (await response.json()) as T
}

export const loadStyleCatalog = async (): Promise<StyleCatalog> => {
  const assets = await requestJson<{ assets?: Array<{ path?: unknown }> }>(
    `${STYLE_SERVICE_BASE}/assets`,
    'FLUX 에셋 목록을 불러오지 못했습니다'
  )
  let objects: Array<{ key?: unknown; label?: unknown }> = []
  try {
    const extracted = await requestJson<{ objects?: Array<{ key?: unknown; label?: unknown }> }>(
      `${STYLE_SERVICE_BASE}/extracted-objects`,
      '추출 오브젝트 목록을 불러오지 못했습니다'
    )
    objects = extracted.objects ?? []
  } catch {
    // Object extraction is optional. Asset styling can still run without it.
  }

  return {
    assets: (assets.assets ?? [])
      .flatMap((asset) =>
        typeof asset.path === 'string' && isStylableAsset(asset.path)
          ? [{ id: asset.path, label: asset.path.split('/').pop() ?? asset.path }]
          : []
      )
      .slice(0, 48),
    objects: objects
      .flatMap((object) =>
        typeof object.key === 'string'
          ? [{ id: object.key, label: typeof object.label === 'string' ? object.label : object.key }]
          : []
      )
      .slice(0, 48)
  }
}

const assetRef = (id: string): string => `asset:${id}`
const objectRef = (id: string): string => `object:${id}`

const getStyleRefs = (catalog: StyleCatalog): string[] => [
  ...catalog.assets.map((asset) => assetRef(asset.id)),
  ...catalog.objects.map((object) => objectRef(object.id))
]

const artDirectionSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    style: { type: 'string', minLength: 1 },
    mood: { type: 'string', minLength: 1 },
    palette: { type: 'string', minLength: 1 }
  },
  required: ['style', 'mood', 'palette']
}

const createStyleTargetsSchema = (catalog: StyleCatalog): object => {
  const styleRefs = getStyleRefs(catalog)
  return {
    type: 'array',
    minItems: styleRefs.length > 0 ? 1 : 0,
    maxItems: Math.min(MAX_STYLE_TARGETS, styleRefs.length),
    items: {
      type: 'object',
      additionalProperties: false,
      properties: {
        target_ref: { type: 'string', enum: styleRefs },
        prompt: { type: 'string', minLength: 1 },
        alpha: { type: 'number', minimum: 0.2, maximum: 1 }
      },
      required: ['target_ref', 'prompt', 'alpha']
    }
  }
}

export const createUnifiedThemeDirectionSchema = (): object => ({
  type: 'object',
  additionalProperties: false,
  properties: {
    theme: { type: 'string', minLength: 1 },
    art_direction: artDirectionSchema
  },
  required: ['theme', 'art_direction']
})

export const createUnifiedThemeStyleTargetSchema = (catalog: StyleCatalog): object => ({
  type: 'object',
  additionalProperties: false,
  properties: {
    style_targets: createStyleTargetsSchema(catalog)
  },
  required: ['style_targets']
})

export const createUnifiedThemePlanSchema = (
  profile: GameStructureProfile,
  catalog: StyleCatalog,
  entity?: GameEntity
): object => {
  return {
    type: 'object',
    additionalProperties: false,
    properties: {
      schema_version: { type: 'integer', const: MY_SAMPLE_RPG_THEME_SCHEMA_VERSION },
      game_id: { type: 'string', const: MY_SAMPLE_RPG_GAME_ID },
      theme: { type: 'string', minLength: 1 },
      art_direction: artDirectionSchema,
      quest: createQuestJsonSchema(profile, entity),
      style_targets: createStyleTargetsSchema(catalog)
    },
    required: [
      'schema_version',
      'game_id',
      'theme',
      'art_direction',
      'quest',
      'style_targets'
    ]
  }
}

const styleCatalogPrompt = (catalog: StyleCatalog): string => {
  const assets = catalog.assets.map((asset) => `- ${assetRef(asset.id)} (${asset.label})`)
  const objects = catalog.objects.map((object) => `- ${objectRef(object.id)} (${object.label})`)
  return [
    `This is a My Sample RPG theme plan. Always set schema_version to ${MY_SAMPLE_RPG_THEME_SCHEMA_VERSION} and game_id to ${MY_SAMPLE_RPG_GAME_ID}.`,
    'The output is consumed by the TypeScript + Vite + PixiJS runtime in this project, not by a generic RPG engine.',
    'Available FLUX style targets. Use only exact target_ref values from this list.',
    'Prefer extracted objects for buildings, trees, props, and other map objects.',
    'For file assets, use only static PNG portraits, monster images, weapons, or armor.',
    'Never target runtime animation sheets, motion files, or any map tileset. Map scenery must be changed through extracted object targets.',
    ...(assets.length > 0 ? ['Asset targets:', ...assets] : ['Asset targets: none']),
    ...(objects.length > 0 ? ['Extracted object targets:', ...objects] : ['Extracted object targets: none'])
  ].join('\n')
}

export const generateUnifiedThemePlan = ({
  apiKey,
  userPrompt,
  profile,
  catalog,
  entity
}: {
  apiKey: string
  userPrompt: string
  profile: GameStructureProfile
  catalog: StyleCatalog
  entity?: GameEntity
}): Promise<UnifiedThemePlan> =>
  generateJson<UnifiedThemePlan>({
    apiKey,
    instructions: [
      createQuestSystemPrompt(profile, entity),
      `This is the current project only: ${MY_SAMPLE_RPG_GAME_ID}, a TypeScript + Vite + PixiJS 2D RPG. Do not design for another engine or invent a portable schema.`,
      'You are the creative director for a 2D web RPG content automation editor.',
      'Turn one natural-language theme into one coherent quest and a small set of FLUX image-editing operations.',
      'Keep the quest story, NPC motivation, environment mood, and art direction consistent with the requested theme.',
      'Use only exact game ids for the quest. Do not invent ids.',
      'Write game-facing quest text in English because the runtime font is ASCII-oriented.',
      'Write every style target prompt in English for FLUX. Preserve pixel-art readability, transparency, identity, canvas dimensions, and frame/tile layout.',
      `Return schema_version=${MY_SAMPLE_RPG_THEME_SCHEMA_VERSION} and game_id="${MY_SAMPLE_RPG_GAME_ID}" exactly.`,
      `Return concise JSON. Use exactly 1 objective, exactly 1 short sentence in each dialogue array, and at most ${MAX_STYLE_TARGETS} style targets. If a target is not useful for the theme, do not include it.`,
      styleCatalogPrompt(catalog)
    ].join('\n'),
    input: userPrompt.trim(),
    schemaName: 'unified_theme_plan',
    schema: createUnifiedThemePlanSchema(profile, catalog, entity)
  })

export const generateUnifiedThemeDirection = ({
  apiKey,
  userPrompt
}: {
  apiKey: string
  userPrompt: string
}): Promise<UnifiedThemeDirection> =>
  generateJson<UnifiedThemeDirection>({
    apiKey,
    instructions: [
      `This is a My Sample RPG theme direction for ${MY_SAMPLE_RPG_GAME_ID}.`,
      'Create only the visual direction for the requested theme; do not create a quest or asset list yet.',
      'Keep the direction concise and suitable for a 2D pixel-art RPG.',
      'Use English for style, mood, and palette so the result can be passed to FLUX.',
      'Return only the requested JSON object.'
    ].join('\n'),
    input: userPrompt.trim(),
    schemaName: 'my_sample_rpg_theme_direction',
    schema: createUnifiedThemeDirectionSchema()
  })

export const generateUnifiedThemeQuest = ({
  apiKey,
  userPrompt,
  profile,
  direction,
  entity
}: {
  apiKey: string
  userPrompt: string
  profile: GameStructureProfile
  direction: UnifiedThemeDirection
  entity?: GameEntity
}): Promise<GeneratedQuestJson> =>
  generateJson<GeneratedQuestJson>({
    apiKey,
    instructions: [
      createQuestSystemPrompt(profile, entity),
      'Implement the accepted My Sample RPG theme direction as exactly one playable quest.',
      'Use exactly one objective so the user can review one clear gameplay loop before applying it.',
      'Use short English dialogue lines because the runtime font is ASCII-oriented.',
      'Do not invent IDs and return only the quest JSON object.',
      `Accepted theme direction:\n${JSON.stringify(direction, null, 2)}`
    ].join('\n'),
    input: userPrompt.trim(),
    schemaName: 'my_sample_rpg_theme_quest',
    schema: createQuestJsonSchema(profile, entity)
  })

export const generateUnifiedThemeStyleTargets = ({
  apiKey,
  userPrompt,
  direction,
  quest,
  catalog
}: {
  apiKey: string
  userPrompt: string
  direction: UnifiedThemeDirection
  quest: GeneratedQuestJson
  catalog: StyleCatalog
}): Promise<UnifiedThemeStyleTargetDraft> =>
  generateJson<UnifiedThemeStyleTargetDraft>({
    apiKey,
    instructions: [
      `Create the FLUX style target list for ${MY_SAMPLE_RPG_GAME_ID}.`,
      'Choose at most two targets that materially improve the requested theme.',
      'Use only exact target_ref values in the catalog.',
      'Prefer extracted map objects for scenery. Use only static PNG files for characters, weapons, or armor.',
      'Never choose a map tileset, animation sheet, or motion file.',
      'Write every prompt in English and preserve pixel-art identity, transparency, dimensions, and object silhouette.',
      `Accepted art direction:\n${JSON.stringify(direction.art_direction, null, 2)}`,
      `Accepted quest:\n${JSON.stringify({ title: quest.title, guide_text: quest.guide_text }, null, 2)}`,
      styleCatalogPrompt(catalog)
    ].join('\n'),
    input: userPrompt.trim(),
    schemaName: 'my_sample_rpg_theme_style_targets',
    schema: createUnifiedThemeStyleTargetSchema(catalog)
  })

export const createUnifiedThemeValidationIssues = (
  plan: UnifiedThemePlan,
  profile: GameStructureProfile,
  catalog: StyleCatalog,
  entity?: GameEntity
): UnifiedThemeValidationIssue[] => {
  const issues: UnifiedThemeValidationIssue[] = [
    ...createGeneratedQuestValidationIssues(plan.quest, profile, {
      selectedEntityId: entity?.kind === 'npc' ? entity.id : undefined
    })
  ]
  if (plan.schema_version !== MY_SAMPLE_RPG_THEME_SCHEMA_VERSION) {
    issues.push({
      path: 'schema_version',
      message: `현재 My Sample RPG 테마 스키마 버전은 ${MY_SAMPLE_RPG_THEME_SCHEMA_VERSION}입니다.`
    })
  }
  if (plan.game_id !== MY_SAMPLE_RPG_GAME_ID) {
    issues.push({
      path: 'game_id',
      message: `현재 프로젝트용 game_id가 아닙니다: ${plan.game_id}`
    })
  }
  const allowedRefs = new Set(getStyleRefs(catalog))
  const seenRefs = new Set<string>()

  if (plan.theme.trim().length === 0) {
    issues.push({ path: 'theme', message: 'theme은 비어 있을 수 없습니다.' })
  }
  plan.style_targets.forEach((target, index) => {
    const path = `style_targets[${index}]`
    if (!allowedRefs.has(target.target_ref)) {
      issues.push({ path: `${path}.target_ref`, message: `존재하지 않는 FLUX 대상입니다: ${target.target_ref}` })
    }
    if (seenRefs.has(target.target_ref)) {
      issues.push({ path: `${path}.target_ref`, message: '같은 스타일 대상을 중복 지정했습니다.' })
    }
    seenRefs.add(target.target_ref)
    if (target.prompt.trim().length === 0) {
      issues.push({ path: `${path}.prompt`, message: '스타일 프롬프트는 비어 있을 수 없습니다.' })
    }
    if (!Number.isFinite(target.alpha) || target.alpha < 0.2 || target.alpha > 1) {
      issues.push({ path: `${path}.alpha`, message: 'alpha는 0.2에서 1 사이여야 합니다.' })
    }
  })
  return issues
}

const resolveStyleTarget = (
  targetRef: string,
  catalog: StyleCatalog
): { kind: 'asset'; path: string } | { kind: 'object'; key: string } | undefined => {
  if (targetRef.startsWith('asset:')) {
    const path = targetRef.slice('asset:'.length)
    return catalog.assets.some((asset) => asset.id === path) ? { kind: 'asset', path } : undefined
  }
  if (targetRef.startsWith('object:')) {
    const key = targetRef.slice('object:'.length)
    return catalog.objects.some((object) => object.id === key) ? { kind: 'object', key } : undefined
  }
  return undefined
}

const buildFluxPrompt = (plan: UnifiedThemePlan, targetPrompt: string, kind: 'asset' | 'object'): string => [
  'Target game: My Sample RPG (TypeScript + Vite + PixiJS).',
  `Art direction: ${plan.art_direction.style}.`,
  `Mood: ${plan.art_direction.mood}.`,
  `Palette: ${plan.art_direction.palette}.`,
  kind === 'object'
    ? 'Preserve the exact object silhouette, transparent background, tile placement, and pixel-art proportions.'
    : 'Preserve the exact canvas dimensions, transparency, sprite frame layout, tile grid, and subject identity.',
  targetPrompt
].join(' ')

export const applyUnifiedThemeStyles = async (
  plan: UnifiedThemePlan,
  catalog: StyleCatalog
): Promise<UnifiedThemeStyleApplyResult> => {
  const applied: string[] = []
  const failed: Array<{ targetRef: string; error: string }> = []

  for (const target of plan.style_targets) {
    const resolved = resolveStyleTarget(target.target_ref, catalog)
    if (!resolved) {
      failed.push({ targetRef: target.target_ref, error: '스타일 대상이 현재 프로젝트에 없습니다.' })
      continue
    }

    try {
      const fluxPrompt = await translatePromptForFlux(buildFluxPrompt(plan, target.prompt, resolved.kind))
      const form = new FormData()
      form.append('prompt', fluxPrompt)
      form.append('alpha', String(Math.min(1, Math.max(0.2, target.alpha))))
      form.append('targets', JSON.stringify([resolved]))
      const response = await fetch(`${STYLE_SERVICE_BASE}/batch-apply`, {
        method: 'POST',
        body: form
      })
      const rawText = await response.text()
      let data: { applied?: string[]; failed?: Array<{ error?: string }> } = {}
      try {
        data = JSON.parse(rawText) as typeof data
      } catch {
        data = {}
      }
      if (!response.ok) {
        throw new Error(rawText || `HTTP ${response.status}`)
      }
      if ((data.failed ?? []).length > 0 || (data.applied ?? []).length === 0) {
        throw new Error(data.failed?.[0]?.error ?? 'FLUX 적용 결과가 없습니다.')
      }
      applied.push(target.target_ref)
    } catch (error) {
      failed.push({
        targetRef: target.target_ref,
        error: error instanceof Error ? error.message : String(error)
      })
    }
  }

  return { applied, failed }
}
