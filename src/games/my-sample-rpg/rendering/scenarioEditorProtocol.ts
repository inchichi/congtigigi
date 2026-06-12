export type ScenarioEditorQuestDraft = {
  title: string
  giver: string
  goal: string
  objectives: string[]
  reward: string
}

export type ScenarioEditorNpcDialogueDraft = {
  npc: string
  context: string
  lines: string[]
}

export type ScenarioEditorResult = {
  summary: string
  quests: ScenarioEditorQuestDraft[]
  npcDialogues: ScenarioEditorNpcDialogueDraft[]
}

const MIN_SCENARIO_EDITOR_QUESTS = 2
const MIN_SCENARIO_EDITOR_NPC_DIALOGUES = 2

export const SCENARIO_EDITOR_SYSTEM_PROMPT = [
  'You design quest content for a 2D web RPG.',
  'Return one valid JSON object only. Do not use markdown or code fences.',
  'Keep the tone game-ready, concise, and easy to import into game data.',
  'If the scenario is written in Korean, answer in Korean too.',
  'If the user prompt includes an available NPC roster, use only those NPC names for quest givers and dialogue speakers.',
  'Do not invent extra NPC titles or roles that are not in the roster.',
  'Use this schema exactly:',
  '{',
  '  "summary": "short scenario summary",',
  '  "quests": [',
  '    {',
  '      "title": "quest title",',
  '      "giver": "npc name",',
  '      "goal": "main goal",',
  '      "objectives": ["step 1", "step 2"],',
  '      "reward": "reward text"',
  '    }',
  '  ],',
  '  "npc_dialogues": [',
  '    {',
  '      "npc": "npc name",',
  '      "context": "when the line is used",',
  '      "lines": ["line 1", "line 2"]',
  '    }',
  '  ]',
  '}'
].join('\n')

export const buildScenarioEditorUserPrompt = (
  scenario: string,
  availableNpcNames: readonly string[] = []
): string => {
  const trimmedScenario = scenario.trim()
  const npcRoster = availableNpcNames
    .map((name) => name.trim())
    .filter((name) => name.length > 0)

  const prompt = [
    'Turn the scenario below into a small set of quests and NPC dialogue.',
    'Write 2 to 4 quests and 2 to 6 NPC dialogue entries.',
    'Keep the result consistent with the scenario and make the lines feel usable in-game.',
    '',
    ...(npcRoster.length > 0
      ? [
          'Available NPC roster:',
          ...npcRoster.map((name) => `- ${name}`),
          'Use only these NPC names for quest givers and dialogue speakers.',
          'Do not invent new NPC titles or roles.'
        ]
      : []),
    ...(npcRoster.length > 0 ? [''] : []),
    'Scenario:',
    trimmedScenario
  ]

  return prompt.join('\n')
}

export const extractScenarioEditorResponseText = (
  response: unknown
): string | undefined => {
  if (!isRecord(response)) {
    return undefined
  }

  const outputText = response.output_text

  if (typeof outputText === 'string' && outputText.trim().length > 0) {
    return outputText.trim()
  }

  const choices = response.choices

  if (Array.isArray(choices)) {
    const firstChoice = choices[0]

    if (isRecord(firstChoice)) {
      const message = firstChoice.message

      if (isRecord(message)) {
        const content = message.content
        const messageText = extractTextFromContent(content)

        if (messageText) {
          return messageText
        }
      }

      const text = firstChoice.text

      if (typeof text === 'string' && text.trim().length > 0) {
        return text.trim()
      }
    }
  }

  const output = response.output

  if (Array.isArray(output)) {
    const parts: string[] = []

    for (const item of output) {
      if (!isRecord(item)) {
        continue
      }

      const text = extractTextFromContent(item.content)

      if (text) {
        parts.push(text)
      }

      if (typeof item.text === 'string' && item.text.trim().length > 0) {
        parts.push(item.text.trim())
      }
    }

    const joined = parts.join('\n').trim()

    if (joined.length > 0) {
      return joined
    }
  }

  return undefined
}

export const parseScenarioEditorResult = (
  responseText: string
): ScenarioEditorResult | undefined => {
  const trimmedText = responseText.trim()

  if (trimmedText.length === 0) {
    return undefined
  }

  const unwrappedText = unwrapCodeFence(trimmedText)
  const jsonText = extractJsonObjectText(unwrappedText)

  if (!jsonText) {
    return undefined
  }

  try {
    return normalizeScenarioEditorResult(JSON.parse(jsonText) as unknown)
  } catch {
    return undefined
  }
}

export const normalizeScenarioEditorResult = (
  value: unknown
): ScenarioEditorResult | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  const summary = toText(value.summary ?? value.scenario_summary)
  const quests = toObjectArray(value.quests).flatMap((item) => {
    const quest = normalizeScenarioEditorQuest(item)

    return quest ? [quest] : []
  })
  const npcDialogues = toObjectArray(
    value.npc_dialogues ?? value.npcDialogues ?? value.dialogues
  ).flatMap((item) => {
    const dialogue = normalizeScenarioEditorNpcDialogue(item)

    return dialogue ? [dialogue] : []
  })

  if (
    summary.length === 0 ||
    quests.length < MIN_SCENARIO_EDITOR_QUESTS ||
    npcDialogues.length < MIN_SCENARIO_EDITOR_NPC_DIALOGUES
  ) {
    return undefined
  }

  return {
    summary,
    quests,
    npcDialogues
  }
}

const normalizeScenarioEditorQuest = (
  value: unknown
): ScenarioEditorQuestDraft | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  const quest = {
    title: toText(value.title ?? value.name ?? value.questTitle),
    giver: toText(value.giver ?? value.npc ?? value.quest_giver),
    goal: toText(value.goal ?? value.objective ?? value.summary),
    objectives: toTextList(value.objectives ?? value.steps ?? value.tasks),
    reward: toText(value.reward ?? value.rewards)
  }

  if (
    quest.title.length === 0 &&
    quest.giver.length === 0 &&
    quest.goal.length === 0 &&
    quest.objectives.length === 0 &&
    quest.reward.length === 0
  ) {
    return undefined
  }

  return quest
}

const normalizeScenarioEditorNpcDialogue = (
  value: unknown
): ScenarioEditorNpcDialogueDraft | undefined => {
  if (!isRecord(value)) {
    return undefined
  }

  const dialogue = {
    npc: toText(value.npc ?? value.name),
    context: toText(value.context ?? value.scene ?? value.moment),
    lines: toTextList(value.lines ?? value.dialogue ?? value.text)
  }

  if (
    dialogue.npc.length === 0 &&
    dialogue.context.length === 0 &&
    dialogue.lines.length === 0
  ) {
    return undefined
  }

  return dialogue
}

const extractTextFromContent = (content: unknown): string | undefined => {
  if (typeof content === 'string') {
    const trimmed = content.trim()

    return trimmed.length > 0 ? trimmed : undefined
  }

  if (!Array.isArray(content)) {
    return undefined
  }

  const parts: string[] = []

  for (const item of content) {
    if (typeof item === 'string') {
      const trimmed = item.trim()

      if (trimmed.length > 0) {
        parts.push(trimmed)
      }

      continue
    }

    if (!isRecord(item)) {
      continue
    }

    const text = toText(item.text ?? item.value ?? item.content)

    if (text.length > 0) {
      parts.push(text)
    }
  }

  const joined = parts.join('\n').trim()

  return joined.length > 0 ? joined : undefined
}

const unwrapCodeFence = (text: string): string => {
  const fenceMatch = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/iu)

  if (fenceMatch) {
    return fenceMatch[1].trim()
  }

  return text
}

const extractJsonObjectText = (text: string): string | undefined => {
  const firstBraceIndex = text.indexOf('{')
  const lastBraceIndex = text.lastIndexOf('}')

  if (firstBraceIndex < 0 || lastBraceIndex < firstBraceIndex) {
    return undefined
  }

  return text.slice(firstBraceIndex, lastBraceIndex + 1)
}

const toObjectArray = (value: unknown): unknown[] => {
  if (Array.isArray(value)) {
    return value
  }

  if (isRecord(value)) {
    return [value]
  }

  return []
}

const toText = (value: unknown): string => {
  if (typeof value === 'string') {
    return value.trim()
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  return ''
}

const toTextList = (value: unknown): string[] => {
  if (!Array.isArray(value)) {
    return []
  }

  return value
    .map((item) => toText(item))
    .filter((item) => item.length > 0)
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null
}
