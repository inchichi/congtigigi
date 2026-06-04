import type { GeneratedEventJson } from '../llm/eventJsonSchema'

export type DynamicEventDefinition = {
  event_json: GeneratedEventJson
  generated_code: string
  created_at: number
}

const dynamicEventDefinitions: DynamicEventDefinition[] = []

export const registerDynamicEventDefinition = (
  definition: DynamicEventDefinition
): void => {
  const existingIndex = dynamicEventDefinitions.findIndex(
    (candidate) => candidate.event_json.event_id === definition.event_json.event_id
  )

  if (existingIndex >= 0) {
    dynamicEventDefinitions.splice(existingIndex, 1, definition)
    return
  }

  dynamicEventDefinitions.push(definition)
}

export const getDynamicEventDefinitions = (): DynamicEventDefinition[] => [
  ...dynamicEventDefinitions
]

export const getLatestDynamicEventDefinition = ():
  | DynamicEventDefinition
  | undefined =>
  dynamicEventDefinitions[dynamicEventDefinitions.length - 1]

export const clearDynamicEventDefinitions = (): void => {
  dynamicEventDefinitions.length = 0
}
