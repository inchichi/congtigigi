import type {
  CharacterAction,
  CharacterMoveDirection
} from './characterState'

export type PlayerControlBindingId =
  | 'move-up'
  | 'move-down'
  | 'move-left'
  | 'move-right'
  | 'interact-enter'
  | 'interact-space'
  | 'attack'
  | 'inventory'
  | 'stat'
  | 'equipment'
  | 'skill'
  | 'quest'
  | 'map'
  | 'pause'
  | 'quickslot-1'
  | 'quickslot-2'
  | 'quickslot-3'
  | 'quickslot-4'
  | 'quickslot-5'
  | 'quickslot-6'

export type PlayerControlBindings = Record<PlayerControlBindingId, string>

export type PlayerControlBindingDefinition = {
  id: PlayerControlBindingId
  label: string
  defaultCode: string
}

const PLAYER_CONTROL_BINDING_DEFINITIONS: readonly PlayerControlBindingDefinition[] =
  [
    { id: 'move-up', label: '이동 위', defaultCode: 'ArrowUp' },
    { id: 'move-down', label: '이동 아래', defaultCode: 'ArrowDown' },
    { id: 'move-left', label: '이동 왼쪽', defaultCode: 'ArrowLeft' },
    { id: 'move-right', label: '이동 오른쪽', defaultCode: 'ArrowRight' },
    { id: 'interact-enter', label: '상호작용 (Enter)', defaultCode: 'Enter' },
    { id: 'interact-space', label: '상호작용 (Space)', defaultCode: 'Space' },
    { id: 'attack', label: '공격', defaultCode: 'KeyA' },
    { id: 'inventory', label: '인벤토리', defaultCode: 'KeyI' },
    { id: 'stat', label: '스탯', defaultCode: 'KeyS' },
    { id: 'equipment', label: '장비', defaultCode: 'KeyU' },
    { id: 'skill', label: '스킬', defaultCode: 'KeyK' },
    { id: 'quest', label: '퀘스트', defaultCode: 'KeyB' },
    { id: 'map', label: '지도', defaultCode: 'KeyM' },
    { id: 'pause', label: '메뉴 닫기', defaultCode: 'Escape' },
    { id: 'quickslot-1', label: '퀵슬롯 1', defaultCode: 'Digit1' },
    { id: 'quickslot-2', label: '퀵슬롯 2', defaultCode: 'Digit2' },
    { id: 'quickslot-3', label: '퀵슬롯 3', defaultCode: 'Digit3' },
    { id: 'quickslot-4', label: '퀵슬롯 4', defaultCode: 'Digit4' },
    { id: 'quickslot-5', label: '퀵슬롯 5', defaultCode: 'Digit5' },
    { id: 'quickslot-6', label: '퀵슬롯 6', defaultCode: 'Digit6' }
  ] as const

const PLAYER_CONTROL_BINDING_INDEX_BY_ID = Object.fromEntries(
  PLAYER_CONTROL_BINDING_DEFINITIONS.map(({ id }, index) => [id, index] as const)
) as Record<PlayerControlBindingId, number>

const PLAYER_CONTROL_BINDING_LABEL_BY_CODE: Record<string, string> = {
  ArrowUp: '↑',
  ArrowDown: '↓',
  ArrowLeft: '←',
  ArrowRight: '→',
  Enter: 'Enter',
  Space: 'Space',
  Escape: 'Esc'
}

export const PLAYER_CONTROL_BINDINGS_STORAGE_KEY =
  'my-sample-rpg:control-bindings'

export const createInitialPlayerControlBindings = (): PlayerControlBindings =>
  Object.fromEntries(
    PLAYER_CONTROL_BINDING_DEFINITIONS.map(({ id, defaultCode }) => [
      id,
      defaultCode
    ])
  ) as PlayerControlBindings

export const normalizeStoredPlayerControlBindings = (
  nextBindings: unknown
): PlayerControlBindings => {
  if (!nextBindings || typeof nextBindings !== 'object') {
    return createInitialPlayerControlBindings()
  }

  const storedBindings = nextBindings as Partial<Record<PlayerControlBindingId, unknown>>
  const normalizedBindings = createInitialPlayerControlBindings()
  const usedCodes = new Set<string>()

  for (const { id, defaultCode } of PLAYER_CONTROL_BINDING_DEFINITIONS) {
    const candidateCode = storedBindings[id]
    const bindingCode =
      typeof candidateCode === 'string' && candidateCode.length > 0
        ? candidateCode
        : defaultCode
    const resolvedCode = usedCodes.has(bindingCode) ? defaultCode : bindingCode

    normalizedBindings[id] = resolvedCode
    usedCodes.add(resolvedCode)
  }

  return normalizedBindings
}

export const setPlayerControlBinding = ({
  bindings,
  bindingId,
  nextCode
}: {
  bindings: PlayerControlBindings
  bindingId: PlayerControlBindingId
  nextCode: string
}): PlayerControlBindings => {
  const currentCode = bindings[bindingId]

  if (currentCode === nextCode) {
    return bindings
  }

  const conflictBindingId = findPlayerControlBindingIdByCode(
    bindings,
    nextCode
  )

  if (!conflictBindingId || conflictBindingId === bindingId) {
    return {
      ...bindings,
      [bindingId]: nextCode
    }
  }

  return {
    ...bindings,
    [bindingId]: nextCode,
    [conflictBindingId]: currentCode
  }
}

export const getPlayerControlBindingDisplayText = (code: string): string => {
  if (code in PLAYER_CONTROL_BINDING_LABEL_BY_CODE) {
    return PLAYER_CONTROL_BINDING_LABEL_BY_CODE[code]
  }

  if (code.startsWith('Key')) {
    return code.slice(3)
  }

  if (code.startsWith('Digit')) {
    return code.slice(5)
  }

  if (code.startsWith('Numpad')) {
    return `Num ${code.slice(6)}`
  }

  return code
}

export const getPlayerControlMovementDirectionFromCode = (
  bindings: PlayerControlBindings,
  code: string
): CharacterMoveDirection | undefined => {
  if (code === bindings['move-up']) {
    return 'up'
  }

  if (code === bindings['move-down']) {
    return 'down'
  }

  if (code === bindings['move-left']) {
    return 'left'
  }

  if (code === bindings['move-right']) {
    return 'right'
  }

  return undefined
}

export const getPlayerControlActionFromCode = (
  bindings: PlayerControlBindings,
  code: string
): CharacterAction | undefined => {
  if (code === bindings['interact-enter'] || code === bindings['interact-space']) {
    return 'interact'
  }

  if (code === bindings.attack) {
    return 'attack'
  }

  return undefined
}

export const getPlayerControlQuickslotIndexFromCode = (
  bindings: PlayerControlBindings,
  code: string
): number | undefined => {
  for (let index = 0; index < 6; index += 1) {
    if (code === bindings[`quickslot-${index + 1}` as PlayerControlBindingId]) {
      return index
    }
  }

  return undefined
}

export const isPlayerControlPauseKey = (
  bindings: PlayerControlBindings,
  code: string
): boolean => code === bindings.pause

export const isPlayerControlCaptureModifierKey = (code: string): boolean =>
  code === 'ShiftLeft' ||
  code === 'ShiftRight' ||
  code === 'ControlLeft' ||
  code === 'ControlRight' ||
  code === 'AltLeft' ||
  code === 'AltRight' ||
  code === 'MetaLeft' ||
  code === 'MetaRight'

export const getPlayerControlBindingDefinitionById = (
  bindingId: PlayerControlBindingId
): PlayerControlBindingDefinition =>
  PLAYER_CONTROL_BINDING_DEFINITIONS[PLAYER_CONTROL_BINDING_INDEX_BY_ID[bindingId]]

export const getPlayerControlBindingDefinitions = (): readonly PlayerControlBindingDefinition[] =>
  PLAYER_CONTROL_BINDING_DEFINITIONS

const findPlayerControlBindingIdByCode = (
  bindings: PlayerControlBindings,
  code: string
): PlayerControlBindingId | undefined => {
  for (const [bindingId, bindingCode] of Object.entries(bindings)) {
    if (bindingCode === code) {
      return bindingId as PlayerControlBindingId
    }
  }

  return undefined
}
