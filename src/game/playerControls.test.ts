import { describe, expect, it } from 'vitest'

import {
  createInitialPlayerControlBindings,
  getPlayerControlActionFromCode,
  getPlayerControlBindingDisplayText,
  getPlayerControlMovementDirectionFromCode,
  getPlayerControlQuickslotIndexFromCode,
  normalizeStoredPlayerControlBindings,
  setPlayerControlBinding
} from './playerControls'

describe('playerControls', () => {
  it('creates the default binding set', () => {
    expect(createInitialPlayerControlBindings()).toEqual({
      'move-up': 'ArrowUp',
      'move-down': 'ArrowDown',
      'move-left': 'ArrowLeft',
      'move-right': 'ArrowRight',
      'interact-enter': 'Enter',
      'interact-space': 'Space',
      attack: 'KeyA',
      inventory: 'KeyI',
      stat: 'KeyS',
      equipment: 'KeyU',
      skill: 'KeyK',
      quest: 'KeyB',
      map: 'KeyM',
      pause: 'Escape',
      'quickslot-1': 'Digit1',
      'quickslot-2': 'Digit2',
      'quickslot-3': 'Digit3',
      'quickslot-4': 'Digit4',
      'quickslot-5': 'Digit5',
      'quickslot-6': 'Digit6'
    })
  })

  it('swaps conflicting bindings when remapping', () => {
    const nextBindings = setPlayerControlBinding({
      bindings: createInitialPlayerControlBindings(),
      bindingId: 'inventory',
      nextCode: 'KeyA'
    })

    expect(nextBindings.inventory).toBe('KeyA')
    expect(nextBindings.attack).toBe('KeyI')
  })

  it('maps codes to gameplay actions and directions', () => {
    const bindings = createInitialPlayerControlBindings()

    expect(getPlayerControlMovementDirectionFromCode(bindings, 'ArrowUp')).toBe(
      'up'
    )
    expect(getPlayerControlActionFromCode(bindings, 'Enter')).toBe('interact')
    expect(getPlayerControlActionFromCode(bindings, 'KeyA')).toBe('attack')
    expect(getPlayerControlActionFromCode(bindings, 'KeyQ')).toBeUndefined()
    expect(getPlayerControlQuickslotIndexFromCode(bindings, 'Digit6')).toBe(5)
  })

  it('formats key labels for the settings screen', () => {
    expect(getPlayerControlBindingDisplayText('ArrowUp')).toBe('↑')
    expect(getPlayerControlBindingDisplayText('KeyA')).toBe('A')
    expect(getPlayerControlBindingDisplayText('Digit1')).toBe('1')
    expect(getPlayerControlBindingDisplayText('Numpad1')).toBe('Num 1')
    expect(getPlayerControlBindingDisplayText('Escape')).toBe('Esc')
  })

  it('normalizes invalid stored bindings back into a usable set', () => {
    expect(
      normalizeStoredPlayerControlBindings({
        inventory: 'ArrowUp',
        attack: 'ArrowUp'
      })
    ).toMatchObject({
      inventory: 'KeyI',
      attack: 'KeyA'
    })
  })
})
