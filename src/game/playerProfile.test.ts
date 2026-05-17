import { describe, expect, it } from 'vitest'

import { createInitialPlayerProfile } from './playerProfile'

describe('createInitialPlayerProfile', () => {
  it('creates the default player HUD profile', () => {
    expect(createInitialPlayerProfile()).toEqual({
      name: 'Arin',
      job: 'Wanderer',
      level: 1,
      hp: {
        current: 24,
        max: 24
      },
      mp: {
        current: 12,
        max: 12
      },
      stats: {
        attack: 5,
        defense: 3,
        agility: 4
      },
      skills: [
        {
          hotkey: '1',
          label: 'Slash',
          description: 'Quick close-range strike'
        },
        {
          hotkey: '2',
          label: 'Guard',
          description: 'Brace against incoming damage'
        },
        {
          hotkey: '3',
          label: 'Dash',
          description: 'Short burst of movement'
        },
        {
          hotkey: '4',
          label: 'Focus',
          description: 'Recover a small amount of mana'
        }
      ]
    })
  })
})
