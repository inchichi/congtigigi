import { describe, expect, it } from 'vitest'

import { createInitialPlayerProfile } from './playerProfile'

describe('createInitialPlayerProfile', () => {
  it('creates the default player HUD profile', () => {
    expect(createInitialPlayerProfile()).toEqual({
      name: '아린',
      job: '방랑자',
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
          label: '베기',
          description: '재빠른 근거리 공격'
        },
        {
          hotkey: '2',
          label: '방어 자세',
          description: '들어오는 피해를 막아냅니다'
        },
        {
          hotkey: '3',
          label: '돌진',
          description: '짧게 빠르게 이동합니다'
        },
        {
          hotkey: '4',
          label: '집중',
          description: '마나를 조금 회복합니다'
        }
      ]
    })
  })
})
