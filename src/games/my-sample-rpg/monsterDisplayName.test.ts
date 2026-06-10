import { describe, expect, it } from 'vitest'

import { getMonsterDisplayName } from './monsterDisplayName'

describe('getMonsterDisplayName', () => {
  it('prefers explicit display text', () => {
    expect(
      getMonsterDisplayName({
        id: '꿀꿀이-1',
        displayText: '숲의 꿀꿀이'
      })
    ).toBe('숲의 꿀꿀이')
  })

  it('removes the trailing instance suffix from monster ids', () => {
    expect(
      getMonsterDisplayName({
        id: '꿀꿀이-1'
      })
    ).toBe('꿀꿀이')
    expect(
      getMonsterDisplayName({
        id: '말캉이-보스'
      })
    ).toBe('말캉이')
  })

  it('keeps ids without a suffix as-is', () => {
    expect(
      getMonsterDisplayName({
        id: 'slime'
      })
    ).toBe('slime')
  })
})
