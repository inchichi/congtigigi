import { describe, expect, it } from 'vitest'

import { getSceneIntroMessage } from './sceneIntro'

describe('getSceneIntroMessage', () => {
  it('returns the town intro text', () => {
    expect(getSceneIntroMessage('town')).toBe('티르코네일 마을')
  })

  it('returns the hunting ground intro text', () => {
    expect(getSceneIntroMessage('hunting-ground')).toBe('슬라임 숲')
  })

  it('returns the cave intro text', () => {
    expect(getSceneIntroMessage('cave')).toBe('동굴')
  })

  it('returns an empty string for unknown scenes', () => {
    expect(getSceneIntroMessage('unknown-scene')).toBe('')
  })
})
