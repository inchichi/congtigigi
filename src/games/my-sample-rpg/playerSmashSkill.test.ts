import { describe, expect, it } from 'vitest'

import {
  getPlayerSmashSkillDirectionVector,
  getPlayerSmashSkillSegmentDistancePixels,
  getPlayerSmashSkillSegmentPlacement
} from './playerSmashSkill'

describe('playerSmashSkill', () => {
  it('keeps the segments 14px apart at launch', () => {
    expect(getPlayerSmashSkillSegmentDistancePixels(0)).toBe(50)
    expect(getPlayerSmashSkillSegmentDistancePixels(1)).toBe(64)
    expect(getPlayerSmashSkillSegmentDistancePixels(2)).toBe(78)
  })

  it('moves the skill forward as progress increases', () => {
    expect(getPlayerSmashSkillSegmentDistancePixels(0, 0)).toBe(50)
    expect(getPlayerSmashSkillSegmentDistancePixels(0, 0.5)).toBe(122)
    expect(getPlayerSmashSkillSegmentDistancePixels(0, 1)).toBe(146)
  })

  it('places the skill in the facing direction', () => {
    expect(getPlayerSmashSkillDirectionVector('up')).toEqual({ x: 0, y: -1 })
    expect(getPlayerSmashSkillDirectionVector('down')).toEqual({ x: 0, y: 1 })
    expect(getPlayerSmashSkillDirectionVector('left')).toEqual({ x: -1, y: 0 })
    expect(getPlayerSmashSkillDirectionVector('right')).toEqual({ x: 1, y: 0 })
  })

  it('travels forward instead of only growing wider', () => {
    const placement = getPlayerSmashSkillSegmentPlacement({
      characterCenterX: 100,
      characterCenterY: 50,
      facing: 'right',
      segmentIndex: 2,
      progress: 0.5
    })

    expect(placement.x).toBe(250)
    expect(placement.y).toBe(50)
    expect(placement.scaleX).toBeCloseTo(0.378, 4)
    expect(placement.scaleY).toBeCloseTo(0.3348, 4)
    expect(placement.alpha).toBe(1)
  })
})
