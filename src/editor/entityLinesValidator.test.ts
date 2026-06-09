import { describe, expect, it } from 'vitest'

import { createEntityLinesValidationIssues } from './entityLinesValidator'

describe('createEntityLinesValidationIssues', () => {
  it('passes a well-formed result that references the selected target', () => {
    const issues = createEntityLinesValidationIssues(
      { entity: 'slime', lines: ['끈적끈적!', '도망쳐!'] },
      { targetEntityName: 'slime' }
    )

    expect(issues).toEqual([])
  })

  it('accepts a target match in either direction (substring)', () => {
    expect(
      createEntityLinesValidationIssues(
        { entity: 'green slime', lines: ['hi'] },
        { targetEntityName: 'slime' }
      )
    ).toEqual([])
  })

  it('flags an empty entity name', () => {
    const issues = createEntityLinesValidationIssues({ entity: '  ', lines: ['hi'] })
    expect(issues).toHaveLength(1)
    expect(issues[0]).toContain('entity')
  })

  it('flags an entity that does not reference the selected target', () => {
    const issues = createEntityLinesValidationIssues(
      { entity: 'dragon', lines: ['roar'] },
      { targetEntityName: 'slime' }
    )
    expect(issues.some((issue) => issue.includes('다릅니다'))).toBe(true)
  })

  it('flags no lines at all and stops there', () => {
    const issues = createEntityLinesValidationIssues({ entity: 'slime', lines: [] })
    expect(issues).toHaveLength(1)
    expect(issues[0]).toContain('최소 1줄')
  })

  it('flags too many lines past the default max of 4', () => {
    const issues = createEntityLinesValidationIssues({
      entity: 'slime',
      lines: ['a', 'b', 'c', 'd', 'e']
    })
    expect(issues.some((issue) => issue.includes('줄 수가 너무 많'))).toBe(true)
  })

  it('flags an empty line in the middle', () => {
    const issues = createEntityLinesValidationIssues({
      entity: 'slime',
      lines: ['first', '   ', 'third']
    })
    expect(issues.some((issue) => issue.includes('lines[1]'))).toBe(true)
  })

  it('flags a line that exceeds the max length', () => {
    const issues = createEntityLinesValidationIssues(
      { entity: 'slime', lines: ['x'.repeat(50)] },
      { maxLineLength: 20 }
    )
    expect(issues.some((issue) => issue.includes('너무 깁니다'))).toBe(true)
  })
})
