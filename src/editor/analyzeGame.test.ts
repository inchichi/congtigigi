import { describe, expect, it } from 'vitest'

import { buildAnalysisEvidence } from './analyzeGame'
import type { GameFile } from './loadGame'

const legendTmx = `<?xml version="1.0" encoding="UTF-8"?>
<map>
  <tileset firstgid="1" name="overworld" tilewidth="16" tileheight="16" tilecount="10" columns="5"/>
  <objectgroup name="Enemies">
    <object id="1" name="slime" type="small"/>
  </objectgroup>
  <objectgroup name="Chests">
    <object id="2" name="chest_1" type="gold"/>
  </objectgroup>
</map>`

describe('buildAnalysisEvidence', () => {
  it('summarizes object groups with types and sample names for the LLM', () => {
    const files: GameFile[] = [
      { name: 'conf.lua', path: 'conf.lua', text: '' },
      { name: 'test.tmx', path: 'maps/test.tmx', text: legendTmx }
    ]

    const evidence = buildAnalysisEvidence(files)

    expect(evidence).toContain('conf.lua')
    expect(evidence).toContain('맵 test.tmx')
    expect(evidence).toContain('object group "Enemies"')
    expect(evidence).toContain('slime')
    expect(evidence).toContain('object group "Chests"')
    expect(evidence).toContain('chest_1')
  })
})
