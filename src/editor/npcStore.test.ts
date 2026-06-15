import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  addNpc,
  clearNpcsForMap,
  loadNpcsForMap,
  removeNpc
} from './npcStore'

// npcStore는 window.localStorage(safeStorage 경유)를 쓴다. 테스트는 node 환경이라
// 인메모리 localStorage를 window에 stub한다(테스트마다 새 저장소).
const makeStorage = (): Storage => {
  const map = new Map<string, string>()
  return {
    getItem: (key) => (map.has(key) ? (map.get(key) as string) : null),
    setItem: (key, value) => {
      map.set(key, String(value))
    },
    removeItem: (key) => {
      map.delete(key)
    },
    clear: () => map.clear(),
    key: (index) => [...map.keys()][index] ?? null,
    get length() {
      return map.size
    }
  } as Storage
}

beforeEach(() => {
  vi.stubGlobal('window', { localStorage: makeStorage() })
})
afterEach(() => {
  vi.unstubAllGlobals()
})

describe('npcStore', () => {
  it('adds an NPC and reads it back for the same map', () => {
    const npc = addNpc(
      'town',
      {
        appearanceType: 'character_wizard_purple',
        name: '마법사',
        dialogueLines: ['안녕하세요.', '좋은 하루 되세요.']
      },
      5,
      7
    )

    expect(npc.id).toMatch(/^n_/)
    expect(loadNpcsForMap('town')).toEqual([
      {
        id: npc.id,
        appearanceType: 'character_wizard_purple',
        name: '마법사',
        dialogueLines: ['안녕하세요.', '좋은 하루 되세요.'],
        col: 5,
        row: 7
      }
    ])
  })

  it('keeps NPCs separate per map', () => {
    addNpc('town', { appearanceType: 'character_villager_brown_tunic' }, 1, 1)
    addNpc('cave', { appearanceType: 'character_ranger_green' }, 2, 2)

    expect(loadNpcsForMap('town')).toHaveLength(1)
    expect(loadNpcsForMap('cave')).toHaveLength(1)
    expect(loadNpcsForMap('town')[0].appearanceType).toBe(
      'character_villager_brown_tunic'
    )
    expect(loadNpcsForMap('cave')[0].appearanceType).toBe(
      'character_ranger_green'
    )
  })

  it('removes a single NPC by id without touching others', () => {
    const first = addNpc('town', { appearanceType: 'character_elder_gray_hair' }, 0, 0)
    const second = addNpc('town', { appearanceType: 'character_ranger_green' }, 1, 0)

    const remaining = removeNpc('town', first.id)

    expect(remaining).toHaveLength(1)
    expect(remaining[0].id).toBe(second.id)
    expect(loadNpcsForMap('town')).toHaveLength(1)
  })

  it('clears all NPCs for a map only', () => {
    addNpc('town', { appearanceType: 'character_wizard_purple' }, 0, 0)
    addNpc('cave', { appearanceType: 'character_ranger_green' }, 0, 0)

    clearNpcsForMap('town')

    expect(loadNpcsForMap('town')).toEqual([])
    expect(loadNpcsForMap('cave')).toHaveLength(1)
  })

  it('returns [] for an unknown map or corrupt storage', () => {
    expect(loadNpcsForMap('nope')).toEqual([])
    window.localStorage.setItem('my-sample-rpg:pending-npcs', 'not json{')
    expect(loadNpcsForMap('town')).toEqual([])
  })
})
