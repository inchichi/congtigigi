import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  addPlacement,
  clearPlacementsForMap,
  loadPlacementsForMap,
  removePlacement
} from './placementStore'

// placementStore는 window.localStorage(safeStorage 경유)를 쓴다. 테스트는 node 환경이라
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

describe('placementStore', () => {
  it('adds a placement and reads it back for the same map', () => {
    const item = addPlacement(
      'town',
      { kind: 'tile', tilesetSource: '../tilesets/town-32.tsx', tileId: 40, label: '지붕' },
      5,
      7
    )

    expect(item.id).toMatch(/^p_/)
    expect(loadPlacementsForMap('town')).toEqual([
      {
        id: item.id,
        kind: 'tile',
        tilesetSource: '../tilesets/town-32.tsx',
        tileId: 40,
        label: '지붕',
        col: 5,
        row: 7
      }
    ])
  })

  it('keeps placements separate per map', () => {
    addPlacement('town', { kind: 'object', imageUrl: '/a.png' }, 1, 1)
    addPlacement('cave', { kind: 'object', imageUrl: '/b.png' }, 2, 2)

    expect(loadPlacementsForMap('town')).toHaveLength(1)
    expect(loadPlacementsForMap('cave')).toHaveLength(1)
    expect(loadPlacementsForMap('town')[0].imageUrl).toBe('/a.png')
    expect(loadPlacementsForMap('cave')[0].imageUrl).toBe('/b.png')
  })

  it('removes a single placement by id without touching others', () => {
    const first = addPlacement('town', { kind: 'tile', tileId: 1 }, 0, 0)
    const second = addPlacement('town', { kind: 'tile', tileId: 2 }, 1, 0)

    const remaining = removePlacement('town', first.id)

    expect(remaining).toHaveLength(1)
    expect(remaining[0].id).toBe(second.id)
    expect(loadPlacementsForMap('town')).toHaveLength(1)
  })

  it('clears all placements for a map only', () => {
    addPlacement('town', { kind: 'tile', tileId: 1 }, 0, 0)
    addPlacement('cave', { kind: 'tile', tileId: 9 }, 0, 0)

    clearPlacementsForMap('town')

    expect(loadPlacementsForMap('town')).toEqual([])
    expect(loadPlacementsForMap('cave')).toHaveLength(1)
  })

  it('returns [] for an unknown map or corrupt storage', () => {
    expect(loadPlacementsForMap('nope')).toEqual([])
    window.localStorage.setItem('my-sample-rpg:pending-placements', 'not json{')
    expect(loadPlacementsForMap('town')).toEqual([])
  })
})
