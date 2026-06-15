import { afterEach, describe, expect, it, vi } from 'vitest'

import { getResponsiveUiScale } from './getResponsiveUiScale'

const setWindowSize = (width: number, height: number): void => {
  vi.stubGlobal('window', { innerWidth: width, innerHeight: height })
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('getResponsiveUiScale', () => {
  it('기준 해상도(1600×900) 이상에서는 최대 배율로 고정된다', () => {
    setWindowSize(1920, 1080)

    expect(getResponsiveUiScale()).toBeCloseTo(1.2)
  })

  it('작은 창에서는 창 크기에 비례해 계속 줄어든다 — 에디터 프리뷰 iframe에서 HUD가 잘리지 않게', () => {
    // 과거엔 하한이 0.72라 700px 창에서도 0.864로 고정 → HUD(설계 폭 780px)가 잘렸다.
    setWindowSize(700, 620)

    expect(getResponsiveUiScale()).toBeCloseTo((700 / 1600) * 1.2)
  })

  it('극단적으로 작은 창에서도 하한 밑으로는 내려가지 않는다', () => {
    setWindowSize(160, 90)

    expect(getResponsiveUiScale()).toBeCloseTo(0.3 * 1.2)
  })
})
