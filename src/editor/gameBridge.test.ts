import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGameBridge, type BridgeApplyMessage } from './gameBridge'

const message: BridgeApplyMessage = {
  id: 'entity_lines-1',
  kind: 'entity_lines',
  target: { id: 'Enemies-105', name: 'slime', kind: 'enemy', mapId: 'testCave' },
  lines: ['크르릉…'],
  generatedAt: 1
}

const makeBridge = (baseUrl = 'http://localhost:17320/') =>
  createGameBridge({ baseUrl, onStatusChange: () => {} })

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('createGameBridge', () => {
  it('base URL의 끝 슬래시를 정리한다', () => {
    const bridge = makeBridge('http://localhost:9000///')
    expect(bridge.getBaseUrl()).toBe('http://localhost:9000')
  })

  it('apply는 {baseUrl}/apply로 메시지를 POST하고 200이면 성공으로 본다', async () => {
    const fetchMock = vi.fn(
      async (_url: string, _init: RequestInit): Promise<Response> =>
        new Response(JSON.stringify({ ok: true }), { status: 200 })
    )
    vi.stubGlobal('fetch', fetchMock)

    const bridge = makeBridge()
    const result = await bridge.apply(message)

    expect(result.ok).toBe(true)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('http://localhost:17320/apply')
    expect(init).toMatchObject({ method: 'POST' })
    expect(JSON.parse(init.body as string)).toEqual(message)
  })

  it('게임이 비-200을 주면 실패로 보고한다', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('nope', { status: 500 }))
    )

    const result = await makeBridge().apply(message)

    expect(result.ok).toBe(false)
    expect(result.error).toContain('500')
  })

  it('게임이 {ok:false,error}를 주면 그 사유를 그대로 전달한다', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify({ ok: false, error: '대상 없음' }), {
          status: 200
        })
      )
    )

    const result = await makeBridge().apply(message)

    expect(result).toEqual({ ok: false, error: '대상 없음' })
  })

  it('게임 미실행(fetch throw)이면 실패로 보고한다(throw 안 함)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('connection refused')
      })
    )

    const result = await makeBridge().apply(message)

    expect(result.ok).toBe(false)
    expect(result.error).toContain('connection refused')
  })
})
