// 실행 중인 외부 게임(예: Love2D legend-of-lua)과 에디터(브라우저)를 잇는 라이브 브리지.
// 게임은 로컬에 작은 HTTP 서버를 띄우고, 에디터가 생성물을 POST해 라이브로 반영한다.
// my-sample-rpg는 같은 origin localStorage로 적용하지만, Lua 게임은 별도 프로세스라
// localStorage를 공유할 수 없어서 이 HTTP 채널이 필요하다. 프로토콜 명세:
// docs/legend-of-lua-bridge-protocol.md

export type BridgeStatus = 'disconnected' | 'connecting' | 'connected'

export type BridgeApplyTarget = {
  id: string
  name: string
  kind: string
  mapId: string
}

// 에디터 → 게임으로 보내는 "적용" 메시지. kind로 종류를 구분한다(엔티티 대사 / 퀘스트).
// 엔티티 대사 적용(기존). 게임은 target.id/mapId로 엔티티를 찾아 lines를 붙인다.
export type BridgeEntityLinesMessage = {
  // 적용 1건의 고유 id(게임이 ack에 echo). 중복 적용 멱등 처리에 쓸 수 있다.
  id: string
  kind: 'entity_lines'
  // 적용 대상(없으면 게임 전역). 게임은 id/mapId로 실제 엔티티를 찾는다.
  target: BridgeApplyTarget | null
  lines: string[]
  generatedAt: number
}

// 퀘스트 목표 1개(게임의 배치 엔티티 id/맵을 가리킨다). 게임-쪽 quest-runtime이 해석한다.
export type BridgeQuestObjective = {
  type: 'defeat' | 'talk' | 'acquire' | 'reach'
  label: string
  required: number
  target: { entityId?: string; mapId?: string }
}

// 퀘스트 적용(신규). 게임-쪽 quest-runtime.lua가 등록·추적·보상한다(docs/legend-of-lua-quest-contract.md).
export type BridgeQuestData = {
  quest_id: string
  title: string
  giver_entity_id: string
  request_text: string
  guide_text: string
  dialogue: {
    start: string[]
    active: string[]
    complete: string[]
  }
  objectives: BridgeQuestObjective[]
  rewards: {
    gold: number
    experience: number
    items: { label: string; quantity: number }[]
  }
}

export type BridgeQuestMessage = {
  id: string
  kind: 'quest'
  quest: BridgeQuestData
  generatedAt: number
}

// NPC 스폰 적용(신규). 실행 중인 legend-of-lua가 받아 NPC를 만든다. 패널(love.js iframe)은 host
// page의 editor-bridge.js가 받아 Emscripten FS에 Lua 청크로 써두고, 게임의 pollEditorInbox가
// 읽어 spawnNPC한다(docs/legend-of-lua-npc-contract.md). nearPlayer면 현재 플레이어 옆에 띄운다.
export type BridgeSpawnNpcData = {
  name: string
  map: string
  appearance: string
  dialogue: string[]
  radius: number
  nearPlayer: boolean
}

export type BridgeSpawnNpcMessage = {
  id: string
  kind: 'spawn_npc'
  npc: BridgeSpawnNpcData
  // 게임이 loadstring으로 바로 로드할 한 줄 Lua 테이블(host page 다리가 'spawn-npc:'로 전달).
  lua: string
  generatedAt: number
}

export type BridgeApplyMessage =
  | BridgeEntityLinesMessage
  | BridgeQuestMessage
  | BridgeSpawnNpcMessage

export type BridgeApplyResponse = {
  ok: boolean
  error?: string
}

export type GameBridge = {
  getStatus: () => BridgeStatus
  getBaseUrl: () => string
  setBaseUrl: (url: string) => void
  start: () => void
  stop: () => void
  apply: (message: BridgeApplyMessage) => Promise<BridgeApplyResponse>
}

const STATUS_POLL_INTERVAL_MS = 2000
const STATUS_TIMEOUT_MS = 2500
const APPLY_TIMEOUT_MS = 5000

const stripTrailingSlash = (url: string): string => url.replace(/\/+$/u, '')

export const createGameBridge = ({
  baseUrl,
  onStatusChange
}: {
  baseUrl: string
  onStatusChange: (status: BridgeStatus) => void
}): GameBridge => {
  let currentBaseUrl = stripTrailingSlash(baseUrl)
  let status: BridgeStatus = 'disconnected'
  let pollTimer: ReturnType<typeof setInterval> | undefined

  const setStatus = (next: BridgeStatus): void => {
    if (next === status) {
      return
    }

    status = next
    onStatusChange(status)
  }

  // 게임이 안 떠 있으면 fetch가 곧장 실패하거나 매달릴 수 있어, 타임아웃으로 끊는다.
  const fetchWithTimeout = async (
    path: string,
    init: RequestInit,
    timeoutMs: number
  ): Promise<Response> => {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)

    try {
      return await fetch(`${currentBaseUrl}${path}`, {
        ...init,
        signal: controller.signal
      })
    } finally {
      clearTimeout(timer)
    }
  }

  const pollStatus = async (): Promise<void> => {
    try {
      const response = await fetchWithTimeout(
        '/status',
        { method: 'GET' },
        STATUS_TIMEOUT_MS
      )
      setStatus(response.ok ? 'connected' : 'disconnected')
    } catch {
      // 연결 거부/타임아웃 = 게임 미실행. 조용히 미연결로 둔다(콘솔 도배 방지).
      setStatus('disconnected')
    }
  }

  return {
    getStatus: () => status,
    getBaseUrl: () => currentBaseUrl,
    setBaseUrl: (url) => {
      currentBaseUrl = stripTrailingSlash(url)

      // 폴링 중이면 새 주소로 즉시 한 번 확인한다.
      if (pollTimer) {
        void pollStatus()
      }
    },
    start: () => {
      if (pollTimer) {
        return
      }

      setStatus('connecting')
      void pollStatus()
      pollTimer = setInterval(() => void pollStatus(), STATUS_POLL_INTERVAL_MS)
    },
    stop: () => {
      if (pollTimer) {
        clearInterval(pollTimer)
        pollTimer = undefined
      }

      setStatus('disconnected')
    },
    apply: async (message) => {
      try {
        const response = await fetchWithTimeout(
          '/apply',
          {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(message)
          },
          APPLY_TIMEOUT_MS
        )

        if (!response.ok) {
          return {
            ok: false,
            error: `게임이 HTTP ${response.status}를 반환했습니다.`
          }
        }

        // 게임이 { ok, error } JSON을 주면 그대로 신뢰하고, 본문이 없으면 성공으로 본다.
        const data = (await response
          .json()
          .catch(() => ({ ok: true }))) as BridgeApplyResponse

        return { ok: data.ok !== false, error: data.error }
      } catch (error) {
        return {
          ok: false,
          error: error instanceof Error ? error.message : String(error)
        }
      }
    }
  }
}
