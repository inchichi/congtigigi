// playerRoll의 구르기 수학 로직을 Lua로 실행하는 래퍼(벡터/객체 입출력 → callJson 마샬링).
// 상수(지속시간/거리)와 시간값은 TS가 소유해 인자로 전달한다(공식만 Lua). normalize는 0 벡터일 때
// Lua가 nil을 반환 → JSON에서 null로 와서 TS undefined로 정규화한다.

import playerRollSource from '../assets/lua/player-roll.lua?raw'

import type { CharacterMoveDirection } from '../characterState'
import {
  PLAYER_ROLL_DURATION_MILLISECONDS,
  PLAYER_ROLL_DISTANCE_TILES,
  type PlayerRollVector,
  type PlayerRollVisualState
} from '../playerRoll'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

export type PlayerRollLua = {
  getPlayerRollDirectionVector: (
    direction: CharacterMoveDirection
  ) => PlayerRollVector
  normalizePlayerRollVector: (
    vector: PlayerRollVector
  ) => PlayerRollVector | undefined
  getPlayerRollProgress: (input: {
    nowMilliseconds: number
    startedAtMilliseconds: number
  }) => number
  getPlayerRollDistanceTiles: (progress: number) => number
  getPlayerRollVisualState: (input: {
    vector: PlayerRollVector
    progress: number
  }) => PlayerRollVisualState
  close: () => void
}

export const createPlayerRollLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerRollLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  // 공유 호스트 경로면 의존성은 facade가 이미 로드함(여기선 의존성 없음). 자기 소스만 로드.
  host.runModule(playerRollSource, '@player-roll.lua')

  return {
    getPlayerRollDirectionVector: (
      direction: CharacterMoveDirection
    ): PlayerRollVector =>
      host.callJson<PlayerRollVector>('player_roll_direction_vector', direction),
    normalizePlayerRollVector: (
      vector: PlayerRollVector
    ): PlayerRollVector | undefined => {
      const result = host.callJson<PlayerRollVector | null>(
        'player_roll_normalize_vector',
        vector
      )

      return result === null ? undefined : result
    },
    getPlayerRollProgress: ({
      nowMilliseconds,
      startedAtMilliseconds
    }: {
      nowMilliseconds: number
      startedAtMilliseconds: number
    }): number =>
      host.callJson<number>(
        'player_roll_progress',
        nowMilliseconds,
        startedAtMilliseconds,
        PLAYER_ROLL_DURATION_MILLISECONDS
      ),
    getPlayerRollDistanceTiles: (progress: number): number =>
      host.callJson<number>(
        'player_roll_distance_tiles',
        progress,
        PLAYER_ROLL_DISTANCE_TILES
      ),
    getPlayerRollVisualState: ({
      vector,
      progress
    }: {
      vector: PlayerRollVector
      progress: number
    }): PlayerRollVisualState =>
      host.callJson<PlayerRollVisualState>(
        'player_roll_visual_state',
        vector,
        progress
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
