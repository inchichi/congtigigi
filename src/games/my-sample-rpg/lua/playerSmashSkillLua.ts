// playerSmashSkill.ts → Lua 래퍼. 스매시 스킬 세그먼트 수학(방향 벡터/거리/배치)을 Lua VM에서 실행한다.
// 순수 수학이라 상수는 TS가 소유하고 init 없이 호출마다 인자로 전달한다. 객체 입출력이라 callJson.

import playerSmashSkillSource from '../assets/lua/player-smash-skill.lua?raw'

import type { CharacterMoveDirection } from '../characterState'
import {
  PLAYER_SMASH_SKILL_FIRST_SEGMENT_DISTANCE_PIXELS,
  PLAYER_SMASH_SKILL_SEGMENT_SPACING_PIXELS,
  PLAYER_SMASH_SKILL_LAUNCH_OFFSET_PIXELS,
  PLAYER_SMASH_SKILL_TRAVEL_DISTANCE_PIXELS,
  PLAYER_SMASH_SKILL_EFFECT_BASE_SCALE_X,
  PLAYER_SMASH_SKILL_EFFECT_BASE_SCALE_Y,
  PLAYER_SMASH_SKILL_EFFECT_SCALE_STEP,
  PLAYER_SMASH_SKILL_EFFECT_SCALE_GROWTH,
  type PlayerSmashSkillSegmentPlacement
} from '../playerSmashSkill'

import {
  createLuaLogicHost,
  type CreateLuaLogicHostInput,
  type LuaLogicHost
} from './luaLogicHost'

// Lua에 한 번에 전달하는 상수 묶음(원본 TS의 export 상수와 1:1).
const PLACEMENT_CONSTANTS = {
  launchOffset: PLAYER_SMASH_SKILL_LAUNCH_OFFSET_PIXELS,
  firstSegmentDistance: PLAYER_SMASH_SKILL_FIRST_SEGMENT_DISTANCE_PIXELS,
  segmentSpacing: PLAYER_SMASH_SKILL_SEGMENT_SPACING_PIXELS,
  travelDistance: PLAYER_SMASH_SKILL_TRAVEL_DISTANCE_PIXELS,
  baseScaleX: PLAYER_SMASH_SKILL_EFFECT_BASE_SCALE_X,
  baseScaleY: PLAYER_SMASH_SKILL_EFFECT_BASE_SCALE_Y,
  scaleStep: PLAYER_SMASH_SKILL_EFFECT_SCALE_STEP,
  scaleGrowth: PLAYER_SMASH_SKILL_EFFECT_SCALE_GROWTH
} as const

export type PlayerSmashSkillLua = {
  getPlayerSmashSkillDirectionVector: (
    facing: CharacterMoveDirection
  ) => {
    x: number
    y: number
  }
  getPlayerSmashSkillSegmentDistancePixels: (
    segmentIndex: number,
    progress?: number
  ) => number
  getPlayerSmashSkillSegmentPlacement: (input: {
    characterCenterX: number
    characterCenterY: number
    facing: CharacterMoveDirection
    segmentIndex: number
    progress?: number
  }) => PlayerSmashSkillSegmentPlacement
  close: () => void
}

export const createPlayerSmashSkillLua = async (
  input: CreateLuaLogicHostInput & { host?: LuaLogicHost } = {}
): Promise<PlayerSmashSkillLua> => {
  const host = input.host ?? (await createLuaLogicHost(input))
  host.runModule(playerSmashSkillSource, '@player-smash-skill.lua')

  return {
    getPlayerSmashSkillDirectionVector: (
      facing: CharacterMoveDirection
    ): { x: number; y: number } =>
      host.callJson<{ x: number; y: number }>(
        'player_smash_skill_direction_vector',
        facing
      ),
    getPlayerSmashSkillSegmentDistancePixels: (
      segmentIndex: number,
      progress = 0
    ): number =>
      host.callJson<number>(
        'player_smash_skill_segment_distance_pixels',
        segmentIndex,
        progress,
        PLACEMENT_CONSTANTS.launchOffset,
        PLACEMENT_CONSTANTS.firstSegmentDistance,
        PLACEMENT_CONSTANTS.segmentSpacing,
        PLACEMENT_CONSTANTS.travelDistance
      ),
    getPlayerSmashSkillSegmentPlacement: ({
      characterCenterX,
      characterCenterY,
      facing,
      segmentIndex,
      progress = 0
    }: {
      characterCenterX: number
      characterCenterY: number
      facing: CharacterMoveDirection
      segmentIndex: number
      progress?: number
    }): PlayerSmashSkillSegmentPlacement =>
      host.callJson<PlayerSmashSkillSegmentPlacement>(
        'player_smash_skill_segment_placement',
        {
          characterCenterX,
          characterCenterY,
          facing,
          segmentIndex,
          progress
        },
        PLACEMENT_CONSTANTS
      ),
    close: (): void => {
      if (!input.host) {
        host.close()
      }
    }
  }
}
