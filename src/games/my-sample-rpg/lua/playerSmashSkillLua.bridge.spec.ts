import { readFile } from 'node:fs/promises'

import { afterEach, describe, expect, it } from 'vitest'

import type { CharacterMoveDirection } from '../characterState'
import {
  getPlayerSmashSkillDirectionVector,
  getPlayerSmashSkillSegmentDistancePixels,
  getPlayerSmashSkillSegmentPlacement
} from '../playerSmashSkill'
import {
  createPlayerSmashSkillLua,
  type PlayerSmashSkillLua
} from './playerSmashSkillLua'

// playerSmashSkill의 세그먼트 수학(객체 입출력 → callJson)이 TS와 동등한지 실제 WASM으로 검증.
const LUA_MODULE_JS_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.mjs',
  import.meta.url
)
const LUA_MODULE_WASM_URL = new URL(
  '../../../../public/vendor/lua/lua-5.3.6.wasm',
  import.meta.url
)

const createLua = async (): Promise<PlayerSmashSkillLua> => {
  const [{ default: createLuaModule }, wasmBinary] = await Promise.all([
    import(/* @vite-ignore */ LUA_MODULE_JS_URL.href),
    readFile(LUA_MODULE_WASM_URL)
  ])

  return createPlayerSmashSkillLua({
    createLuaModuleFactory: async () => createLuaModule,
    createLuaModuleOptions: { wasmBinary }
  })
}

// 각 enum 분기 + 알려지지 않은 값(기본 분기) 검증용.
const DIRECTIONS: CharacterMoveDirection[] = ['up', 'down', 'left', 'right']
const UNKNOWN_FACINGS = ['', 'diagonal', 'UP'] as unknown as CharacterMoveDirection[]

// 정상 + 경계(0, 1, 음수, 1 초과, fadeOut 경계 0.78, 클램프 직전/직후).
const PROGRESSES = [0, 0.1, 0.25, 0.5, 0.78, 0.779, 0.781, 0.9, 1, -0.5, 1.5, 0.123456789]
const SEGMENT_INDICES = [0, 1, 2, 3, 5, -1, 2.5]
const CENTERS = [0, 16, 123.5, -40, 1000]

describe('playerSmashSkillLua (real wasm bridge)', () => {
  let lua: PlayerSmashSkillLua | undefined

  afterEach(() => {
    lua?.close()
    lua = undefined
  })

  it('matches TS getPlayerSmashSkillDirectionVector for every direction + default branch', async () => {
    lua = await createLua()

    for (const facing of [...DIRECTIONS, ...UNKNOWN_FACINGS]) {
      expect(lua.getPlayerSmashSkillDirectionVector(facing)).toEqual(
        getPlayerSmashSkillDirectionVector(facing)
      )
    }
  })

  it('matches TS getPlayerSmashSkillSegmentDistancePixels across segments + progresses', async () => {
    lua = await createLua()

    for (const segmentIndex of SEGMENT_INDICES) {
      // 기본값(progress 생략) 경로.
      expect(lua.getPlayerSmashSkillSegmentDistancePixels(segmentIndex)).toBe(
        getPlayerSmashSkillSegmentDistancePixels(segmentIndex)
      )

      for (const progress of PROGRESSES) {
        expect(
          lua.getPlayerSmashSkillSegmentDistancePixels(segmentIndex, progress)
        ).toBe(getPlayerSmashSkillSegmentDistancePixels(segmentIndex, progress))
      }
    }
  })

  it('matches TS getPlayerSmashSkillSegmentPlacement across directions, segments, progresses, centers', async () => {
    lua = await createLua()

    for (const facing of [...DIRECTIONS, ...UNKNOWN_FACINGS]) {
      for (const segmentIndex of SEGMENT_INDICES) {
        for (const progress of PROGRESSES) {
          for (const center of CENTERS) {
            const input = {
              characterCenterX: center,
              characterCenterY: center + 7.5,
              facing,
              segmentIndex,
              progress
            }

            expect(lua.getPlayerSmashSkillSegmentPlacement(input)).toEqual(
              getPlayerSmashSkillSegmentPlacement(input)
            )
          }
        }
      }
    }
  })

  it('matches TS getPlayerSmashSkillSegmentPlacement with default progress', async () => {
    lua = await createLua()

    for (const facing of DIRECTIONS) {
      const input = {
        characterCenterX: 100,
        characterCenterY: 200,
        facing,
        segmentIndex: 1
      }

      expect(lua.getPlayerSmashSkillSegmentPlacement(input)).toEqual(
        getPlayerSmashSkillSegmentPlacement(input)
      )
    }
  })
})
