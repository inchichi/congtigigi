import { describe, expect, it } from 'vitest'
import {
  hasBlockingLuaControllerIssue,
  validateLuaControllerSource
} from './luaControllerValidator'

// 실제 컨트롤러와 같은 형태의 유효한 샘플(게임의 reply-with-message 와 동형).
const VALID_CONTROLLER = `
local controllers = {}
local controller = {}

function controller.register(id, home_x, home_y, radius)
  controllers[id] = true
end

function controller.step(id, dt, x, y)
  return 0, 0
end

function controller.interact(id, source_id)
  engine.ui.show_message("안녕하신가, 풋내기.", 2.5)
end

return controller
`

describe('validateLuaControllerSource', () => {
  it('accepts a well-formed controller (no issues)', () => {
    expect(validateLuaControllerSource(VALID_CONTROLLER)).toEqual([])
    expect(
      hasBlockingLuaControllerIssue(validateLuaControllerSource(VALID_CONTROLLER))
    ).toBe(false)
  })

  it('rejects empty source', () => {
    const issues = validateLuaControllerSource('   ')
    expect(hasBlockingLuaControllerIssue(issues)).toBe(true)
    expect(issues[0].message).toContain('비어')
  })

  it('rejects a controller that does not return the table', () => {
    const issues = validateLuaControllerSource(
      'local controller = {}\nfunction controller.step(id, dt, x, y) return 0, 0 end'
    )
    expect(issues.some((i) => i.severity === 'error' && i.message.includes('return'))).toBe(true)
  })

  it('rejects a controller missing the required step method', () => {
    const issues = validateLuaControllerSource(
      'local controller = {}\nfunction controller.interact(id, s) end\nreturn controller'
    )
    expect(issues.some((i) => i.severity === 'error' && i.message.includes('step'))).toBe(true)
  })

  it('rejects sandbox-escaping APIs (io/os/require/load)', () => {
    for (const bad of ['io.open("x")', 'os.execute("x")', 'require("socket")', 'load("x")()']) {
      const source = `local controller = {}\nfunction controller.step(id) ${bad}\nreturn 0,0 end\nreturn controller`
      expect(hasBlockingLuaControllerIssue(validateLuaControllerSource(source))).toBe(true)
    }
  })

  it('warns on a break-less while true loop', () => {
    const issues = validateLuaControllerSource(
      'local controller = {}\nfunction controller.step(id) while true do end return 0,0 end\nreturn controller'
    )
    expect(issues.some((i) => i.severity === 'warning' && i.message.includes('while true'))).toBe(true)
  })
})
