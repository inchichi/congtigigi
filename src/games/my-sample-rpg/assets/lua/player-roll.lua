-- TS playerRoll.ts → Lua. 플레이어 구르기(roll)의 순수 수학 로직(방향 벡터/정규화/진행도/거리/비주얼).
-- 상수(지속시간/거리)는 TS가 소유하고 인자로 전달한다(공식만 Lua). 시간값도 모두 인자로 받는다(시계 읽기 없음).
-- 객체/벡터 입출력이라 callJson 마샬링을 쓴다. normalize는 0 벡터일 때 nil을 반환(TS undefined와 동등).

local function clamp01(value)
  if value < 0 then
    return 0
  end
  if value > 1 then
    return 1
  end
  return value
end

-- direction(string) → {x, y}. 미지의 값은 0 벡터(TS default와 동일).
function player_roll_direction_vector(direction)
  if direction == 'up' then
    return { x = 0, y = -1 }
  elseif direction == 'down' then
    return { x = 0, y = 1 }
  elseif direction == 'left' then
    return { x = -1, y = 0 }
  elseif direction == 'right' then
    return { x = 1, y = 0 }
  end
  return { x = 0, y = 0 }
end

-- 0 벡터면 nil(→ TS undefined). 아니면 단위 벡터로 정규화(hypot = Math.hypot).
function player_roll_normalize_vector(vector)
  if vector.x == 0 and vector.y == 0 then
    return nil
  end

  local length = math.sqrt(vector.x * vector.x + vector.y * vector.y)

  return {
    x = vector.x / length,
    y = vector.y / length
  }
end

-- 경과/지속시간으로 진행도 [0,1] 클램프.
function player_roll_progress(now_ms, started_at_ms, duration_ms)
  return clamp01((now_ms - started_at_ms) / duration_ms)
end

-- 큐빅 ease-out: distance_tiles * (1 - (1 - p)^3).
function player_roll_distance_tiles(progress, distance_tiles)
  local p = clamp01(progress)
  return distance_tiles * (1 - (1 - p) ^ 3)
end

-- 비주얼 상태: 진행도에 따른 lift(sin)와 방향 부호 기반 회전.
function player_roll_visual_state(vector, progress)
  local p = clamp01(progress)
  local direction_sign
  if vector.x < 0 or (vector.x == 0 and vector.y < 0) then
    direction_sign = -1
  else
    direction_sign = 1
  end
  local lift_amount = math.sin(p * math.pi)

  return {
    offsetX = 0,
    offsetY = -math.abs(lift_amount) * 4,
    rotation = direction_sign * p * math.pi * 2,
    scaleX = 1,
    scaleY = 1
  }
end
