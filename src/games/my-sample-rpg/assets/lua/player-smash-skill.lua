-- TS playerSmashSkill.ts → Lua. 스매시 스킬의 세그먼트 배치 수학(방향 벡터 / 거리 / 배치).
-- 순수 수학이라 상수는 TS에서 인자로 전달(상수 소유는 TS, 공식만 Lua). 객체 입출력이라 JSON 마샬링.
-- facing 방향 스위치, 거리(min/max + 정수 제곱), 배치(sin(progress*pi)).

local function clamp01(value)
  if value < 0 then
    return 0
  end
  if value > 1 then
    return 1
  end
  return value
end

-- facing -> { x, y } 단위 벡터. (TS의 switch 기본값은 {0,0}.)
function player_smash_skill_direction_vector(facing)
  if facing == 'up' then
    return { x = 0, y = -1 }
  elseif facing == 'down' then
    return { x = 0, y = 1 }
  elseif facing == 'left' then
    return { x = -1, y = 0 }
  elseif facing == 'right' then
    return { x = 1, y = 0 }
  end
  return { x = 0, y = 0 }
end

-- 세그먼트 거리(px). progress는 0..1로 클램프, (1-(1-p)^2) 이징 곡선.
-- TS의 ** 2 는 double 곱이라 (1-p)*(1-p) 와 비트 동일.
function player_smash_skill_segment_distance_pixels(
  segment_index,
  progress,
  launch_offset,
  first_segment_distance,
  segment_spacing,
  travel_distance
)
  local normalized = clamp01(progress)
  local inverse = 1 - normalized
  return launch_offset
    + first_segment_distance
    + segment_index * segment_spacing
    + (1 - inverse * inverse) * travel_distance
end

-- 세그먼트 배치 결과 { x, y, scaleX, scaleY, alpha }.
function player_smash_skill_segment_placement(input, constants)
  local direction = player_smash_skill_direction_vector(input.facing)
  local normalized = clamp01(input.progress)
  local distance = player_smash_skill_segment_distance_pixels(
    input.segmentIndex,
    normalized,
    constants.launchOffset,
    constants.firstSegmentDistance,
    constants.segmentSpacing,
    constants.travelDistance
  )
  local pulse = math.sin(normalized * math.pi)
  local growth_multiplier = 1 + pulse * constants.scaleGrowth
  local base_scale_x = constants.baseScaleX + input.segmentIndex * constants.scaleStep
  local base_scale_y = constants.baseScaleY + input.segmentIndex * constants.scaleStep
  local fade_out_start = 0.78
  local alpha
  if normalized < fade_out_start then
    alpha = 1
  else
    local faded = 1 - (normalized - fade_out_start) / (1 - fade_out_start)
    alpha = faded > 0 and faded or 0
  end

  return {
    x = input.characterCenterX + direction.x * distance,
    y = input.characterCenterY + direction.y * distance,
    scaleX = base_scale_x * growth_multiplier,
    scaleY = base_scale_y * growth_multiplier,
    alpha = alpha
  }
end
