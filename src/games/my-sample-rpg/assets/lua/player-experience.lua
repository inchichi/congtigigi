-- 플레이어 레벨업 경험치 규칙 (TS playerExperience.ts에서 Lua로 변환).
-- 순수 산술 — 호스트가 전역 함수를 level/상수 인자로 호출해 number를 받는다.
-- playerExperience.ts와 동일: normalized = max(1, floor(level)).
-- 상수(max_level, base_exp, per_level_exp)는 TS가 보유하고 인자로 넘긴다 — Lua는 공식만 갖는다.

function player_experience_to_next_level(level, max_level, base_exp, per_level_exp)
  local normalized_level = math.max(1, math.floor(level))

  if normalized_level >= max_level then
    return 0
  end

  return base_exp + (normalized_level - 1) * per_level_exp
end

-- profile 의 모든 키를 얕게 복사(중첩 객체는 새로 만든다 — TS spread 패턴).
local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- ── grantPlayerExperience ── (객체 in/out: callJson 경로)
-- 레벨업 보상은 progression_grant_level_up_rewards(다른 모듈 전역)에 위임한다 — 이 호스트에
-- player-progression.lua 가 이미 로드돼 있어야 한다(standalone 래퍼가 의존성으로 로드).
-- 상수 소유는 TS: max_level/base_exp/per_level_exp 는 경험치 공식용,
-- level_up_stat_points/level_up_hp_bonus 는 레벨업 보상용으로 모두 인자로 전달한다.
function player_experience_grant(
  profile,
  experience,
  max_level,
  base_exp,
  per_level_exp,
  level_up_stat_points,
  level_up_hp_bonus
)
  local gained_experience = math.max(0, math.floor(experience))

  if gained_experience == 0 then
    if profile.level >= max_level then
      local next_profile = shallow_copy(profile)
      next_profile.experience = { current = 0 }
      return { nextProfile = next_profile, levelsGained = 0, gainedExperience = 0 }
    end

    return { nextProfile = profile, levelsGained = 0, gainedExperience = 0 }
  end

  if profile.level >= max_level then
    local next_profile = shallow_copy(profile)
    next_profile.experience = { current = 0 }
    return { nextProfile = next_profile, levelsGained = 0, gainedExperience = gained_experience }
  end

  local next_profile = profile
  local remaining_experience = profile.experience.current + gained_experience
  local levels_gained = 0

  while next_profile.level < max_level do
    local required_experience =
      player_experience_to_next_level(next_profile.level, max_level, base_exp, per_level_exp)

    if remaining_experience < required_experience then
      break
    end

    remaining_experience = remaining_experience - required_experience
    next_profile = progression_grant_level_up_rewards(
      next_profile,
      1,
      max_level,
      level_up_stat_points,
      level_up_hp_bonus
    )
    levels_gained = levels_gained + 1
  end

  if next_profile.level >= max_level then
    remaining_experience = 0
  end

  local result_profile = shallow_copy(next_profile)
  result_profile.experience = { current = remaining_experience }

  return {
    nextProfile = result_profile,
    levelsGained = levels_gained,
    gainedExperience = gained_experience
  }
end
