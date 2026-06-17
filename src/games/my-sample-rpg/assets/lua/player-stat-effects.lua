-- TS playerStatEffects.ts → Lua. profile.stats(객체)를 받아 파생 스탯을 계산한다(JSON 마샬링).
-- 상수는 TS에서 인자로 전달(상수 소유는 TS, 공식만 Lua). shouldPlayerEvadeDamage의 난수 비교는
-- 난수값을 인자로 받는 형태라 TS 래퍼에서 처리한다(여기선 회피 확률만 제공).

local function clamp(value, min_value, max_value)
  if value < min_value then
    return min_value
  end
  if value > max_value then
    return max_value
  end
  return value
end

function player_physical_attack_power(profile)
  return math.max(1, math.floor(profile.stats.strength))
end

function player_movement_speed(profile, default_speed, agility_bonus_per_point, min_speed, max_speed)
  return clamp(
    default_speed + (profile.stats.agility - 4) * agility_bonus_per_point,
    min_speed,
    max_speed
  )
end

function player_evade_chance(profile, base_chance, bonus_per_point, max_chance)
  return clamp(base_chance + profile.stats.luck * bonus_per_point, 0, max_chance)
end
