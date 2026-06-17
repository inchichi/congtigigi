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
