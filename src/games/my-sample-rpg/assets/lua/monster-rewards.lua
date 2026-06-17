-- 몬스터 처치 보상 규칙 (TS monsterRewards.ts에서 Lua로 변환).
-- 순수 산술 — 호스트가 전역 함수를 monster_level 인자로 호출해 number를 받는다.
-- monsterRewards.ts와 동일: level = max(1, floor(monster_level)).

local GOLD_DROP_BASE = 10
local GOLD_DROP_PER_LEVEL = 4
local EXPERIENCE_DROP_BASE = 12
local EXPERIENCE_DROP_PER_LEVEL = 6
local SKILL_POINT_DROP_BASE = 0

local function clamp_level(monster_level)
  local level = math.floor(monster_level)
  if level < 1 then
    return 1
  end
  return level
end

function monster_rewards_gold(monster_level)
  return GOLD_DROP_BASE + clamp_level(monster_level) * GOLD_DROP_PER_LEVEL
end

function monster_rewards_experience(monster_level)
  return EXPERIENCE_DROP_BASE
    + clamp_level(monster_level) * EXPERIENCE_DROP_PER_LEVEL
end

function monster_rewards_skill_point(monster_level)
  return SKILL_POINT_DROP_BASE + clamp_level(monster_level)
end
