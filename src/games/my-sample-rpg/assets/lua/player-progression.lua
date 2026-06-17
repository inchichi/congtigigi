-- TS playerProgression.ts → Lua. PlayerProfile(객체)를 받아 레벨업/스탯/스킬 보상 규칙을 계산한다(JSON 마샬링).
-- 상수 소유는 TS(공식만 Lua): PLAYER_MAX_LEVEL, PLAYER_LEVEL_UP_STAT_POINTS, PLAYER_LEVEL_UP_HP_BONUS,
-- PLAYER_BASE_INTELLIGENCE_STAT, PLAYER_INTELLIGENCE_MP_BONUS_PER_POINT 는 모두 인자로 전달된다.
-- 예외: PLAYER_SKILL_USER_LEVEL_BASE_MANA_BY_LEVEL 은 TS에서 private인 작고 고정된 맵이라 여기 하드코딩한다
--   ({1:12,2:16,3:20,4:24,5:28}, level>5 는 28 + (level-5)*4). TS 동작과 정확히 일치.
-- skill_index 는 JS 0-based, Lua skills 배열은 1-based. spend_* 의 실패(undefined)는 json_null 로 반환한다.

-- profile 의 모든 키를 얕게 복사한다(중첩 객체는 호출부에서 새로 만든다 — TS의 spread 패턴).
local function shallow_copy(t)
  local out = {}
  for k, v in pairs(t) do
    out[k] = v
  end
  return out
end

-- ── getPlayerSkillPointCost ──
function progression_skill_point_cost(skill)
  if skill.level >= skill.maxLevel then
    return 0
  end
  if skill.level <= 1 then
    return 2
  end
  if skill.level == 2 then
    return 3
  end
  return 4
end

-- ── getPlayerSkillLevelLabel ──
function progression_skill_level_label(skill)
  if skill.level >= skill.maxLevel then
    return 'MAX'
  end
  return skill.level .. ' / ' .. skill.maxLevel
end

-- ── getPlayerSkillUserLevel ──
function progression_skill_user_level(total_skill_points_earned)
  return math.max(1, math.floor(total_skill_points_earned) + 1)
end

-- ── getPlayerMaxManaBySkillUserLevel ── (작은 고정 맵 하드코딩)
local SKILL_USER_LEVEL_BASE_MANA_BY_LEVEL = { [1] = 12, [2] = 16, [3] = 20, [4] = 24, [5] = 28 }

function progression_max_mana_by_skill_user_level(skill_user_level)
  local normalized_level = math.max(1, math.floor(skill_user_level))
  local base = SKILL_USER_LEVEL_BASE_MANA_BY_LEVEL[normalized_level]
  if base ~= nil then
    return base
  end
  return SKILL_USER_LEVEL_BASE_MANA_BY_LEVEL[5] + (normalized_level - 5) * 4
end

-- ── getPlayerMaxManaForProfile ──
-- base_intelligence_stat, intelligence_mp_bonus_per_point 는 TS 상수에서 인자로 전달.
function progression_max_mana_for_profile(profile, base_intelligence_stat, intelligence_mp_bonus_per_point)
  local skill_user_level = progression_skill_user_level(profile.totalSkillPointsEarned)
  return progression_max_mana_by_skill_user_level(skill_user_level)
    + math.max(0, profile.stats.intelligence - base_intelligence_stat) * intelligence_mp_bonus_per_point
end

-- 내부: syncPlayerManaFromSkillProgress — mp.max 가 바뀌면 늘어난 만큼 current 도 올린다.
local function sync_mana_from_skill_progress(profile, base_intelligence_stat, intelligence_mp_bonus_per_point)
  local next_mp_max =
    progression_max_mana_for_profile(profile, base_intelligence_stat, intelligence_mp_bonus_per_point)

  if next_mp_max == profile.mp.max then
    return profile
  end

  local next_mp_current = math.min(
    next_mp_max,
    profile.mp.current + math.max(0, next_mp_max - profile.mp.max)
  )

  local out = shallow_copy(profile)
  out.mp = { current = next_mp_current, max = next_mp_max }
  return out
end

-- ── grantPlayerLevelUpRewards ──
function progression_grant_level_up_rewards(
  profile,
  levels,
  max_level,
  level_up_stat_points,
  level_up_hp_bonus
)
  local next_levels = math.max(0, math.floor(levels))
  local applied_levels = math.min(next_levels, math.max(0, max_level - profile.level))

  if applied_levels == 0 then
    return profile
  end

  local next_hp_max = profile.hp.max + applied_levels * level_up_hp_bonus

  local out = shallow_copy(profile)
  out.level = profile.level + applied_levels
  out.statPoints = profile.statPoints + applied_levels * level_up_stat_points
  out.hp = { current = next_hp_max, max = next_hp_max }
  return out
end

-- ── grantPlayerSkillPoints ──
function progression_grant_skill_points(
  profile,
  gained_skill_points,
  base_intelligence_stat,
  intelligence_mp_bonus_per_point
)
  local normalized_skill_points = math.max(0, math.floor(gained_skill_points))

  if normalized_skill_points == 0 then
    return profile
  end

  local next_profile = shallow_copy(profile)
  next_profile.availableSkillPoints = profile.availableSkillPoints + normalized_skill_points
  next_profile.totalSkillPointsEarned = profile.totalSkillPointsEarned + normalized_skill_points

  return sync_mana_from_skill_progress(
    next_profile,
    base_intelligence_stat,
    intelligence_mp_bonus_per_point
  )
end

-- ── spendPlayerStatPoint ── (실패 시 json_null)
function progression_spend_stat_point(
  profile,
  stat_id,
  base_intelligence_stat,
  intelligence_mp_bonus_per_point
)
  if profile.statPoints <= 0 then
    return json_null
  end

  -- stats 는 새 테이블로(중첩 spread). 해당 stat_id 만 +1.
  local next_stats = shallow_copy(profile.stats)
  next_stats[stat_id] = profile.stats[stat_id] + 1

  local next_profile = shallow_copy(profile)
  next_profile.stats = next_stats

  -- TS: mp 는 intelligence 일 때만 sync 후 mp, 아니면 원래 profile.mp.
  local result = shallow_copy(next_profile)
  result.statPoints = profile.statPoints - 1
  if stat_id == 'intelligence' then
    local synced = sync_mana_from_skill_progress(
      next_profile,
      base_intelligence_stat,
      intelligence_mp_bonus_per_point
    )
    result.mp = synced.mp
  else
    result.mp = profile.mp
  end
  return result
end

-- ── spendPlayerSkillPoint ── (실패 시 json_null) skill_index 0-based.
function progression_spend_skill_point(profile, skill_index)
  local skill = profile.skills[skill_index + 1]

  if skill == nil or skill == json_null then
    return json_null
  end

  local skill_point_cost = progression_skill_point_cost(skill)

  if skill_point_cost <= 0 or profile.availableSkillPoints < skill_point_cost then
    return json_null
  end

  -- skills 배열을 새로 만들되, 대상 인덱스만 level+1 한 새 스킬로 교체(TS의 map).
  local next_skills = json_array({})
  for i = 1, #profile.skills do
    if i == skill_index + 1 then
      local updated = shallow_copy(profile.skills[i])
      updated.level = profile.skills[i].level + 1
      next_skills[i] = updated
    else
      next_skills[i] = profile.skills[i]
    end
  end

  local out = shallow_copy(profile)
  out.availableSkillPoints = profile.availableSkillPoints - skill_point_cost
  out.skills = next_skills
  return out
end
