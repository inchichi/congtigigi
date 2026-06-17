-- TS playerSkills.ts → Lua (순수 로직 6개 함수). Vite 에셋 URL이 필요한 display info는 제외.
-- profile.skills 는 {level, maxLevel, ...} 객체 배열 (JS 0-based 인덱스).
-- 결과는 callJson 경유(JSON 마샬링): boolean → true/false, 미발견 number → json_null.

local PLAYER_SMASH_SKILL_ID    = 'smash'
local PLAYER_PROTECT_SKILL_ID  = 'protect'
local PLAYER_SKILL_UNLOCK_LEVEL = 1

-- skillId → profile.skills 배열 인덱스 (0-based, Lua 내부에선 +1 해서 접근)
local PROFILE_INDEX_BY_ID = {
  smash   = 0,
  protect = 1,
}

local SMASH_MANA_COST_BY_LEVEL = {
  [1] = 4, [2] = 5, [3] = 6, [4] = 7, [5] = 8,
}

local SMASH_DAMAGE_BY_LEVEL = {
  [1] = 10, [2] = 14, [3] = 18, [4] = 23, [5] = 28,
}

local PROTECT_DURATION_BY_LEVEL = {
  [1] = 1800, [2] = 2200, [3] = 2600, [4] = 3000, [5] = 3400,
}

-- ── 내부 헬퍼 ──

local function smash_damage_by_level(skill_level)
  local nl = math.max(1, math.floor(skill_level))
  local v = SMASH_DAMAGE_BY_LEVEL[nl]
  if v ~= nil then return v end
  return SMASH_DAMAGE_BY_LEVEL[5] + (nl - 5) * 5
end

local function smash_mana_cost_by_level(skill_level)
  local nl = math.max(1, math.floor(skill_level))
  local v = SMASH_MANA_COST_BY_LEVEL[nl]
  if v ~= nil then return v end
  return SMASH_MANA_COST_BY_LEVEL[5] + (nl - 5)
end

-- ── 공개 함수 ──

-- getPlayerSkillLevelById: number or json_null
function player_skills_get_level(profile, skill_id)
  local profile_index = PROFILE_INDEX_BY_ID[skill_id]
  if profile_index == nil then
    return json_null
  end
  local skill = profile.skills[profile_index + 1]
  if skill == nil or skill == json_null then
    return json_null
  end
  return math.min(skill.level, skill.maxLevel)
end

-- isPlayerSkillUnlockedInProfile: boolean
function player_skills_is_unlocked(profile, skill_id)
  local profile_index = PROFILE_INDEX_BY_ID[skill_id]
  if profile_index == nil then
    return false
  end
  local skill = profile.skills[profile_index + 1]
  if skill == nil or skill == json_null then
    return false
  end
  return skill.level >= PLAYER_SKILL_UNLOCK_LEVEL
end

-- getPlayerSkillDamageById: number (0 if not smash or unknown)
function player_skills_damage(profile, skill_id)
  local level = player_skills_get_level(profile, skill_id)
  if level == json_null then
    return 0
  end
  if skill_id ~= PLAYER_SMASH_SKILL_ID then
    return 0
  end
  return smash_damage_by_level(level)
end

-- getPlayerSkillDamageBonusById: same as damage
function player_skills_damage_bonus(profile, skill_id)
  return player_skills_damage(profile, skill_id)
end

-- getPlayerSkillManaCostById: number
function player_skills_mana_cost(profile, skill_id)
  local level = player_skills_get_level(profile, skill_id)
  if level == json_null then
    return 0
  end
  if skill_id == PLAYER_SMASH_SKILL_ID then
    return smash_mana_cost_by_level(level)
  end
  if skill_id == PLAYER_PROTECT_SKILL_ID then
    return 2
  end
  return 0
end

-- getPlayerProtectSkillDurationByLevel: number
function player_skills_protect_duration(skill_level)
  local nl = math.max(1, math.floor(skill_level))
  local v = PROTECT_DURATION_BY_LEVEL[nl]
  if v ~= nil then return v end
  return PROTECT_DURATION_BY_LEVEL[5] + (nl - 5) * 400
end
