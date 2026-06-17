-- TS playerProfile.ts → Lua. 순수 헬퍼만 변환(createInitialPlayerProfile은 데이터 구성이라 TS 유지).
-- 작은 고정 룩업/상수는 여기 하드코딩(값은 TS와 동일하게 유지).
-- string|undefined 반환 함수는 undefined일 때 json_null을 돌려주고, TS 래퍼가 null→undefined로 정규화한다.

-- 상수(TS PLAYER_JOB_PROMOTION_LEVEL / PLAYER_MAX_LEVEL와 동일).
local PLAYER_JOB_PROMOTION_LEVEL = 10
local PLAYER_MAX_LEVEL = 100

-- TS PLAYER_JOB_PRIMARY_STAT_ID_BY_NAME와 동일한 고정 맵.
local PLAYER_JOB_PRIMARY_STAT_ID_BY_NAME = {
  ['전사'] = 'strength',
  ['궁수'] = 'agility',
  ['마법사'] = 'intelligence',
  ['도적'] = 'luck'
}

-- TS PLAYER_STAT_LABEL_BY_ID와 동일한 고정 맵.
local PLAYER_STAT_LABEL_BY_ID = {
  strength = '힘',
  agility = '민첩',
  intelligence = '지력',
  luck = '행운'
}

function player_profile_job_display_name(profile)
  if profile.level >= PLAYER_JOB_PROMOTION_LEVEL then
    return profile.job .. ' · 전직 가능'
  end
  return profile.job
end

function player_profile_job_promotion_available(profile)
  return profile.level >= PLAYER_JOB_PROMOTION_LEVEL
end

function player_profile_at_max_level(profile)
  return profile.level >= PLAYER_MAX_LEVEL
end

function player_profile_job_primary_stat_id(job)
  local stat_id = PLAYER_JOB_PRIMARY_STAT_ID_BY_NAME[job]
  if stat_id == nil then
    return json_null
  end
  return stat_id
end

function player_profile_job_primary_stat_label(job)
  local stat_id = PLAYER_JOB_PRIMARY_STAT_ID_BY_NAME[job]
  if stat_id == nil then
    return json_null
  end
  return PLAYER_STAT_LABEL_BY_ID[stat_id]
end
