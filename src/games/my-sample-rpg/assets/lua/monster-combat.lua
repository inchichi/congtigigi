-- 몬스터 전투 상태 규칙 (TS monsterCombat.ts에서 Lua로 변환).
-- 순수 산술 — 호스트가 전역 함수를 원시값 인자로 호출해 number를 받는다.
-- monsterCombat.ts와 동일: level = max(1, floor(monster_level)).

local MONSTER_BASE_HP = 10
local MONSTER_HP_PER_LEVEL = 2

function monster_combat_create(monster_level, hp_multiplier, damage_multiplier)
  local level = math.max(1, math.floor(monster_level))
  local hp_mult = math.max(1, hp_multiplier)
  local dmg_mult = math.max(1, damage_multiplier)
  local max_hp = (MONSTER_BASE_HP + level * MONSTER_HP_PER_LEVEL) * hp_mult
  local contact_damage = math.max(1, math.ceil(level / 2)) * dmg_mult

  return max_hp, max_hp, contact_damage
end

function monster_combat_apply_damage(current_hp, damage)
  local next_damage = math.max(0, math.floor(damage))

  if next_damage == 0 or current_hp == 0 then
    return current_hp
  end

  return math.max(0, current_hp - next_damage)
end

function monster_combat_is_defeated(current_hp)
  if current_hp <= 0 then
    return 1
  end
  return 0
end
