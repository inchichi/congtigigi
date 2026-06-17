-- 몬스터 장비 드롭 인덱스 규칙 (TS monsterEquipmentDrops.ts에서 Lua로 변환).
-- 순수 산술 — 호스트가 (드롭 확률 통과 후) index_roll과 정의 개수를 넘겨 드롭 인덱스를 받는다.
-- monsterEquipmentDrops.ts와 동일: idx = min(count - 1, floor(index_roll * count)).
-- 확률 게이트와 정의 배열은 TS 래퍼에 남는다(이 모듈은 인덱스 수학만).

function monster_equipment_drop_index(index_roll, count)
  local idx = math.floor(index_roll * count)
  if idx > count - 1 then
    idx = count - 1
  end
  return idx
end
