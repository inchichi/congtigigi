-- 몬스터 순찰 상태 (TS monsterPatrol.ts에서 Lua로 변환).
-- 변환 대상은 createMonsterPatrolState뿐 — 순수 초기화 함수다.
-- (stepMonsterPatrol은 가변 횟수의 Math.random 추첨에 의존해 동등 검증이 불가능하므로 TS에 남긴다.)
--
-- character.position.x/y를 평범한 객체로 복사하고 pauseUntilMilliseconds=0으로 둔다.
-- 객체 인자/반환이라 호스트의 callJson(__logic_call_json)으로 마샬링한다.

function monster_patrol_create_state(character)
  local x = character.position.x
  local y = character.position.y

  return {
    homePosition = { x = x, y = y },
    targetPosition = { x = x, y = y },
    pauseUntilMilliseconds = 0
  }
end
