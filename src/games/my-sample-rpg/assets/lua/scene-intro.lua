-- 씬 인트로 메시지 규칙 (TS sceneIntro.ts에서 Lua로 변환).
-- 순수 문자열 — 호스트가 전역 함수를 (scene_id) 인자로 호출해 string을 받는다.
-- sceneIntro.ts와 동일: 알려진 sceneId면 해당 메시지, 아니면 빈 문자열을 돌려준다.

local SCENE_INTRO_MESSAGES = {
  ['town'] = '티르코네일 마을',
  ['hunting-ground'] = '슬라임 숲',
  ['cave'] = '동굴'
}

function scene_intro_message(scene_id)
  -- TS: map[sceneId] ?? '' — 없는 키는 nil → ''.
  return SCENE_INTRO_MESSAGES[scene_id] or ''
end
