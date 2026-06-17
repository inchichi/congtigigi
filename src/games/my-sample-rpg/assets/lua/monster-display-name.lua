-- 몬스터 표시 이름 규칙 (TS monsterDisplayName.ts에서 Lua로 변환).
-- 순수 문자열 — 호스트가 전역 함수를 (id, display_text) 인자로 호출해 string을 받는다.
-- monsterDisplayName.ts와 동일: display_text가 비어있지 않으면 그대로, 아니면 마지막 '-'
-- 앞부분(첫 글자가 아닌 위치에 '-'가 있을 때)으로 자른다. display_text 미지정은 ''로 들어온다.

function monster_display_name(id, display_text)
  if display_text ~= nil and display_text ~= '' then
    return display_text
  end

  -- TS lastIndexOf('-') (0-based). 없으면 -1, 있으면 0-based 인덱스.
  -- Lua는 1-indexed이므로 마지막 '-'의 1-based 위치를 찾아 0-based로 환산한다.
  local separator_position = nil
  local search_from = 1

  while true do
    local found = string.find(id, '-', search_from, true)
    if found == nil then
      break
    end
    separator_position = found
    search_from = found + 1
  end

  -- separator_position가 nil이면 TS의 lastIndexOf == -1 (구분자 없음).
  -- TS의 0-based index = separator_position - 1. index > 0 조건은 0-based가 1 이상,
  -- 즉 1-based 위치가 2 이상(첫 글자가 아님)이라는 뜻.
  if separator_position ~= nil and separator_position > 1 then
    -- TS id.slice(0, index) — 0-based index 앞까지. 1-based로는 [1, position - 1].
    return string.sub(id, 1, separator_position - 1)
  end

  return id
end
