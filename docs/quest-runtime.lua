-- 에디터가 생성한 "퀘스트"를 실행 중인 legend-of-lua에서 등록·추적·완료시키는 게임-쪽 드롭인.
-- 계약 명세: docs/legend-of-lua-quest-contract.md  (브리지 전송: docs/legend-of-lua-bridge.lua)
--
-- 이 모듈은 게임 내부 구조를 모른다. 그래서 게임이 다음 4개의 훅을 자기 코드에서 호출해줘야
-- 퀘스트가 진행된다(아래 "게임이 연결해야 할 곳" 참고). 그 연결만 해주면 목표 추적·완료·보상은
-- 이 모듈이 처리한다.
--
-- ── 설치 ──
--   1) 이 파일을 게임 폴더에 `quest_runtime.lua`로 둔다.
--   2) main.lua에서 한 번 설정한다:
--        local quest_runtime = require("quest_runtime")
--        quest_runtime.configure({
--          show = function(text) -- 게임의 메시지/대사 UI 함수에 연결
--            game.ui.show_message(text)
--          end,
--          give_reward = function(rewards) -- 게임의 보상 지급에 연결
--            player.gold = player.gold + (rewards.gold or 0)
--            player.exp  = player.exp  + (rewards.experience or 0)
--            -- rewards.items = { { label=..., quantity=... }, ... }
--          end,
--        })
--   3) 브리지로 받은 퀘스트를 등록한다(bridge.lua의 applyToGame에서):
--        if message.kind == "quest" then return quest_runtime.register(message.quest) end
--
-- ── 게임이 연결해야 할 곳 (훅 4개) ──
--   몬스터가 죽을 때          → quest_runtime.on_defeat(entityId)
--   아이템/상자를 획득할 때   → quest_runtime.on_acquire(entityId)
--   맵에 진입할 때            → quest_runtime.on_reach(mapId)
--   NPC와 대화할 때           → quest_runtime.on_talk(entityId)  -- 기버면 시작/완료 처리
--   (entityId는 에디터가 보낸 target.entityId와 같은 값: "Enemies-105" 등 TMX "{group}-{objId}")

local quest_runtime = {}

-- questId -> { def = <bridge quest data>, status = "...", progress = { [objIndex] = count } }
local quests = {}
local config = {
  show = function(_) end,
  give_reward = function(_) end
}

function quest_runtime.configure(options)
  options = options or {}
  if options.show then config.show = options.show end
  if options.give_reward then config.give_reward = options.give_reward end
end

-- 에디터/브리지가 보낸 퀘스트를 등록한다(멱등: 같은 quest_id는 덮어쓴다). 반환은 bridge 계약과 동일.
function quest_runtime.register(quest)
  if type(quest) ~= "table" or type(quest.quest_id) ~= "string" then
    return false, "잘못된 퀘스트 데이터"
  end

  local progress = {}
  for index = 1, #(quest.objectives or {}) do
    progress[index] = 0
  end

  quests[quest.quest_id] = {
    def = quest,
    status = "not-started",
    progress = progress
  }
  return true
end

local function all_objectives_done(entry)
  for index, objective in ipairs(entry.def.objectives or {}) do
    if (entry.progress[index] or 0) < (objective.required or 1) then
      return false
    end
  end
  return true
end

-- 활성 퀘스트의 목표를 술어로 찾아 1씩 진행. 다 차면 ready-to-turn-in으로 바꾸고 안내한다.
local function advance(matches)
  for _, entry in pairs(quests) do
    if entry.status == "active" then
      for index, objective in ipairs(entry.def.objectives or {}) do
        if (entry.progress[index] or 0) < (objective.required or 1)
          and matches(objective) then
          entry.progress[index] = (entry.progress[index] or 0) + 1
          config.show(("%s (%d/%d)"):format(
            objective.label or "목표",
            entry.progress[index],
            objective.required or 1
          ))
          if all_objectives_done(entry) then
            entry.status = "ready-to-turn-in"
            config.show("목표 완료! 의뢰한 NPC에게 돌아가세요.")
          end
        end
      end
    end
  end
end

function quest_runtime.on_defeat(entityId)
  advance(function(o) return o.type == "defeat" and o.target and o.target.entityId == entityId end)
end

function quest_runtime.on_acquire(entityId)
  advance(function(o) return o.type == "acquire" and o.target and o.target.entityId == entityId end)
end

function quest_runtime.on_reach(mapId)
  advance(function(o) return o.type == "reach" and o.target and o.target.mapId == mapId end)
end

-- NPC 대화: 기버면 (미시작→시작 대사 / 완료가능→완료+보상), 그 외엔 talk 목표 진행.
function quest_runtime.on_talk(entityId)
  for _, entry in pairs(quests) do
    if entry.def.giver_entity_id == entityId then
      if entry.status == "not-started" then
        entry.status = "active"
        for _, line in ipairs(entry.def.dialogue and entry.def.dialogue.start or {}) do
          config.show(line)
        end
        return
      elseif entry.status == "ready-to-turn-in" then
        entry.status = "completed"
        for _, line in ipairs(entry.def.dialogue and entry.def.dialogue.complete or {}) do
          config.show(line)
        end
        config.give_reward(entry.def.rewards or {})
        return
      elseif entry.status == "active" then
        for _, line in ipairs(entry.def.dialogue and entry.def.dialogue.active or {}) do
          config.show(line)
        end
        return
      end
    end
  end

  advance(function(o) return o.type == "talk" and o.target and o.target.entityId == entityId end)
end

-- 표시용 조회(선택): 현재 퀘스트 상태 스냅샷.
function quest_runtime.get(questId)
  return quests[questId]
end

return quest_runtime
