# legend-of-lua 퀘스트 계약 (에디터 ↔ 게임)

에디터(시나리오 공방)가 자연어로 만든 **단계별 퀘스트**를 실행 중인 legend-of-lua에 라이브로 적용하기 위한 규약이다. 대사 적용(`entity_lines`)과 같은 브리지 채널을 쓰되, `kind: "quest"`로 구분한다.

- 전송(에디터→게임): HTTP 브리지 `POST /apply` 또는 love.js iframe `postMessage('editor:apply')` — 둘 다 같은 페이로드. ([gameBridge.ts](../src/editor/gameBridge.ts), [legend-of-lua-bridge.lua](legend-of-lua-bridge.lua))
- 게임-쪽 처리: 드롭인 [quest-runtime.lua](quest-runtime.lua)가 등록·추적·완료·보상.

## 1. 브리지 메시지 (kind: "quest")

```jsonc
{
  "id": "quest-hunt_cave_slimes-1718200000000",
  "kind": "quest",
  "generatedAt": 1718200000000,
  "quest": {
    "quest_id": "hunt_cave_slimes",          // 영문 snake_case
    "title": "동굴의 슬라임 사냥",
    "giver_entity_id": "NPCs-12",            // 퀘스트를 주는 NPC(이 게임에 배치된 엔티티 id)
    "request_text": "...",
    "guide_text": "...",
    "dialogue": {
      "start":    ["수락 대사 ..."],
      "active":   ["진행 중 대사 ..."],
      "complete": ["완료 대사 ..."]
    },
    "objectives": [
      { "type": "defeat",  "label": "슬라임 처치", "required": 3, "target": { "entityId": "Enemies-105", "mapId": "testCave" } },
      { "type": "acquire", "label": "상자 열기",   "required": 1, "target": { "entityId": "Chests-12" } },
      { "type": "talk",    "label": "촌장과 대화", "required": 1, "target": { "entityId": "NPCs-3" } },
      { "type": "reach",   "label": "동굴 진입",   "required": 1, "target": { "mapId": "testCave" } }
    ],
    "rewards": { "gold": 100, "experience": 50, "items": [ { "label": "강철 검", "quantity": 1 } ] }
  }
}
```

- `target.entityId`는 에디터가 그 게임의 TMX에서 만든 `"{objectGroup}-{objectId}"`(예: `"Enemies-105"`) — 게임의 배치 엔티티와 같은 규칙.
- 목표 타입: `defeat`(처치) / `acquire`(획득) / `talk`(대화) / `reach`(맵 진입).
- 응답은 기존과 동일: `{ "ok": true }` 또는 `{ "ok": false, "error": "..." }`.

## 2. 게임-쪽 설치 (drop-in)

1. [quest-runtime.lua](quest-runtime.lua)를 게임 폴더에 `quest_runtime.lua`로 둔다.
2. `main.lua`에서 표시/보상 함수를 연결한다:
   ```lua
   local quest_runtime = require("quest_runtime")
   quest_runtime.configure({
     show = function(text) game.ui.show_message(text) end,
     give_reward = function(r) player.gold = player.gold + (r.gold or 0) end,
   })
   ```
3. 브리지 핸들러에서 'quest'를 등록한다([legend-of-lua-bridge.lua](legend-of-lua-bridge.lua) applyToGame):
   ```lua
   if message.kind == "quest" then return quest_runtime.register(message.quest) end
   ```

## 3. 게임이 연결해야 할 훅 4개 (마지막 연결)

런타임은 게임 내부를 모르므로, 게임이 자기 이벤트에서 아래를 호출해야 목표가 진행된다:

| 게임 이벤트 | 호출 |
|---|---|
| 몬스터 사망 | `quest_runtime.on_defeat(entityId)` |
| 아이템/상자 획득 | `quest_runtime.on_acquire(entityId)` |
| 맵 진입 | `quest_runtime.on_reach(mapId)` |
| NPC 대화 | `quest_runtime.on_talk(entityId)` (기버면 시작/완료·보상 자동) |

`entityId`는 위 `target.entityId`와 같은 값이어야 한다. 게임이 몬스터/오브젝트를 TMX의 `{group}-{id}`로 식별할 수 있으면 그대로 넘기면 된다.

## 4. 한계

- 이 계약/런타임은 **목표 추적·완료·보상**까지 제공한다. 단, "킬을 어떻게 감지하나"는 게임마다 달라 **훅 4개 연결은 게임 작성자 몫**이다(소스 필요).
- 보상 아이템은 라벨만 전달한다(게임이 자기 아이템 시스템에 매핑). 목표 타깃은 에디터가 그 게임에 **실재하는 배치 엔티티**만 검증해 보낸다.
