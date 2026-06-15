# 라이브 게임 브리지 프로토콜 (에디터 ↔ 외부 게임)

시나리오 에디터(브라우저)가 **실행 중인 외부 게임**(예: Love2D `legend-of-lua`)에 생성물을
라이브로 적용하기 위한 규약이다. `my-sample-rpg`는 같은 origin이라 localStorage로 적용하지만,
별도 프로세스로 도는 Lua/Love2D 게임은 localStorage를 공유할 수 없어 이 HTTP 채널을 쓴다.

- 에디터-쪽 구현: [`src/editor/gameBridge.ts`](../src/editor/gameBridge.ts)
- 게임-쪽 구현: **이 문서를 보고 게임 측에서 작성**(LÖVE는 `luasocket`이 번들로 들어 있어 작은 HTTP 서버를 띄울 수 있다)

---

## 1. 트랜스포트

- **게임이 로컬 HTTP 서버를 띄운다.** 기본 주소 `http://localhost:17320` (에디터 설정 ⚙ 에서 변경 가능).
- 에디터가 이 서버에 요청한다. 두 엔드포인트만 구현하면 된다:
  - `GET /status` — 연결 확인(에디터가 2초마다 폴링해 헤더의 연결 표시등을 갱신).
  - `POST /apply` — 생성물 1건을 받아 게임에 라이브 반영.

### CORS (중요)

에디터는 `http://localhost:5173`(Vite)에서 돌고, 게임 서버는 다른 포트라서 **교차 출처**다.
게임 서버는 모든 응답에 아래 헤더를 넣고, `POST`의 프리플라이트(`OPTIONS`)에 200으로 답해야 한다:

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: content-type
```

> CORS가 부담되면, 대안으로 Vite 프록시(`vite.config.ts`의 `server.proxy`)에 브리지 경로를
> 추가해 같은 origin으로 우회할 수도 있다. 그 경우 에디터의 브리지 URL을 프록시 경로로 설정한다.

---

## 2. 엔드포인트

### `GET /status`

게임이 살아 있으면 **HTTP 200**과 함께 JSON을 돌려준다(본문은 표시용, 200이면 "연결됨"으로 간주).

```jsonc
// 200 OK
{ "game": "legend-of-lua", "version": "0.1.0" }
```

게임이 안 떠 있으면 연결 자체가 실패하고, 에디터는 "게임 미연결"로 표시한다(자동 재시도).

### `POST /apply`

요청 본문(JSON) = `BridgeApplyMessage`:

```jsonc
{
  "id": "entity_lines-1718200000000",   // 적용 1건의 고유 id(아래 ack에 그대로 echo)
  "kind": "entity_lines",                // 메시지 종류(현재 1종, 앞으로 확장)
  "target": {                            // 적용 대상. 전역이면 null
    "id": "Enemies-105",                 // 에디터 엔티티 트리의 id (group-objectId 규칙)
    "name": "slime",
    "kind": "enemy",                     // enemy | npc | chest | loot ...
    "mapId": "testCave"                  // .tmx 파일명(확장자 제외)
  },
  "lines": [                             // 생성된 대사/설명 1~4줄
    "크르릉… 누구냐.",
    "이 동굴은 내 구역이다."
  ],
  "generatedAt": 1718200000000
}
```

응답(JSON):

```jsonc
// 200 OK — 적용 성공
{ "ok": true }

// 200 OK — 적용 실패(게임이 대상 못 찾음 등). 에디터가 error를 그대로 보여준다.
{ "ok": false, "error": "id=Enemies-105 인 적 엔티티를 찾지 못했습니다." }
```

- 본문이 비어 있어도(`Content-Length: 0`) 200이면 성공으로 간주한다.
- `ok`가 `false`거나 200이 아니면 에디터가 "적용 실패"로 표시한다.

---

## 3. `target` 매칭 규칙

`target.id`는 에디터가 TMX 객체에서 만든 id다([`gameAdapter.ts`](../src/editor/gameAdapter.ts)의
`legendOfLuaAdapter.extractEntities` 참고):

- 형식: `"{objectGroup}-{objectId}"` (예: `"Enemies-105"`).
- `objectGroup`은 TMX `<objectgroup name="...">`의 이름(`Enemies` / `NPCs` / `Chests` / `Loot`).
- `objectId`는 그 객체의 `<object id="...">`.

게임은 `target.mapId`(맵) + `target.id`(객체)를 그 맵의 실제 엔티티로 되짚어 대사/설명을
교체하거나 부여하면 된다. 매칭에 실패하면 `{ "ok": false, "error": "..." }`로 알린다.

---

## 4. LÖVE 측 구현 (드롭인 제공)

게임-쪽 HTTP 서버는 **그대로 넣어 쓰는 구현체**가 있다: [`docs/legend-of-lua-bridge.lua`](legend-of-lua-bridge.lua).

설치 3단계:

1. `legend-of-lua-bridge.lua`를 게임 폴더에 `bridge.lua`로 둔다.
2. JSON 라이브러리 [`rxi/json.lua`](https://github.com/rxi/json.lua)(단일 파일, MIT)를 `json.lua`로 같이 둔다.
3. `main.lua`에서 연결한다:

```lua
local bridge = require("bridge")

function love.load()
  bridge.start(17320, applyToGame)   -- applyToGame(message) → true | false,"사유"
end

function love.update(dt)
  bridge.update()                    -- 매 프레임 요청 처리(논블로킹)
end
```

`conf.lua`에서 socket 모듈이 켜져 있어야 한다(LÖVE 기본 켜짐): `t.modules.socket = true`.

> 핵심은 "에디터가 보낸 `lines`를 `target`이 가리키는 엔티티에 라이브로 적용"하는 것.
> 적용 자체(대사 부여 방식)는 `applyToGame`에서 게임 내부 구조에 맞춰 구현한다(예시는 `bridge.lua` 하단).

---

## 5. 확장

새 콘텐츠 종류가 생기면 `kind`를 추가하고(예: `"event"`, `"quest"`), 에디터 쪽
`BridgeApplyMessage`와 `gameAdapter`의 `bridgePayload` 생성부를 함께 늘린다. 게임은 모르는
`kind`를 받으면 `{ "ok": false, "error": "unsupported kind" }`로 답하면 된다.
