-- 에디터 ↔ legend-of-lua 라이브 브리지 (게임-쪽, LÖVE 드롭인 구현)
-- 프로토콜 명세: docs/legend-of-lua-bridge-protocol.md
--
-- 이 파일을 legend-of-lua 게임에 그대로 넣으면, 실행 중인 게임이 로컬 HTTP 서버를 열어
-- 에디터가 생성한 대사/설명을 라이브로 받아 적용한다.
--
-- ── 설치 (3단계) ──
--   1) 이 파일을 게임 폴더에 `bridge.lua`로 둔다.
--   2) JSON 라이브러리를 `json.lua`로 같이 둔다. (rxi/json.lua, MIT, 단일 파일:
--      https://github.com/rxi/json.lua → json.lua 하나만 받아 게임 폴더에 둔다)
--   3) main.lua에서 아래처럼 연결한다:
--
--        local bridge = require("bridge")
--
--        function love.load()
--          -- 두 번째 인자(applyToGame)가 실제 "적용" 로직. 게임 구조에 맞게 구현한다.
--          bridge.start(17320, applyToGame)
--        end
--
--        function love.update(dt)
--          bridge.update()   -- 매 프레임 새 요청을 논블로킹으로 처리
--        end
--
--   conf.lua에서 socket 모듈이 켜져 있어야 한다(LÖVE 기본값은 켜짐):
--        function love.conf(t) t.modules.socket = true end
--
-- ── applyToGame(message) 계약 ──
--   message = {
--     id = "entity_lines-...", kind = "entity_lines",
--     target = { id="Enemies-105", name="slime", kind="enemy", mapId="testCave" } 또는 nil,
--     lines = { "첫 줄", "둘째 줄" }, generatedAt = 1718200000000,
--   }
--   반환: true            → 적용 성공
--         false, "사유"   → 적용 실패(에디터가 사유를 그대로 보여줌)
--
--   target.id는 "{objectGroup}-{objectId}" 규칙(예: "Enemies-105")이다. 게임은 target.mapId(맵)와
--   target.id로 실제 엔티티를 되짚어 message.lines를 대사/설명으로 부여하면 된다.

local socket = require("socket")
local json = require("json")

local bridge = {}

local server
local onApply

local CORS_HEADERS =
  "Access-Control-Allow-Origin: *\r\n" ..
  "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ..
  "Access-Control-Allow-Headers: content-type\r\n"

-- port: 게임 HTTP 서버가 들을 포트(에디터 설정의 "게임 브리지 URL" 포트와 같아야 함).
-- 기본 17320. 다른 프로그램과 겹치면 다른 빈 포트로 바꾸고 에디터 설정도 같이 바꾼다.
function bridge.start(port, applyCallback)
  onApply = applyCallback
  server = assert(socket.bind("127.0.0.1", port or 17320))
  server:settimeout(0) -- 논블로킹: accept가 즉시 반환
end

function bridge.stop()
  if server then
    server:close()
    server = nil
  end
end

local function sendResponse(client, status, body)
  local payload = body or ""
  client:send(
    ("HTTP/1.1 %s\r\n"):format(status) ..
    CORS_HEADERS ..
    "Content-Type: application/json\r\n" ..
    ("Content-Length: %d\r\n"):format(#payload) ..
    "Connection: close\r\n\r\n" ..
    payload
  )
end

local function readRequest(client)
  local requestLine = client:receive("*l")
  if not requestLine then
    return nil
  end

  local method, path = requestLine:match("^(%u+)%s+(%S+)")

  -- 헤더를 읽어 Content-Length만 챙긴다(빈 줄까지).
  local contentLength = 0
  while true do
    local line = client:receive("*l")
    if not line or line == "" then
      break
    end
    local key, value = line:match("^([%w%-]+):%s*(.+)$")
    if key and key:lower() == "content-length" then
      contentLength = tonumber(value) or 0
    end
  end

  local body = ""
  if contentLength > 0 then
    body = client:receive(contentLength) or "" -- 정확히 N바이트(UTF-8 한글 포함)
  end

  return method, path, body
end

function bridge.update()
  if not server then
    return
  end

  local client = server:accept()
  if not client then
    return -- 이번 프레임엔 새 연결 없음
  end

  client:settimeout(1)
  local method, path, body = readRequest(client)

  if method == "OPTIONS" then
    -- CORS 프리플라이트
    sendResponse(client, "204 No Content", "")
  elseif method == "GET" and path == "/status" then
    sendResponse(client, "200 OK", json.encode({ game = "legend-of-lua", version = "0.1.0" }))
  elseif method == "POST" and path == "/apply" then
    local ok, message = pcall(json.decode, body)
    if not ok or type(message) ~= "table" then
      sendResponse(client, "200 OK", json.encode({ ok = false, error = "잘못된 JSON 본문" }))
    else
      local applied, err = true, nil
      if onApply then
        applied, err = onApply(message)
      end
      sendResponse(client, "200 OK", json.encode({ ok = applied ~= false, error = err }))
    end
  else
    sendResponse(client, "404 Not Found", json.encode({ ok = false, error = "알 수 없는 경로" }))
  end

  client:close()
end

return bridge

--[[ ── applyToGame 예시 (게임 구조에 맞게 고쳐 쓰기) ──

-- 맵별 엔티티 레지스트리가 { [mapId] = { [entityId] = entity } } 형태라고 가정.
function applyToGame(message)
  if message.kind ~= "entity_lines" then
    return false, "지원하지 않는 종류: " .. tostring(message.kind)
  end

  local target = message.target
  if not target then
    return false, "적용 대상이 없습니다."
  end

  local mapEntities = MAP_ENTITIES[target.mapId]
  local entity = mapEntities and mapEntities[target.id]
  if not entity then
    return false, ("대상을 찾지 못함: %s @ %s"):format(target.id, target.mapId)
  end

  entity.dialogue = message.lines  -- 라이브 반영
  return true
end
--]]
