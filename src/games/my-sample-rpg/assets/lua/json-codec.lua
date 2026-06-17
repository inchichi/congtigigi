-- 최소 JSON 코덱 + 로직 디스패처 (TS <-> Lua 구조적 마샬링용).
-- 게임 로직 마샬링에 필요한 타입만 지원: null, boolean, number, string,
-- array(순차 정수 키 테이블), object(문자열 키 테이블). 중첩 가능.
-- 빈 배열 모호성: 빈 테이블은 기본 object({})로 인코딩하므로, 비어도 배열이어야 하는 값은
-- json_array(...)로 감싼다(디코더는 모든 배열을 json_array로 표시해 왕복이 안정적).

local json = {}

-- 배열의 null 구멍(TS undefined→JSON null)을 보존하기 위한 센티넬.
-- Lua nil은 "없음"(키/슬롯 사라짐)이라 배열 길이가 깨지므로, JSON null은 이 표시값으로 디코드한다.
-- 모듈 코드는 `slot == json_null`로 빈 슬롯을 판정한다.
local JSON_NULL = setmetatable({}, { __json = 'null' })
json_null = JSON_NULL

-- ── 인코드 ──
local escape_map = {
  ['"'] = '\\"', ['\\'] = '\\\\', ['\b'] = '\\b',
  ['\f'] = '\\f', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t'
}

local function escape_char(c)
  return escape_map[c] or string.format('\\u%04x', string.byte(c))
end

local function encode_string(s)
  return '"' .. s:gsub('[%c"\\]', escape_char) .. '"'
end

local function encode_number(n)
  if n ~= n or n == math.huge or n == -math.huge then
    error('cannot encode non-finite number')
  end
  if math.type(n) == 'integer' then
    return string.format('%d', n)
  end
  -- 정수값 float는 JS number처럼 정수로 표기(5.0 -> "5").
  if n == math.floor(n) and math.abs(n) < 1e15 then
    return string.format('%d', n)
  end
  -- %.17g는 double을 정확히 왕복시킨다(JS Number로 파싱하면 동일 비트).
  return string.format('%.17g', n)
end

local function is_array(t)
  local mt = getmetatable(t)
  if mt and mt.__json == 'array' then return true end
  if mt and mt.__json == 'object' then return false end
  local count = 0
  for k in pairs(t) do
    if type(k) ~= 'number' then return false end
    count = count + 1
  end
  if count == 0 then return false end -- 표시 없는 빈 테이블은 object {}
  return count == #t
end

local encode_value

local function encode_table(t)
  if is_array(t) then
    local parts = {}
    for i = 1, #t do
      parts[i] = encode_value(t[i])
    end
    return '[' .. table.concat(parts, ',') .. ']'
  end
  local parts = {}
  for k, v in pairs(t) do
    parts[#parts + 1] = encode_string(tostring(k)) .. ':' .. encode_value(v)
  end
  return '{' .. table.concat(parts, ',') .. '}'
end

encode_value = function(v)
  if rawequal(v, JSON_NULL) then
    return 'null'
  end
  local tv = type(v)
  if v == nil then
    return 'null'
  elseif tv == 'boolean' then
    return v and 'true' or 'false'
  elseif tv == 'number' then
    return encode_number(v)
  elseif tv == 'string' then
    return encode_string(v)
  elseif tv == 'table' then
    return encode_table(v)
  end
  error('cannot encode lua type ' .. tv)
end

function json.encode(v)
  return encode_value(v)
end

-- 비어도 배열로 인코딩되게 표시. 모듈이 배열을 만들 때 사용(특히 빈 배열).
function json_array(t)
  return setmetatable(t or {}, { __json = 'array' })
end

-- ── 디코드 (재귀 하강) ──
local parse, parse_string, parse_array, parse_object

local function next_nonws(s, i)
  while i <= #s do
    local c = s:sub(i, i)
    if c ~= ' ' and c ~= '\t' and c ~= '\n' and c ~= '\r' then
      return i
    end
    i = i + 1
  end
  return i
end

local unescape_map = {
  ['"'] = '"', ['\\'] = '\\', ['/'] = '/', ['b'] = '\b',
  ['f'] = '\f', ['n'] = '\n', ['r'] = '\r', ['t'] = '\t'
}

parse_string = function(s, i)
  i = i + 1 -- '"' 건너뛰기
  local parts = {}
  while i <= #s do
    local c = s:sub(i, i)
    if c == '"' then
      return table.concat(parts), i + 1
    elseif c == '\\' then
      local e = s:sub(i + 1, i + 1)
      if e == 'u' then
        local code = tonumber(s:sub(i + 2, i + 5), 16)
        parts[#parts + 1] = utf8.char(code)
        i = i + 6
      else
        parts[#parts + 1] = unescape_map[e] or e
        i = i + 2
      end
    else
      parts[#parts + 1] = c
      i = i + 1
    end
  end
  error('json decode: unterminated string')
end

local function parse_number(s, i)
  local token = s:match('^%-?%d+%.?%d*[eE]?[%-%+]?%d*', i)
  if not token or token == '' then
    error('json decode: invalid number at ' .. i)
  end
  return tonumber(token), i + #token
end

parse_array = function(s, i)
  i = i + 1 -- '[' 건너뛰기
  local arr = json_array({})
  i = next_nonws(s, i)
  if s:sub(i, i) == ']' then
    return arr, i + 1
  end
  while true do
    local value
    value, i = parse(s, i)
    arr[#arr + 1] = value
    i = next_nonws(s, i)
    local c = s:sub(i, i)
    if c == ',' then
      i = i + 1
    elseif c == ']' then
      return arr, i + 1
    else
      error('json decode: expected , or ] at ' .. i)
    end
  end
end

parse_object = function(s, i)
  i = i + 1 -- '{' 건너뛰기
  local obj = {}
  i = next_nonws(s, i)
  if s:sub(i, i) == '}' then
    return obj, i + 1
  end
  while true do
    i = next_nonws(s, i)
    local key
    key, i = parse_string(s, i)
    i = next_nonws(s, i)
    if s:sub(i, i) ~= ':' then
      error('json decode: expected : at ' .. i)
    end
    local value
    value, i = parse(s, i + 1)
    obj[key] = value
    i = next_nonws(s, i)
    local c = s:sub(i, i)
    if c == ',' then
      i = i + 1
    elseif c == '}' then
      return obj, i + 1
    else
      error('json decode: expected , or } at ' .. i)
    end
  end
end

parse = function(s, i)
  i = next_nonws(s, i)
  local c = s:sub(i, i)
  if c == '{' then
    return parse_object(s, i)
  elseif c == '[' then
    return parse_array(s, i)
  elseif c == '"' then
    return parse_string(s, i)
  elseif c == 't' then
    return true, i + 4 -- true
  elseif c == 'f' then
    return false, i + 5 -- false
  elseif c == 'n' then
    return JSON_NULL, i + 4 -- null → 센티넬(배열 구멍 보존)
  end
  return parse_number(s, i)
end

function json.decode(s)
  local value = parse(s, 1)
  return value
end

-- ── 로직 디스패처 ──
-- TS가 callJson(fn, ...args)로 호출: args를 JSON 배열로 받아 디코드 후 전역 fn(args...)을 부르고,
-- 결과를 {value=...}로 감싸 인코딩해 돌려준다(결과가 nil/숫자/문자열/테이블이든 균일 처리).
function __logic_call_json(fn_name, args_json)
  local fn = _G[fn_name]
  if type(fn) ~= 'function' then
    error('logic function not found: ' .. tostring(fn_name))
  end
  local args = json.decode(args_json)
  local result = fn(table.unpack(args, 1, #args))
  return json.encode({ value = result })
end

-- 다른 모듈 청크에서 쓰도록 전역으로도 노출.
__json_encode = json.encode
__json_decode = json.decode
