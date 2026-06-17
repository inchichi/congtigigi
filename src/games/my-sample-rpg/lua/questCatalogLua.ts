// Phase 5: 퀘스트 카탈로그(또는 임의의 데이터)를 Lua 데이터 모듈 문자열로 직렬화한다.
// 이렇게 만든 Lua 를 런타임의 loadDataModule 로 다시 읽으면 원본과 동일한 JS 값이 나온다
// (golden 라운드트립 테스트로 보장). 즉 퀘스트 카탈로그를 "Lua 데이터"로 표현·왕복할 수 있어,
// 향후 LLM 이 퀘스트/로스터를 Lua 데이터로 생성하는 길을 연다.

const escapeLuaString = (value: string): string =>
  '"' +
  value
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t') +
  '"'

const toLuaValue = (value: unknown): string => {
  if (value === null || value === undefined) {
    return 'nil'
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false'
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : 'nil'
  }
  if (typeof value === 'string') {
    return escapeLuaString(value)
  }
  if (Array.isArray(value)) {
    return '{' + value.map((item) => toLuaValue(item)).join(', ') + '}'
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, entryValue]) => entryValue !== undefined)
      .map(([key, entryValue]) => `[${escapeLuaString(key)}] = ${toLuaValue(entryValue)}`)
    return '{' + entries.join(', ') + '}'
  }
  return 'nil'
}

// 데이터를 그대로 반환하는 Lua 모듈 소스를 만든다.
export const serializeToLuaDataModule = (value: unknown): string =>
  `return ${toLuaValue(value)}\n`
