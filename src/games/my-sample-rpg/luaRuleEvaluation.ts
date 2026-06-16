// 게임 규칙을 Lua 로 실행하는 공용 헬퍼. Lua 소스(규칙 정의)에 `return <expression>` 을 붙여
// 실행하고 결과를 받는다. 실패하거나 유한수가 아니면 TS 폴백으로 안전하게 떨어진다.
export type LoadLuaDataModule = (source: string) => unknown

export const evaluateLuaNumber = (
  loadDataModule: LoadLuaDataModule,
  luaSource: string,
  expression: string,
  fallback: number
): number => {
  try {
    const result = loadDataModule(`${luaSource}\nreturn ${expression}`)
    return typeof result === 'number' && Number.isFinite(result) ? result : fallback
  } catch {
    return fallback
  }
}
