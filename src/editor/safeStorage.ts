// localStorage는 프라이빗 모드·저장소 차단 정책·용량 초과(QuotaExceededError)에서 throw할 수 있다.
// 그 예외가 에디터 초기화(통째로 안 뜸)나 입력 핸들러(먹통)를 깨뜨리지 않도록 읽기/쓰기를 감싼다.
export const readLocalStorage = (key: string): string | null => {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

// 저장 성공 여부를 boolean으로 돌려준다. 실패해도 throw하지 않으므로 호출부가 흐름을 이어갈 수 있다.
export const writeLocalStorage = (key: string, value: string): boolean => {
  try {
    window.localStorage.setItem(key, value)
    return true
  } catch {
    return false
  }
}
