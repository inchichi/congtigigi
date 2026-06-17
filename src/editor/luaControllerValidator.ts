// 생성된 Lua 컨트롤러를 "적용 전에" 결정적으로 검증한다(생성/검증 분리 원칙).
// 게임 측 Lua 런타임이 로드 시 진짜 컴파일/계약 검증을 하지만, 여기서 명백한 문제를
// 사람이 읽을 수 있는 이슈로 미리 잡아준다. Lua 를 실행하지 않고 정적 패턴만 검사한다.

export type LuaControllerValidationIssue = {
  severity: 'error' | 'warning'
  message: string
}

// 샌드박스(런타임 env)가 이미 io/os 등을 노출하지 않지만, 생성물이 이런 토큰을 쓰면
// 거의 항상 잘못 생성된 것이므로 적용 전에 막아 사람이 바로 알 수 있게 한다.
const DENYLIST: { pattern: RegExp; message: string }[] = [
  { pattern: /\brequire\s*\(/, message: 'require 사용 불가 (외부 모듈 로드 금지).' },
  { pattern: /\bdofile\s*\(/, message: 'dofile 사용 불가.' },
  { pattern: /\bloadfile\s*\(/, message: 'loadfile 사용 불가.' },
  { pattern: /\bloadstring\s*\(/, message: 'loadstring 사용 불가.' },
  { pattern: /\bload\s*\(/, message: 'load 사용 불가 (동적 코드 실행 금지).' },
  { pattern: /\bio\s*\./, message: 'io.* 사용 불가 (파일 입출력 금지).' },
  { pattern: /\bos\s*\./, message: 'os.* 사용 불가.' },
  { pattern: /\bpackage\s*\./, message: 'package.* 사용 불가.' },
  { pattern: /\bdebug\s*\./, message: 'debug.* 사용 불가.' },
  { pattern: /\bsetfenv\s*\(|\bgetfenv\s*\(/, message: 'setfenv/getfenv 사용 불가.' },
  { pattern: /\b_G\b/, message: '_G(전역 테이블) 접근 불가.' }
]

const hasStepMethod = (source: string): boolean =>
  /function\s+[A-Za-z_]\w*\s*\.\s*step\b/.test(source) ||
  /\bstep\s*=\s*function\b/.test(source) ||
  /\[\s*['"]step['"]\s*\]\s*=/.test(source)

// 생성된 Lua 컨트롤러 소스의 이슈 목록을 돌려준다. error 가 하나라도 있으면 적용을 막아야 한다.
export const validateLuaControllerSource = (
  source: string
): LuaControllerValidationIssue[] => {
  const issues: LuaControllerValidationIssue[] = []

  if (source.trim().length === 0) {
    return [{ severity: 'error', message: '생성된 Lua 소스가 비어 있습니다.' }]
  }

  // 최상위(들여쓰기 없는) `return controller` / `return {` 만 모듈 반환으로 인정한다.
  // 함수 내부의 `return 0, 0` 같은 들여쓴 return 은 모듈 반환이 아니다.
  if (!/(^|\n)return\s+([A-Za-z_]\w*|\{)/.test(source)) {
    issues.push({
      severity: 'error',
      message: '컨트롤러 테이블을 return 하지 않습니다 (마지막에 `return controller` 필요).'
    })
  }

  if (!hasStepMethod(source)) {
    issues.push({
      severity: 'error',
      message: '필수 메서드 step 이 없습니다 (제자리면 `return 0, 0`).'
    })
  }

  for (const { pattern, message } of DENYLIST) {
    if (pattern.test(source)) {
      issues.push({ severity: 'error', message })
    }
  }

  if (/\bwhile\s+true\b/.test(source) && !/\bbreak\b/.test(source)) {
    issues.push({
      severity: 'warning',
      message: 'break 없는 `while true` 무한 루프가 의심됩니다.'
    })
  }

  return issues
}

export const hasBlockingLuaControllerIssue = (
  issues: LuaControllerValidationIssue[]
): boolean => issues.some((issue) => issue.severity === 'error')
