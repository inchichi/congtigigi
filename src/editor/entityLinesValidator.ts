// 생성과 분리된 결정적 검증(이사님 #1: 생성/검증 분리)을 rpg 외 모든 게임에도 적용하기 위한 모듈.
// generic/legend 어댑터의 공용 생성물({entity, lines})을, 게임 구조에 근거해 결정적으로 점검한다.
// LLM 호출 없이 순수 함수로만 동작하므로 테스트가 쉽고, 생성이 흔들려도 검증은 안정적이다.

export type GeneratedEntityLines = {
  entity: string
  lines: string[]
}

export type EntityLinesValidationOptions = {
  // 선택된 대상 엔티티 이름(있으면, 생성물이 그 대상을 가리키는지 근거 검사한다).
  targetEntityName?: string
  // 한 번에 허용하는 최대 줄 수. 공용 생성 지시는 "1~4줄"이라 기본 4.
  maxLines?: number
  // 게임 대사 UI에 들어갈 한 줄의 최대 길이(문자). 기본 200.
  maxLineLength?: number
}

const DEFAULT_MAX_LINES = 4
const DEFAULT_MAX_LINE_LENGTH = 200

const referencesTarget = (entity: string, target: string): boolean => {
  const a = entity.trim().toLowerCase()
  const b = target.trim().toLowerCase()
  if (a.length === 0 || b.length === 0) {
    return false
  }
  return a.includes(b) || b.includes(a)
}

// 통과하면 빈 배열. 각 문자열은 사람이 읽을 수 있는 한국어 이슈 설명이다(rpg Validator와 톤 일치).
export const createEntityLinesValidationIssues = (
  generated: GeneratedEntityLines,
  options: EntityLinesValidationOptions = {}
): string[] => {
  const issues: string[] = []
  const maxLines = options.maxLines ?? DEFAULT_MAX_LINES
  const maxLineLength = options.maxLineLength ?? DEFAULT_MAX_LINE_LENGTH

  if (typeof generated.entity !== 'string' || generated.entity.trim().length === 0) {
    issues.push('entity - 생성 대상 이름이 비어 있습니다.')
  } else if (
    options.targetEntityName !== undefined &&
    options.targetEntityName.trim().length > 0 &&
    !referencesTarget(generated.entity, options.targetEntityName)
  ) {
    issues.push(
      `entity - 생성 대상("${generated.entity}")이 선택한 엔티티("${options.targetEntityName}")와 다릅니다.`
    )
  }

  if (!Array.isArray(generated.lines) || generated.lines.length === 0) {
    issues.push('lines - 생성된 대사/설명이 없습니다. 최소 1줄이 필요합니다.')
    return issues
  }

  if (generated.lines.length > maxLines) {
    issues.push(`lines - 줄 수가 너무 많습니다(${generated.lines.length}줄). 최대 ${maxLines}줄.`)
  }

  generated.lines.forEach((line, index) => {
    if (typeof line !== 'string' || line.trim().length === 0) {
      issues.push(`lines[${index}] - 빈 줄입니다.`)
      return
    }
    if (line.length > maxLineLength) {
      issues.push(
        `lines[${index}] - 한 줄이 너무 깁니다(${line.length}자). 최대 ${maxLineLength}자.`
      )
    }
  })

  return issues
}
