# LLM 게임 패널 작업 요약

이 문서는 캡스톤 프로젝트 `my-sample-rpg`에 추가한 LLM 기반 게임 패널 작업을 한국어로 정리한 것이다.

## 작업 목표

- 게임 실행 중 `L` 키로 LLM 분석 패널을 열고 닫을 수 있게 한다.
- 현재 게임 구조를 분석해서 `Game Structure Profile`과 `GUS(Game Understanding Score)`를 계산한다.
- `GUS`가 기준을 넘으면 자연어 기반 이벤트 생성 패널을 활성화한다.
- 자연어 입력을 JSON 초안으로 변환하고, JSON을 수정한 뒤 코드 미리보기와 게임 적용까지 이어지게 한다.

## 구현한 기능

### 1. L 키 기반 패널

- 게임 실행 중 `L` 키를 누르면 LLM 분석 패널이 열린다.
- 다시 `L` 키를 누르면 패널이 닫힌다.
- 패널은 데스크톱 창처럼 동작하도록 구성했다.

### 2. 윈도우 창 UI

- 상단바에 제목과 창 제어 버튼을 넣었다.
  - 최소화
  - 최대화/복원
  - 닫기
- 상단바 드래그로 위치를 옮길 수 있다.
- 창 크기는 좌우/상하 가장자리 드래그로 조절할 수 있다.
- 위치와 크기는 `localStorage`에 저장한다.

### 3. API 키 입력 게이트

- `L` 키를 눌렀을 때 바로 분석을 시작하지 않고, 먼저 OpenAI API 키 입력 창을 띄운다.
- 키를 입력하고 `확인 후 분석 시작`을 눌러야 구조 분석이 시작된다.
- API 키는 브라우저 `localStorage`에 저장한다.

### 4. 게임 구조 분석

- 현재 프로젝트 구조를 모의 분석하는 기능을 추가했다.
- 분석 결과로 `Game Structure Profile`을 보여준다.
- 분석 과정과 진행률도 패널에서 확인할 수 있다.

### 5. GUS 계산

- `QA_correctness`: 게임 구조를 정확히 파악했는지 평가한다.
- `QA_completeness`: 맵, NPC, 아이템, 이벤트, 대화 시스템, 이벤트 시스템을 충분히 포괄했는지 평가한다.
- `Dependency_F1`: NPC-맵, 파일-시스템, 이벤트-대화 관계를 얼마나 잘 파악했는지 평가한다.
- `Localization_Accuracy`: 이벤트 생성에 필요한 코드와 데이터 파일 위치를 얼마나 잘 찾았는지 평가한다.
- `Trace_Correctness`: 입력, 이벤트, 대화, 적용 흐름을 추적할 수 있는지 평가한다.
- `Grounding_Score`: 분석 결과가 실제 프로젝트 파일에 근거하고 있는지 평가한다.

위 항목을 가중합해 `GUS`를 계산한다.

- `GUS` 기본 기준은 `75점`이다.
- `GUS`가 기준 미만이면 이벤트 생성 기능이 잠긴다.
- `GUS`가 기준 이상이면 다음 단계로 진행할 수 있다.

### 6. 이벤트 생성 파이프라인

- 자연어 입력
- JSON 초안 생성
- JSON 내용 수정
- JSON 검증
- 코드 생성
- 게임 적용

이 흐름으로 이벤트 생성을 실험할 수 있게 구성했다.

### 7. JSON 편집 UI

- JSON 필드는 한 번에 하나씩 보는 방식 대신 토글형 카드로 수정할 수 있게 했다.
- 필드를 펼치면 해당 값을 직접 편집할 수 있다.
- 여러 필드를 동시에 펼쳐서 수정할 수 있다.

### 8. 코드 미리보기와 게임 적용

- JSON을 기반으로 코드 미리보기를 생성한다.
- 생성된 코드는 런타임 레지스트리에 등록된다.
- 일부 이벤트는 실제 게임 씬에 적용해 확인할 수 있다.

### 9. 모드 전환

- 규칙 기반 모드와 LLM 모드를 구분했다.
- 규칙 기반 모드는 키워드 기반 초안 생성에 사용한다.
- LLM 모드는 OpenAI API를 사용해 JSON 초안을 생성한다.

## 최근 UI 정리

- 패널은 3개 페이지로 정리했다.
  - `현 게임 분석`
  - `JSON 생성`
  - `코드 생성`
- 각 페이지에는 해당 단계의 기능만 남기고 중복 버튼은 제거했다.

## 현재 상태

- 프로젝트 구조 분석, GUS 계산, JSON 생성, JSON 검증, 코드 미리보기, 게임 적용까지 연결된 상태다.
- 패널 UI는 윈도우 창처럼 동작하며 드래그와 리사이즈를 지원한다.
- 현재 브랜치 기준 변경 사항을 `develop-yc`에 정리해 두었다.

## 관련 파일

- [`src/ui/createLlmPanel.ts`](../src/ui/createLlmPanel.ts)
- [`src/llm/gameStructureProfile.ts`](../src/llm/gameStructureProfile.ts)
- [`src/llm/gusCalculator.ts`](../src/llm/gusCalculator.ts)
- [`src/llm/eventJsonSchema.ts`](../src/llm/eventJsonSchema.ts)
- [`src/llm/eventJsonGenerator.ts`](../src/llm/eventJsonGenerator.ts)
- [`src/llm/eventCodeGenerator.ts`](../src/llm/eventCodeGenerator.ts)
- [`src/llm/gameStructureAnalyzer.ts`](../src/llm/gameStructureAnalyzer.ts)
- [`src/events/DynamicEventManager.ts`](../src/events/DynamicEventManager.ts)
- [`notes/llm-event-generation-pipeline.md`](./llm-event-generation-pipeline.md)
- [`notes/llm-event-code-milestones.md`](./llm-event-code-milestones.md)
