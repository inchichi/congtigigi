# LLM 이벤트 생성 파이프라인 노트

프로젝트 주제:

- `LLM 및 스타일 전이 기반 게임 콘텐츠 제작 자동화 시스템 개발`

이 문서는 캡스톤 프로젝트 중 `LLM 기반 이벤트 코드 생성` 파트의 진행 방향과 설계 내용을 계속 기록하는 노트다.
구현 방향이 바뀌거나 실험이 진행되면 이 문서를 갱신한다.

## 현재 목표

자연어로 입력한 게임 이벤트 설명을 아래 순서로 변환하는 흐름을 만든다.

1. 구조화된 JSON 생성
2. 게임이 읽을 수 있는 설정 데이터로 변환
3. Lua 스크립트 또는 템플릿 기반 코드 생성
4. 필요 시 NPC 대화, 퀘스트, 보상까지 연결

## 현재 저장소 상황

- 게임은 이미 Tiled 오브젝트 레이어로 NPC를 스폰한다.
- NPC 행동은 Lua 컨트롤러로 제어된다.
- 현재 Lua 컨트롤러는 주로 대화와 이동 행동을 지원한다.
- 퀘스트 진행 상태도 게임 코드에 이미 존재한다.
- 따라서 LLM 이벤트 파이프라인은 `대화 이벤트`부터 시작하고, 이후 퀘스트와 보상으로 확장하는 방식이 적합하다.

## 추천 개발 순서

### 1. 이벤트 JSON 스키마 정의

먼저 생성과 검증이 쉬운 좁은 범위의 스키마를 고정한다.

초기 스키마 필드 후보:

- `event_name`
- `event_type`
- `npc`
- `trigger`
- `dialogue`
- `reward`
- `duration`

#### 이벤트 스키마 v1

현재 게임에서는 우선 `NPC 대화 이벤트`에 집중한다.
이렇게 하면 기존 엔진 구조와 잘 맞고, 새로운 런타임 API를 바로 추가하지 않아도 된다.

허용할 트리거 타입:

- `talk`
- `scene_enter`
- `quest_start`
- `quest_complete`

허용할 보상 타입:

- `none`
- `item`
- `gold`
- `experience`

권장 스키마 예시:

```json
{
  "event_name": "christmas_event",
  "event_type": "holiday_dialogue",
  "npc": {
    "id": "santa",
    "display_name": "산타",
    "appearance_type": "character_villager_red_tunic"
  },
  "trigger": {
    "type": "talk",
    "target_scene": "town"
  },
  "dialogue": {
    "opening_lines": [
      "메리 크리스마스!",
      "선물을 받아가라."
    ],
    "active_lines": [],
    "completion_lines": []
  },
  "reward": {
    "type": "item",
    "id": "gift",
    "count": 1
  },
  "duration": 7
}
```

#### 현재 게임에 맞춘 매핑

- `npc.id` -> Tiled 오브젝트 `name`
- `npc.display_name` -> `displayText`
- `npc.appearance_type` -> Tiled 오브젝트 `type`
- `dialogue.opening_lines` -> `controller.dialogueLines`
- `duration` -> `controller.messageDurationSeconds`
- `trigger.type = talk` -> 대화용 Lua 컨트롤러 `scriptId`

보상 필드는 현재 단계에서는 논리적 출력으로만 둬도 된다.
실제 반영은 이후 퀘스트 로직이나 엔진 API 확장으로 처리한다.

### 2. LLM 출력은 JSON으로 제한

처음부터 Lua 코드를 직접 생성하게 하지 않는다.
1차 목표는 자연어 설명을 위 JSON 스키마로만 변환하는 것이다.

이 방식의 장점:

- 검증이 쉽다
- 잘못된 출력 복구가 쉽다
- 베이스라인 비교가 쉽다
- 여러 이벤트 타입으로 재사용하기 쉽다

### 3. 검증과 복구 단계 추가

JSON 생성 후 아래 항목을 검사한다.

- 필수 필드 존재 여부
- 필드 타입 일치 여부
- 허용 값 범위 여부
- 지원하는 이벤트 타입인지 여부

#### 검증 규칙 v1

다음 규칙을 우선 적용한다.

- `event_name`은 비어 있으면 안 된다.
- `event_name`은 영어 소문자와 밑줄 중심으로 정규화한다.
- `event_type`은 사전에 정의된 타입만 허용한다.
- `npc.id`는 문자열이어야 한다.
- `npc.display_name`은 사용자에게 보이는 이름이므로 비어 있으면 안 된다.
- `npc.appearance_type`은 현재 타일셋에 존재하는 외형 키여야 한다.
- `trigger.type`은 허용 목록 안에 있어야 한다.
- `dialogue.opening_lines`는 최소 1개 이상의 문자열 배열이어야 한다.
- `duration`은 양의 정수여야 한다.
- `reward.type`이 `item`이면 `id`와 `count`가 필요하다.
- `reward.type`이 `gold`이면 `count`가 필요하다.
- `reward.type`이 `experience`이면 `count`가 필요하다.

검증 실패 시 처리 순서:

1. 누락 필드 확인
2. 타입 오류 확인
3. 허용 값 범위 확인
4. 보정 가능한 경우 자동 수정
5. 보정 불가하면 재생성

검증 실패 시 선택할 수 있는 방식:

- JSON 재생성
- 두 번째 프롬프트로 자동 보정

### 4. JSON을 게임 데이터로 변환

현재 프로젝트에서는 완전 자유형 코드 생성보다 템플릿 기반 변환이 현실적이다.
JSON을 아래 형태로 변환하는 흐름이 적합하다.

- Tiled NPC 오브젝트 속성
- Lua 컨트롤러 설정값
- 퀘스트 정의 일부

#### 변환 규칙 v1

`event_type = holiday_dialogue` 인 경우:

- Tiled 오브젝트 `name`은 `npc.id`를 사용
- Tiled 오브젝트 `type`은 `npc.appearance_type`을 사용
- `displayText`는 `npc.display_name`을 사용
- `controller.scriptId`는 대화용 Lua 스크립트를 사용
- `controller.dialogueLines`는 `dialogue.opening_lines`를 사용
- `controller.messageDurationSeconds`는 `duration`을 사용

Lua 스크립트 쪽은 우선 템플릿 기반으로 둔다.
즉, LLM이 직접 Lua 전체를 쓰는 대신 스키마를 채우고, 엔진은 그 값을 받아 미리 정해둔 템플릿에 넣는다.

예상 출력 예시:

```xml
<object name="santa" type="character_villager_red_tunic" x="400" y="516">
  <properties>
    <property name="displayText" value="산타"/>
    <property name="controller.scriptId" value="reply-with-message"/>
    <property name="controller.dialogueLines" type="list">
      <item value="메리 크리스마스!"/>
      <item value="선물을 받아가라."/>
    </property>
    <property name="controller.messageDurationSeconds" type="float" value="2.5"/>
  </properties>
</object>
```

### 5. 대화 이벤트에서 게임 플레이 훅으로 확장

대화 이벤트가 안정적으로 동작하면 다음 기능으로 확장한다.

- 퀘스트 시작
- 퀘스트 완료
- 아이템 보상
- 장면 진입 트리거
- NPC 상태 배지 표시

## 크리스마스 이벤트 예시 v1

이 예시는 현재 게임 구조에 가장 쉽게 붙는 형태다.
우선은 `대화 + 외형 + 이벤트 식별자`까지만 확정하고, 보상은 논리 정보로 남겨둔다.

### 자연어 입력

```text
크리스마스 이벤트를 만들어줘.
산타 NPC가 등장하고, 말을 걸면 크리스마스 인사를 하고 선물 이야기를 하게 해줘.
```

### LLM 출력 JSON

```json
{
  "event_name": "christmas_event",
  "event_type": "holiday_dialogue",
  "npc": {
    "id": "santa",
    "display_name": "산타",
    "appearance_type": "character_villager_red_tunic"
  },
  "trigger": {
    "type": "talk",
    "target_scene": "town"
  },
  "dialogue": {
    "opening_lines": [
      "메리 크리스마스!",
      "오늘은 특별한 날이란다."
    ],
    "active_lines": [],
    "completion_lines": []
  },
  "reward": {
    "type": "item",
    "id": "gift",
    "count": 1
  },
  "duration": 7
}
```

### Tiled 반영 예시

```xml
<object name="santa" type="character_villager_red_tunic" x="400" y="516">
  <properties>
    <property name="displayText" value="산타"/>
    <property name="controller.scriptId" value="reply-with-message"/>
    <property name="controller.dialogueLines" type="list">
      <item value="메리 크리스마스!"/>
      <item value="오늘은 특별한 날이란다."/>
    </property>
    <property name="controller.messageDurationSeconds" type="float" value="2.5"/>
  </properties>
</object>
```

### Lua 컨트롤러 역할

현재 구조에서는 Lua가 직접 보상을 지급하기보다, 아래 역할을 맡는 것이 적절하다.

- 상호작용 시 대사 출력
- 이벤트 식별자 유지
- 필요 시 다음 대사로 순환

즉, 보상 지급이나 퀘스트 완료는 이후 엔진 API 확장 또는 퀘스트 로직에서 처리한다.

## 패널 용어 설명

현재 게임 화면에 있는 입력 패널에는 두 가지 결과가 표시된다.

### JSON 초안

- 자연어 입력을 구조화된 이벤트 데이터로 바꾼 중간 결과다.
- LLM이 가장 먼저 생성해야 하는 표준 이벤트 표현이다.
- 아직 게임 맵에 직접 들어가는 최종 데이터는 아니다.

예:

```json
{
  "event_name": "christmas_event",
  "event_type": "holiday_dialogue",
  "npc": {
    "id": "santa",
    "display_name": "산타",
    "appearance_type": "character_villager_brown_tunic"
  },
  "trigger": {
    "type": "talk",
    "target_scene": "town"
  },
  "dialogue": {
    "opening_lines": [
      "메리 크리스마스!",
      "오늘은 특별한 날이란다."
    ],
    "active_lines": [],
    "completion_lines": []
  },
  "reward": {
    "type": "item",
    "id": "gift",
    "count": 1
  },
  "duration": 7
}
```

### Tiled 미리보기

- JSON 초안을 현재 게임이 읽을 수 있는 Tiled NPC 오브젝트 형태로 바꾼 결과다.
- `name`, `type`, `displayText`, `controller.scriptId` 같은 값으로 변환된다.
- 이 결과를 그대로 맵 데이터에 넣으면 NPC가 게임 안에서 동작할 수 있다.

예:

```xml
<object name="santa" type="character_villager_brown_tunic">
  <properties>
    <property name="displayText" value="산타"/>
    <property name="controller.scriptId" value="reply-with-message"/>
    <property name="controller.dialogueLines" type="list">
      <item value="메리 크리스마스!"/>
      <item value="오늘은 특별한 날이란다."/>
    </property>
    <property name="controller.messageDurationSeconds" type="float" value="2.5"/>
  </properties>
</object>
```

## 비교 실험 후보

- 규칙 기반 템플릿 생성기
- Zero-shot LLM JSON 생성
- Few-shot LLM JSON 생성
- 스키마 기반 LLM 생성 + 검증

## 평가 지표

- JSON 유효성 통과율
- 필수 필드 충족률
- 코드/템플릿 생성 성공률
- 수동 수정 비율
- 같은 입력에 대한 결과 일관성

## 이 저장소에서의 적용 아이디어

### 안 1. NPC 대화 이벤트 생성

가장 먼저 시도할 방식이다.
기존 Lua 컨트롤러와 Tiled NPC 파이프라인을 활용해서 이벤트 NPC를 만든다.

예:

- 크리스마스 NPC
- 할로윈 NPC
- 이벤트 안내 NPC

### 안 2. 퀘스트 연계 이벤트 생성

대화 이벤트가 동작한 뒤 확장하는 단계다.
퀘스트 로그와 연결되는 이벤트 묶음을 생성할 수 있다.

### 안 3. 에디터 지원

후반 단계에서 추가할 수 있다.
간단한 웹 기반 에디터에서 아래 항목을 보여준다.

- 자연어 입력
- 생성된 JSON
- 생성된 Lua 또는 설정값
- 검증 오류

### 안 4. LLM 모드와 규칙 모드 분리

현재 개발 중인 패널은 두 가지 모드로 동작한다.

- 규칙 모드: 키워드를 기준으로 미리 정의한 초안을 즉시 생성
- LLM 모드: OpenAI Responses API를 호출해 구조화된 JSON 초안을 생성

이 방식의 장점은 다음과 같다.

- 규칙 모드로 엔드투엔드 흐름을 빠르게 검증할 수 있다.
- LLM 모드로 실제 자연어 이해 성능을 실험할 수 있다.
- 같은 JSON 검증기와 Tiled 변환기를 공용으로 재사용할 수 있다.

### OpenAI API 연동 방식

- `gpt-5.4-mini` 모델을 기본값으로 사용한다.
- Responses API의 `json_schema` 구조화 출력 형식을 사용한다.
- 브라우저에서 직접 OpenAI로 보내지 않고 Vite 개발 서버 프록시를 통해 요청한다.
- API 키는 패널에서 입력하고 브라우저 localStorage에 저장한다.

### 1차 구현 범위의 현재 상태

- `L` 키를 누르면 LLM 분석 패널이 열린다.
- 패널이 열리면 현재 프로젝트의 `Game Structure Profile`을 모의 분석한다.
- 분석 결과를 바탕으로 `GUS(Game Understanding Score)`를 계산한다.
- GUS가 기준점 75점 이상이면 이벤트 생성 영역이 활성화된다.
- 자연어 입력을 바탕으로 mock `JSON 초안`을 생성한다.
- JSON 필드 편집 UI에서 필드를 직접 수정할 수 있다.
- JSON 검증 결과를 화면에 표시한다.
- JSON 확정 후 코드 미리보기를 생성한다.
- 생성된 코드는 현재는 런타임 레지스트리에 보관하고, 현재 씬에는 호환 가능한 대화 이벤트로 적용한다.

### 씬 적용 방식

- 생성된 이벤트는 우선 `JSON 초안`으로 보여준다.
- 같은 초안을 `Tiled 미리보기`로 변환한다.
- 현재 town 씬에서는 적용 대상 NPC를 `santa`로 고정해 실시간 변경이 눈에 보이도록 했다.
- 따라서 할로윈이나 크리스마스 이벤트도 산타 NPC의 대사와 설정이 즉시 바뀌는 형태로 확인할 수 있다.

## 다음 작업

현재 단계에서 이어서 할 작업은 다음과 같다.

1. 크리스마스 이벤트를 기준으로 예시 1개를 완성형으로 구체화
2. JSON 검증 규칙을 더 세밀하게 정리
3. JSON -> Tiled property / Lua config 변환 규칙 작성

## 업데이트 로그

### 2026-06-04

- 저장소 내부 구조 확인 완료
  - Tiled 기반 NPC 스폰
  - Lua 컨트롤러 기반 NPC 행동
  - 퀘스트 로그 지원
- 1차 개발 목표 확정
  - 자연어 -> 구조화된 이벤트 JSON
- 1차 적용 경로 확정
  - NPC 대화 이벤트 우선
  - 퀘스트/보상은 이후 확장
- 이벤트 스키마 v1 초안 작성 완료
- 코드 구현 시작
  - `src/game/eventGeneration.ts`
  - `src/game/eventGeneration.test.ts`
  - `src/game/eventDrafting.ts`
  - `src/game/eventDrafting.test.ts`
  - `src/main.ts`
  - `src/assets/maps/town.tmx`
- OpenAI API 연동 시작
  - `src/game/openaiHolidayEventDraft.ts`
  - `vite.config.ts`
  - 이벤트 초안 패널에 `규칙 모드 / LLM 모드` 전환 추가
  - OpenAI API 키 입력칸 추가
  - Responses API + structured outputs 기반 JSON 생성 경로 추가
  - 실제 씬 적용 대상 NPC를 `santa`로 고정해 화면 반영 확인 가능하게 조정
- L 키 기반 LLM 분석 패널 추가
  - `src/llm/gameStructureProfile.ts`
  - `src/llm/currentGameProjectSnapshot.ts`
  - `src/llm/gusCalculator.ts`
  - `src/llm/eventJsonSchema.ts`
  - `src/llm/eventJsonGenerator.ts`
  - `src/llm/eventCodeGenerator.ts`
  - `src/llm/gameStructureAnalyzer.ts`
  - `src/events/DynamicEventManager.ts`
  - `src/ui/createLlmPanel.ts`
  - L 키로 패널 열고 닫기
  - Game Structure Profile 모의 분석
  - GUS 계산 및 표시
  - GUS 통과 시 이벤트 생성 패널 활성화
  - 자연어 -> JSON 초안 생성
  - JSON 필드 편집 UI
  - JSON 검증
  - 코드 미리보기와 런타임 등록
- 구현 범위
  - 이벤트 JSON 타입 정의
  - JSON 검증 규칙 v1
  - Tiled NPC 오브젝트 변환
  - 자연어 -> 이벤트 초안 생성기
  - 게임 화면 내 미니 입력 패널
  - town 맵에 크리스마스 NPC 추가
  - LLM 모드 / 규칙 모드 전환
  - OpenAI LLM 기반 이벤트 초안 생성
  - L 키 기반 LLM 분석 패널
  - GUS 기반 게이팅
  - JSON 필드 수정 UI
  - JSON 기반 코드 미리보기
- 검증 결과
  - 새 테스트 6개 통과
  - TypeScript type-check 통과
- 신규 테스트
  - GUS 계산 테스트
  - mock JSON 생성 테스트
- 다음 작업
  - LLM 출력 오류 복구 절차 추가
  - 이벤트 스키마 확장
  - 실제 맵 저장 기능 검토
