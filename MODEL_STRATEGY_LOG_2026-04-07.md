# AdaIN 모델 전략 정리 (최근 기록 기반)

작성일: 2026-04-07  
대상 프로젝트: `C:\Users\praisy\Desktop\26-1\캡스톤\AdaIN\인치연\congtigigi`

## 1) 기록 출처
- `~/.codex/history.jsonl`에서 `adain|mask|interpolate|output2|grounded` 키워드로 최근 질의 추출
- `~/.codex/sessions/2026/04/07/rollout-2026-04-07T02-05-05-019d63c1-4c0b-77f1-b463-e29702acdc7b.jsonl` 확인
- 프로젝트 코드/문서:
  - `README.md`, `run_batch_interp.sh`, `evaluate_batch_metrics.py`
  - `scripts/generate_grounded_sam_masks.py`
  - `test.py` (mask 내부 적용 수정본)

## 2) 최근 작업 타임라인 (핵심)
- 2026-03-28: AdaIN 원본/Torch 레포 참조 및 환경 구축 시작
- 2026-03-29~30: 게임 타일 대상 스타일 전이 실험 방향 수립
  - 계절별 스타일(2D/real) 적용
  - 출력 해상도/타일 정렬/스프라이트시트 후처리 고민
  - 단일 스타일 대신 다중 스타일 보간(interpolation) 전략으로 이동
- 2026-03-30: 문제 가설 정리
  - 색감은 변하지만 객체 형태 인식이 약함
  - 개선 실험축: `파라미터 튜닝` vs `mask 도입`
- 2026-03-31: `congtigigi` 기준 대량 배치 실행 및 정량 평가
  - 콘텐츠 타일 일괄 처리
  - 계절/도메인별 보간 결과 생성
  - Content Loss / Style Loss / FPS 계산 스크립트 반영
- 2026-04-06: GroundingDINO + SAM 기반 객체 마스크 생성 요청/적용
  - 클래스: tree / rock / lake
  - semantic 및 overlay 마스크 산출
- 2026-04-06~07: mask 적용 방식 고도화
  - (초기) 후처리 합성 방식으로 `output2` 생성
  - (개선) `test.py` 내부 feature 단계 마스킹 방식으로 변경
  - 최종 재실행: `output2` 8개 그룹 x 100장 = 총 800장 생성

## 3) 전략 변화 요약
### 단계 A. 전역(Global) AdaIN
- 기본 AdaIN은 이미지 전체 feature 통계를 스타일 통계에 맞춤
- 장점: 빠르고 단순
- 한계: 타일 내 객체별(나무/돌/호수) 제어가 어려움

### 단계 B. 계절 스타일 보간
- 각 계절/도메인(2D, real)별 스타일 5장을 동일 가중치로 보간
- 목표: 단일 스타일 편향 완화, 계절 톤의 대표성 확보
- 구현: `run_batch_interp.sh` + `test.py --style ... --style_interpolation_weights ...`

### 단계 C. 정량 평가 루프 도입
- `evaluate_batch_metrics.py`로 타일 단위 지표 계산
- 지표: Content Loss, Style Loss, FPS
- 목적: “좋아 보인다” 대신 수치 기반 비교 가능하게 전환

### 단계 D. 객체 단위 마스크 파이프라인
- `scripts/generate_grounded_sam_masks.py`로 클래스 마스크 생성
- 산출물:
  - 클래스별 바이너리 마스크(`tree`, `rock`, `lake`)
  - semantic 마스크
  - overlay 시각화

### 단계 E. AdaIN 내부 마스크 적용 (현재 전략)
- `test.py` 수정으로 mask를 feature 단계에서 적용:
  - `feat = feat * mask + content_f * (1 - mask)`
  - mask 영역만 스타일 강하게, 비마스크 영역은 콘텐츠 유지
- 신규 옵션:
  - `--mask`, `--mask_dir`, `--mask_suffix`
  - `--mask_threshold`, `--invert_mask`
- 안정성 보강:
  - 콘텐츠/스타일 이미지를 모두 `RGB`로 강제 변환(RGBA 충돌 방지)

## 4) 현재 운영 중인 표준 실행 전략
1. 입력 준비
- 콘텐츠 타일: `input/content/PNG`
- 스타일 셋: `input/style/{spring,summer,fall,winter}/{2D,real}`
- 마스크: `input/mask/grounded_sam/semantic/*_semantic.png`

2. 추론 설정
- 콘텐츠/스타일 크기: `256`, `--crop` 사용
- 스타일 보간: 그룹별 5장 equal weight
- 마스크: semantic > 0 영역 스타일 적용

3. 출력 구조
- `output2/{season}_{kind}/<tile>_interpolation.jpg`
- 최종 기준 산출량: 그룹별 100장, 총 800장

## 5) 현재 코드 기준 핵심 포인트
- `test.py`
  - AdaIN 전역 적용 + feature-level mask 혼합을 함께 지원
  - 단일 마스크 파일/폴더 기반 자동 매칭 모두 지원
- `scripts/generate_grounded_sam_masks.py`
  - GroundingDINO 탐지 + SAM 분할 결과를 semantic으로 통합
- `evaluate_batch_metrics.py`
  - 배치 결과를 손실/FPS로 정량화해 비교 기준 제공

## 6) 다음 실험 우선순위(권장)
- 클래스별 가중 마스킹:
  - 현재는 semantic > 0 일괄 적용이므로, `tree/rock/lake`별 다른 강도(alpha) 실험 권장
- 계절별 스타일 가중치 튜닝:
  - 균등 가중치(1,1,1,1,1) 대신 가중치 최적화
- 지표-품질 연계:
  - 손실/FPS와 사람이 느끼는 타일 품질 상관관계 점검용 샘플 세트 구축

---
이 문서는 2026-04-07 기준, Codex 로컬 히스토리와 현재 저장소 상태를 합쳐 작성한 전략 요약본이다.
