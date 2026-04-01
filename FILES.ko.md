# 파일 구조 안내 (정리 버전)

이 문서는 현재 저장소의 핵심 구조를 빠르게 파악하기 위한 안내입니다.

## 루트 핵심 파일
- `README.md`: 메인 사용 설명서(영문)
- `README.ko.md`: 메인 사용 설명서(국문)
- `requirements.txt`: 실행/학습/리포트 생성에 필요한 Python 의존성
- `run.sh`: RGBA 입력을 RGB로 자동 변환해 `test.py` 호출
- `run_best.sh`: `best_decoder` 기반 추론 실행 래퍼
- `train_generalize.sh`: 개선 학습 프리셋 실행 스크립트

## 코드
- `test.py`: 이미지 스타일 변환 추론
- `test_video.py`: 비디오 스타일 변환
- `train.py`: 기본 학습
- `train_improved.py`: 일반화 성능 개선 학습
- `net.py`: 인코더/디코더 네트워크 정의
- `function.py`: AdaIN 핵심 함수 및 보조 함수
- `sampler.py`: 무한 샘플러
- `torch_to_pytorch.py`: Torch -> PyTorch 변환 스크립트

## 데이터/모델/산출물
- `input/`: 콘텐츠/스타일/비디오 샘플
- `models/`: 추론용 사전학습 가중치 위치
- `output/`: 생성 결과 이미지(깃 추적 제외)
- `experiments_improved/`: 개선 학습 체크포인트(깃 추적 제외)
- `logs_improved/`: 개선 학습 로그(깃 추적 제외)

## 리포트
- `report_assets/data/`: 지표 계산/그래프 생성 스크립트
- `report_assets/figures/`, `report_assets/images/`: 생성된 리포트 산출물(깃 추적 제외)
- `docs/papers/`: 참고 논문 PDF
- `docs/reports/`: 보고서(md/html/pdf)

## 실행 흐름 요약
1. 가상환경 생성 후 `pip install -r requirements.txt`
2. `models/`에 `decoder.pth`, `vgg_normalised.pth` 배치
3. `run_best.sh` 또는 `test.py`로 추론 실행
4. 필요 시 `train_generalize.sh`로 학습 진행
5. `report_assets/data/*.py`로 지표/그래프 생성
