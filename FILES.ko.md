# pytorch-AdaIN 파일 안내

이 문서는 `pytorch-AdaIN` 폴더 안 주요 파일/폴더가 무엇을 의미하는지 정리한 안내서입니다.

## 루트 파일/폴더
- `.git/`: Git 버전 관리 메타데이터
- `.gitignore`: Git 추적 제외 규칙
- `.venv/`: Python 가상환경(패키지 설치 위치)
- `LICENSE`: 오픈소스 라이선스 문서
- `README.md`: 원본 영어 사용 설명서
- `README.ko.md`: 한국어 번역 설명서
- `FILES.ko.md`: 현재 파일(구성 안내)
- `__pycache__/`: Python 실행 시 생성되는 캐시 파일

## 코드 파일
- `test.py`: 이미지 스타일 변환(추론) 메인 스크립트
- `test_video.py`: 비디오 스타일 변환 스크립트
- `train.py`: 모델 학습 스크립트
- `net.py`: 네트워크 구조(인코더/디코더 정의)
- `function.py`: AdaIN 핵심 함수 및 보조 함수
- `sampler.py`: 학습/데이터 샘플링 보조 코드
- `torch_to_pytorch.py`: Torch 모델을 PyTorch 형식으로 변환할 때 쓰는 스크립트

## 데이터/모델/결과
- `input/`: 예제 입력 데이터
  - `input/content/`: 콘텐츠 이미지 샘플
  - `input/style/`: 스타일 이미지 샘플
  - `input/mask/`: 마스크 샘플
  - `input/videos/`: 비디오 샘플
- `models/`: 사전학습 가중치 저장 폴더
  - `decoder.pth`: 디코더 가중치
  - `vgg_normalised.pth`: VGG 인코더 가중치
- `output_pytorch/`: `test.py` 실행 결과 이미지가 저장되는 폴더

## 기타
- `requirements.txt`: Python 의존성 목록(원본 저장소 기준)
- `results.png`: README에 쓰인 결과 예시 이미지

## 실행 흐름 요약
1. `models/`에 가중치(`decoder.pth`, `vgg_normalised.pth`) 준비
2. `test.py`로 `input/` 또는 사용자 이미지를 스타일 변환
3. 결과는 `output_pytorch/`에 저장
