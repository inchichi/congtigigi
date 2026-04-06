# textile 폴더 파일 정리

기준 경로: `C:\Users\praisy\Desktop\26-1\캡스톤\textile`

## 루트 파일/폴더
- `.git/`: Git 버전관리 메타데이터.
- `.venv/`: 프로젝트용 Python 가상환경(패키지 설치 위치).
- `LICENSE`: MIT 라이선스 원문.
- `README.md`: 프로젝트 소개, 논문 링크, 설치/사용 예시.
- `requirements.txt`: 실행에 필요한 핵심 의존성 목록.
- `setup.py`: 패키징/배포 설정(`textile-metric`), 설치 의존성/포함 파일 정의.
- `textile/`: 실제 라이브러리 소스 코드 패키지.

## textile 패키지 내부
- `textile/__init__.py`: 외부에서 바로 쓰는 주요 API(`Textile`, 이미지 유틸 등) 노출.
- `textile/textile.py`: 핵심 클래스 `Textile(nn.Module)` 구현.
  - 사전학습 가중치가 없으면 자동 다운로드
  - 입력 이미지를 타일링/리사이즈/정규화 후 점수 계산
  - 타일러블리티 점수를 `(0, 1)` 범위로 변환

## 모델/아키텍처 관련
- `textile/models/__init__.py`: 패키지 인식용 빈 파일.
- `textile/architectures/__init__.py`: 패키지 인식용 빈 파일.
- `textile/architectures/layers/__init__.py`: 패키지 인식용 빈 파일.
- `textile/architectures/layers/attention/__init__.py`: 패키지 인식용 빈 파일.
- `textile/architectures/layers/attention/attention.py`: 커스텀 어텐션 레이어 구현.
  - `LayerNorm`
  - `LinearAttention` (ConvNeXt feature 단계에 삽입되는 경량 attention)

## 유틸리티
- `textile/utils/__init__.py`: 패키지 인식용 빈 파일.
- `textile/utils/create_model.py`: 모델 생성 함수 `CreateModel`.
  - `convnext_base` 백본 생성
  - 분류 헤드(1차원 출력)로 교체
  - 중간 feature 블록에 `LinearAttention` 삽입
  - `.pth` 가중치 로드
- `textile/utils/image_utils.py`: 이미지 로딩/전처리 유틸.
  - OpenCV로 읽기(RGB 변환, 0~1 정규화)
  - Kornia tensor 변환
- `textile/utils/dataset.py`: 학습/평가용 `TilingDataset`.
  - 이미지 로드 -> 타일링 -> 리사이즈 -> 정규화
- `textile/utils/misc.py`: 모델 다운로드 진행률 표시용 `MyProgressBar`.

## 데이터 파일
- `textile/data/__init__.py`: 패키지 인식용 빈 파일.
- `textile/data/texture_368.jpg`: 샘플 텍스처 이미지(1024x1024 JPEG).
- `textile/data/teaser.png`: README/프로젝트 소개용 티저 이미지(2450x902 PNG).

## 참고
- `requirements.txt`에는 핵심 의존성 6개만 있고,
  `setup.py`에는 추가로 `progressbar>=2.5`가 포함되어 있음.
