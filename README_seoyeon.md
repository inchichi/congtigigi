# README Seoyeon

## 목적

이 브랜치는 TexTile 점수를 이용해 게임 타일의 연속성을 해석하고, 이후 하이퍼파라미터 튜닝과 모델 개선 방향을 잡기 위한 실험 내용을 정리한 문서입니다.

TexTile은 "같은 타일을 반복 배치했을 때 경계가 자연스럽게 이어지는가"를 평가하는 모델입니다.  
즉, 단순히 오브젝트가 규칙적으로 반복되는지보다 **경계가 seam 없이 이어지는 텍스처성**을 더 중요하게 보는 경향이 있습니다.

## 지금까지의 핵심 결과

### 1. Controlled Eval에서 확인한 모델 성향

- `perfect_tileable`로 생각한 타일도 점수가 항상 매우 높게 나오지는 않았습니다.
  - 대략 `0.62 ~ 0.68` 수준이 나왔고, 이는 현재 점수 변환이 압축되어 있고 모델이 "완전한 seamless texture"에 더 엄격하기 때문으로 해석했습니다.
- `seam_shift_1px ~ 8px` 실험에서는 어긋남이 커질수록 점수가 내려가는 경향이 있었습니다.
  - 다만 감소폭은 크지 않았고, 이는 모델이 seam 변화에 반응은 하지만 민감도가 아주 높지는 않다는 뜻으로 봤습니다.
- `color_discontinuous` 실험에서는 예상과 달리 점수가 오르는 사례도 나왔습니다.
  - 이는 TexTile이 색 변화보다 구조적 경계 어긋남을 더 중요하게 보거나, 경계 색 변화를 부드러운 저주파 변화로 해석했을 가능성을 시사합니다.
- `structure_broken`은 사람이 보기엔 더 많이 망가져 보여도 `seam_shift_2px`와 비슷한 점수가 나온 경우가 있었습니다.
  - 즉, local artifact 개수보다 global seam consistency를 더 보는 경향이 있는 것으로 해석했습니다.

### 2. Game Tile에서 확인한 추가 성향

- 무지 배경 위에 하나의 오브젝트가 배치된 타일은, 사람이 보기엔 규칙적으로 반복되어도 TexTile 점수가 높지 않을 수 있습니다.
- 이유는 이 모델이 "오브젝트 반복의 규칙성"보다 "경계를 넘어 이어지는 텍스처 단서"를 더 중요하게 볼 가능성이 크기 때문입니다.
- 따라서 게임 타일 해석 시에는:
  - 바닥/벽/잔디처럼 텍스처형 타일에는 TexTile 점수를 더 신뢰할 수 있고
  - 문/기둥/장식처럼 오브젝트형 타일은 점수만으로 판단하기 어렵습니다.

## 팀원용 실행 준비

### 1. 저장소 받기

```bash
git clone <repo-url>
cd textile
```

### 2. Git LFS 받기

이 저장소의 `CNN_Images/controlled_eval` 이미지는 Git LFS로 관리됩니다.

```bash
git lfs install
git lfs pull
```

### 3. 가상환경 준비

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 이미지 폴더 정책

- `CNN_Images/controlled_eval/`
  - 저장소에 포함된 실험용 이미지
  - clone 후 `git lfs pull`로 받으면 바로 사용 가능
- `CNN_Images/gametile/`
  - 로컬 전용 폴더
  - `.gitignore`에 포함되어 있어서 자동으로 커밋되지 않음
  - 각자 Good/Bad 타일 이미지를 넣어서 사용
- `gradcam_outputs/`
  - Grad-CAM 실행 결과가 저장되는 생성 산출물
  - 재생성 가능하므로 저장소에는 올리지 않고 `.gitignore`로 관리

## `test.py` 설명

현재 [`test.py`](./test.py)는 TexTile 점수를 빠르게 확인하는 가장 단순한 스크립트입니다.

### 현재 동작

1. `GAME_TILE_EVAL_CONFIG`로 평가 설정을 잡습니다.
2. 지정된 이미지 1장을 읽습니다.
3. TexTile 점수와 raw logit을 출력합니다.

현재 기본 설정:

```python
GAME_TILE_EVAL_CONFIG = {
    "lambda_value": 0.25,
    "resolution": (512, 512),
    "number_tiles": 2,
}
```

현재 기본 이미지 경로:

```python
image = read_and_process_image("CNN_Images/gametile/Bad/rpgTile000_stylized_winter2Dgame.jpg")
```

### 실행 방법

```bash
cd textile
source venv/bin/activate
python test.py
```

### 출력 해석

- `raw logit`
  - 모델의 내부 판단값
  - `0`보다 크면 양성 쪽, `0`보다 작으면 음성 쪽
- `타일 점수`
  - raw logit을 `0~1`로 변환한 값
  - 사람이 보기 쉽게 만든 점수

### 주의사항

- `CNN_Images/gametile/`은 git에서 추적하지 않기 때문에, clone 후 바로 실행하려면 해당 경로에 이미지를 직접 넣어야 합니다.
- 바로 테스트만 해보고 싶다면 경로를 `controlled_eval` 이미지로 바꿔서 실행하면 됩니다.

예시:

```python
image = read_and_process_image("CNN_Images/controlled_eval/perfect_tileable/Dot_image.jpg")
```

## `gradcam_textile.py` 설명

[`gradcam_textile.py`](./gradcam_textile.py)는 TexTile이 **어느 위치를 보고 현재 점수를 냈는지** 시각화하는 스크립트입니다.

### 왜 쓰는가

- 모델이 경계를 실제로 보고 있는지 확인
- 오브젝트형 게임 타일에서 어디를 중요하게 보는지 확인
- 좋은 타일/나쁜 타일의 해석 차이를 시각적으로 비교

### 기본 실행 방법

```bash
cd textile
source venv/bin/activate
python gradcam_textile.py CNN_Images/gametile/Good/rpgTile173_stylized_Fall.jpg
```

### 자주 쓰는 옵션

```bash
python gradcam_textile.py CNN_Images/gametile/Good/rpgTile173_stylized_Fall.jpg \
  --resolution 512 \
  --number-tiles 2 \
  --target-layer features.9 \
  --output-dir gradcam_outputs
```

### 출력 파일

기본적으로 `gradcam_outputs/` 아래에 3개 파일이 저장됩니다.

- `*_preview.png`
  - TexTile이 실제로 본 2x2 tiled preview
- `*_heatmap.png`
  - 중요 위치만 색으로 표시한 heatmap
- `*_overlay.png`
  - preview 위에 heatmap을 겹쳐 놓은 이미지

## Grad-CAM 결과 해석 방법

### 색 의미

- 빨강/노랑
  - 현재 raw logit을 올리는 데 더 크게 기여한 위치
- 파랑/보라
  - 상대적으로 덜 중요하거나 기여가 작은 위치

### 해석할 때 주의할 점

- Grad-CAM은 "모델이 맞았는지"를 보여주는 것이 아니라,
  **현재 예측을 만들 때 어디를 많이 참고했는지**를 보여줍니다.
- 즉 모델이 틀린 예측을 했더라도, 빨간 부분은 "그 오답을 만드는 데 기여한 위치"일 수 있습니다.
- 그래서 좋은 타일/나쁜 타일을 비교할 때는:
  - seam 주변을 보는지
  - 오브젝트 내부만 보는지
  - 배경 전체를 보는지
  를 같이 확인해야 합니다.

### 현재까지 관찰한 예시

- `rpgTile173_stylized_Fall.jpg`의 경우, 빈 배경보다 오브젝트의 테두리와 코너 쪽을 더 강하게 보는 경향이 있었습니다.
- 이는 이 타일에서 모델이 경계 전체보다 오브젝트 윤곽에 더 주목하고 있을 가능성을 보여줍니다.

## 현재 해석 원칙

- TexTile 점수는 "절대적 정답"이 아니라 tileability에 대한 모델의 추정값입니다.
- 특히 오브젝트형 게임 타일은 사람이 느끼는 "규칙적 반복"과 TexTile의 "seamless texture" 정의가 다를 수 있습니다.
- 따라서 실험 시에는 아래 순서를 권장합니다.

1. 사람이 먼저 Good / Bad / Ambiguous를 정함
2. `test.py`로 raw logit과 score를 확인함
3. `gradcam_textile.py`로 모델이 실제로 어디를 보는지 확인함
4. 점수와 heatmap이 사람 판단과 얼마나 맞는지 비교함

## 앞으로의 권장 방향

- 게임 타일용 하이퍼파라미터 튜닝은 "점수 상승" 자체보다 Good/Bad 분리력이 좋아지는지를 기준으로 해야 합니다.
- `lambda_value`는 점수 표시 스케일을 바꾸는 값이고, `resolution`은 모델 실제 판단을 바꿀 수 있습니다.
- 따라서 튜닝 시에는:
  - 좋은 타일 평균이 올라가는지
  - 나쁜 타일도 같이 올라가지는 않는지
  - raw logit 기준 분리 간격이 더 좋아지는지
  를 함께 확인해야 합니다.
