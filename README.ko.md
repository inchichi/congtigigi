# pytorch-AdaIN

이 저장소는 다음 논문의 비공식 PyTorch 구현입니다.

- X. Huang, S. Belongie, *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization* (ICCV 2017)

Torch 원본 구현:
https://github.com/xunhuang1995/AdaIN-style

![Results](results.png)

## 1) 환경 설정

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

권장 Python 버전: **3.10 ~ 3.12**

## 2) 프로젝트 구조

```text
.
├── input/                      # 콘텐츠/스타일 샘플 데이터
├── models/                     # 사전학습 가중치(decoder.pth, vgg_normalised.pth)
├── output/                     # 생성 결과(깃 추적 제외)
├── report_assets/              # 리포트 생성 스크립트 + 생성 산출물
├── docs/
│   ├── papers/                 # 참고 논문
│   └── reports/                # 생성된 리포트 문서
├── test.py                     # 이미지 스타일 변환
├── test_video.py               # 비디오 스타일 변환
├── train.py                    # 기본 학습
├── train_improved.py           # 개선/일반화 학습
├── train_generalize.sh         # 개선 학습 프리셋
├── run.sh                      # RGBA PNG 안전 변환 래퍼
└── run_best.sh                 # best decoder 추론 래퍼
```

## 3) 모델 준비

`models/` 아래에 가중치를 두세요.

- `models/decoder.pth`
- `models/vgg_normalised.pth`

기본 가중치 릴리스:
https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0

## 4) 추론

### 단일 콘텐츠 + 단일 스타일

```bash
python test.py \
  --content input/content/cornell.jpg \
  --style input/style/woman_with_hat_matisse.jpg \
  --output output
```

### 폴더 배치: 모든 콘텐츠 x 모든 스타일 조합

```bash
python test.py \
  --content_dir input/content/PNG \
  --style_dir input/style/test \
  --output output
```

### 스타일 여러 장을 1개 스타일로 합성

```bash
python test.py \
  --content_dir input/content/PNG \
  --style input/style/a.jpg,input/style/b.jpg,input/style/c.jpg \
  --style_interpolation_weights 1,1,1 \
  --content_size 64 --style_size 64 --crop \
  --output output
```

주의: 스타일 합성 모드에서는 스타일 텐서 크기가 같아야 합니다. 안전하게 `--content_size`, `--style_size`, `--crop`를 함께 지정하세요.

### 권장 실행 래퍼(RGBA -> RGB 자동 변환)

```bash
./run_best.sh \
  --decoder experiments_improved/<run_name>/best_decoder.pth.tar \
  --content_dir input/content/PNG \
  --style_dir input/style/test \
  --output output/new_run
```

## 5) 학습

### 기본 학습

```bash
python train.py --content_dir <content_dir> --style_dir <style_dir>
```

### 일반화 개선 학습 프리셋

```bash
./train_generalize.sh
```

로그/체크포인트 출력 위치:

- `logs_improved/`
- `experiments_improved/`

## 6) 리포트

리포트 지표/그래프 생성:

```bash
python report_assets/data/generate_report_assets.py
python report_assets/data/generate_generalization_report.py
```

리포트 문서 위치:

- `docs/reports/research_paper_adain_improvement.md`
- `docs/reports/research_paper_adain_improvement.html`
- `docs/reports/research_paper_adain_improvement.pdf`

## 참고

- [1] X. Huang and S. Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*, ICCV 2017.
- [2] https://github.com/xunhuang1995/AdaIN-style
