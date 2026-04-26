# AdaIN Generalization Improvement Report (Updated)

## 1. 목표
처음 보는 스타일(holdout style)에도 잘 작동하도록 AdaIN decoder를 개선하고, 기존 코드 대비 수정점이 성능에 어떤 영향을 줬는지 정리한다.

## 2. 기존 코드 대비 수정 사항
### 2.1 학습 코드 구조 변경
- 기존 `train.py`는 `style_dir` 최상위 파일만 읽음(`glob('*')`).
- 개선 `train_improved.py`는 `--style_recursive`로 하위 폴더까지 재귀 로딩(`rglob('*')`) 가능.
- `collect_image_paths`, `split_train_holdout`를 추가해 학습/검증 분할을 코드 내에서 명시적으로 관리.

### 2.2 일반화 중심 검증/베스트 선택
- `--holdout_style_ratio 0.2`로 스타일 20%를 학습에서 제외하고 holdout 검증셋으로 사용.
- `--best_metric_target holdout_style`로 best checkpoint를 seen이 아니라 holdout 기준으로 저장.
- `eval_seen`, `eval_holdout_style`, `eval/tracked_metric`를 로그로 남겨 일반화 성능을 직접 추적.

### 2.3 학습 안정화/품질 개선 항목
- `RandomHorizontalFlip` 추가(학습 다양성 증가).
- 추가 손실 항목 도입: `gram_style_weight`, `recon_weight`, `tv_weight`(선택).
- EMA decoder(`--use_ema_eval`)로 평가 안정화.

### 2.4 실행 스크립트
- `train_generalize.sh`를 추가해 일반화 목적 하이퍼파라미터를 고정 재현 가능하게 구성.
- 실제 실행: `./train_generalize.sh --n_threads 4`

## 3. 이번 학습 설정
- Run ID: `generalize_20260401_172313`
- Steps: 10,000
- Content split: train 230 / holdout 0 (`input/content/PNG`)
- Style split: train 54 / holdout 14 (`input/style`, recursive)
- Best checkpoint from log: **iter 7500** (tracked holdout-style metric 최소)

## 4. 정량 결과
### 4.1 학습 중 곡선 (로그 기반)
![Train curve](../../report_assets/figures/generalize_train_curve.png)

### 4.2 과적합 지표 (Generalization gap)
![Gap curve](../../report_assets/figures/generalize_gap_curve.png)

- Log best: iter 7500, holdout_style=18.755139, seen=16.387644
- Final(10000): holdout_style=20.296628, seen=13.645930
- 해석: 후반부(7500 이후)에는 seen은 더 좋아지지만 holdout은 악화되어 과적합이 관찰됨.

### 4.3 동일 분할 기준 비교 평가 (추가 재평가)
![Baseline vs Best vs Final](../../report_assets/figures/generalize_baseline_best_final.png)

| Model | Step | Seen Metric | Holdout-style Metric | Gap(Holdout-Seen) |
|---|---:|---:|---:|---:|
| Baseline | 0 | 31.9552 | 28.7422 | -3.2130 |
| Best checkpoint | 7500 | 13.7401 | 20.0685 | 6.3284 |
| Final checkpoint | 10000 | 13.4022 | 20.0088 | 6.6066 |

- Baseline 대비 Best(holdout 기준) 개선율: **30.18%**
- Baseline 대비 Best(seen 기준) 개선율: **57.00%**

## 5. 시각 결과 (Before vs After)
### Sample A
![Sample A](../../report_assets/figures/compare_photo_generalize.png)

### Sample B
![Sample B](../../report_assets/figures/compare_tile_generalize.png)

## 6. 결론
- 코드 수정으로 스타일 데이터 커버리지가 늘고, holdout 기반 선택이 가능해져 일반화 지표 자체를 개선할 수 있었다.
- 단, 10k까지 계속 학습하면 holdout이 다시 악화되므로 이번 런에서는 `best_decoder.pth.tar`(iter 7500)를 사용하는 것이 타당하다.

## 7. 재현 산출물
- Report summary JSON: `../../report_assets/data/generalize_report_summary.json`
- Eval curve CSV: `../../report_assets/data/generalize_eval_curve.csv`
- Best model: `experiments_improved/generalize_20260401_172313/best_decoder.pth.tar`