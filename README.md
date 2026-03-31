# pytorch-AdaIN

이 저장소는 Huang 등(ICCV 2017)의 논문
"Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"의
비공식 PyTorch 구현입니다.

저자 분들의 Torch 원본 구현([AdaIN-style](https://github.com/xunhuang1995/AdaIN-style))을 많이 참고했고,
큰 도움을 받았습니다.

![Results](results.png)

## 요구 사항
아래 명령으로 의존성을 설치하세요.

```bash
pip install -r requirements.txt
```

- Python 3.5+
- PyTorch 0.4+
- TorchVision
- Pillow

(학습 시 선택)
- tqdm
- TensorboardX

## 사용 방법

### 모델 다운로드
[release](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0)에서
`decoder.pth`, `vgg_normalised.pth`를 다운로드한 뒤 `models/` 아래에 넣으세요.

### 테스트
콘텐츠 이미지와 스타일 이미지를 각각 `--content`, `--style`로 지정합니다.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg
```

`--content_dir`, `--style_dir`를 사용하면 폴더 단위로 모든 조합을 생성할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content_dir input/content --style_dir input/style
```

아래는 `--style`과 `--style_interpolation_weights`를 이용해
4개 스타일을 혼합하는 예시입니다.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/avril.jpg --style input/style/picasso_self_portrait.jpg,input/style/impronte_d_artista.jpg,input/style/trial.jpg,input/style/antimonocromatismo.jpg --style_interpolation_weights 1,1,1,1 --content_size 512 --style_size 512 --crop
```

기타 주요 옵션:
- `--content_size`: 콘텐츠 이미지 최소 크기(0이면 원본 유지)
- `--style_size`: 스타일 이미지 최소 크기(0이면 원본 유지)
- `--alpha`: 스타일 강도 조절(기본값 1.0, 범위 0.0~1.0)
- `--preserve_color`: 콘텐츠 이미지 색상 보존

### 학습
콘텐츠/스타일 이미지 디렉터리를 지정해 학습합니다.

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --content_dir <content_dir> --style_dir <style_dir>
```

세부 파라미터는 `--help`를 참고하세요.

이 코드로 학습된 모델(`iter_1000000.pth`)은
[release](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0)에서 제공합니다.

## 참고 문헌
- [1] X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", ICCV, 2017.
- [2] [Torch 원본 구현](https://github.com/xunhuang1995/AdaIN-style)

## 최근 작업 정리 (2026-04-01)

### 1) 데이터/브랜치 정리
- `input/content/PNG` 내 `*:Zone.Identifier` 파일 229개를 정리했습니다.
- 작업 브랜치 기준으로 `main`, `adain_yc` 브랜치를 구성했고, 현재 작업 내용은 `adain_yc`에 반영합니다.

### 2) 스타일 변환 배치 실행
- 콘텐츠: `input/content/PNG`의 타일 이미지 229장
- 스타일: 계절별 `2D`, `real` 폴더 (`spring/summer/fall/winter`)
- 실행 방식: 스타일 보간(`--style_interpolation_weights`) 적용
- 결과 폴더: `output/batch_interp/<season>_<type>`
- 생성 수량: 각 폴더 229장씩

### 3) 평가 지표 계산
타일별로 아래 지표를 계산했습니다.
- Content Loss: VGG `relu4_1`에서 AdaIN target feature와 출력 feature의 MSE
- Style Loss: VGG `relu1_1`~`relu4_1`에서 평균/표준편차 MSE 합
- FPS: 단일 추론(encoder+AdaIN+decoder) 기준 처리 속도

요약 평균:

| group | count | content_loss_mean | style_loss_mean | fps_mean |
|---|---:|---:|---:|---:|
| spring_2D | 229 | 3.211147 | 1.720949 | 4.650787 |
| spring_real | 229 | 1.999379 | 0.991212 | 4.347069 |
| summer_2D | 229 | 2.505165 | 3.326356 | 4.594342 |
| summer_real | 229 | 2.161388 | 1.443016 | 4.372921 |
| fall_2D | 229 | 2.947417 | 2.012072 | 4.358770 |
| fall_real | 229 | 2.747492 | 1.264818 | 4.443353 |
| winter_2D | 229 | 2.087853 | 1.324905 | 4.273178 |
| winter_real | 229 | 1.723523 | 0.982994 | 4.628149 |
| OVERALL | 1832 | 2.422921 | 1.633290 | 4.458571 |

세부 결과 파일:
- `output/batch_interp/metrics_per_tile.csv`
- `output/batch_interp/metrics_summary.csv`

### 4) 추가 스크립트
- `run_batch_interp.sh`: 계절/스타일 타입별 보간 배치 실행 스크립트
- `evaluate_batch_metrics.py`: Content/Style/FPS 지표 계산 및 CSV 저장 스크립트
