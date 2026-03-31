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
