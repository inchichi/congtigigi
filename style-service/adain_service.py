"""AdaIN 스타일 트랜스퍼 래퍼.

ADAIN 원본 코드는 수정하지 않는다 — 순수 모듈인 net.py/function.py만 import하고,
test.py의 추론 흐름(VGG 인코딩 → AdaIN → alpha 블렌딩 → 디코딩)을 여기서 재현한다.
(test.py는 __main__ 가드 없이 모듈 최상위에서 argparse와 배치 추론을 실행하므로
직접 import할 수 없다.)

경로는 config.json + 환경변수로만 정한다(하드코딩 금지):
  ADAIN_DIR / ADAIN_VGG_PATH / ADAIN_DECODER_PATH
  STYLE_SERVICE_HOST / STYLE_SERVICE_PORT
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import transforms

_SERVICE_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _SERVICE_DIR / "config.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_lock = threading.Lock()
_models: tuple[nn.Module, nn.Module] | None = None
_config: dict | None = None


def get_config() -> dict:
    """config.json을 읽고 환경변수로 덮어쓴 뒤 절대 경로로 정규화한다."""
    global _config
    if _config is not None:
        return _config

    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))

    adain_dir = Path(os.environ.get("ADAIN_DIR", raw["adainDir"]))
    if not adain_dir.is_absolute():
        adain_dir = (_SERVICE_DIR / adain_dir).resolve()

    def _resolve_weight(env_name: str, key: str) -> Path:
        value = Path(os.environ.get(env_name, raw[key]))
        return value if value.is_absolute() else (adain_dir / value).resolve()

    project_dir = Path(os.environ.get("EDITOR_PROJECT_DIR", raw["projectDir"]))
    if not project_dir.is_absolute():
        project_dir = (_SERVICE_DIR / project_dir).resolve()

    _config = {
        "adain_dir": adain_dir,
        "vgg_path": _resolve_weight("ADAIN_VGG_PATH", "vggPath"),
        "decoder_path": _resolve_weight("ADAIN_DECODER_PATH", "decoderPath"),
        "project_dir": project_dir,
        "assets_subdir": raw.get("assetsSubdir", "src/assets"),
        "host": os.environ.get("STYLE_SERVICE_HOST", raw["host"]),
        "port": int(os.environ.get("STYLE_SERVICE_PORT", raw["port"])),
    }
    return _config


def _get_models() -> tuple[nn.Module, nn.Module]:
    """vgg(relu4-1까지 절단)와 decoder를 최초 1회만 로드한다. 스레드 안전."""
    global _models
    with _lock:
        if _models is None:
            config = get_config()
            for key in ("vgg_path", "decoder_path"):
                if not config[key].is_file():
                    raise FileNotFoundError(f"모델 가중치가 없습니다: {config[key]}")

            sys.path.insert(0, str(config["adain_dir"]))
            try:
                import net  # ADAIN 원본 모듈 (부작용 없는 모델 정의)
            finally:
                sys.path.pop(0)

            decoder = net.decoder
            decoder.load_state_dict(torch.load(config["decoder_path"], map_location=device))
            vgg = net.vgg
            vgg.load_state_dict(torch.load(config["vgg_path"], map_location=device))
            # test.py:179와 동일 — relu4-1 출력까지만 사용한다.
            vgg = nn.Sequential(*list(vgg.children())[:31])
            vgg.to(device).eval()
            decoder.to(device).eval()
            _models = (vgg, decoder)
    return _models


def _adain():
    """function.py의 adaptive_instance_normalization을 ADAIN 폴더에서 가져온다."""
    config = get_config()
    sys.path.insert(0, str(config["adain_dir"]))
    try:
        from function import adaptive_instance_normalization
    finally:
        sys.path.pop(0)
    return adaptive_instance_normalization


def _to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    """test.py의 test_transform과 동일: (옵션) 짧은 변 기준 리사이즈 + [0,1] 텐서화."""
    steps = []
    if size > 0:
        steps.append(transforms.Resize(size))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)(image.convert("RGB")).unsqueeze(0).to(device)


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    )


def erode_alpha(alpha_channel: Image.Image, pixels: int) -> Image.Image:
    """알파 마스크 형태학적 침식 — 경계의 스타일 색 번짐 완화용.

    MinFilter는 창 안의 최솟값만 취하므로 새 중간값(안티앨리어싱)을 만들지 않는다 —
    픽셀아트의 깔끔한 경계가 유지된다. pixels 0이면 그대로 반환.
    """
    if pixels <= 0:
        return alpha_channel
    return alpha_channel.filter(ImageFilter.MinFilter(pixels * 2 + 1))


def style_transfer_image(
    content: Image.Image,
    style: Image.Image,
    alpha: float = 1.0,
    content_size: int = 512,
    style_size: int = 512,
    alpha_erode: int = 0,
    preserve_size: bool = False,
) -> Image.Image:
    """PIL 이미지를 받아 스타일이 적용된 PIL 이미지를 반환한다. size 0은 원본 크기 유지.

    콘텐츠에 알파 채널이 있으면(투명 배경 스프라이트): RGB만 모델에 넣되 투명부를 중성
    회색으로 합성해 넣고(검정 배경 오염 방지), 결과를 원본 해상도로 되돌린 뒤 원본 알파를
    재합성한 RGBA를 반환한다 — 누끼가 원본과 동일하게 유지된다. alpha_erode(px)는 합성 전
    알파를 침식해 경계 색 번짐을 줄이는 옵션.

    preserve_size=True면 알파가 없어도 결과를 원본 해상도로 정확히 되돌린다 — 게임 에셋/
    스프라이트 시트를 덮어쓸 때 프레임 좌표가 어긋나지 않게 하려는 용도(예: 몬스터 시트는
    하드코딩된 프레임 rect로 잘리므로 크기가 1px만 달라도 안 됨). 알파 없는 일반 파일
    업로드(미리보기용)는 기존처럼 모델 출력 크기를 그대로 둔다.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha는 0.0~1.0 사이여야 합니다: {alpha}")

    source_alpha: Image.Image | None = None
    original_size = content.size
    if _has_alpha(content):
        rgba = content.convert("RGBA")
        source_alpha = rgba.getchannel("A")
        gray = Image.new("RGB", rgba.size, (128, 128, 128))
        gray.paste(rgba, mask=source_alpha)
        content = gray

    vgg, decoder = _get_models()
    adain = _adain()
    content_tensor = _to_tensor(content, content_size)
    style_tensor = _to_tensor(style, style_size)

    # CPU 추론을 직렬화한다 — 동시 요청이 메모리를 두 배로 쓰는 것을 막는다.
    with _lock, torch.no_grad():
        content_feat = vgg(content_tensor)
        style_feat = vgg(style_tensor)
        feat = adain(content_feat, style_feat)
        feat = feat * alpha + content_feat * (1 - alpha)
        output = decoder(feat)

    output = output.clamp(0, 1).squeeze(0).cpu()
    result = transforms.ToPILImage()(output)

    # 원본 해상도 복원: 알파 스프라이트(누끼 보존)거나 preserve_size(에셋 덮어쓰기)일 때.
    # VGG가 8배 다운/업샘플(ceil)이라 출력이 모델 입력보다 약간 클 수 있어, 입력 크기로 자른 뒤
    # 원본 해상도로 되돌려야 픽셀/프레임 좌표가 정확히 맞는다.
    if source_alpha is not None or preserve_size:
        model_height, model_width = content_tensor.shape[-2:]
        result = result.crop((0, 0, model_width, model_height))
        if result.size != original_size:
            resample = Image.BOX if result.width >= original_size[0] else Image.LANCZOS
            result = result.resize(original_size, resample)

    if source_alpha is None:
        return result

    result = result.convert("RGBA")
    result.putalpha(erode_alpha(source_alpha, alpha_erode))
    return result


def style_transfer(
    content_path: str | Path,
    style_path: str | Path,
    alpha: float = 1.0,
    output_path: str | Path | None = None,
    content_size: int = 512,
    style_size: int = 512,
    alpha_erode: int = 0,
) -> str:
    """파일 경로 기반 API: 결과 PNG를 저장하고 그 경로를 반환한다."""
    content_path = Path(content_path)
    style_path = Path(style_path)
    result = style_transfer_image(
        Image.open(content_path),
        Image.open(style_path),
        alpha=alpha,
        content_size=content_size,
        style_size=style_size,
        alpha_erode=alpha_erode,
    )
    if output_path is None:
        output_path = content_path.with_name(
            f"{content_path.stem}_stylized_{style_path.stem}.png"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    return str(output_path)
