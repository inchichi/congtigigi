"""몬스터 스프라이트 시트 스타일 변환 — 배경 보존.

게임은 몬스터 시트를 매 로드마다 색(밝기·채도) 기반 전경 판정으로 프레임을 잘라낸다
(loadMonsterPigAnimationTextures.ts / loadMonsterSlimeAnimationTextures.ts). 따라서 시트
전체를 AdaIN으로 칠하면 배경색이 바뀌어 프레임 슬라이싱이 깨지고, 몬스터가 사각형 덩어리로
렌더된다.

해결: 게임과 동일한 전경 판정으로 마스크를 만들어, 전경(캐릭터)만 스타일 결과로 바꾸고
배경 픽셀은 원본 그대로 둔다. 그러면 게임의 슬라이싱이 동일하게 동작해 몬스터 모양이
유지되고 캐릭터 색만 바뀐다.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

import adain_service

# 게임 로더의 isPig/isSlimeForegroundPixel과 동일한 "배경" 조건. brightness=(r+g+b)/3,
# saturation=max(r,g,b)-min(r,g,b). 전경 = NOT(배경) (+ 알파 0은 항상 배경).
def _background_mask(rgb: np.ndarray, monster_key: str) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    brightness = (r + g + b) / 3.0
    saturation = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    if monster_key == "pig":
        return ((brightness < 18) & (saturation < 18)) | (
            (brightness > 76) & (saturation < 28)
        )
    if monster_key == "slime":
        return (
            ((brightness > 220) & (saturation < 28))
            | ((saturation < 24) & (brightness < 165))
            | ((brightness < 70) & (saturation < 18))
        )
    raise ValueError(f"알 수 없는 몬스터 종류: {monster_key}")


def stylize_monster_sheet(
    sheet: Image.Image,
    style: Image.Image,
    monster_key: str,
    alpha: float = 1.0,
    alpha_erode: int = 0,
) -> Image.Image:
    """전경만 스타일, 배경은 원본 보존. 원본과 정확히 같은 크기를 반환한다."""
    has_alpha = adain_service._has_alpha(sheet)
    original = sheet.convert("RGBA") if has_alpha else sheet.convert("RGB")

    # 전체 시트 스타일 변환(원본 크기 보존). 합성은 원본 해상도에서 한다.
    styled = adain_service.style_transfer_image(
        sheet, style, alpha=alpha, content_size=512, style_size=512, preserve_size=True
    )

    orig_rgb = np.array(original.convert("RGB"))
    styled_rgb = np.array(styled.convert("RGB"))

    foreground = ~_background_mask(orig_rgb, monster_key)
    if has_alpha:
        foreground &= np.array(original.getchannel("A")) != 0

    # 경계 침식(옵션): 전경을 살짝 줄여 배경 경계의 색 번짐을 줄인다.
    if alpha_erode > 0:
        mask_img = Image.fromarray((foreground * 255).astype(np.uint8), "L").filter(
            ImageFilter.MinFilter(alpha_erode * 2 + 1)
        )
        foreground = np.array(mask_img) > 127

    out_rgb = np.where(foreground[..., None], styled_rgb, orig_rgb).astype(np.uint8)

    if has_alpha:
        out = Image.fromarray(out_rgb, "RGB").convert("RGBA")
        out.putalpha(original.getchannel("A"))
        return out
    return Image.fromarray(out_rgb, "RGB")
