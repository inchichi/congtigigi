"""맵 오브젝트(타일 군집) 부분 스타일 변환.

에디터가 보낸 셀 목록(맵 좌표 + 타일셋 로컬 타일 id)으로 오브젝트의 타일들을 맵 배치 그대로
한 장으로 조립해 AdaIN을 적용하고, 변환된 픽셀을 타일셋 이미지의 해당 타일 자리에 되써넣는다.
타일셋을 패치하므로 같은 타일을 쓰는 동일 오브젝트(예: 같은 모양의 나무 전부)가 함께 바뀐다.
"""

from __future__ import annotations

from PIL import Image

import adain_service

# 픽셀아트 타일은 원본이 매우 작아(타일 몇 장 = 수십 px) 그대로는 스타일 통계가 빈약하다.
# 정수배 NEAREST 업스케일(픽셀 경계 유지) 후 변환하고, BOX 다운스케일로 되돌린다.
DEFAULT_WORK_SIZE = 512

# 변환 결과에서 원본이 완전 투명했던 픽셀은 되살리지 않는다 — 타일 모양(알파)은 보존된다.
_BACKGROUND_GRAY = (128, 128, 128)


def _tile_rect(tile_id: int, columns: int, tile_width: int, tile_height: int) -> tuple[int, int, int, int]:
    left = (tile_id % columns) * tile_width
    top = (tile_id // columns) * tile_height
    return (left, top, left + tile_width, top + tile_height)


def compose_object_canvas(
    tileset_image: Image.Image,
    cells: list[dict],
    columns: int,
    tile_width: int,
    tile_height: int,
) -> tuple[Image.Image, int, int]:
    """오브젝트의 셀들을 맵 배치 그대로 투명 캔버스에 조립한다 — 결과가 곧 누끼 RGBA.

    (canvas, min_col, min_row)를 반환한다. min_col/min_row는 역패치(셀 → 캔버스 영역)에 필요.
    타일셋 원본 타일이 알파를 갖고 있으므로 별도 배경 제거가 필요 없다.
    """
    tileset_image = tileset_image.convert("RGBA")
    for cell in cells:
        rect = _tile_rect(cell["tileId"], columns, tile_width, tile_height)
        if rect[2] > tileset_image.width or rect[3] > tileset_image.height:
            raise ValueError(f"타일 id {cell['tileId']}가 타일셋 이미지 범위를 벗어납니다.")

    min_col = min(cell["col"] for cell in cells)
    min_row = min(cell["row"] for cell in cells)
    width_tiles = max(cell["col"] for cell in cells) - min_col + 1
    height_tiles = max(cell["row"] for cell in cells) - min_row + 1
    canvas_width = width_tiles * tile_width
    canvas_height = height_tiles * tile_height
    if canvas_width * canvas_height > 4096 * 4096:
        raise ValueError("오브젝트가 너무 큽니다(셀 좌표 범위 초과).")
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    for cell in cells:
        tile = tileset_image.crop(_tile_rect(cell["tileId"], columns, tile_width, tile_height))
        canvas.paste(tile, ((cell["col"] - min_col) * tile_width, (cell["row"] - min_row) * tile_height))
    return canvas, min_col, min_row


def patch_tileset_from_object(
    tileset_image: Image.Image,
    styled_object: Image.Image,
    cells: list[dict],
    columns: int,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    """스타일이 적용된 오브젝트 RGBA를 타일별로 잘라 타일셋의 해당 타일 자리에 되써넣는다.

    styled_object는 compose_object_canvas와 같은 크기여야 한다(크기로 검증).
    알파는 styled_object의 것을 그대로 쓴다 — 알파 보존/침식이 이미 반영된 상태.
    """
    tileset_image = tileset_image.convert("RGBA")
    # compose와 동일한 경계 검증 — 추출 이후 타일셋이 교체되어 작아졌으면 PIL paste가
    # 조용히 클리핑해 "성공"으로 보이는 무패치가 된다. 명시적으로 실패시킨다(→422).
    for cell in cells:
        rect = _tile_rect(cell["tileId"], columns, tile_width, tile_height)
        if rect[2] > tileset_image.width or rect[3] > tileset_image.height:
            raise ValueError(f"타일 id {cell['tileId']}가 타일셋 이미지 범위를 벗어납니다.")
    min_col = min(cell["col"] for cell in cells)
    min_row = min(cell["row"] for cell in cells)
    width_tiles = max(cell["col"] for cell in cells) - min_col + 1
    height_tiles = max(cell["row"] for cell in cells) - min_row + 1
    expected = (width_tiles * tile_width, height_tiles * tile_height)
    if styled_object.size != expected:
        raise ValueError(
            f"오브젝트 이미지 크기가 맞지 않습니다: {styled_object.size} (기대: {expected})"
        )

    styled_object = styled_object.convert("RGBA")
    patched = tileset_image.copy()
    seen_tile_ids: set[int] = set()
    for cell in cells:
        if cell["tileId"] in seen_tile_ids:
            continue
        seen_tile_ids.add(cell["tileId"])
        region = (
            (cell["col"] - min_col) * tile_width,
            (cell["row"] - min_row) * tile_height,
            (cell["col"] - min_col + 1) * tile_width,
            (cell["row"] - min_row + 1) * tile_height,
        )
        patched.paste(styled_object.crop(region), _tile_rect(cell["tileId"], columns, tile_width, tile_height))
    return patched


def stylize_tiles(
    tileset_image: Image.Image,
    cells: list[dict],
    columns: int,
    tile_width: int,
    tile_height: int,
    style_image: Image.Image,
    alpha: float,
    work_size: int = DEFAULT_WORK_SIZE,
    alpha_erode: int = 0,
) -> tuple[Image.Image, Image.Image]:
    """(오브젝트 미리보기 RGBA, 패치된 타일셋 RGBA)를 반환한다."""
    # 1) 오브젝트 조립(누끼 캔버스).
    canvas, _min_col, _min_row = compose_object_canvas(
        tileset_image, cells, columns, tile_width, tile_height
    )

    # 업스케일 배율을 먼저 계산해 작업 면적에 상한을 둔다 — 멀리 떨어진 셀 두 개만으로
    # bbox가 거대해지고(셀 극값 기준) 거기에 업스케일까지 곱해지면 메모리가 폭주한다.
    scale = max(1, round(work_size / min(canvas.size))) if work_size > 0 else 1
    if canvas.width * canvas.height * scale * scale > 4096 * 4096:
        raise ValueError("변환 대상이 너무 큽니다(셀 좌표 범위 초과). 더 작은 오브젝트를 선택하세요.")

    # 2) 정수배 업스케일 + 중성 회색 배경 합성(투명부가 검정으로 새는 halo 방지) 후 변환.
    work = (
        canvas.resize((canvas.width * scale, canvas.height * scale), Image.NEAREST)
        if scale > 1
        else canvas
    )
    base = Image.new("RGB", work.size, _BACKGROUND_GRAY)
    base.paste(work, mask=work.getchannel("A"))
    result_rgb = adain_service.style_transfer_image(
        base, style_image, alpha=alpha, content_size=0, style_size=512
    )
    # VGG가 8배 다운/업샘플(ceil)이라 출력이 입력보다 약간 클 수 있다 — 원래 크기로 자른다.
    result_rgb = result_rgb.crop((0, 0, work.width, work.height))
    if scale > 1:
        result_rgb = result_rgb.resize(canvas.size, Image.BOX)

    # 3) 알파: 조립 캔버스 전체에서 침식한다 — 타일 경계가 맞닿은 안쪽(불투명)은 깎이지
    #    않고 오브젝트의 진짜 외곽(투명 경계)만 깎여, 타일별 침식 때 생기는 이음새가 없다.
    object_alpha = adain_service.erode_alpha(canvas.getchannel("A"), alpha_erode)

    # 오브젝트 미리보기: 변환 RGB + (침식된) 원본 알파 — 모양 보존.
    preview = result_rgb.convert("RGBA")
    preview.putalpha(object_alpha)

    # 4) 타일셋 패치: 미리보기(RGBA)를 타일별로 잘라 되써넣는다.
    patched = patch_tileset_from_object(
        tileset_image, preview, cells, columns, tile_width, tile_height
    )

    return preview, patched
