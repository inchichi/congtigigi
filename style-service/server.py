"""AdaIN 스타일 트랜스퍼 로컬 HTTP 서비스.

에디터(Vite dev 서버)의 /api/style 프록시가 이 서버를 가리킨다.
실행: python server.py  (style-service 폴더에서)

엔드포인트
  GET  /health          서비스 상태 + 디바이스 정보
  POST /style-transfer  multipart(content, style 이미지 + alpha, content_size, style_size)
                        → 결과 PNG 바이트
  GET  /assets          프로젝트 src/games/my-sample-rpg/assets의 PNG 목록 (에디터 '게임 에셋' 선택용)
  POST /stylize-object  맵 오브젝트(타일 군집) 부분 변환 → 오브젝트 미리보기 + 패치된 타일셋
  POST /apply-asset     결과 PNG를 src/games/my-sample-rpg/assets의 기존 파일에 백업 후 덮어쓰기 (게임 즉시 반영)

동기 엔드포인트는 FastAPI가 스레드풀에서 실행하므로 추론 중에도
서버(및 health 체크)는 블로킹되지 않는다.
"""

import base64
import io
import json
from urllib.parse import urlparse

import torch
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

import adain_service
import asset_store
import object_extract
import tile_stylize

app = FastAPI(title="AdaIN style-transfer service")

# 드라이브-바이 방어: multipart POST는 CORS preflight 없이 어느 웹사이트에서든 127.0.0.1로
# 직접 보낼 수 있다(응답은 못 읽어도 쓰기는 성공). Origin 헤더가 있는 변조 요청은 로컬 출처
# (에디터 dev 서버 — Vite가 포트를 5174 등으로 옮겨도 허용)만 통과시킨다.
# Origin이 없는 요청(curl 등 비브라우저 도구)은 브라우저 위협 모델 밖이라 통과.
def _is_local_origin(origin: str) -> bool:
    host = urlparse(origin).hostname
    return host in ("localhost", "127.0.0.1", "::1")


@app.middleware("http")
async def reject_foreign_origins(request, call_next):
    origin = request.headers.get("origin")
    if request.method == "POST" and origin is not None and not _is_local_origin(origin):
        return JSONResponse(status_code=403, content={"error": f"허용되지 않은 출처입니다: {origin}"})
    return await call_next(request)


@app.get("/health")
def health() -> dict:
    # 모델은 첫 요청에서 지연 로드되므로, 가중치 존재 여부를 여기서 미리 확인해
    # 변환이 100% 실패할 상태("degraded")를 에디터에 정직하게 알린다.
    config = adain_service.get_config()
    missing = [
        str(config[key])
        for key in ("vgg_path", "decoder_path")
        if not config[key].is_file()
    ]
    return {
        "status": "ok" if not missing else "degraded",
        "missing_weights": missing,
        "device": str(adain_service.device),
        "torch": torch.__version__,
    }


@app.post("/style-transfer")
def style_transfer(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    alpha: float = Form(1.0),
    content_size: int = Form(512),
    style_size: int = Form(512),
    alpha_erode: int = Form(0),
):
    if not 0.0 <= alpha <= 1.0:
        return JSONResponse(status_code=422, content={"error": "alpha는 0.0~1.0 사이여야 합니다."})
    if not 0 <= alpha_erode <= 3:
        return JSONResponse(status_code=422, content={"error": "alpha_erode는 0~3 사이여야 합니다."})
    # 64 미만은 VGG의 maxpool/reflection pad에서 크래시, 과대 값은 CPU 메모리 폭주.
    for name, value in (("content_size", content_size), ("style_size", style_size)):
        if value != 0 and not 64 <= value <= 2048:
            return JSONResponse(
                status_code=422,
                content={"error": f"{name}은 0(원본 유지) 또는 64~2048 사이여야 합니다."},
            )

    try:
        # PIL은 헤더만 읽고 본문 디코딩을 미루므로, load()로 즉시 디코딩해
        # 잘린 이미지가 추론 중 500으로 새지 않고 여기서 422로 잡히게 한다.
        content_image = Image.open(io.BytesIO(content.file.read()))
        content_image.load()
        style_image = Image.open(io.BytesIO(style.file.read()))
        style_image.load()
    except OSError:
        return JSONResponse(status_code=422, content={"error": "이미지 파일을 해석할 수 없습니다."})

    # 원본 유지(0)는 업로드 해상도가 곧 추론 해상도 — 상한 없이는 CPU 메모리가 무제한이다.
    for name, image, size in (("content", content_image, content_size), ("style", style_image, style_size)):
        if size == 0 and image.width * image.height > 4096 * 4096:
            return JSONResponse(
                status_code=422,
                content={"error": f"{name} 이미지가 원본 유지 한도(4096×4096 픽셀)를 초과합니다. 변환 크기를 지정하세요."},
            )

    try:
        result = adain_service.style_transfer_image(
            content_image,
            style_image,
            alpha=alpha,
            content_size=content_size,
            style_size=style_size,
            alpha_erode=alpha_erode,
        )
    except FileNotFoundError as error:
        # 가중치/ADAIN 경로 미설정 — 원인 메시지를 그대로 전달한다.
        return JSONResponse(status_code=503, content={"error": str(error)})
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.get("/assets")
def list_assets() -> dict:
    return {"assets": asset_store.list_assets()}


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.post("/stylize-object")
def stylize_object(
    style: UploadFile = File(...),
    alpha: float = Form(1.0),
    tileset_path: str = Form(...),
    tile_width: int = Form(...),
    tile_height: int = Form(...),
    columns: int = Form(...),
    cells: str = Form(...),
    work_size: int = Form(tile_stylize.DEFAULT_WORK_SIZE),
    alpha_erode: int = Form(0),
):
    if not 0.0 <= alpha <= 1.0:
        return JSONResponse(status_code=422, content={"error": "alpha는 0.0~1.0 사이여야 합니다."})
    if not 0 <= alpha_erode <= 3:
        return JSONResponse(status_code=422, content={"error": "alpha_erode는 0~3 사이여야 합니다."})
    if not 1 <= tile_width <= 512 or not 1 <= tile_height <= 512 or columns < 1:
        return JSONResponse(status_code=422, content={"error": "타일 크기/열 수가 올바르지 않습니다."})
    if work_size != 0 and not 64 <= work_size <= 2048:
        return JSONResponse(status_code=422, content={"error": "work_size는 0 또는 64~2048 사이여야 합니다."})

    try:
        cell_list = json.loads(cells)
        assert isinstance(cell_list, list) and 0 < len(cell_list) <= 2048
        for cell in cell_list:
            assert isinstance(cell["col"], int) and isinstance(cell["row"], int)
            assert isinstance(cell["tileId"], int) and cell["tileId"] >= 0
    except (AssertionError, KeyError, TypeError, json.JSONDecodeError):
        return JSONResponse(status_code=422, content={"error": "cells 형식이 올바르지 않습니다."})

    try:
        tileset_file = asset_store.resolve_asset_path(tileset_path)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    if not tileset_file.is_file():
        return JSONResponse(status_code=404, content={"error": f"타일셋 이미지가 없습니다: {tileset_path}"})

    try:
        tileset_image = Image.open(tileset_file)
        tileset_image.load()
        style_image = Image.open(io.BytesIO(style.file.read()))
        style_image.load()
    except OSError:
        return JSONResponse(status_code=422, content={"error": "이미지 파일을 해석할 수 없습니다."})

    try:
        preview, patched = tile_stylize.stylize_tiles(
            tileset_image,
            cell_list,
            columns=columns,
            tile_width=tile_width,
            tile_height=tile_height,
            style_image=style_image,
            alpha=alpha,
            work_size=work_size,
            alpha_erode=alpha_erode,
        )
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=503, content={"error": str(error)})

    return {"object_png": _png_b64(preview), "tileset_png": _png_b64(patched)}


@app.get("/asset-status")
def asset_status(path: str):
    try:
        return asset_store.asset_status(path)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})


def _validate_cells(cell_list) -> bool:
    if not isinstance(cell_list, list) or not 0 < len(cell_list) <= 2048:
        return False
    for cell in cell_list:
        if not isinstance(cell, dict):
            return False
        if not all(isinstance(cell.get(field), int) for field in ("col", "row", "tileId")):
            return False
        if cell["tileId"] < 0:
            return False
    return True


@app.post("/extract-objects")
def extract_objects(payload: dict = Body(...)):
    """맵 인식 시점의 배치 누끼 추출. 이미 추출된 오브젝트는 건너뛴다."""
    tileset_path = payload.get("tileset_path")
    tile_width = payload.get("tile_width")
    tile_height = payload.get("tile_height")
    columns = payload.get("columns")
    objects = payload.get("objects")
    if (
        not isinstance(tileset_path, str)
        or not isinstance(tile_width, int)
        or not isinstance(tile_height, int)
        or not isinstance(columns, int)
        or not 1 <= tile_width <= 512
        or not 1 <= tile_height <= 512
        or columns < 1
        or not isinstance(objects, list)
        or not 0 < len(objects) <= 256
    ):
        return JSONResponse(status_code=422, content={"error": "추출 요청 형식이 올바르지 않습니다."})
    for entry in objects:
        if not isinstance(entry, dict) or "id" not in entry or not _validate_cells(entry.get("cells")):
            return JSONResponse(status_code=422, content={"error": "오브젝트 셀 형식이 올바르지 않습니다."})

    try:
        return object_extract.extract_objects(tileset_path, tile_width, tile_height, columns, objects)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})


@app.get("/extracted-objects")
def list_extracted_objects() -> dict:
    return {"objects": object_extract.list_objects()}


@app.get("/extracted-objects/{key}.png")
def extracted_object_png(key: str):
    try:
        return Response(content=object_extract.read_png(key), media_type="image/png")
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})


@app.post("/apply-object")
def apply_object(file: UploadFile = File(...), object_key: str = Form(...)):
    """스타일 적용된 오브젝트 PNG를 타일셋에 역패치해 게임 에셋에 반영한다(백업 포함)."""
    data = file.file.read()
    try:
        probe = Image.open(io.BytesIO(data))
        probe.load()
    except OSError:
        return JSONResponse(status_code=422, content={"error": "PNG로 해석할 수 없는 데이터입니다."})

    try:
        return object_extract.apply_styled_object(object_key, data)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})


@app.post("/revert-asset")
def revert_asset(path: str = Form(...)):
    try:
        asset_store.revert_asset(path)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})
    return {"ok": True}


@app.post("/apply-asset")
def apply_asset(file: UploadFile = File(...), path: str = Form(...)):
    data = file.file.read()
    try:
        # 깨진 데이터로 게임 에셋을 덮어쓰지 않도록 먼저 PNG로 디코딩되는지 확인한다.
        probe = Image.open(io.BytesIO(data))
        probe.load()
    except OSError:
        return JSONResponse(status_code=422, content={"error": "PNG로 해석할 수 없는 데이터입니다."})

    try:
        backup = asset_store.backup_and_write(path, data)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})
    return {"ok": True, "backup": backup}


if __name__ == "__main__":
    import uvicorn

    config = adain_service.get_config()
    uvicorn.run(app, host=config["host"], port=config["port"])
