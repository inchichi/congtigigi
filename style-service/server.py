"""FLUX.1 Kontext 기반 로컬 HTTP 스타일 편집 서비스.

에디터(Vite dev 서버)의 /api/style 프록시가 이 서버를 가리킨다.
실행: python server.py  (style-service 폴더에서)
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

import asset_store
import external_assets
import kontext_client
import object_extract
import tile_stylize

app = FastAPI(title="FLUX.1 Kontext style-edit service")


def _config() -> dict[str, object]:
    path = Path(__file__).with_name("config.json")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {
        "host": data.get("host", "127.0.0.1"),
        "port": int(data.get("port", 8765)),
    }

def _load_uploaded_image(upload: UploadFile) -> Image.Image:
    image = Image.open(io.BytesIO(upload.file.read()))
    image.load()
    return image


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _edit_image(image: Image.Image, prompt: str, strength: float) -> Image.Image:
    return kontext_client.edit_image(image, prompt, strength=strength, config=kontext_client.get_kontext_config())


def _compose_object_canvas(
    tileset_image: Image.Image,
    cells: list[dict],
    columns: int,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    canvas, _, _ = tile_stylize.compose_object_canvas(
        tileset_image, cells, columns, tile_width, tile_height
    )
    return canvas


def _patch_tileset(
    tileset_image: Image.Image,
    styled_object: Image.Image,
    cells: list[dict],
    columns: int,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    return tile_stylize.patch_tileset_from_object(
        tileset_image,
        styled_object,
        cells,
        columns=columns,
        tile_width=tile_width,
        tile_height=tile_height,
    )


def _is_local_origin(origin: str) -> bool:
    from urllib.parse import urlparse

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
    status = kontext_client.get_runtime_status()
    missing = []
    if not status["cuda_available"]:
        missing.append("CUDA")
    return {
        "status": "ok" if not missing else "degraded",
        "missing": missing,
        "service": "FLUX.1-Kontext-dev",
        **status,
    }


@app.get("/assets")
def list_assets() -> dict:
    return {"assets": asset_store.list_assets()}


def _prompt_strength(alpha: float | None) -> float:
    if alpha is None:
        return 1.0
    return max(0.0, min(1.0, alpha))


def _apply_prompt_to_content(
    content_image: Image.Image,
    prompt: str,
    alpha: float,
) -> Image.Image:
    return _edit_image(content_image, prompt, _prompt_strength(alpha))


@app.post("/style-transfer")
def style_transfer(
    content: UploadFile = File(...),
    prompt: str = Form(...),
    alpha: float = Form(1.0),
):
    try:
        content_image = _load_uploaded_image(content)
    except OSError:
        return JSONResponse(status_code=422, content={"error": "콘텐츠 이미지 파일을 읽을 수 없습니다."})

    try:
        result = _apply_prompt_to_content(content_image, prompt, alpha)
    except (FileNotFoundError, TimeoutError) as error:
        return JSONResponse(status_code=503, content={"error": str(error)})
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except RuntimeError as error:
        return JSONResponse(status_code=502, content={"error": str(error)})

    return Response(content=_image_to_png_bytes(result), media_type="image/png")


@app.post("/stylize-object")
def stylize_object(
    prompt: str = Form(...),
    tileset_path: str = Form(...),
    tile_width: int = Form(...),
    tile_height: int = Form(...),
    columns: int = Form(...),
    cells: str = Form(...),
    alpha: float = Form(1.0),
):
    try:
        cell_list = json.loads(cells)
        assert isinstance(cell_list, list) and cell_list
    except (AssertionError, json.JSONDecodeError, TypeError):
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
        canvas = _compose_object_canvas(tileset_image, cell_list, columns, tile_width, tile_height)
        preview = _apply_prompt_to_content(canvas, prompt, alpha)
        patched = _patch_tileset(tileset_image, preview, cell_list, columns, tile_width, tile_height)
    except (FileNotFoundError, TimeoutError) as error:
        return JSONResponse(status_code=503, content={"error": str(error)})
    except (ValueError, RuntimeError, OSError) as error:
        return JSONResponse(status_code=422, content={"error": str(error)})

    return {
        "object_png": base64.b64encode(_image_to_png_bytes(preview)).decode("ascii"),
        "tileset_png": base64.b64encode(_image_to_png_bytes(patched)).decode("ascii"),
    }


@app.post("/stylize-monster")
def stylize_monster(
    prompt: str = Form(...),
    sheet_path: str = Form(...),
    monster_key: str = Form(...),
    alpha: float = Form(1.0),
):
    if monster_key not in ("pig", "slime"):
        return JSONResponse(status_code=422, content={"error": f"지원하지 않는 몬스터 종류: {monster_key}"})
    if asset_store.is_runtime_animation_asset(sheet_path):
        return JSONResponse(
            status_code=422,
            content={"error": "Runtime animation sheets and map tilesets cannot be overwritten by FLUX. Use a static portrait or extracted object target instead."},
        )

    try:
        sheet_bytes = asset_store.read_original_or_current(sheet_path)
        sheet_image = Image.open(io.BytesIO(sheet_bytes))
        sheet_image.load()
        result = _apply_prompt_to_content(sheet_image, prompt, alpha)
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})
    except (ValueError, RuntimeError, OSError, TimeoutError) as error:
        return JSONResponse(status_code=422, content={"error": str(error)})

    return Response(content=_image_to_png_bytes(result), media_type="image/png")


@app.post("/batch-apply")
def batch_apply(
    prompt: str = Form(...),
    targets: str = Form(...),
    alpha: float = Form(1.0),
):
    try:
        target_list = json.loads(targets)
        assert isinstance(target_list, list) and 0 < len(target_list) <= 64
    except (json.JSONDecodeError, AssertionError):
        return JSONResponse(status_code=422, content={"error": "targets 형식이 올바르지 않습니다."})

    applied: list[str] = []
    failed: list[dict] = []
    for target in target_list:
        try:
            kind = target.get("kind")
            if kind == "asset":
                path = target["path"]
                if asset_store.is_runtime_animation_asset(path):
                    failed.append({"target": str(target), "error": "Runtime animation sheets and map tilesets cannot be overwritten by FLUX."})
                    continue
                source = Image.open(asset_store.resolve_asset_path(path))
                source.load()
                result = _apply_prompt_to_content(source, prompt, alpha)
                asset_store.backup_and_write(path, _image_to_png_bytes(result))
                applied.append(path)
            elif kind == "object":
                key = target["key"]
                meta = object_extract.read_meta(key)
                cutout = Image.open(io.BytesIO(object_extract.read_png(key)))
                cutout.load()
                result = _apply_prompt_to_content(cutout, prompt, alpha)
                patched = _patch_tileset(
                    Image.open(asset_store.resolve_asset_path(meta["tilesetPath"])),
                    result,
                    meta["cells"],
                    columns=meta["columns"],
                    tile_width=meta["tileWidth"],
                    tile_height=meta["tileHeight"],
                )
                # Object targets are extracted static regions. Patch their
                # recorded cells back into the protected tileset, while direct
                # full-tileset and runtime-sheet overwrites remain blocked.
                asset_store.backup_and_write(
                    meta["tilesetPath"],
                    _image_to_png_bytes(patched),
                    allow_protected_tileset_patch=True,
                )
                applied.append(key)
            elif kind == "monster":
                path = target["sheet_path"]
                if asset_store.is_runtime_animation_asset(path):
                    failed.append({"target": str(target), "error": "Runtime animation sheets and map tilesets cannot be overwritten by FLUX."})
                    continue
                source = Image.open(io.BytesIO(asset_store.read_original_or_current(path)))
                source.load()
                result = _apply_prompt_to_content(source, prompt, alpha)
                asset_store.backup_and_write(path, _image_to_png_bytes(result))
                applied.append(path)
            else:
                failed.append({"target": str(target), "error": "지원하지 않는 대상입니다."})
        except (KeyError, TypeError, ValueError, FileNotFoundError, OSError, RuntimeError, TimeoutError) as error:
            failed.append({"target": str(target), "error": str(error)})
    return {"applied": applied, "failed": failed, "written": applied}


@app.get("/asset-status")
def asset_status(path: str):
    try:
        return asset_store.asset_status(path)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})


def _object_canvas_from_tileset(meta: dict, tileset_bytes: bytes) -> Image.Image:
    tileset_image = Image.open(io.BytesIO(tileset_bytes)).convert("RGBA")
    tileset_image.load()
    return _compose_object_canvas(
        tileset_image,
        meta["cells"],
        columns=meta["columns"],
        tile_width=meta["tileWidth"],
        tile_height=meta["tileHeight"],
    )


def _object_variant_bytes(key: str, variant: str) -> bytes:
    meta = object_extract.read_meta(key)
    tileset_path = meta["tilesetPath"]
    status = asset_store.asset_status(tileset_path)
    if variant == "before":
        if not status["hasOriginal"]:
            raise FileNotFoundError(f"원본이 없는 오브젝트입니다: {key}")
        tileset_bytes = asset_store.read_original_or_current(tileset_path)
    else:
        tileset_bytes = asset_store.resolve_asset_path(tileset_path).read_bytes()
    return _image_to_png_bytes(_object_canvas_from_tileset(meta, tileset_bytes))


@app.get("/styled-objects")
def styled_objects() -> dict:
    objects: list[dict] = []
    for meta in object_extract.list_objects():
        try:
            key = str(meta["key"])
            tileset_path = str(meta["tilesetPath"])
            status = asset_store.asset_status(tileset_path)
            if not status["hasOriginal"] or not status["styled"]:
                continue
            before = _object_canvas_from_tileset(meta, asset_store.read_original_or_current(tileset_path))
            after = _object_canvas_from_tileset(
                meta,
                asset_store.resolve_asset_path(tileset_path).read_bytes(),
            )
            if before.convert("RGBA").tobytes() == after.convert("RGBA").tobytes():
                continue
            objects.append({
                "key": key,
                "label": str(meta.get("label", key)),
                "tilesetPath": tileset_path,
                "beforeUrl": f"/api/style/styled-objects/{key}/before.png",
                "afterUrl": f"/api/style/styled-objects/{key}/after.png",
            })
        except (KeyError, TypeError, ValueError, FileNotFoundError, OSError):
            continue
    return {"objects": objects}


@app.get("/styled-objects/{key}/{variant}.png")
def styled_object_png(key: str, variant: str):
    if variant not in {"before", "after"}:
        return JSONResponse(status_code=422, content={"error": "variant는 before 또는 after여야 합니다."})
    try:
        return Response(content=_object_variant_bytes(key, variant), media_type="image/png")
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except (FileNotFoundError, OSError) as error:
        return JSONResponse(status_code=404, content={"error": str(error)})


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
            return JSONResponse(status_code=422, content={"error": "오브젝트 형식이 올바르지 않습니다."})

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
    data = file.file.read()
    try:
        probe = Image.open(io.BytesIO(data))
        probe.load()
    except OSError:
        return JSONResponse(status_code=422, content={"error": "PNG로 읽을 수 없는 데이터입니다."})

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


@app.get("/styled-assets")
def styled_assets() -> dict:
    return {"assets": asset_store.list_styled_assets()}


@app.post("/revert-assets")
def revert_assets(payload: dict = Body(...)):
    paths = payload.get("paths")
    if not isinstance(paths, list) or not 0 < len(paths) <= 1024:
        return JSONResponse(status_code=422, content={"error": "paths 형식이 올바르지 않습니다."})
    reverted: list[str] = []
    failed: list[dict] = []
    for path in paths:
        if not isinstance(path, str):
            failed.append({"path": str(path), "error": "경로가 문자열이 아닙니다."})
            continue
        try:
            asset_store.revert_asset(path)
            reverted.append(path)
        except (ValueError, FileNotFoundError) as error:
            failed.append({"path": path, "error": str(error)})
    return {"reverted": reverted, "failed": failed}


@app.post("/apply-asset")
def apply_asset(file: UploadFile = File(...), path: str = Form(...)):
    data = file.file.read()
    try:
        probe = Image.open(io.BytesIO(data))
        probe.load()
    except OSError:
        return JSONResponse(status_code=422, content={"error": "PNG로 읽을 수 없는 데이터입니다."})

    try:
        backup = asset_store.backup_and_write(path, data)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})
    return {"ok": True, "backup": backup}


@app.get("/ext/projects")
def ext_projects() -> dict:
    return {"projects": external_assets.get_projects()}


@app.get("/ext/assets")
def ext_assets(project: str):
    try:
        return {"assets": external_assets.list_assets(project)}
    except ValueError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})


@app.get("/ext/asset")
def ext_asset(project: str, path: str):
    try:
        return Response(content=external_assets.read_png(project, path), media_type="image/png")
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})


@app.get("/ext/asset-status")
def ext_asset_status(project: str, path: str):
    try:
        return external_assets.asset_status(project, path)
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})


@app.get("/ext/styled")
def ext_styled(project: str):
    try:
        return {"assets": external_assets.list_styled_assets(project)}
    except ValueError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})


@app.post("/ext/apply")
def ext_apply(
    prompt: str = Form(...),
    project: str = Form(...),
    path: str = Form(...),
    alpha: float = Form(1.0),
):
    try:
        source_bytes = external_assets.read_original_or_current(project, path)
        source = Image.open(io.BytesIO(source_bytes))
        source.load()
        result = _apply_prompt_to_content(source, prompt, alpha)
        backup = external_assets.backup_and_write(project, path, _image_to_png_bytes(result))
    except ValueError as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    except FileNotFoundError as error:
        return JSONResponse(status_code=404, content={"error": str(error)})
    except (OSError, RuntimeError, TimeoutError) as error:
        return JSONResponse(status_code=422, content={"error": str(error)})
    return {"ok": True, "backup": backup}


@app.post("/ext/batch-apply")
def ext_batch_apply(
    prompt: str = Form(...),
    project: str = Form(...),
    paths: str = Form(...),
    alpha: float = Form(1.0),
):
    try:
        path_list = json.loads(paths)
        assert isinstance(path_list, list) and 0 < len(path_list) <= 256
    except (json.JSONDecodeError, AssertionError):
        return JSONResponse(status_code=422, content={"error": "paths 형식이 올바르지 않습니다."})

    applied: list[str] = []
    failed: list[dict] = []
    for path in path_list:
        try:
            source = Image.open(io.BytesIO(external_assets.read_original_or_current(project, path)))
            source.load()
            result = _apply_prompt_to_content(source, prompt, alpha)
            external_assets.backup_and_write(project, path, _image_to_png_bytes(result))
            applied.append(path)
        except (ValueError, FileNotFoundError, OSError, RuntimeError, TimeoutError) as error:
            failed.append({"path": str(path), "error": str(error)})
    return {"applied": applied, "failed": failed}


@app.post("/ext/revert")
def ext_revert(payload: dict = Body(...)):
    project = payload.get("project")
    paths = payload.get("paths")
    if not isinstance(project, str) or not isinstance(paths, list) or not 0 < len(paths) <= 1024:
        return JSONResponse(status_code=422, content={"error": "project/paths 형식이 올바르지 않습니다."})
    reverted: list[str] = []
    failed: list[dict] = []
    for path in paths:
        if not isinstance(path, str):
            failed.append({"path": str(path), "error": "경로가 문자열이 아닙니다."})
            continue
        try:
            external_assets.revert_asset(project, path)
            reverted.append(path)
        except (ValueError, FileNotFoundError) as error:
            failed.append({"path": path, "error": str(error)})
    return {"reverted": reverted, "failed": failed}


if __name__ == "__main__":
    import uvicorn

    config = _config()
    uvicorn.run(app, host=config["host"], port=config["port"])
