"""맵 오브젝트별 누끼 PNG 자동 추출 저장소.

에디터가 맵을 인식할 때(scene-changed) 오브젝트들의 셀 정보를 배치로 보내오면,
타일셋에서 타일을 조립해 투명 배경 RGBA PNG로 저장한다. 타일 원본이 알파를 갖고
있어 별도 배경 제거는 필요 없다.

저장 위치가 Vite 감시 밖(style-service/extracted-objects/)인 이유: src/games/my-sample-rpg/assets에 쓰면
Vite가 에디터·게임 페이지를 전체 리로드해 "인식 → 추출 → 리로드 → 재인식" 루프가 된다.

PNG 옆에 사이드카 JSON({key}.json)을 함께 저장한다 — 셀·타일셋 정보가 있어야
스타일 적용본을 다시 타일셋에 역패치(게임 반영)할 수 있다.
"""

from __future__ import annotations

import io
import json
import re
import threading
from pathlib import Path

from PIL import Image

import asset_store
import tile_stylize

_EXTRACT_DIR = Path(__file__).resolve().parent / "extracted-objects"
_lock = threading.Lock()

# 파일명·URL 경로로 안전한 키만 허용 — 경로 탈출 방지의 단일 지점.
_SAFE_KEY_PATTERN = re.compile(r"^[\w가-힣-]+$")


def object_key(object_id: str) -> str:
    """오브젝트 id('tile:tree:12,34', 'tree_1' 등)를 파일명 안전 키로 만든다."""
    key = re.sub(r"[^\w가-힣-]+", "_", object_id).strip("_")
    if not key or not _SAFE_KEY_PATTERN.match(key):
        raise ValueError(f"오브젝트 id를 키로 만들 수 없습니다: {object_id}")
    return key


def _png_path(key: str) -> Path:
    return _EXTRACT_DIR / f"{key}.png"


def _meta_path(key: str) -> Path:
    return _EXTRACT_DIR / f"{key}.json"


def extract_objects(
    tileset_path: str,
    tile_width: int,
    tile_height: int,
    columns: int,
    objects: list[dict],
) -> dict:
    """오브젝트 배치를 추출한다. 이미 추출된 키는 건너뛴다(중복 저장 방지)."""
    tileset_file = asset_store.resolve_asset_path(tileset_path)
    if not tileset_file.is_file():
        raise FileNotFoundError(f"타일셋 이미지가 없습니다: {tileset_path}")

    extracted: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    tileset_image: Image.Image | None = None

    with _lock:
        _EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
        for entry in objects:
            try:
                key = object_key(str(entry["id"]))
                # 중복 검사는 파일 존재가 아니라 내용 일치로 한다 — TMX가 편집되어 셀/타일
                # 구성이 바뀌면 같은 id라도 재추출해야 stale 사이드카로 엉뚱한 타일을
                # 패치하는 사고가 없다.
                if _png_path(key).is_file() and _meta_path(key).is_file():
                    try:
                        old = json.loads(_meta_path(key).read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        old = None
                    if (
                        old is not None
                        and old.get("cells") == entry["cells"]
                        and old.get("tilesetPath") == tileset_path
                        and old.get("columns") == columns
                        and old.get("tileWidth") == tile_width
                        and old.get("tileHeight") == tile_height
                    ):
                        skipped.append(key)
                        continue
                if tileset_image is None:
                    tileset_image = Image.open(tileset_file)
                    tileset_image.load()
                canvas, _, _ = tile_stylize.compose_object_canvas(
                    tileset_image, entry["cells"], columns, tile_width, tile_height
                )
                canvas.save(_png_path(key))
                _meta_path(key).write_text(
                    json.dumps(
                        {
                            "id": entry["id"],
                            "key": key,
                            "label": entry.get("label", entry["id"]),
                            "tilesetPath": tileset_path,
                            "tileWidth": tile_width,
                            "tileHeight": tile_height,
                            "columns": columns,
                            "cells": entry["cells"],
                            "sharedOutsideCells": int(entry.get("sharedOutsideCells", 0)),
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                extracted.append(key)
            except (KeyError, TypeError, ValueError):
                failed.append(str(entry.get("id", "?")))
    return {"extracted": extracted, "skipped": skipped, "failed": failed}


def list_objects() -> list[dict]:
    """추출된 오브젝트 목록(사이드카 메타) — 에디터의 썸네일 그리드용."""
    if not _EXTRACT_DIR.is_dir():
        return []
    out = []
    for meta_file in sorted(_EXTRACT_DIR.glob("*.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if _png_path(meta["key"]).is_file():
                out.append(meta)
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return out


def read_png(key: str) -> bytes:
    if not _SAFE_KEY_PATTERN.match(key):
        raise ValueError(f"올바르지 않은 오브젝트 키입니다: {key}")
    path = _png_path(key)
    if not path.is_file():
        raise FileNotFoundError(f"추출된 오브젝트가 없습니다: {key}")
    return path.read_bytes()


def apply_styled_object(key: str, styled_png: bytes) -> dict:
    """스타일 적용된 오브젝트 PNG를 사이드카의 셀 정보로 타일셋에 역패치하고,
    기존 백업/원본 시드 경로(asset_store.backup_and_write)로 게임 에셋에 반영한다."""
    if not _SAFE_KEY_PATTERN.match(key):
        raise ValueError(f"올바르지 않은 오브젝트 키입니다: {key}")
    meta_file = _meta_path(key)
    if not meta_file.is_file():
        raise FileNotFoundError(f"추출된 오브젝트가 없습니다: {key}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    styled = Image.open(io.BytesIO(styled_png))
    styled.load()

    # read→patch→write 전체를 락 안에서 — 병렬 적용의 lost update/부분 읽기 방지.
    def _patch(current_tileset_bytes: bytes) -> bytes:
        tileset_image = Image.open(io.BytesIO(current_tileset_bytes))
        tileset_image.load()
        patched = tile_stylize.patch_tileset_from_object(
            tileset_image,
            styled,
            meta["cells"],
            columns=meta["columns"],
            tile_width=meta["tileWidth"],
            tile_height=meta["tileHeight"],
        )
        buffer = io.BytesIO()
        patched.save(buffer, format="PNG")
        return buffer.getvalue()

    backup = asset_store.backup_and_transform(meta["tilesetPath"], _patch)
    return {
        "ok": True,
        "backup": backup,
        "tilesetPath": meta["tilesetPath"],
        "sharedOutsideCells": int(meta.get("sharedOutsideCells", 0)),
    }
