"""외부 게임 프로젝트(예: Love2D 'Legend of Lua')의 스프라이트 읽기·쓰기 게이트.

my-sample-rpg 전용 asset_store와 의도적으로 분리했다 — 같은 안전 패턴(경로 탈출 차단,
PNG만, 덮어쓰기 전 원본/백업 보존)을 쓰되 별도 네임스페이스(originals-ext/, backups-ext/)에
저장해 기존 에셋 샌드박스·되돌리기·테스트에 어떤 영향도 주지 않는다.

허용 루트는 config.json의 externalProjects[*].root로만 정해진다(하드코딩 금지).
파일명 키는 "{projectId}__상대경로(/→__)"라 프로젝트끼리도 충돌하지 않는다.
"""

from __future__ import annotations

import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

import adain_service

_SERVICE_DIR = Path(__file__).resolve().parent
_BACKUP_DIR = _SERVICE_DIR / "backups-ext"
_ORIGINALS_DIR = _SERVICE_DIR / "originals-ext"

# asset_store와 동일하게 백업+쓰기를 직렬화한다(병렬 적용 경합 방지).
_write_lock = threading.Lock()

ALLOWED_SUFFIXES = {".png"}


def _projects() -> dict:
    return adain_service.get_config().get("external_projects", {})


def get_projects() -> list[dict]:
    """설정에 등록되고 실제 폴더가 존재하는 외부 프로젝트 목록."""
    return [
        {"id": pid, "name": project["name"]}
        for pid, project in _projects().items()
        if project["root"].is_dir()
    ]


def _project(project_id: str) -> dict:
    projects = _projects()
    if project_id not in projects:
        raise ValueError(f"알 수 없는 외부 프로젝트입니다: {project_id}")
    return projects[project_id]


def project_root(project_id: str) -> Path:
    return _project(project_id)["root"]


def resolve(project_id: str, relative: str) -> Path:
    """프로젝트 루트 기준 상대 경로를 검증해 절대 경로로 푼다. 탈출/비PNG는 ValueError."""
    root = project_root(project_id)
    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise ValueError(f"프로젝트 폴더 밖의 경로는 다룰 수 없습니다: {relative}")
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"PNG 파일만 다룰 수 있습니다: {relative}")
    return target


def _key(project_id: str, relative: str) -> str:
    return f"{project_id}__" + relative.replace("/", "__")


def _original_path(project_id: str, relative: str) -> Path:
    return _ORIGINALS_DIR / _key(project_id, relative)


def list_assets(project_id: str) -> list[dict]:
    """프로젝트(또는 그 assetsSubdir) 안의 PNG 목록 — 루트 상대 경로 + 크기."""
    project = _project(project_id)
    root = project["root"]
    if not root.is_dir():
        return []
    subdir = project.get("assets_subdir")
    base = (root / subdir) if subdir else root
    if not base.is_dir():
        base = root
    out = []
    for path in sorted(base.rglob("*.png")):
        relative = path.relative_to(root).as_posix()
        out.append({"path": relative, "size": path.stat().st_size})
    return out


def read_png(project_id: str, relative: str) -> bytes:
    target = resolve(project_id, relative)
    if not target.is_file():
        raise FileNotFoundError(f"스프라이트가 없습니다: {relative}")
    return target.read_bytes()


def read_original_or_current(project_id: str, relative: str) -> bytes:
    """최초 원본이 시드돼 있으면 그 바이트를, 없으면 현재 파일 바이트를 돌려준다.

    스타일은 매번 깨끗한 원본에서 입혀야 색이 누적/오염되지 않으므로, 재적용 시 이 함수를 쓴다.
    """
    target = resolve(project_id, relative)
    original_path = _original_path(project_id, relative)
    if original_path.is_file():
        return original_path.read_bytes()
    if target.is_file():
        return target.read_bytes()
    raise FileNotFoundError(f"스프라이트를 찾을 수 없습니다: {relative}")


def _backup_and_write_locked(target: Path, project_id: str, relative: str, data: bytes) -> str:
    """_write_lock 안에서만 호출 — 최초 원본 시드 + 타임스탬프 백업 + 덮어쓰기."""
    original_path = _original_path(project_id, relative)
    if not original_path.exists():
        _ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, original_path)

    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    flattened = _key(project_id, relative)
    backup_path = _BACKUP_DIR / f"{stamp}__{flattened}"
    counter = 1
    while backup_path.exists():
        backup_path = _BACKUP_DIR / f"{stamp}_{counter}__{flattened}"
        counter += 1
    shutil.copy2(target, backup_path)

    target.write_bytes(data)
    return str(backup_path)


def backup_and_write(project_id: str, relative: str, data: bytes) -> str:
    """기존 스프라이트를 백업한 뒤 덮어쓴다. 새 파일 생성은 허용하지 않는다(오타 경로 방지)."""
    target = resolve(project_id, relative)
    if not target.is_file():
        raise FileNotFoundError(f"덮어쓸 스프라이트가 없습니다: {relative}")
    with _write_lock:
        return _backup_and_write_locked(target, project_id, relative, data)


def asset_status(project_id: str, relative: str) -> dict:
    target = resolve(project_id, relative)
    original_path = _original_path(project_id, relative)
    has_original = original_path.is_file()
    styled = (
        has_original
        and target.is_file()
        and original_path.read_bytes() != target.read_bytes()
    )
    return {"hasOriginal": has_original, "styled": styled}


def revert_asset(project_id: str, relative: str) -> None:
    """최초 원본으로 복원한다(원본 파일은 보존 — 여러 번 되돌리기 가능)."""
    target = resolve(project_id, relative)
    original_path = _original_path(project_id, relative)
    if not original_path.is_file():
        raise FileNotFoundError(f"되돌릴 원본이 없습니다(스타일이 적용된 적 없음): {relative}")
    with _write_lock:
        shutil.copy2(original_path, target)


def list_styled_assets(project_id: str) -> list[dict]:
    """현재 스타일이 적용된(원본과 내용이 다른) 스프라이트 목록. originals-ext/가 단일 진실 소스."""
    if not _ORIGINALS_DIR.is_dir():
        return []
    prefix = f"{project_id}__"
    out = []
    for original in sorted(_ORIGINALS_DIR.glob(f"{project_id}__*.png")):
        if not original.name.startswith(prefix):
            continue
        relative = original.name[len(prefix):].replace("__", "/")
        try:
            target = resolve(project_id, relative)
        except ValueError:
            continue
        if target.is_file() and original.read_bytes() != target.read_bytes():
            out.append({"path": relative, "size": target.stat().st_size})
    return out
