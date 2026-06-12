"""게임 에셋(src/assets) 읽기·쓰기 게이트.

이 서비스가 디스크에 쓰는 모든 경로는 여기를 거친다 — 프로젝트의 에셋 폴더 밖으로는
절대 쓰지 않고(경로 탈출 차단), 덮어쓰기 전에는 반드시 backups/에 원본을 복사한다.
"""

from __future__ import annotations

import shutil
import threading
from datetime import datetime
from pathlib import Path

import adain_service

_BACKUP_DIR = Path(__file__).resolve().parent / "backups"
# 에셋별 "최초 원본" 보관소 — 첫 적용 때 한 번만 시드되고 이후 적용은 건드리지 않으므로,
# 여러 번 스타일을 입혀도 되돌리기 한 번에 진짜 원본으로 복귀한다.
_ORIGINALS_DIR = Path(__file__).resolve().parent / "originals"

# 백업+쓰기를 직렬화한다 — FastAPI 동기 엔드포인트는 스레드풀에서 병렬 실행되므로,
# 같은 파일에 동시 적용되면 백업 복사와 덮어쓰기가 경합할 수 있다.
_write_lock = threading.Lock()

ALLOWED_SUFFIXES = {".png"}


def _original_path(relative: str) -> Path:
    return _ORIGINALS_DIR / relative.replace("/", "__")


def assets_root() -> Path:
    config = adain_service.get_config()
    return (config["project_dir"] / config["assets_subdir"]).resolve()


def resolve_asset_path(relative: str) -> Path:
    """프로젝트 상대 경로('src/assets/...')를 검증해 절대 경로로 푼다. 탈출 시 ValueError."""
    config = adain_service.get_config()
    target = (config["project_dir"] / relative).resolve()
    root = assets_root()
    if root != target and root not in target.parents:
        raise ValueError(f"에셋 폴더({config['assets_subdir']}) 밖의 경로는 다룰 수 없습니다: {relative}")
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"PNG 파일만 다룰 수 있습니다: {relative}")
    return target


def list_assets() -> list[dict]:
    """에셋 폴더의 PNG 목록(프로젝트 상대 경로, 크기). 에디터의 '게임 에셋' 선택 드롭다운용."""
    config = adain_service.get_config()
    root = assets_root()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.rglob("*.png")):
        relative = path.relative_to(config["project_dir"]).as_posix()
        out.append({"path": relative, "size": path.stat().st_size})
    return out


def backup_and_write(relative: str, data: bytes) -> str:
    """기존 에셋을 backups/에 복사한 뒤 덮어쓴다. 새 파일 생성은 허용하지 않는다(오타 경로 방지)."""
    target = resolve_asset_path(relative)
    if not target.is_file():
        raise FileNotFoundError(f"덮어쓸 에셋이 없습니다: {relative}")

    with _write_lock:
        # 첫 적용이면 최초 원본을 시드한다(이미 있으면 보존) — 되돌리기의 복원 지점.
        original_path = _original_path(relative)
        if not original_path.exists():
            _ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, original_path)

        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        # 마이크로초 + 충돌 시 카운터 — 같은 초 안의 재적용이 직전 백업(원본일 수 있음)을
        # 덮어써 영구 소실시키는 것을 막는다.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        flattened = relative.replace("/", "__")
        backup_path = _BACKUP_DIR / f"{stamp}__{flattened}"
        counter = 1
        while backup_path.exists():
            backup_path = _BACKUP_DIR / f"{stamp}_{counter}__{flattened}"
            counter += 1
        shutil.copy2(target, backup_path)

        target.write_bytes(data)
    return str(backup_path)


def asset_status(relative: str) -> dict:
    """원본 보유 여부와 "현재 스타일이 적용된 상태"(원본과 내용이 다름) 여부."""
    target = resolve_asset_path(relative)
    original_path = _original_path(relative)
    has_original = original_path.is_file()
    styled = (
        has_original
        and target.is_file()
        and original_path.read_bytes() != target.read_bytes()
    )
    return {"hasOriginal": has_original, "styled": styled}


def revert_asset(relative: str) -> None:
    """최초 원본으로 복원한다. 원본 파일은 보존한다(여러 번 되돌리기 가능)."""
    target = resolve_asset_path(relative)
    original_path = _original_path(relative)
    if not original_path.is_file():
        raise FileNotFoundError(f"되돌릴 원본이 없습니다(스타일이 적용된 적 없음): {relative}")
    with _write_lock:
        shutil.copy2(original_path, target)
