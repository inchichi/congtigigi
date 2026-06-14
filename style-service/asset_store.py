"""게임 에셋(src/assets) 읽기·쓰기 게이트.

이 서비스가 디스크에 쓰는 모든 경로는 여기를 거친다 — 프로젝트의 에셋 폴더 밖으로는
절대 쓰지 않고(경로 탈출 차단), 덮어쓰기 전에는 반드시 backups/에 원본을 복사한다.
"""

from __future__ import annotations

import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

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


def _backup_and_write_locked(target: Path, relative: str, data: bytes) -> str:
    """_write_lock 안에서만 호출 — 원본 시드 + 백업 + 덮어쓰기."""
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


def backup_and_write(relative: str, data: bytes) -> str:
    """기존 에셋을 backups/에 복사한 뒤 덮어쓴다. 새 파일 생성은 허용하지 않는다(오타 경로 방지)."""
    target = resolve_asset_path(relative)
    if not target.is_file():
        raise FileNotFoundError(f"덮어쓸 에셋이 없습니다: {relative}")
    with _write_lock:
        return _backup_and_write_locked(target, relative, data)


def backup_and_transform(relative: str, transform: Callable[[bytes], bytes]) -> str:
    """현재 에셋 바이트를 읽어 변환한 결과로 덮어쓴다 — read-modify-write 전체를 락 안에서.

    apply가 락 밖에서 파일을 읽으면 병렬 적용 시 lost update(나중 쓰기가 먼저 적용된
    패치를 패치 전 픽셀로 되돌림)와 부분 읽기가 생긴다. 변환 함수까지 락 안에서 돌려
    /apply-asset·/revert-asset과도 직렬화한다.
    """
    target = resolve_asset_path(relative)
    if not target.is_file():
        raise FileNotFoundError(f"덮어쓸 에셋이 없습니다: {relative}")
    with _write_lock:
        data = transform(target.read_bytes())
        return _backup_and_write_locked(target, relative, data)


def read_original_or_current(relative: str) -> bytes:
    """최초 원본이 시드돼 있으면 그 바이트를, 없으면 현재 파일 바이트를 돌려준다.

    몬스터 시트처럼 "스타일은 매번 깨끗한 원본에서" 입혀야 하는 경우에 쓴다 — 이미 한 번
    스타일이 입혀진(또는 깨진) 현재 파일을 다시 변환하면 색이 누적/오염되기 때문이다.
    """
    target = resolve_asset_path(relative)
    original_path = _original_path(relative)
    if original_path.is_file():
        return original_path.read_bytes()
    if target.is_file():
        return target.read_bytes()
    raise FileNotFoundError(f"에셋을 찾을 수 없습니다: {relative}")


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


def list_styled_assets() -> list[dict]:
    """현재 스타일이 적용된(=원본과 내용이 다른) 에셋 목록.

    originals/가 곧 "스타일이 적용된 적 있는 에셋"의 단일 진실 소스다. 파일명은
    relative 경로의 '/'를 '__'로 치환한 것이라 역으로 풀어 검증한 뒤, 현재 파일과
    바이트 비교로 styled만 남긴다(되돌린 뒤 그대로면 제외). 깨진 항목은 조용히 건너뛴다.
    """
    if not _ORIGINALS_DIR.is_dir():
        return []
    out = []
    for original in sorted(_ORIGINALS_DIR.glob("*.png")):
        relative = original.name.replace("__", "/")
        try:
            target = resolve_asset_path(relative)
        except ValueError:
            continue
        if target.is_file() and original.read_bytes() != target.read_bytes():
            out.append({"path": relative, "size": target.stat().st_size})
    return out
