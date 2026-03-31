#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
  PY="$ROOT_DIR/.venv/bin/python3"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  PY="python3"
fi

rgb_dir="/tmp/congtigigi_content_rgb"
mkdir -p "$rgb_dir"
"$PY" - <<'PY'
from pathlib import Path
from PIL import Image
src = Path("input/content/PNG")
dst = Path("/tmp/congtigigi_content_rgb")
for p in sorted(src.iterdir()):
    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        Image.open(p).convert("RGB").save(dst / p.name)
print("[INFO] RGB conversion complete")
PY

for season in spring summer fall winter; do
  for kind in 2D real; do
    style_dir="input/style/${season}/${kind}"
    mapfile -t style_files < <(find "$style_dir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)
    if [[ ${#style_files[@]} -eq 0 ]]; then
      echo "[SKIP] ${season}/${kind}"
      continue
    fi

    styles=$(printf "%s," "${style_files[@]}")
    styles=${styles%,}
    weights=""
    for _ in "${style_files[@]}"; do
      weights+="1,"
    done
    weights=${weights%,}

    out_dir="output/batch_interp/${season}_${kind}"
    mkdir -p "$out_dir"
    echo "[RUN] ${season}/${kind}"
    "$PY" test.py \
      --content_dir "$rgb_dir" \
      --style "$styles" \
      --style_interpolation_weights "$weights" \
      --content_size 256 \
      --style_size 256 \
      --crop \
      --output "$out_dir"
    echo "[DONE] ${season}/${kind}"
  done
done

echo "[ALL DONE]"
