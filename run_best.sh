#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "[run_best.sh] python not found in PATH" >&2
  exit 1
fi

args=("$@")
new_args=()

content_path=""
content_dir=""
style_arg=""
style_dir=""

has_output=false
has_content_size=false
has_style_size=false
has_alpha=false
has_save_ext=false

for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --content)
      if (( i + 1 >= ${#args[@]} )); then
        echo "[run_best.sh] Missing value for --content" >&2
        exit 1
      fi
      content_path="${args[i+1]}"
      new_args+=("--content" "__BEST_CONTENT__")
      ((i+=1))
      ;;
    --content_dir)
      if (( i + 1 >= ${#args[@]} )); then
        echo "[run_best.sh] Missing value for --content_dir" >&2
        exit 1
      fi
      content_dir="${args[i+1]}"
      new_args+=("--content_dir" "__BEST_CONTENT_DIR__")
      ((i+=1))
      ;;
    --style)
      if (( i + 1 >= ${#args[@]} )); then
        echo "[run_best.sh] Missing value for --style" >&2
        exit 1
      fi
      style_arg="${args[i+1]}"
      new_args+=("--style" "__BEST_STYLE__")
      ((i+=1))
      ;;
    --style_dir)
      if (( i + 1 >= ${#args[@]} )); then
        echo "[run_best.sh] Missing value for --style_dir" >&2
        exit 1
      fi
      style_dir="${args[i+1]}"
      new_args+=("--style_dir" "__BEST_STYLE_DIR__")
      ((i+=1))
      ;;
    --output)
      has_output=true
      new_args+=("${args[i]}")
      if (( i + 1 < ${#args[@]} )); then
        new_args+=("${args[i+1]}")
        ((i+=1))
      fi
      ;;
    --content_size)
      has_content_size=true
      new_args+=("${args[i]}")
      if (( i + 1 < ${#args[@]} )); then
        new_args+=("${args[i+1]}")
        ((i+=1))
      fi
      ;;
    --style_size)
      has_style_size=true
      new_args+=("${args[i]}")
      if (( i + 1 < ${#args[@]} )); then
        new_args+=("${args[i+1]}")
        ((i+=1))
      fi
      ;;
    --alpha)
      has_alpha=true
      new_args+=("${args[i]}")
      if (( i + 1 < ${#args[@]} )); then
        new_args+=("${args[i+1]}")
        ((i+=1))
      fi
      ;;
    --save_ext)
      has_save_ext=true
      new_args+=("${args[i]}")
      if (( i + 1 < ${#args[@]} )); then
        new_args+=("${args[i+1]}")
        ((i+=1))
      fi
      ;;
    *)
      new_args+=("${args[i]}")
      ;;
  esac
done

if ! $has_content_size; then
  new_args+=("--content_size" "0")
fi
if ! $has_style_size; then
  new_args+=("--style_size" "0")
fi
if ! $has_alpha; then
  new_args+=("--alpha" "0.9")
fi
if ! $has_save_ext; then
  new_args+=("--save_ext" ".png")
fi
if ! $has_output; then
  new_args+=("--output" "output_best_$(date +%Y%m%d_%H%M%S)")
fi

# No conversion target provided; just run with quality defaults.
if [[ -z "${content_path}" && -z "${content_dir}" && -z "${style_arg}" && -z "${style_dir}" ]]; then
  exec python test.py "${new_args[@]}"
fi

tmp_root="$(mktemp -d -t adain_best_XXXXXX)"
cleanup() {
  rm -rf "${tmp_root}"
}
trap cleanup EXIT

convert_file_rgb() {
  local src="$1"
  local dst="$2"
  python - "$src" "$dst" <<'PY'
import sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
Image.open(src).convert("RGB").save(dst)
PY
}

convert_dir_rgb() {
  local src_dir="$1"
  local dst_dir="$2"
  python - "$src_dir" "$dst_dir" <<'PY'
from pathlib import Path
import sys
from PIL import Image

src_dir = Path(sys.argv[1])
dst_dir = Path(sys.argv[2])
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

if not src_dir.is_dir():
    raise SystemExit(f"Not a directory: {src_dir}")

dst_dir.mkdir(parents=True, exist_ok=True)
count = 0
for p in sorted(src_dir.iterdir()):
    if p.is_file() and p.suffix.lower() in exts:
        Image.open(p).convert('RGB').save(dst_dir / f"{p.stem}.png")
        count += 1
if count == 0:
    raise SystemExit(f"No image files found in: {src_dir}")
print(count)
PY
}

converted_content="${content_path}"
if [[ -n "${content_path}" ]]; then
  if [[ ! -f "${content_path}" ]]; then
    echo "[run_best.sh] Content file not found: ${content_path}" >&2
    exit 1
  fi
  converted_content="${tmp_root}/content/$(basename "${content_path%.*}").png"
  mkdir -p "$(dirname "${converted_content}")"
  convert_file_rgb "${content_path}" "${converted_content}"
fi

converted_content_dir="${content_dir}"
if [[ -n "${content_dir}" ]]; then
  converted_content_dir="${tmp_root}/content_dir"
  convert_dir_rgb "${content_dir}" "${converted_content_dir}" >/dev/null
fi

converted_style="${style_arg}"
if [[ -n "${style_arg}" ]]; then
  IFS=',' read -r -a styles <<< "${style_arg}"
  converted_styles=()
  mkdir -p "${tmp_root}/style"
  for idx in "${!styles[@]}"; do
    s="${styles[idx]}"
    if [[ ! -f "${s}" ]]; then
      echo "[run_best.sh] Style file not found: ${s}" >&2
      exit 1
    fi
    stem="$(basename "${s%.*}")"
    if (( ${#styles[@]} > 1 )); then
      out="${tmp_root}/style/${stem}_${idx}.png"
    else
      out="${tmp_root}/style/${stem}.png"
    fi
    convert_file_rgb "${s}" "${out}"
    converted_styles+=("${out}")
  done
  converted_style="$(IFS=,; echo "${converted_styles[*]}")"
fi

converted_style_dir="${style_dir}"
if [[ -n "${style_dir}" ]]; then
  converted_style_dir="${tmp_root}/style_dir"
  convert_dir_rgb "${style_dir}" "${converted_style_dir}" >/dev/null
fi

for ((i=0; i<${#new_args[@]}; i++)); do
  case "${new_args[i]}" in
    __BEST_CONTENT__) new_args[i]="${converted_content}" ;;
    __BEST_CONTENT_DIR__) new_args[i]="${converted_content_dir}" ;;
    __BEST_STYLE__) new_args[i]="${converted_style}" ;;
    __BEST_STYLE_DIR__) new_args[i]="${converted_style_dir}" ;;
  esac
done

exec python test.py "${new_args[@]}"
