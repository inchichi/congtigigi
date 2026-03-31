#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "[run.sh] python not found in PATH" >&2
  exit 1
fi

args=("$@")
new_args=()

content_path=""
style_arg=""

for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --content)
      if (( i + 1 >= ${#args[@]} )); then
        echo "[run.sh] Missing value for --content" >&2
        exit 1
      fi
      content_path="${args[i+1]}"
      new_args+=("--content" "__RUNSH_CONTENT__")
      ((i+=1))
      ;;
    --style)
      if (( i + 1 >= ${#args[@]} )); then
        echo "[run.sh] Missing value for --style" >&2
        exit 1
      fi
      style_arg="${args[i+1]}"
      new_args+=("--style" "__RUNSH_STYLE__")
      ((i+=1))
      ;;
    *)
      new_args+=("${args[i]}")
      ;;
  esac
done

# If user did not pass --content/--style, forward args as-is.
if [[ -z "${content_path}" && -z "${style_arg}" ]]; then
  exec python test.py "${args[@]}"
fi

tmp_dir="$(mktemp -d -t adain_rgb_XXXXXX)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

convert_to_rgb() {
  local src="$1"
  local dst="$2"
  python - "$src" "$dst" <<'PY'
import sys
from PIL import Image

src, dst = sys.argv[1], sys.argv[2]
Image.open(src).convert("RGB").save(dst)
PY
}

converted_content="${content_path}"
if [[ -n "${content_path}" ]]; then
  if [[ ! -f "${content_path}" ]]; then
    echo "[run.sh] Content file not found: ${content_path}" >&2
    exit 1
  fi
  content_name="$(basename "${content_path}")"
  content_stem="${content_name%.*}"
  converted_content="${tmp_dir}/${content_stem}.png"
  convert_to_rgb "${content_path}" "${converted_content}"
fi

converted_style="${style_arg}"
if [[ -n "${style_arg}" ]]; then
  IFS=',' read -r -a styles <<< "${style_arg}"
  converted_styles=()
  for idx in "${!styles[@]}"; do
    s="${styles[idx]}"
    if [[ ! -f "${s}" ]]; then
      echo "[run.sh] Style file not found: ${s}" >&2
      exit 1
    fi
    style_name="$(basename "${s}")"
    style_stem="${style_name%.*}"
    if (( ${#styles[@]} > 1 )); then
      out="${tmp_dir}/${style_stem}_${idx}.png"
    else
      out="${tmp_dir}/${style_stem}.png"
    fi
    convert_to_rgb "${s}" "${out}"
    converted_styles+=("${out}")
  done
  converted_style="$(IFS=,; echo "${converted_styles[*]}")"
fi

for ((i=0; i<${#new_args[@]}; i++)); do
  if [[ "${new_args[i]}" == "__RUNSH_CONTENT__" ]]; then
    new_args[i]="${converted_content}"
  elif [[ "${new_args[i]}" == "__RUNSH_STYLE__" ]]; then
    new_args[i]="${converted_style}"
  fi
done

exec python test.py "${new_args[@]}"
