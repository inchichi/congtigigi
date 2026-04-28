#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "input" / "content" / "PNG"
SEMANTIC_DIR = ROOT / "input" / "mask" / "grounded_sam" / "semantic"
STYLE_OUTPUT_ROOT = ROOT / "output"
FINAL_OUTPUT_ROOT = ROOT / "output2"

STYLE_GROUPS = [
    "spring_2D",
    "spring_real",
    "summer_2D",
    "summer_real",
    "fall_2D",
    "fall_real",
    "winter_2D",
    "winter_real",
]


def main() -> None:
    FINAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    produced = 0
    skipped_no_mask = 0
    skipped_no_content = 0
    skipped_no_style = 0

    for sem_path in sorted(SEMANTIC_DIR.glob("*_semantic.png")):
        stem = sem_path.name.replace("_semantic.png", "")

        content_path = CONTENT_DIR / f"{stem}.png"
        if not content_path.exists():
            skipped_no_content += 1
            continue

        semantic = Image.open(sem_path).convert("L")
        content = Image.open(content_path).convert("RGB")

        for group in STYLE_GROUPS:
            style_path = STYLE_OUTPUT_ROOT / group / f"{stem}_interpolation.jpg"
            if not style_path.exists():
                skipped_no_style += 1
                continue

            styled = Image.open(style_path).convert("RGB")

            if semantic.size != styled.size:
                semantic_resized = semantic.resize(styled.size, resample=Image.NEAREST)
            else:
                semantic_resized = semantic

            if content.size != styled.size:
                content_resized = content.resize(styled.size, resample=Image.BICUBIC)
            else:
                content_resized = content

            # semantic>0 영역만 스타일 적용
            binary_mask = semantic_resized.point(lambda p: 255 if p > 0 else 0)
            if binary_mask.getbbox() is None:
                skipped_no_mask += 1
                continue

            merged = Image.composite(styled, content_resized, binary_mask)

            out_dir = FINAL_OUTPUT_ROOT / group
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stem}_masked.jpg"
            merged.save(out_path, quality=95)
            produced += 1

    print(f"[DONE] produced={produced}")
    print(f"[INFO] skipped_no_mask={skipped_no_mask}")
    print(f"[INFO] skipped_no_content={skipped_no_content}")
    print(f"[INFO] skipped_no_style={skipped_no_style}")


if __name__ == "__main__":
    main()
