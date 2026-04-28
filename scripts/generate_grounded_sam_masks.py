#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

try:
    from groundingdino.util.inference import load_model, load_image, predict
except ImportError as e:
    raise SystemExit(
        "groundingdino가 설치되어 있지 않습니다.\n"
        "설치 예시: pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
    ) from e

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    raise SystemExit(
        "segment-anything가 설치되어 있지 않습니다.\n"
        "설치 예시: pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


@dataclass
class ClassSpec:
    name: str
    keywords: List[str]


def parse_class_specs(raw: str) -> List[ClassSpec]:
    specs: List[ClassSpec] = []
    # format: tree=tree,oak,pine;rock=rock,stone,boulder;lake=lake,water,pond
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"잘못된 class 형식: {token}")
        name, kws = token.split("=", 1)
        name = name.strip().lower()
        keywords = [k.strip().lower() for k in kws.split(",") if k.strip()]
        if not name or not keywords:
            raise ValueError(f"클래스 또는 키워드 비어 있음: {token}")
        specs.append(ClassSpec(name=name, keywords=keywords))

    if not specs:
        raise ValueError("최소 1개 클래스가 필요합니다.")
    return specs


def phrase_to_class(phrase: str, specs: List[ClassSpec]) -> str | None:
    p = phrase.lower()
    for spec in specs:
        if any(k in p for k in spec.keywords):
            return spec.name
    return None


def build_text_prompt(specs: List[ClassSpec]) -> str:
    # GroundingDINO는 점(.) 기준으로 구문 분리하는 방식이 안정적이다.
    parts = [spec.keywords[0] for spec in specs]
    return " . ".join(parts) + " ."


def ensure_dirs(base_out: Path, specs: List[ClassSpec]) -> Dict[str, Path]:
    per_class_dir: Dict[str, Path] = {}
    for spec in specs:
        d = base_out / spec.name
        d.mkdir(parents=True, exist_ok=True)
        per_class_dir[spec.name] = d
    (base_out / "semantic").mkdir(parents=True, exist_ok=True)
    (base_out / "overlay").mkdir(parents=True, exist_ok=True)
    return per_class_dir


def overlay_mask(image_bgr: np.ndarray, semantic: np.ndarray, specs: List[ClassSpec]) -> np.ndarray:
    palette = {
        "tree": (34, 139, 34),
        "rock": (128, 128, 128),
        "lake": (255, 120, 0),
    }
    out = image_bgr.copy()
    alpha = 0.35
    for idx, spec in enumerate(specs, start=1):
        color = palette.get(spec.name, tuple(int(x) for x in np.random.randint(0, 255, size=3)))
        mask = semantic == idx
        if np.any(mask):
            colored = np.zeros_like(out)
            colored[:, :] = color
            out[mask] = cv2.addWeighted(out, 1.0 - alpha, colored, alpha, 0)[mask]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="GroundingDINO + SAM 기반 타일 오브젝트 마스크 생성")
    parser.add_argument("--input-dir", type=Path, required=True, help="입력 타일 이미지 폴더")
    parser.add_argument("--output-dir", type=Path, default=Path("input/mask"), help="마스크 출력 폴더")
    parser.add_argument("--dino-config", type=Path, required=True, help="GroundingDINO config.py 경로")
    parser.add_argument("--dino-checkpoint", type=Path, required=True, help="GroundingDINO .pth 경로")
    parser.add_argument("--sam-checkpoint", type=Path, required=True, help="SAM checkpoint(.pth) 경로")
    parser.add_argument("--sam-model-type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--classes",
        type=str,
        default="tree=tree,oak,pine;rock=rock,stone,boulder;lake=lake,water,pond",
        help="형식: name=kw1,kw2;name2=kw1,kw2",
    )
    parser.add_argument("--glob", type=str, default="*.png", help="입력 파일 glob")
    parser.add_argument("--skip-existing", action="store_true", help="이미 semantic 마스크가 있으면 스킵")

    args = parser.parse_args()

    specs = parse_class_specs(args.classes)
    prompt = build_text_prompt(specs)
    print(f"[INFO] text prompt: {prompt}")

    input_paths = sorted(args.input_dir.glob(args.glob))
    if not input_paths:
        raise SystemExit(f"입력 이미지가 없습니다: {args.input_dir} ({args.glob})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}, images={len(input_paths)}")

    dino_model = load_model(str(args.dino_config), str(args.dino_checkpoint), device=device)
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    class_dirs = ensure_dirs(args.output_dir, specs)

    for i, image_path in enumerate(input_paths, start=1):
        stem = image_path.stem
        sem_path = args.output_dir / "semantic" / f"{stem}_semantic.png"
        if args.skip_existing and sem_path.exists():
            if i % 20 == 0 or i == len(input_paths):
                print(f"[INFO] {i}/{len(input_paths)} skip-existing: {image_path.name}")
            continue

        image_source, image_dino = load_image(str(image_path))
        boxes, logits, phrases = predict(
            model=dino_model,
            image=image_dino,
            caption=prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=device,
        )

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] skip unreadable: {image_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        predictor.set_image(rgb)

        # class-wise binary masks
        class_masks = {spec.name: np.zeros((h, w), dtype=np.uint8) for spec in specs}

        if boxes.numel() > 0:
            boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            scale = torch.tensor([w, h, w, h], device=boxes_xyxy.device)
            boxes_xyxy = boxes_xyxy * scale

            for box, phrase in zip(boxes_xyxy, phrases):
                class_name = phrase_to_class(phrase, specs)
                if class_name is None:
                    continue

                box = box.unsqueeze(0)
                transformed = predictor.transform.apply_boxes_torch(box, (h, w)).to(device)
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed,
                    multimask_output=False,
                )
                m = masks[0, 0].detach().cpu().numpy().astype(np.uint8)
                class_masks[class_name] = np.maximum(class_masks[class_name], m)

        # save class masks
        semantic = np.zeros((h, w), dtype=np.uint8)
        for idx, spec in enumerate(specs, start=1):
            mask = class_masks[spec.name]
            semantic[(mask > 0) & (semantic == 0)] = idx
            out_path = class_dirs[spec.name] / f"{stem}_{spec.name}.png"
            cv2.imwrite(str(out_path), (mask * 255).astype(np.uint8))

        cv2.imwrite(str(sem_path), semantic)

        ov = overlay_mask(bgr, semantic, specs)
        ov_path = args.output_dir / "overlay" / f"{stem}_overlay.png"
        cv2.imwrite(str(ov_path), ov)

        if i % 20 == 0 or i == len(input_paths):
            print(f"[INFO] {i}/{len(input_paths)} done: {image_path.name}")

    print("[DONE] masks generated")


if __name__ == "__main__":
    main()
