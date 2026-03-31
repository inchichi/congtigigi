#!/usr/bin/env python3
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

import net
from function import calc_mean_std

ROOT = Path(__file__).resolve().parent
CONTENT_DIR = ROOT / "input" / "content" / "PNG"
STYLE_ROOT = ROOT / "input" / "style"
OUTPUT_ROOT = ROOT / "output" / "batch_interp"
MODELS_DIR = ROOT / "models"

GROUPS = [
    "spring_2D",
    "spring_real",
    "summer_2D",
    "summer_real",
    "fall_2D",
    "fall_real",
    "winter_2D",
    "winter_real",
]

EPS = 1e-5


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((a - b) ** 2)


def weighted_stats(means: List[torch.Tensor], stds: List[torch.Tensor], weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # means/stds: list of (1,C,1,1), weights: (N,)
    stacked_mean = torch.stack([m.squeeze(0) for m in means], dim=0)  # N,C,1,1
    stacked_std = torch.stack([s.squeeze(0) for s in stds], dim=0)
    w = weights.view(-1, 1, 1, 1)
    target_mean = (stacked_mean * w).sum(dim=0, keepdim=True)
    target_std = (stacked_std * w).sum(dim=0, keepdim=True)
    return target_mean, target_std


def encode_with_intermediate(x: torch.Tensor, enc_1, enc_2, enc_3, enc_4) -> List[torch.Tensor]:
    r1 = enc_1(x)
    r2 = enc_2(r1)
    r3 = enc_3(r2)
    r4 = enc_4(r3)
    return [r1, r2, r3, r4]


def find_content_map(content_dir: Path) -> Dict[str, Path]:
    m: Dict[str, Path] = {}
    for p in sorted(content_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            continue
        m[p.stem] = p
    return m


def parse_group(group: str) -> Tuple[str, str]:
    season, kind = group.split("_", 1)
    return season, kind


def load_rgb_tensor(path: Path, tf, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(MODELS_DIR / "decoder.pth", map_location=device))
    vgg.load_state_dict(torch.load(MODELS_DIR / "vgg_normalised.pth", map_location=device))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    decoder.to(device)
    vgg.to(device)

    enc_layers = list(vgg.children())
    enc_1 = nn.Sequential(*enc_layers[:4]).to(device).eval()
    enc_2 = nn.Sequential(*enc_layers[4:11]).to(device).eval()
    enc_3 = nn.Sequential(*enc_layers[11:18]).to(device).eval()
    enc_4 = nn.Sequential(*enc_layers[18:31]).to(device).eval()

    for module in (enc_1, enc_2, enc_3, enc_4, decoder):
        for p in module.parameters():
            p.requires_grad = False

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    content_map = find_content_map(CONTENT_DIR)

    metrics_rows = []

    with torch.no_grad():
        for group in GROUPS:
            season, kind = parse_group(group)
            style_dir = STYLE_ROOT / season / kind
            output_dir = OUTPUT_ROOT / group

            style_paths = sorted([p for p in style_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}])
            if not style_paths:
                continue

            weights = torch.ones(len(style_paths), dtype=torch.float32, device=device)
            weights = weights / weights.sum()

            style_feats_per_layer: List[List[torch.Tensor]] = [[], [], [], []]
            for sp in style_paths:
                s = load_rgb_tensor(sp, tf, device)
                s_feats = encode_with_intermediate(s, enc_1, enc_2, enc_3, enc_4)
                for i in range(4):
                    style_feats_per_layer[i].append(s_feats[i])

            target_style_stats = []
            for i in range(4):
                means = []
                stds = []
                for feat in style_feats_per_layer[i]:
                    m, s = calc_mean_std(feat, eps=EPS)
                    means.append(m)
                    stds.append(s)
                target_style_stats.append(weighted_stats(means, stds, weights))

            target_mean4, target_std4 = target_style_stats[3]

            out_paths = sorted([p for p in output_dir.glob("*_interpolation.jpg") if p.is_file()])
            for out_path in out_paths:
                stem = out_path.name.replace("_interpolation.jpg", "")
                content_path = content_map.get(stem)
                if content_path is None:
                    continue

                content = load_rgb_tensor(content_path, tf, device)
                output = load_rgb_tensor(out_path, tf, device)

                c_feats = encode_with_intermediate(content, enc_1, enc_2, enc_3, enc_4)
                c4 = c_feats[3]

                c_mean, c_std = calc_mean_std(c4, eps=EPS)
                t = (c4 - c_mean) / c_std * target_std4 + target_mean4

                g_feats = encode_with_intermediate(output, enc_1, enc_2, enc_3, enc_4)
                content_loss = mse(g_feats[3], t).item()

                style_loss_val = 0.0
                for i in range(4):
                    g_mean, g_std = calc_mean_std(g_feats[i], eps=EPS)
                    tm, ts = target_style_stats[i]
                    style_loss_val += (mse(g_mean, tm) + mse(g_std, ts)).item()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                c4_sp = encode_with_intermediate(content, enc_1, enc_2, enc_3, enc_4)[3]
                c_mean_sp, c_std_sp = calc_mean_std(c4_sp, eps=EPS)
                t_sp = (c4_sp - c_mean_sp) / c_std_sp * target_std4 + target_mean4
                _ = decoder(t_sp)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = max(time.perf_counter() - t0, 1e-9)
                fps = 1.0 / dt

                metrics_rows.append({
                    "group": group,
                    "tile": stem,
                    "content_loss": content_loss,
                    "style_loss": style_loss_val,
                    "fps": fps,
                })

    metrics_csv = OUTPUT_ROOT / "metrics_per_tile.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "tile", "content_loss", "style_loss", "fps"])
        writer.writeheader()
        writer.writerows(metrics_rows)

    summary_rows = []
    for group in GROUPS:
        g_rows = [r for r in metrics_rows if r["group"] == group]
        if not g_rows:
            continue
        n = len(g_rows)
        summary_rows.append({
            "group": group,
            "count": n,
            "content_loss_mean": sum(r["content_loss"] for r in g_rows) / n,
            "style_loss_mean": sum(r["style_loss"] for r in g_rows) / n,
            "fps_mean": sum(r["fps"] for r in g_rows) / n,
        })

    if metrics_rows:
        n_all = len(metrics_rows)
        summary_rows.append({
            "group": "OVERALL",
            "count": n_all,
            "content_loss_mean": sum(r["content_loss"] for r in metrics_rows) / n_all,
            "style_loss_mean": sum(r["style_loss"] for r in metrics_rows) / n_all,
            "fps_mean": sum(r["fps"] for r in metrics_rows) / n_all,
        })

    summary_csv = OUTPUT_ROOT / "metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "count", "content_loss_mean", "style_loss_mean", "fps_mean"])
        writer.writeheader()
        writer.writerows(summary_rows)

    def fmt(x: float) -> str:
        return f"{x:.6f}"

    print("| group | count | content_loss_mean | style_loss_mean | fps_mean |")
    print("|---|---:|---:|---:|---:|")
    for r in summary_rows:
        print(
            f"| {r['group']} | {r['count']} | {fmt(r['content_loss_mean'])} | {fmt(r['style_loss_mean'])} | {fmt(r['fps_mean'])} |"
        )

    print(f"\nSaved: {metrics_csv}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
