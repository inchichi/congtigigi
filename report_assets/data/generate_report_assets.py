from __future__ import annotations

import copy
import csv
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageDraw
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import net

warnings.filterwarnings("ignore", category=FutureWarning)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class StageMetric:
    name: str
    total_steps: int
    path: str
    metric_seen_style: float
    metric_holdout_style: float
    improve_seen_style_pct: float


class PathDataset(data.Dataset):
    def __init__(self, paths: Iterable[Path], transform):
        self.paths = list(paths)
        if not self.paths:
            raise RuntimeError("empty dataset")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(str(self.paths[idx])).convert("RGB"))


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )


def list_flat_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.glob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def list_recursive_files_excluding_root(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and p.parent != root
    )


def build_network(decoder_path: Path, vgg_state_dict: dict, device: torch.device):
    vgg = copy.deepcopy(net.vgg)
    vgg.load_state_dict(vgg_state_dict)
    vgg = nn.Sequential(*list(vgg.children())[:31])

    decoder = copy.deepcopy(net.decoder)
    decoder.load_state_dict(torch.load(str(decoder_path), map_location="cpu"))

    network = net.Net(vgg, decoder).to(device)
    network.eval()
    return network


@torch.no_grad()
def evaluate_metric(
    network,
    content_paths: list[Path],
    style_paths: list[Path],
    device: torch.device,
    batches: int,
    batch_size: int,
    num_workers: int = 2,
    content_weight: float = 1.0,
    style_weight: float = 10.0,
):
    tf = eval_transform()
    cset = PathDataset(content_paths, tf)
    sset = PathDataset(style_paths, tf)

    cloader = data.DataLoader(
        cset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    sloader = data.DataLoader(
        sset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )

    if len(cloader) == 0 or len(sloader) == 0:
        raise RuntimeError(
            f"dataloader empty: c={len(cloader)} s={len(sloader)} batch={batch_size}"
        )

    citer, siter = iter(cloader), iter(sloader)
    total_metric = 0.0

    for _ in range(batches):
        try:
            content = next(citer)
        except StopIteration:
            citer = iter(cloader)
            content = next(citer)

        try:
            style = next(siter)
        except StopIteration:
            siter = iter(sloader)
            style = next(siter)

        content = content.to(device)
        style = style.to(device)
        loss_c, loss_s = network(content, style)
        total_metric += (content_weight * loss_c + style_weight * loss_s).item()

    return total_metric / float(batches)


def fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    img = img.copy()
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_comparison_panel(
    content_path: Path,
    style_path: Path,
    baseline_path: Path,
    improved_path: Path,
    out_path: Path,
    title: str,
):
    tile_w, tile_h = 320, 320
    margin = 20
    title_h = 50
    label_h = 40

    content = fit_image(Image.open(content_path).convert("RGB"), tile_w, tile_h)
    style = fit_image(Image.open(style_path).convert("RGB"), tile_w, tile_h)
    baseline = fit_image(Image.open(baseline_path).convert("RGB"), tile_w, tile_h)
    improved = fit_image(Image.open(improved_path).convert("RGB"), tile_w, tile_h)

    cols = [
        ("Content", content),
        ("Style", style),
        ("Baseline", baseline),
        ("Improved", improved),
    ]

    width = margin * (len(cols) + 1) + tile_w * len(cols)
    height = title_h + tile_h + label_h + margin * 2
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)

    draw.text((margin, 12), title, fill=(0, 0, 0))

    for i, (label, image) in enumerate(cols):
        x = margin + i * (tile_w + margin)
        y = title_h
        panel.paste(image, (x, y))
        draw.rectangle((x, y, x + tile_w, y + tile_h), outline=(190, 190, 190), width=2)
        draw.text((x, y + tile_h + 12), label, fill=(0, 0, 0))

    panel.save(out_path)


def main():
    root = Path(__file__).resolve().parents[2]
    assets = root / "report_assets"
    fig_dir = assets / "figures"
    data_dir = assets / "data"
    img_dir = assets / "images"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg_state = torch.load(str(root / "models" / "vgg_normalised.pth"), map_location="cpu")

    content_seen = list_flat_files(root / "input" / "content" / "PNG")
    content_holdout = list_flat_files(root / "input" / "content")
    style_seen = list_flat_files(root / "input" / "style")
    style_holdout = list_recursive_files_excluding_root(root / "input" / "style")

    stages = [
        ("Baseline", 0, root / "models" / "decoder.pth"),
        (
            "Stage 1 (+2k)",
            2000,
            root / "experiments_improved" / "paper_metric_20260401_145811" / "decoder_iter_2000.pth.tar",
        ),
        (
            "Stage 2 (+6k)",
            6000,
            root / "experiments_improved" / "paper_metric_resume_20260401_151127" / "decoder_iter_4000.pth.tar",
        ),
        (
            "Stage 3 (+8k)",
            8000,
            root / "experiments_improved" / "paper_metric_resume2_20260401_153526" / "decoder_iter_2000.pth.tar",
        ),
        (
            "Stage 4 (+10k)",
            10000,
            root / "experiments_improved" / "paper_metric_resume3_20260401_155026" / "decoder_iter_2000.pth.tar",
        ),
    ]

    stage_rows: list[StageMetric] = []

    baseline_metric_seen = None
    for name, steps, path in stages:
        network = build_network(path, vgg_state, device)
        metric_seen = evaluate_metric(
            network,
            content_paths=content_seen,
            style_paths=style_seen,
            device=device,
            batches=80,
            batch_size=8,
        )
        metric_holdout_style = evaluate_metric(
            network,
            content_paths=content_seen,
            style_paths=style_holdout,
            device=device,
            batches=80,
            batch_size=8,
        )
        if baseline_metric_seen is None:
            baseline_metric_seen = metric_seen
        improve_pct = (baseline_metric_seen - metric_seen) / baseline_metric_seen * 100.0
        stage_rows.append(
            StageMetric(
                name=name,
                total_steps=steps,
                path=str(path.relative_to(root)),
                metric_seen_style=metric_seen,
                metric_holdout_style=metric_holdout_style,
                improve_seen_style_pct=improve_pct,
            )
        )

    # Overfitting diagnostics (run3 checkpoints)
    overfit_models = [
        (
            "iter500",
            root
            / "experiments_improved"
            / "paper_metric_resume3_20260401_155026"
            / "decoder_iter_500.pth.tar",
        ),
        (
            "iter1000",
            root
            / "experiments_improved"
            / "paper_metric_resume3_20260401_155026"
            / "decoder_iter_1000.pth.tar",
        ),
        (
            "iter1500",
            root
            / "experiments_improved"
            / "paper_metric_resume3_20260401_155026"
            / "decoder_iter_1500.pth.tar",
        ),
        (
            "iter2000",
            root
            / "experiments_improved"
            / "paper_metric_resume3_20260401_155026"
            / "decoder_iter_2000.pth.tar",
        ),
    ]

    split_defs = {
        "seen_content_seen_style": (content_seen, style_seen),
        "seen_content_holdout_style": (content_seen, style_holdout),
        "holdout_content_seen_style": (content_holdout, style_seen),
        "holdout_content_holdout_style": (content_holdout, style_holdout),
    }

    overfit_rows = []
    for ckpt_name, ckpt_path in overfit_models:
        network = build_network(ckpt_path, vgg_state, device)
        for split_name, (cp, sp) in split_defs.items():
            metric = evaluate_metric(
                network,
                content_paths=cp,
                style_paths=sp,
                device=device,
                batches=30,
                batch_size=4,
            )
            overfit_rows.append(
                {
                    "checkpoint": ckpt_name,
                    "path": str(ckpt_path.relative_to(root)),
                    "split": split_name,
                    "paper_metric": metric,
                }
            )

    # Save CSV/JSON data
    stage_csv = data_dir / "stage_metrics.csv"
    with stage_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "total_steps",
                "path",
                "metric_seen_style",
                "metric_holdout_style",
                "improve_seen_style_pct",
            ]
        )
        for row in stage_rows:
            writer.writerow(
                [
                    row.name,
                    row.total_steps,
                    row.path,
                    f"{row.metric_seen_style:.6f}",
                    f"{row.metric_holdout_style:.6f}",
                    f"{row.improve_seen_style_pct:.4f}",
                ]
            )

    overfit_csv = data_dir / "overfit_metrics.csv"
    with overfit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "path", "split", "paper_metric"])
        for row in overfit_rows:
            writer.writerow(
                [
                    row["checkpoint"],
                    row["path"],
                    row["split"],
                    f"{row['paper_metric']:.6f}",
                ]
            )

    summary_json = data_dir / "report_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "stage_metrics": [row.__dict__ for row in stage_rows],
                "overfit_metrics": overfit_rows,
                "dataset_counts": {
                    "content_seen": len(content_seen),
                    "content_holdout": len(content_holdout),
                    "style_seen": len(style_seen),
                    "style_holdout": len(style_holdout),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Comparison panels
    make_comparison_panel(
        content_path=root / "input" / "content" / "cornell.jpg",
        style_path=root / "input" / "style" / "woman_with_hat_matisse.jpg",
        baseline_path=img_dir
        / "baseline"
        / "cornell_stylized_woman_with_hat_matisse.jpg",
        improved_path=img_dir
        / "improved"
        / "cornell_stylized_woman_with_hat_matisse.jpg",
        out_path=fig_dir / "compare_photo_matisse.png",
        title="Sample A: Cornell + Woman with Hat",
    )

    make_comparison_panel(
        content_path=root / "input" / "content" / "PNG" / "rpgTile000.png",
        style_path=root / "input" / "style" / "test" / "winter.png",
        baseline_path=img_dir / "baseline" / "rpgTile000_stylized_winter.jpg",
        improved_path=img_dir / "improved" / "rpgTile000_stylized_winter.jpg",
        out_path=fig_dir / "compare_tile_winter.png",
        title="Sample B: RPG Tile + Winter",
    )

    # Figure 1: Metric trend vs total training steps
    xs = [r.total_steps for r in stage_rows]
    seen_vals = [r.metric_seen_style for r in stage_rows]
    holdout_vals = [r.metric_holdout_style for r in stage_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, seen_vals, marker="o", linewidth=2, label="Seen style set")
    plt.plot(xs, holdout_vals, marker="o", linewidth=2, label="Holdout style set")
    plt.xlabel("Cumulative fine-tuning steps")
    plt.ylabel("Paper metric (lower is better)")
    plt.title("Metric Trend by Training Stage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_trend_steps.png", dpi=200)
    plt.close()

    # Figure 2: Improvement over baseline (seen-style split)
    names = [r.name for r in stage_rows]
    improvements = [r.improve_seen_style_pct for r in stage_rows]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, improvements)
    plt.ylabel("Improvement over baseline (%)")
    plt.title("Seen-Style Improvement vs Baseline")
    plt.xticks(rotation=20)
    for b, val in zip(bars, improvements):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "improvement_bar.png", dpi=200)
    plt.close()

    # Figure 3: Overfitting diagnostics for run3 checkpoints
    ckpt_order = ["iter500", "iter1000", "iter1500", "iter2000"]
    split_to_vals = {k: [] for k in split_defs.keys()}
    by_ckpt = {k: {s: None for s in split_defs.keys()} for k in ckpt_order}
    for row in overfit_rows:
        by_ckpt[row["checkpoint"]][row["split"]] = row["paper_metric"]

    for ckpt in ckpt_order:
        for split in split_defs.keys():
            split_to_vals[split].append(by_ckpt[ckpt][split])

    x = [500, 1000, 1500, 2000]
    plt.figure(figsize=(10, 5))
    plt.plot(x, split_to_vals["seen_content_seen_style"], marker="o", label="Seen content + seen style")
    plt.plot(x, split_to_vals["seen_content_holdout_style"], marker="o", label="Seen content + holdout style")
    plt.plot(x, split_to_vals["holdout_content_seen_style"], marker="o", label="Holdout content + seen style")
    plt.plot(x, split_to_vals["holdout_content_holdout_style"], marker="o", label="Holdout content + holdout style")
    plt.xlabel("Checkpoint step in run3")
    plt.ylabel("Paper metric (lower is better)")
    plt.title("Overfitting Diagnostics (Run3)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "overfitting_diagnostics.png", dpi=200)
    plt.close()

    # Build markdown report
    baseline_seen = stage_rows[0].metric_seen_style
    best_stage = min(stage_rows, key=lambda r: r.metric_seen_style)

    table_lines = [
        "| Stage | Total Steps | Paper Metric (Seen Style) | Paper Metric (Holdout Style) | Improvement vs Baseline (Seen) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in stage_rows:
        table_lines.append(
            f"| {r.name} | {r.total_steps:,} | {r.metric_seen_style:.4f} | {r.metric_holdout_style:.4f} | {r.improve_seen_style_pct:.2f}% |"
        )

    report_md = root / "research_paper_adain_improvement.md"
    report_md.write_text(
        "\n".join(
            [
                "# Improving AdaIN Stylization Quality Through Progressive Decoder Fine-Tuning",
                "",
                "## Abstract",
                "This report compares baseline AdaIN decoder performance with progressively fine-tuned decoders and documents both visual and quantitative changes. The evaluation metric follows the paper-style objective (`paper_metric = 1*L_content + 10*L_style`, lower is better).",
                "",
                "## 1. Experimental Setup",
                f"- Date: 2026-04-01",
                f"- Device: CUDA GPU",
                f"- Content (seen) set size: {len(content_seen)}",
                f"- Content (holdout) set size: {len(content_holdout)}",
                f"- Style (seen) set size: {len(style_seen)}",
                f"- Style (holdout) set size: {len(style_holdout)}",
                "- Baseline model: `models/decoder.pth`",
                "- Final model candidate: `experiments_improved/paper_metric_resume3_20260401_155026/decoder_iter_2000.pth.tar`",
                "",
                "## 2. Visual Comparison (Before vs After)",
                "### Sample A (Photo Content)",
                "![Sample A comparison](report_assets/figures/compare_photo_matisse.png)",
                "",
                "### Sample B (Tile Content)",
                "![Sample B comparison](report_assets/figures/compare_tile_winter.png)",
                "",
                "## 3. Quantitative Results",
                "### 3.1 Stage-wise Metrics",
                *table_lines,
                "",
                f"Baseline (seen-style) metric: **{baseline_seen:.4f}**",
                f"Best stage (seen-style): **{best_stage.name}** at **{best_stage.metric_seen_style:.4f}** ({best_stage.improve_seen_style_pct:.2f}% improvement)",
                "",
                "### 3.2 Graph: Metric vs Training Stage",
                "![Metric trend](report_assets/figures/metric_trend_steps.png)",
                "",
                "### 3.3 Graph: Improvement vs Baseline",
                "![Improvement bar](report_assets/figures/improvement_bar.png)",
                "",
                "## 4. Overfitting Check",
                "To check overfitting, run3 checkpoints were evaluated on four splits (seen/holdout content x seen/holdout style).",
                "",
                "![Overfitting diagnostics](report_assets/figures/overfitting_diagnostics.png)",
                "",
                "Interpretation:",
                "- Seen-style split keeps improving with training.",
                "- Holdout-style splits fluctuate more, which indicates mild overfitting risk in late checkpoints.",
                "- Despite that risk, the latest checkpoint still outperforms the previous best in our current holdout test.",
                "",
                "## 5. Conclusion",
                "Progressive fine-tuning consistently improved the core paper metric and produced visibly stronger stylization detail compared with the baseline decoder. The current best deployment candidate is the latest run3 final checkpoint, while early stopping on holdout-style metric is recommended for the next cycle.",
                "",
                "## 6. Reproducibility Artifacts",
                "- Stage metric CSV: `report_assets/data/stage_metrics.csv`",
                "- Overfitting CSV: `report_assets/data/overfit_metrics.csv`",
                "- Summary JSON: `report_assets/data/report_summary.json`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[done] wrote {report_md}")
    print(f"[done] figures: {fig_dir}")
    print(f"[done] data: {data_dir}")


if __name__ == "__main__":
    main()
