from __future__ import annotations

import copy
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageDraw
from torchvision import transforms

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import net

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class EvalRow:
    step: int
    seen: float
    tracked: float
    best_so_far: float
    holdout_style: float


def collect_image_paths(root: Path, recursive: bool = False) -> list[Path]:
    candidates = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def split_train_holdout(paths: list[Path], holdout_ratio: float, seed: int):
    if holdout_ratio <= 0.0:
        return list(paths), []

    holdout_count = max(1, int(round(len(paths) * holdout_ratio)))
    holdout_count = min(holdout_count, len(paths) - 1)

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)

    holdout = sorted(shuffled[:holdout_count])
    train = sorted(shuffled[holdout_count:])
    return train, holdout


def parse_eval_rows(log_path: Path) -> list[EvalRow]:
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    pattern = re.compile(
        r"\[eval\s+(\d+)\]\s+seen=([0-9.]+),\s+tracked=([0-9.]+),\s+best=([0-9.]+),\s+holdout_style=([0-9.]+)"
    )
    rows = []
    for m in pattern.finditer(text):
        rows.append(
            EvalRow(
                step=int(m.group(1)),
                seen=float(m.group(2)),
                tracked=float(m.group(3)),
                best_so_far=float(m.group(4)),
                holdout_style=float(m.group(5)),
            )
        )

    # Keep unique by step (last one wins) then sort.
    dedup = {r.step: r for r in rows}
    return [dedup[k] for k in sorted(dedup.keys())]


class PathDataset(data.Dataset):
    def __init__(self, paths: list[Path], tf):
        self.paths = list(paths)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.tf(Image.open(str(self.paths[idx])).convert("RGB"))


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]
    )


def build_network(decoder_path: Path, device: torch.device):
    vgg = copy.deepcopy(net.vgg)
    vgg.load_state_dict(
        torch.load(str(PROJECT_ROOT / "models" / "vgg_normalised.pth"), map_location="cpu")
    )
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
    batches: int = 40,
    batch_size: int = 8,
):
    tf = eval_transform()
    cset = PathDataset(content_paths, tf)
    sset = PathDataset(style_paths, tf)

    cbs = min(batch_size, len(cset))
    sbs = min(batch_size, len(sset))

    cloader = data.DataLoader(cset, batch_size=cbs, shuffle=False, num_workers=2, drop_last=False)
    sloader = data.DataLoader(sset, batch_size=sbs, shuffle=False, num_workers=2, drop_last=False)

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

        b = min(content.size(0), style.size(0))
        content = content[:b].to(device)
        style = style[:b].to(device)

        loss_c, loss_s = network(content, style)
        total_metric += (1.0 * loss_c + 10.0 * loss_s).item()

    return total_metric / float(batches)


def fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    img = img.copy()
    img.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_panel(
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
        ("Best Decoder", improved),
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
    root = PROJECT_ROOT

    report_md = root / "research_paper_adain_improvement.md"
    figure_dir = root / "report_assets" / "figures"
    data_dir = root / "report_assets" / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    log_path = root / "logs_improved" / "train_generalize_20260401_172313.log"

    eval_rows = parse_eval_rows(log_path)
    if not eval_rows:
        raise RuntimeError("No eval rows found in training log")

    # Data split used by train_generalize.sh
    content_all = collect_image_paths(root / "input" / "content" / "PNG", recursive=False)
    style_all = collect_image_paths(root / "input" / "style", recursive=True)

    content_train, content_holdout = split_train_holdout(content_all, 0.0, 42 + 11)
    style_train, style_holdout = split_train_holdout(style_all, 0.2, 42 + 29)

    # Evaluate comparable baseline/best/final metrics on identical split.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = build_network(root / "models" / "decoder.pth", device)
    best_model = build_network(
        root / "experiments_improved" / "generalize_20260401_172313" / "best_decoder.pth.tar",
        device,
    )
    final_model = build_network(
        root / "experiments_improved" / "generalize_20260401_172313" / "decoder_iter_10000.pth.tar",
        device,
    )

    baseline_seen = evaluate_metric(baseline_model, content_train, style_train, device)
    baseline_holdout = evaluate_metric(baseline_model, content_train, style_holdout, device)

    best_seen = evaluate_metric(best_model, content_train, style_train, device)
    best_holdout = evaluate_metric(best_model, content_train, style_holdout, device)

    final_seen = evaluate_metric(final_model, content_train, style_train, device)
    final_holdout = evaluate_metric(final_model, content_train, style_holdout, device)

    # Save parsed eval CSV
    eval_csv = data_dir / "generalize_eval_curve.csv"
    with eval_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "seen", "tracked", "best_so_far", "holdout_style", "gap"])
        for r in eval_rows:
            writer.writerow(
                [
                    r.step,
                    f"{r.seen:.6f}",
                    f"{r.tracked:.6f}",
                    f"{r.best_so_far:.6f}",
                    f"{r.holdout_style:.6f}",
                    f"{(r.holdout_style - r.seen):.6f}",
                ]
            )

    summary_json = data_dir / "generalize_report_summary.json"
    best_row = min(eval_rows, key=lambda r: r.tracked)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_eval_from_log": best_row.__dict__,
                "final_eval_from_log": eval_rows[-1].__dict__,
                "split_counts": {
                    "content_train": len(content_train),
                    "content_holdout": len(content_holdout),
                    "style_train": len(style_train),
                    "style_holdout": len(style_holdout),
                },
                "comparable_metrics": {
                    "baseline": {"seen": baseline_seen, "holdout_style": baseline_holdout},
                    "best_7500": {"seen": best_seen, "holdout_style": best_holdout},
                    "final_10000": {"seen": final_seen, "holdout_style": final_holdout},
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Figure A: Training metric curve
    steps = [r.step for r in eval_rows]
    seen_vals = [r.seen for r in eval_rows]
    holdout_vals = [r.holdout_style for r in eval_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(steps, seen_vals, marker="o", linewidth=2, label="Seen-style metric")
    plt.plot(steps, holdout_vals, marker="o", linewidth=2, label="Holdout-style metric")
    plt.xlabel("Training step")
    plt.ylabel("Paper metric (lower is better)")
    plt.title("Generalization Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "generalize_train_curve.png", dpi=200)
    plt.close()

    # Figure B: Generalization gap
    gaps = [h - s for h, s in zip(holdout_vals, seen_vals)]
    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, gaps, marker="o", linewidth=2, color="#C0392B")
    plt.axhline(0.0, linestyle="--", linewidth=1, color="#555555")
    plt.xlabel("Training step")
    plt.ylabel("Holdout - Seen gap")
    plt.title("Generalization Gap (Overfitting Indicator)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_dir / "generalize_gap_curve.png", dpi=200)
    plt.close()

    # Figure C: Comparable baseline/best/final metrics
    labels = ["Baseline", "Best(7500)", "Final(10000)"]
    seen_bar = [baseline_seen, best_seen, final_seen]
    holdout_bar = [baseline_holdout, best_holdout, final_holdout]

    x = range(len(labels))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], seen_bar, width=width, label="Seen-style")
    plt.bar([i + width / 2 for i in x], holdout_bar, width=width, label="Holdout-style")
    plt.xticks(list(x), labels)
    plt.ylabel("Paper metric (lower is better)")
    plt.title("Baseline vs Best vs Final (same data split)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "generalize_baseline_best_final.png", dpi=200)
    plt.close()

    # Comparison panels (visual before/after)
    make_panel(
        content_path=root / "input" / "content" / "cornell.jpg",
        style_path=root / "input" / "style" / "woman_with_hat_matisse.jpg",
        baseline_path=root
        / "report_assets"
        / "images"
        / "baseline_generalize"
        / "cornell_stylized_woman_with_hat_matisse.jpg",
        improved_path=root
        / "report_assets"
        / "images"
        / "improved_generalize"
        / "cornell_stylized_woman_with_hat_matisse.jpg",
        out_path=figure_dir / "compare_photo_generalize.png",
        title="Sample A: Cornell + Woman with Hat",
    )

    make_panel(
        content_path=root / "input" / "content" / "PNG" / "rpgTile000.png",
        style_path=root / "input" / "style" / "test" / "winter.png",
        baseline_path=root
        / "report_assets"
        / "images"
        / "baseline_generalize"
        / "rpgTile000_stylized_winter.jpg",
        improved_path=root
        / "report_assets"
        / "images"
        / "improved_generalize"
        / "rpgTile000_stylized_winter.jpg",
        out_path=figure_dir / "compare_tile_generalize.png",
        title="Sample B: RPG Tile + Winter",
    )

    # Build report markdown
    holdout_improve_vs_baseline = (baseline_holdout - best_holdout) / baseline_holdout * 100.0
    seen_improve_vs_baseline = (baseline_seen - best_seen) / baseline_seen * 100.0

    table_rows = [
        ("Baseline", 0, baseline_seen, baseline_holdout),
        ("Best checkpoint", best_row.step, best_seen, best_holdout),
        ("Final checkpoint", 10000, final_seen, final_holdout),
    ]

    report_md.write_text(
        "\n".join(
            [
                "# AdaIN Generalization Improvement Report (Updated)",
                "",
                "## 1. 목표",
                "처음 보는 스타일(holdout style)에도 잘 작동하도록 AdaIN decoder를 개선하고, 기존 코드 대비 수정점이 성능에 어떤 영향을 줬는지 정리한다.",
                "",
                "## 2. 기존 코드 대비 수정 사항",
                "### 2.1 학습 코드 구조 변경",
                "- 기존 `train.py`는 `style_dir` 최상위 파일만 읽음(`glob('*')`).",
                "- 개선 `train_improved.py`는 `--style_recursive`로 하위 폴더까지 재귀 로딩(`rglob('*')`) 가능.",
                "- `collect_image_paths`, `split_train_holdout`를 추가해 학습/검증 분할을 코드 내에서 명시적으로 관리.",
                "",
                "### 2.2 일반화 중심 검증/베스트 선택",
                "- `--holdout_style_ratio 0.2`로 스타일 20%를 학습에서 제외하고 holdout 검증셋으로 사용.",
                "- `--best_metric_target holdout_style`로 best checkpoint를 seen이 아니라 holdout 기준으로 저장.",
                "- `eval_seen`, `eval_holdout_style`, `eval/tracked_metric`를 로그로 남겨 일반화 성능을 직접 추적.",
                "",
                "### 2.3 학습 안정화/품질 개선 항목",
                "- `RandomHorizontalFlip` 추가(학습 다양성 증가).",
                "- 추가 손실 항목 도입: `gram_style_weight`, `recon_weight`, `tv_weight`(선택).",
                "- EMA decoder(`--use_ema_eval`)로 평가 안정화.",
                "",
                "### 2.4 실행 스크립트",
                "- `train_generalize.sh`를 추가해 일반화 목적 하이퍼파라미터를 고정 재현 가능하게 구성.",
                "- 실제 실행: `./train_generalize.sh --n_threads 4`",
                "",
                "## 3. 이번 학습 설정",
                "- Run ID: `generalize_20260401_172313`",
                "- Steps: 10,000",
                "- Content split: train 230 / holdout 0 (`input/content/PNG`)",
                "- Style split: train 54 / holdout 14 (`input/style`, recursive)",
                "- Best checkpoint from log: **iter 7500** (tracked holdout-style metric 최소)",
                "",
                "## 4. 정량 결과",
                "### 4.1 학습 중 곡선 (로그 기반)",
                "![Train curve](report_assets/figures/generalize_train_curve.png)",
                "",
                "### 4.2 과적합 지표 (Generalization gap)",
                "![Gap curve](report_assets/figures/generalize_gap_curve.png)",
                "",
                f"- Log best: iter {best_row.step}, holdout_style={best_row.holdout_style:.6f}, seen={best_row.seen:.6f}",
                f"- Final(10000): holdout_style={eval_rows[-1].holdout_style:.6f}, seen={eval_rows[-1].seen:.6f}",
                "- 해석: 후반부(7500 이후)에는 seen은 더 좋아지지만 holdout은 악화되어 과적합이 관찰됨.",
                "",
                "### 4.3 동일 분할 기준 비교 평가 (추가 재평가)",
                "![Baseline vs Best vs Final](report_assets/figures/generalize_baseline_best_final.png)",
                "",
                "| Model | Step | Seen Metric | Holdout-style Metric | Gap(Holdout-Seen) |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {name} | {step} | {seen:.4f} | {holdout:.4f} | {holdout-seen:.4f} |"
                    for name, step, seen, holdout in table_rows
                ],
                "",
                f"- Baseline 대비 Best(holdout 기준) 개선율: **{holdout_improve_vs_baseline:.2f}%**",
                f"- Baseline 대비 Best(seen 기준) 개선율: **{seen_improve_vs_baseline:.2f}%**",
                "",
                "## 5. 시각 결과 (Before vs After)",
                "### Sample A",
                "![Sample A](report_assets/figures/compare_photo_generalize.png)",
                "",
                "### Sample B",
                "![Sample B](report_assets/figures/compare_tile_generalize.png)",
                "",
                "## 6. 결론",
                "- 코드 수정으로 스타일 데이터 커버리지가 늘고, holdout 기반 선택이 가능해져 일반화 지표 자체를 개선할 수 있었다.",
                "- 단, 10k까지 계속 학습하면 holdout이 다시 악화되므로 이번 런에서는 `best_decoder.pth.tar`(iter 7500)를 사용하는 것이 타당하다.",
                "",
                "## 7. 재현 산출물",
                "- Report summary JSON: `report_assets/data/generalize_report_summary.json`",
                "- Eval curve CSV: `report_assets/data/generalize_eval_curve.csv`",
                "- Best model: `experiments_improved/generalize_20260401_172313/best_decoder.pth.tar`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[done] report: {report_md}")
    print(f"[done] figures: {figure_dir}")
    print(f"[done] data: {data_dir}")


if __name__ == "__main__":
    main()
