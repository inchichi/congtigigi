import argparse
import copy
import random
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import net
from function import adaptive_instance_normalization as adain
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def train_transform(resize=512, crop=256, hflip_prob=0.5):
    return transforms.Compose([
        transforms.Resize(size=(resize, resize)),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.ToTensor(),
    ])


def eval_transform(resize=512, crop=256):
    return transforms.Compose([
        transforms.Resize(size=(resize, resize)),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
    ])


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform, paths=None, recursive=False):
        super().__init__()
        self.root = root
        self.transform = transform

        if paths is None:
            self.paths = collect_image_paths(self.root, recursive=recursive)
        else:
            self.paths = list(paths)
            if not self.paths:
                raise RuntimeError(f"No image files provided for dataset rooted at {self.root}")

        if not self.paths:
            raise RuntimeError(f"No image files found in {self.root}")

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert("RGB")
        return self.transform(img)

    def __len__(self):
        return len(self.paths)


def collect_image_paths(root, recursive=False):
    root_path = Path(root)
    if not root_path.exists():
        raise RuntimeError(f"Directory does not exist: {root}")
    if not root_path.is_dir():
        raise RuntimeError(f"Not a directory: {root}")

    candidates = root_path.rglob("*") if recursive else root_path.glob("*")
    paths = [
        p for p in sorted(candidates) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    if not paths:
        mode = "recursive" if recursive else "flat"
        raise RuntimeError(f"No image files found in {root} ({mode})")
    return paths


def split_train_holdout(paths, holdout_ratio, seed):
    if holdout_ratio <= 0.0:
        return list(paths), []
    if holdout_ratio >= 1.0:
        raise ValueError("holdout_ratio must be < 1.0")
    if len(paths) < 2:
        raise ValueError("Need at least 2 images for holdout splitting")

    holdout_count = max(1, int(round(len(paths) * holdout_ratio)))
    holdout_count = min(holdout_count, len(paths) - 1)

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)

    holdout_paths = sorted(shuffled[:holdout_count])
    train_paths = sorted(shuffled[holdout_count:])
    return train_paths, holdout_paths


def adjust_learning_rate_inverse(optimizer, base_lr, lr_decay, iteration_count):
    lr = base_lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param.data, alpha=1.0 - decay)
    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.copy_(buf)


def gram_matrix(feat):
    n, c, h, w = feat.size()
    feat = feat.view(n, c, h * w)
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / (c * h * w)


def calc_gram_style_loss(mse_loss, input_feats, target_feats):
    loss = 0.0
    for input_feat, target_feat in zip(input_feats, target_feats):
        loss = loss + mse_loss(gram_matrix(input_feat), gram_matrix(target_feat))
    return loss


def total_variation_loss(image):
    loss_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return loss_h + loss_w


def compute_main_losses(network, content, style, alpha=1.0):
    style_feats = network.encode_with_intermediate(style)
    content_feat = network.encode(content)

    t = adain(content_feat, style_feats[-1])
    t = alpha * t + (1.0 - alpha) * content_feat

    g_t = network.decoder(t)
    g_t_feats = network.encode_with_intermediate(g_t)

    loss_c = network.calc_content_loss(g_t_feats[-1], t)
    loss_s = network.calc_style_loss(g_t_feats[0], style_feats[0])
    for i in range(1, 4):
        loss_s = loss_s + network.calc_style_loss(g_t_feats[i], style_feats[i])

    return loss_c, loss_s, g_t, g_t_feats, style_feats, content_feat


@torch.no_grad()
def evaluate_paper_metric(
    network,
    content_loader,
    style_loader,
    device,
    eval_batches,
    content_weight,
    style_weight,
):
    network.eval()

    total_c = 0.0
    total_s = 0.0
    total_metric = 0.0

    content_iter = iter(content_loader)
    style_iter = iter(style_loader)

    for _ in range(eval_batches):
        try:
            content_images = next(content_iter)
        except StopIteration:
            content_iter = iter(content_loader)
            content_images = next(content_iter)

        try:
            style_images = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_images = next(style_iter)

        content_images = content_images.to(device)
        style_images = style_images.to(device)

        if content_images.size(0) != style_images.size(0):
            b = min(content_images.size(0), style_images.size(0))
            content_images = content_images[:b]
            style_images = style_images[:b]

        loss_c, loss_s = network(content_images, style_images)
        metric = content_weight * loss_c + style_weight * loss_s

        total_c += loss_c.item()
        total_s += loss_s.item()
        total_metric += metric.item()

    denom = float(eval_batches)
    return {
        "loss_content": total_c / denom,
        "loss_style": total_s / denom,
        "paper_metric": total_metric / denom,
    }


def state_dict_to_cpu(module):
    state_dict = module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.device("cpu"))
    return state_dict


parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--content_dir", type=str, required=True)
parser.add_argument("--style_dir", type=str, required=True)
parser.add_argument(
    "--content_recursive",
    action="store_true",
    help="Recursively load content images from subdirectories",
)
parser.add_argument(
    "--style_recursive",
    action="store_true",
    help="Recursively load style images from subdirectories",
)
parser.add_argument("--vgg", type=str, default="models/vgg_normalised.pth")
parser.add_argument(
    "--decoder_init",
    type=str,
    default="models/decoder.pth",
    help="Optional decoder checkpoint to initialize/fine-tune from",
)

# Output options
parser.add_argument("--save_dir", default="./experiments_improved")
parser.add_argument("--log_dir", default="./logs_improved")
parser.add_argument("--save_model_interval", type=int, default=5000)

# Optimization options
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--lr_decay", type=float, default=5e-5)
parser.add_argument("--lr_schedule", choices=["inverse", "cosine"], default="cosine")
parser.add_argument("--max_iter", type=int, default=160000)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--n_threads", type=int, default=8)

# Loss weights
parser.add_argument("--content_weight", type=float, default=1.0)
parser.add_argument("--style_weight", type=float, default=10.0)
parser.add_argument("--gram_style_weight", type=float, default=0.0)
parser.add_argument("--tv_weight", type=float, default=0.0)
parser.add_argument("--recon_weight", type=float, default=0.0)

# EMA / evaluation options
parser.add_argument("--ema_decay", type=float, default=0.999)
parser.add_argument("--use_ema_eval", action="store_true")
parser.add_argument("--eval_interval", type=int, default=5000)
parser.add_argument("--eval_batches", type=int, default=20)
parser.add_argument("--val_content_dir", type=str, default="")
parser.add_argument("--val_style_dir", type=str, default="")
parser.add_argument(
    "--holdout_content_ratio",
    type=float,
    default=0.0,
    help="Fraction of content images held out from training for eval diagnostics",
)
parser.add_argument(
    "--holdout_style_ratio",
    type=float,
    default=0.0,
    help="Fraction of style images held out from training for unseen-style eval",
)
parser.add_argument("--split_seed", type=int, default=42)
parser.add_argument(
    "--best_metric_target",
    choices=["seen", "holdout_style", "blended"],
    default="seen",
    help="Metric target used to select best_decoder",
)
parser.add_argument(
    "--holdout_style_blend",
    type=float,
    default=0.7,
    help="Blend weight for holdout_style in blended target (0~1)",
)

args = parser.parse_args()

if not (0.0 <= args.holdout_content_ratio < 1.0):
    raise ValueError("--holdout_content_ratio must be in [0, 1)")
if not (0.0 <= args.holdout_style_ratio < 1.0):
    raise ValueError("--holdout_style_ratio must be in [0, 1)")
if not (0.0 <= args.holdout_style_blend <= 1.0):
    raise ValueError("--holdout_style_blend must be in [0, 1]")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
log_dir = Path(args.log_dir)
log_dir.mkdir(exist_ok=True, parents=True)
writer = SummaryWriter(log_dir=str(log_dir))

# Build network
vgg = net.vgg
vgg.load_state_dict(torch.load(args.vgg, map_location="cpu"))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, net.decoder)
network.train()
network.to(device)

if args.decoder_init:
    decoder_init = Path(args.decoder_init)
    if decoder_init.is_file():
        network.decoder.load_state_dict(
            torch.load(str(decoder_init), map_location="cpu")
        )
        print(f"[train_improved] Loaded decoder init: {decoder_init}")
    else:
        print(
            f"[train_improved] decoder_init not found, training from scratch: {decoder_init}"
        )

# EMA decoder for smoother/better metric checkpoints
ema_decoder = copy.deepcopy(network.decoder)
ema_decoder.eval()
for param in ema_decoder.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)
if args.lr_schedule == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iter, eta_min=args.min_lr
    )
else:
    scheduler = None

mse_loss = nn.MSELoss()

# Data
content_tf = train_transform()
style_tf = train_transform()

all_content_paths = collect_image_paths(
    args.content_dir, recursive=args.content_recursive
)
all_style_paths = collect_image_paths(args.style_dir, recursive=args.style_recursive)

train_content_paths, holdout_content_paths = split_train_holdout(
    all_content_paths, args.holdout_content_ratio, args.split_seed + 11
)
train_style_paths, holdout_style_paths = split_train_holdout(
    all_style_paths, args.holdout_style_ratio, args.split_seed + 29
)

print(
    "[train_improved] content: total={} train={} holdout={} (recursive={})".format(
        len(all_content_paths),
        len(train_content_paths),
        len(holdout_content_paths),
        args.content_recursive,
    )
)
print(
    "[train_improved] style: total={} train={} holdout={} (recursive={})".format(
        len(all_style_paths),
        len(train_style_paths),
        len(holdout_style_paths),
        args.style_recursive,
    )
)

content_dataset = FlatFolderDataset(args.content_dir, content_tf, paths=train_content_paths)
style_dataset = FlatFolderDataset(args.style_dir, style_tf, paths=train_style_paths)

content_iter = iter(
    data.DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads,
    )
)
style_iter = iter(
    data.DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads,
    )
)

# Validation data for metric tracking
def make_eval_loader(paths):
    dataset = FlatFolderDataset("<eval>", eval_transform(), paths=paths)
    eval_batch_size = min(args.batch_size, len(dataset))
    return data.DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=True,
        num_workers=max(1, args.n_threads // 2),
        # Keep tensor shapes stable across content/style loader pairs.
        drop_last=True,
    )

if args.val_content_dir:
    seen_eval_content_paths = collect_image_paths(
        args.val_content_dir, recursive=args.content_recursive
    )
else:
    seen_eval_content_paths = train_content_paths

if args.val_style_dir:
    seen_eval_style_paths = collect_image_paths(
        args.val_style_dir, recursive=args.style_recursive
    )
else:
    seen_eval_style_paths = train_style_paths

val_content_loader = make_eval_loader(seen_eval_content_paths)
val_style_loader = make_eval_loader(seen_eval_style_paths)

holdout_style_loader = (
    make_eval_loader(holdout_style_paths) if holdout_style_paths else None
)
holdout_content_loader = (
    make_eval_loader(holdout_content_paths) if holdout_content_paths else None
)

if args.best_metric_target != "seen" and holdout_style_loader is None:
    print(
        "[train_improved] holdout_style split is empty. "
        "best_metric_target will fallback to seen metric."
    )

best_metric = float("inf")

for i in tqdm(range(args.max_iter)):
    if args.lr_schedule == "inverse":
        lr = adjust_learning_rate_inverse(optimizer, args.lr, args.lr_decay, i)
    else:
        lr = optimizer.param_groups[0]["lr"]

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)

    loss_c, loss_s, g_t, g_t_feats, style_feats, content_feat = compute_main_losses(
        network, content_images, style_images
    )

    loss_gram = calc_gram_style_loss(mse_loss, g_t_feats, style_feats)
    loss_tv = total_variation_loss(g_t)

    # Reconstruction regularization helps decoder better invert VGG features.
    recon_content = network.decoder(content_feat)
    style_feat = network.encode(style_images)
    recon_style = network.decoder(style_feat)
    loss_recon = mse_loss(recon_content, content_images) + mse_loss(recon_style, style_images)

    total_loss = (
        args.content_weight * loss_c
        + args.style_weight * loss_s
        + args.gram_style_weight * loss_gram
        + args.tv_weight * loss_tv
        + args.recon_weight * loss_recon
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    update_ema(ema_decoder, network.decoder, args.ema_decay)

    writer.add_scalar("train/loss_content", loss_c.item(), i + 1)
    writer.add_scalar("train/loss_style", loss_s.item(), i + 1)
    writer.add_scalar("train/loss_gram", loss_gram.item(), i + 1)
    writer.add_scalar("train/loss_tv", loss_tv.item(), i + 1)
    writer.add_scalar("train/loss_recon", loss_recon.item(), i + 1)
    writer.add_scalar("train/loss_total", total_loss.item(), i + 1)
    writer.add_scalar("train/lr", lr, i + 1)

    if (i + 1) % args.eval_interval == 0 or (i + 1) == 1:
        decoder_state = None
        if args.use_ema_eval:
            decoder_state = copy.deepcopy(network.decoder.state_dict())
            network.decoder.load_state_dict(ema_decoder.state_dict())

        seen_eval_stats = evaluate_paper_metric(
            network,
            val_content_loader,
            val_style_loader,
            device,
            args.eval_batches,
            args.content_weight,
            args.style_weight,
        )

        writer.add_scalar("eval_seen/loss_content", seen_eval_stats["loss_content"], i + 1)
        writer.add_scalar("eval_seen/loss_style", seen_eval_stats["loss_style"], i + 1)
        writer.add_scalar("eval_seen/paper_metric", seen_eval_stats["paper_metric"], i + 1)

        holdout_style_stats = None
        if holdout_style_loader is not None:
            holdout_style_stats = evaluate_paper_metric(
                network,
                val_content_loader,
                holdout_style_loader,
                device,
                args.eval_batches,
                args.content_weight,
                args.style_weight,
            )
            writer.add_scalar(
                "eval_holdout_style/loss_content",
                holdout_style_stats["loss_content"],
                i + 1,
            )
            writer.add_scalar(
                "eval_holdout_style/loss_style",
                holdout_style_stats["loss_style"],
                i + 1,
            )
            writer.add_scalar(
                "eval_holdout_style/paper_metric",
                holdout_style_stats["paper_metric"],
                i + 1,
            )

        holdout_joint_stats = None
        if holdout_style_loader is not None and holdout_content_loader is not None:
            holdout_joint_stats = evaluate_paper_metric(
                network,
                holdout_content_loader,
                holdout_style_loader,
                device,
                args.eval_batches,
                args.content_weight,
                args.style_weight,
            )
            writer.add_scalar(
                "eval_holdout_joint/loss_content",
                holdout_joint_stats["loss_content"],
                i + 1,
            )
            writer.add_scalar(
                "eval_holdout_joint/loss_style",
                holdout_joint_stats["loss_style"],
                i + 1,
            )
            writer.add_scalar(
                "eval_holdout_joint/paper_metric",
                holdout_joint_stats["paper_metric"],
                i + 1,
            )

        tracked_metric = seen_eval_stats["paper_metric"]
        if args.best_metric_target == "holdout_style" and holdout_style_stats is not None:
            tracked_metric = holdout_style_stats["paper_metric"]
        elif args.best_metric_target == "blended" and holdout_style_stats is not None:
            tracked_metric = (
                (1.0 - args.holdout_style_blend) * seen_eval_stats["paper_metric"]
                + args.holdout_style_blend * holdout_style_stats["paper_metric"]
            )
        writer.add_scalar("eval/tracked_metric", tracked_metric, i + 1)

        if tracked_metric < best_metric:
            best_metric = tracked_metric
            best_decoder = ema_decoder if args.use_ema_eval else network.decoder
            torch.save(state_dict_to_cpu(best_decoder), save_dir / "best_decoder.pth.tar")
            msg = (
                f"[train_improved] new best @iter {i + 1}: "
                f"tracked={tracked_metric:.6f}, seen={seen_eval_stats['paper_metric']:.6f}"
            )
            if holdout_style_stats is not None:
                msg += f", holdout_style={holdout_style_stats['paper_metric']:.6f}"
            if holdout_joint_stats is not None:
                msg += f", holdout_joint={holdout_joint_stats['paper_metric']:.6f}"
            print(msg)

        summary = (
            f"[eval {i + 1}] seen={seen_eval_stats['paper_metric']:.6f}, "
            f"tracked={tracked_metric:.6f}, best={best_metric:.6f}"
        )
        if holdout_style_stats is not None:
            summary += f", holdout_style={holdout_style_stats['paper_metric']:.6f}"
        if holdout_joint_stats is not None:
            summary += f", holdout_joint={holdout_joint_stats['paper_metric']:.6f}"
        print(summary)

        if decoder_state is not None:
            network.decoder.load_state_dict(decoder_state)
        network.train()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        torch.save(
            state_dict_to_cpu(network.decoder),
            save_dir / f"decoder_iter_{i + 1}.pth.tar",
        )
        torch.save(
            state_dict_to_cpu(ema_decoder),
            save_dir / f"decoder_ema_iter_{i + 1}.pth.tar",
        )

writer.close()
