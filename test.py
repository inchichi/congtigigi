import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None, mask=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)

    if mask is not None:
        # Resize mask to feature-map resolution and apply style only on masked region.
        mask = F.interpolate(mask, size=feat.shape[-2:], mode='nearest')
        feat = feat * mask + content_f * (1 - mask)

    return decoder(feat)


def resolve_mask_path(content_path: Path, args):
    if args.mask:
        p = Path(args.mask)
        return p if p.exists() else None
    if not args.mask_dir:
        return None

    mask_dir = Path(args.mask_dir)
    stem = content_path.stem
    candidates = [
        mask_dir / f"{stem}{args.mask_suffix}.png",
        mask_dir / f"{stem}{args.mask_suffix}.jpg",
        mask_dir / f"{stem}{args.mask_suffix}.jpeg",
        mask_dir / f"{stem}{args.mask_suffix}.bmp",
        mask_dir / f"{stem}{args.mask_suffix}.webp",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}.jpeg",
        mask_dir / f"{stem}.bmp",
        mask_dir / f"{stem}.webp",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_binary_mask(mask_path: Path, size: int, crop: bool, threshold: int, invert: bool):
    # Use the same geometric transform as content for alignment.
    mask_tf = test_transform(size, crop)
    mask = mask_tf(Image.open(str(mask_path)).convert("L"))[:1, :, :]
    th = threshold / 255.0
    mask = (mask > th).float()
    if invert:
        mask = 1.0 - mask
    return mask


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--content_dir', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--style_dir', type=str,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')
parser.add_argument('--mask', type=str,
                    help='File path to mask image (white: style, black: preserve content)')
parser.add_argument('--mask_dir', type=str,
                    help='Directory path to per-content masks')
parser.add_argument('--mask_suffix', type=str, default='_semantic',
                    help='Suffix added to content stem when loading masks from --mask_dir')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')
parser.add_argument('--mask_threshold', type=int, default=0,
                    help='Mask threshold in [0,255]. Pixels > threshold are stylized')
parser.add_argument('--invert_mask', action='store_true',
                    help='Invert mask meaning (black: style, white: preserve content)')

args = parser.parse_args()
assert 0 <= args.mask_threshold <= 255

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = Path(args.output)
output_dir.mkdir(exist_ok=True, parents=True)

# Either --content or --contentDir should be given.
assert (args.content or args.content_dir)
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --styleDir should be given.
assert (args.style or args.style_dir)
if args.style:
    style_paths = args.style.split(',')
    if len(style_paths) == 1:
        style_paths = [Path(args.style)]
    else:
        do_interpolation = True
        assert (args.style_interpolation_weights != ''), \
            'Please specify interpolation weights'
        weights = [int(i) for i in args.style_interpolation_weights.split(',')]
        interpolation_weights = [w / sum(weights) for w in weights]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)
style_tf = test_transform(args.style_size, args.crop)

for content_path in content_paths:
    mask_path = resolve_mask_path(content_path, args)
    content_mask = None
    if mask_path is not None:
        content_mask = load_binary_mask(
            mask_path, args.content_size, args.crop,
            args.mask_threshold, args.invert_mask
        ).unsqueeze(0).to(device)

    if do_interpolation:  # one content image, N style image
        style = torch.stack([style_tf(Image.open(str(p)).convert("RGB")) for p in style_paths])
        base_content = content_tf(Image.open(str(content_path)).convert("RGB"))
        content = base_content.unsqueeze(0).expand_as(style)
        style = style.to(device)
        content = content.to(device)
        with torch.no_grad():
            output = style_transfer(vgg, decoder, content, style,
                                    args.alpha, interpolation_weights,
                                    mask=content_mask)
        output = output.cpu()
        output_name = output_dir / '{:s}_interpolation{:s}'.format(
            content_path.stem, args.save_ext)
        save_image(output, str(output_name))

    else:  # process one content and one style
        for style_path in style_paths:
            content = content_tf(Image.open(str(content_path)).convert("RGB"))
            style = style_tf(Image.open(str(style_path)).convert("RGB"))
            if args.preserve_color:
                style = coral(style, content)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha, mask=content_mask)
            output = output.cpu()

            output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                content_path.stem, style_path.stem, args.save_ext)
            save_image(output, str(output_name))
