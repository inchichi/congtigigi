import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import textile
from textile.utils.image_utils import read_and_process_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM heatmap for a TexTile prediction."
    )
    parser.add_argument("image_path", help="Path to the input tile image.")
    parser.add_argument(
        "--output-dir",
        default="gradcam_outputs",
        help="Directory to save the preview, heatmap, and overlay images.",
    )
    parser.add_argument(
        "--target-layer",
        default="features.9",
        help=(
            "Module path inside `loss_textile.model` used for Grad-CAM. "
            "Default: final ConvNeXt stage output."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Square resolution used after TexTile tiles the image.",
    )
    parser.add_argument(
        "--number-tiles",
        type=int,
        default=2,
        help="How many times TexTile repeats the tile in each axis before scoring.",
    )
    parser.add_argument(
        "--lambda-value",
        type=float,
        default=0.25,
        help="TexTile lambda used only for the reported 0-1 score.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay strength. Higher means stronger heatmap colors.",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print named modules inside the backbone and exit.",
    )
    return parser.parse_args()


def resolve_module(root, path):
    module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def tensor_to_rgb_image(tensor):
    image = tensor.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def write_rgb_image(path, image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self.forward_handle = target_module.register_forward_hook(self._forward_hook)
        self.backward_handle = target_module.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def compute(self, model_input):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(model_input)
        target = logits.squeeze()
        target.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=model_input.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        if torch.isclose(cam_max, cam_min):
            cam = torch.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return target.detach(), cam.detach().cpu().numpy()


def save_cam_outputs(output_dir, stem, preview_rgb, cam, alpha):
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(
        preview_rgb,
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0.0,
    )

    preview_path = output_dir / f"{stem}_preview.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_overlay.png"

    write_rgb_image(preview_path, preview_rgb)
    write_rgb_image(heatmap_path, heatmap_rgb)
    write_rgb_image(overlay_path, overlay)

    return preview_path, heatmap_path, overlay_path


def main():
    args = parse_args()
    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loss_textile = textile.Textile(
        lambda_value=args.lambda_value,
        resolution=(args.resolution, args.resolution),
        number_tiles=args.number_tiles,
    )

    if args.list_layers:
        for name, module in loss_textile.model.named_modules():
            if name:
                print(name, "->", module.__class__.__name__)
        return

    target_module = resolve_module(loss_textile.model, args.target_layer)
    image = read_and_process_image(str(image_path))
    tiled_image = torch.tile(image, (1, 1, args.number_tiles, args.number_tiles))
    resized_image = loss_textile.t_resized.forward(tiled_image)
    normalized_image = loss_textile.transform(resized_image).float().cuda()

    gradcam = GradCAM(loss_textile.model, target_module)
    try:
        raw_logit, cam = gradcam.compute(normalized_image)
    finally:
        gradcam.remove()

    score = 1 / (1 + torch.exp(-loss_textile.lambda_value * raw_logit))
    preview_rgb = tensor_to_rgb_image(resized_image)
    stem = image_path.stem
    preview_path, heatmap_path, overlay_path = save_cam_outputs(
        output_dir, stem, preview_rgb, cam, args.alpha
    )

    print("image:", image_path)
    print("target layer:", args.target_layer)
    print("resolution:", args.resolution)
    print("number_tiles:", args.number_tiles)
    print("raw logit:", raw_logit.item())
    print("score:", score.item())
    print("preview:", preview_path)
    print("heatmap:", heatmap_path)
    print("overlay:", overlay_path)


if __name__ == "__main__":
    main()
