from __future__ import annotations

import os
import random
import threading
from dataclasses import dataclass

import torch
from PIL import Image
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

try:
    from diffusers import FluxKontextPipeline
except ImportError as exc:  # pragma: no cover - import-time guard
    FluxKontextPipeline = None  # type: ignore[assignment]
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None


MODEL_ID = os.environ.get("FLUX_KONTEXT_MODEL_ID", "black-forest-labs/FLUX.1-Kontext-dev")

_PIPELINE: FluxKontextPipeline | None = None
_PIPELINE_LOCK = threading.Lock()


@dataclass(frozen=True)
class KontextConfig:
    model_id: str
    steps: int
    guidance_scale: float
    max_side: int
    device: str
    dtype: str
    use_cpu_offload: bool


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported FLUX_KONTEXT_DTYPE: {name}") from exc


def get_kontext_config() -> KontextConfig:
    cuda_available = torch.cuda.is_available()
    device = os.environ.get("FLUX_KONTEXT_DEVICE", "cuda" if cuda_available else "cpu")
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported FLUX_KONTEXT_DEVICE: {device}")

    dtype = _resolve_dtype(os.environ.get("FLUX_KONTEXT_DTYPE", "float16"))
    if device == "cpu":
        dtype = torch.float32
    steps = int(os.environ.get("FLUX_KONTEXT_STEPS", "4"))
    guidance_scale = float(os.environ.get("FLUX_KONTEXT_GUIDANCE", "2.5"))
    max_side = int(os.environ.get("FLUX_KONTEXT_MAX_SIDE", "768"))
    use_cpu_offload = _parse_bool(os.environ.get("FLUX_KONTEXT_CPU_OFFLOAD"), default=False)
    if device != "cuda":
        use_cpu_offload = False
    return KontextConfig(
        model_id=MODEL_ID,
        steps=steps,
        guidance_scale=guidance_scale,
        max_side=max_side,
        device=device,
        dtype={torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype],
        use_cpu_offload=use_cpu_offload,
    )


def _ensure_pipeline(config: KontextConfig) -> FluxKontextPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    if FluxKontextPipeline is None:
        raise RuntimeError(
            "diffusers is not installed. Install the style-service requirements first."
        ) from _DIFFUSERS_IMPORT_ERROR

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        dtype = _resolve_dtype(config.dtype)
        try:
            # FLUX.1-Kontext-dev의 체크포인트는 크기가 커서, 로딩 중 state dict를
            # 메모리에 두 벌 유지하면 Qwen과 함께 실행하는 3090 서버에서 피크가 난다.
            # accelerate의 저메모리 로딩과 GPU/CPU 분산으로 피크 사용량을 낮춘다.
            load_kwargs: dict[str, object] = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "offload_state_dict": False,
            }
            if config.use_cpu_offload:
                load_kwargs["device_map"] = "balanced"
                load_kwargs["max_memory"] = {0: "20GiB", "cpu": "48GiB"}
            pipeline = FluxKontextPipeline.from_pretrained(config.model_id, **load_kwargs)
        except (GatedRepoError, HfHubHTTPError) as exc:
            raise RuntimeError(
                "FLUX.1-Kontext-dev is gated on Hugging Face. "
                "Accept the model terms and sign in with `huggingface-cli login` "
                "or set `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`."
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "A required runtime dependency is missing while loading FLUX.1-Kontext-dev. "
                "Please reinstall style-service requirements (protobuf, sentencepiece, transformers, diffusers)."
            ) from exc
        pipeline.set_progress_bar_config(disable=True)
        if config.use_cpu_offload:
            # device_map이 적용된 경우 accelerate dispatch hook이 이미 배치와 이동을
            # 담당하므로 enable_model_cpu_offload()를 중복 호출하지 않는다.
            if not getattr(pipeline, "hf_device_map", None):
                pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(config.device)
        _PIPELINE = pipeline
        return pipeline


def _build_prompt(prompt: str, strength: float) -> str:
    text = prompt.strip()
    if not text:
        raise ValueError("Prompt must not be empty.")

    if strength <= 0.33:
        prefix = "Make a subtle edit. Keep the original structure and layout mostly intact. "
    elif strength <= 0.66:
        prefix = "Make a balanced edit. Preserve the main structure while clearly applying the requested style. "
    else:
        prefix = "Make a strong edit. Transform the image noticeably while keeping it coherent. "
    return prefix + text


def _resize_to_max_side(image: Image.Image, max_side: int) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image, image.size

    scale = max_side / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.LANCZOS), image.size


def _restore_alpha(original: Image.Image, edited: Image.Image) -> Image.Image:
    has_alpha = original.mode in ("RGBA", "LA", "PA") or (
        original.mode == "P" and "transparency" in original.info
    )
    if not has_alpha:
        return edited

    original_rgba = original.convert("RGBA")
    original_alpha = original_rgba.getchannel("A")
    restored = edited.convert("RGBA")
    if restored.size != original_alpha.size:
        original_alpha = original_alpha.resize(restored.size, Image.LANCZOS)
    restored.putalpha(original_alpha)
    return restored


def edit_image(
    content: Image.Image,
    prompt: str,
    *,
    strength: float = 1.0,
    config: KontextConfig | None = None,
) -> Image.Image:
    config = config or get_kontext_config()
    pipeline = _ensure_pipeline(config)

    source = content.convert("RGB")
    source, original_size = _resize_to_max_side(source, config.max_side)
    built_prompt = _build_prompt(prompt, strength)

    call_kwargs = {
        "image": source,
        "prompt": built_prompt,
        "width": source.size[0],
        "height": source.size[1],
        "num_inference_steps": config.steps,
        "guidance_scale": config.guidance_scale,
        "generator": torch.Generator().manual_seed(random.randint(0, 2**31 - 1)),
    }

    with torch.inference_mode():
        result = pipeline(**call_kwargs)

    image = result.images[0]
    if image.size != original_size:
        image = image.resize(original_size, Image.LANCZOS)
    return _restore_alpha(content, image)


def get_runtime_status() -> dict[str, object]:
    config = get_kontext_config()
    cuda_available = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "model_id": config.model_id,
        "device": config.device,
        "dtype": config.dtype,
        "steps": config.steps,
        "guidance_scale": config.guidance_scale,
        "max_side": config.max_side,
        "use_cpu_offload": config.use_cpu_offload,
        "cuda_available": cuda_available,
        "cuda_name": cuda_name,
        "loaded": _PIPELINE is not None,
    }
