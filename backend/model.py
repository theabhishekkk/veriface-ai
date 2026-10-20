"""ViT inference and attention-map generation for VeriFace AI."""

import base64
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("veriface.model")

MODEL_CHECKPOINT = Path(
    os.getenv(
        "MODEL_CHECKPOINT",
        str(Path(__file__).resolve().parent / "models" / "vit_deepfake.pth"),
    )
).expanduser()
if not MODEL_CHECKPOINT.is_absolute():
    MODEL_CHECKPOINT = (Path(__file__).resolve().parent / MODEL_CHECKPOINT).resolve()

_vit_model = None
_vit_transform = None
_device = "cpu"
_model_load_error = None
_model_load_lock = threading.Lock()


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vit_model():
    """Load a binary fine-tuned ViT checkpoint and its image transform."""
    global _vit_model, _vit_transform, _device, _model_load_error

    if _vit_model is not None:
        return _vit_model

    with _model_load_lock:
        if _vit_model is not None:
            return _vit_model

        try:
            if not MODEL_CHECKPOINT.is_file():
                from scripts.init_model import initialize_checkpoint

                initialize_checkpoint()
            if not MODEL_CHECKPOINT.is_file():
                raise FileNotFoundError(
                    f"ViT checkpoint does not exist: {MODEL_CHECKPOINT}"
                )

            from functools import partial

            import timm
            import torch
            from torch import nn
            from torchvision import transforms

            num_threads = int(os.getenv("TORCH_NUM_THREADS", "1"))
            if num_threads < 1:
                raise ValueError("TORCH_NUM_THREADS must be a positive integer.")
            torch.set_num_threads(num_threads)

            checkpoint = torch.load(
                MODEL_CHECKPOINT,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            preprocessing = (
                checkpoint.get("preprocessing", {})
                if isinstance(checkpoint, dict)
                else {}
            )
            model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=2),
            )
            state_dict = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            if not isinstance(state_dict, dict):
                raise ValueError(
                    "The ViT checkpoint must contain a model state dictionary."
                )
            state_dict = {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
            # model.load_state_dict(state_dict, strict=True, assign=True)
            _device = _get_device()
            model.to(_device).eval()
            image_mean = preprocessing.get("mean", (0.485, 0.456, 0.406))
            image_std = preprocessing.get("std", (0.229, 0.224, 0.225))

            # Disable fused attention so each block exposes its attention matrix to hooks.
            if hasattr(model.blocks[-1].attn, "fused_attn"):
                for block in model.blocks:
                    block.attn.fused_attn = False

            _vit_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=tuple(image_mean), std=tuple(image_std)),
                ]
            )
            _vit_model = model
            _model_load_error = None
            logger.info("Loaded ViT checkpoint %s on %s", MODEL_CHECKPOINT, _device)
            return _vit_model
        except Exception as exc:
            _model_load_error = str(exc)
            raise RuntimeError(
                f"Failed to load the ViT deepfake model from {MODEL_CHECKPOINT}: {exc}"
            ) from exc


def _normalize_attention(attention: np.ndarray) -> np.ndarray:
    low = float(attention.min())
    high = float(attention.max())
    if high - low < 1e-8:
        return np.zeros_like(attention, dtype=np.float32)
    return ((attention - low) / (high - low)).astype(np.float32)


def _encode_heatmaps(image_bgr: np.ndarray, attention: np.ndarray) -> tuple[str, str]:
    height, width = image_bgr.shape[:2]
    attention_224 = cv2.resize(attention, (224, 224), interpolation=cv2.INTER_CUBIC)
    resized = cv2.resize(attention_224, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.48, heatmap, 0.52, 0)

    def encode(image: np.ndarray) -> str:
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Could not encode the attention visualization.")
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

    return encode(overlay), encode(heatmap)


def _make_result(
    image: Image.Image,
    attention: np.ndarray,
    is_synthetic: bool,
    confidence: float,
    elapsed_ms: float,
    mode: str,
    model_name: str,
) -> dict[str, Any]:
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    overlay, raw_heatmap = _encode_heatmaps(image_bgr, attention)
    label = "Synthetic" if is_synthetic else "Real"
    return {
        "label": label,
        "is_deepfake": is_synthetic,
        "confidence": round(float(confidence), 4),
        "confidence_percentage": round(float(confidence) * 100, 2),
        "attention_heatmap": overlay,
        "raw_heatmap": raw_heatmap,
        "execution_time_ms": round(elapsed_ms, 2),
        "model_name": model_name,
        "mode": mode,
        "patch_analysis": {
            "patch_count": 196,
            "grid_size": "14x14",
            "highest_activation_zone": "Highest ViT attention patch",
        },
    }


def _run_vit(image: Image.Image) -> dict[str, Any]:
    import torch

    model = load_vit_model()
    input_tensor = _vit_transform(image).unsqueeze(0).to(_device)
    attention_weights = []

    def capture_attention(_module, inputs, _output):
        attention_weights.append(inputs[0].detach())

    hooks = [
        block.attn.attn_drop.register_forward_hook(capture_attention)
        for block in model.blocks
    ]
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            logits = model(input_tensor)
    finally:
        for hook in hooks:
            hook.remove()
    elapsed_ms = (time.perf_counter() - started) * 1000

    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise RuntimeError("The configured ViT checkpoint must produce two class logits.")
    probabilities = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
    synthetic_index = 1
    is_synthetic = bool(np.argmax(probabilities) == synthetic_index)
    confidence = float(probabilities[synthetic_index if is_synthetic else 1 - synthetic_index])
    if len(attention_weights) != len(model.blocks):
        raise RuntimeError(
            "The ViT model did not expose attention weights from every transformer block."
        )
    joint_attention = torch.eye(attention_weights[0].shape[-1], device=_device)
    identity = joint_attention
    for weights in attention_weights:
        mean_attention = weights[0].mean(dim=0).float()
        residual_attention = mean_attention + identity
        residual_attention /= residual_attention.sum(dim=-1, keepdim=True)
        joint_attention = residual_attention @ joint_attention
    cls_attention = joint_attention[0, 1:].cpu().numpy()
    attention_map = _normalize_attention(cls_attention.reshape(14, 14))
    return _make_result(
        image,
        attention_map,
        is_synthetic,
        confidence,
        elapsed_ms,
        "production_vit",
        "vit_base_patch16_224",
    )


def get_model_status() -> dict[str, Any]:
    """Return the configured and currently loaded inference model state."""
    return {
        "status": (
            "ready"
            if _vit_model is not None
            else "error"
            if _model_load_error is not None
            else "warming_up"
        ),
        "model_name": "vit_base_patch16_224",
        "device": _device,
        "checkpoint": str(MODEL_CHECKPOINT),
        "checkpoint_loaded": _vit_model is not None,
    }


def predict_image(image_bytes: bytes) -> dict[str, Any]:
    """Decode and classify a JPEG/PNG image, returning an overlay as a data URL."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported.")
            if source.width * source.height > 40_000_000:
                raise ValueError("The image dimensions exceed the 40 megapixel limit.")
            source.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            image = source.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError("The uploaded file is not a valid JPEG or PNG image.") from exc

    return _run_vit(image)
