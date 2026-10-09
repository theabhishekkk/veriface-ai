"""ViT inference and attention-map generation for VeriFace AI."""

import base64
import io
import logging
import os
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger("veriface.model")

USE_MOCK_MODEL = os.getenv("USE_MOCK_MODEL", "True").strip().lower() in {
    "true",
    "1",
    "yes",
}
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "").strip()

_vit_model = None
_vit_transform = None
_device = "cpu"


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vit_model():
    """Load a binary fine-tuned ViT checkpoint and its image transform."""
    global _vit_model, _vit_transform, _device

    if USE_MOCK_MODEL:
        return None
    if not MODEL_CHECKPOINT:
        raise RuntimeError(
            "USE_MOCK_MODEL is disabled, but MODEL_CHECKPOINT is not configured. "
            "Provide a fine-tuned two-class ViT checkpoint."
        )
    if _vit_model is not None:
        return _vit_model

    import torch
    import timm
    from torchvision import transforms

    if not os.path.isfile(MODEL_CHECKPOINT):
        raise FileNotFoundError(f"ViT checkpoint does not exist: {MODEL_CHECKPOINT}")

    _device = _get_device()
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=2,
    )
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("The ViT checkpoint must contain a model state dictionary.")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(_device)

    # Disable fused attention in the final block so timm exposes attention weights.
    if hasattr(model.blocks[-1].attn, "fused_attn"):
        model.blocks[-1].attn.fused_attn = False

    _vit_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    _vit_model = model
    logger.info("Loaded fine-tuned ViT checkpoint on %s", _device)
    return _vit_model


def _normalize_attention(attention: np.ndarray) -> np.ndarray:
    low = float(attention.min())
    high = float(attention.max())
    if high - low < 1e-8:
        return np.zeros_like(attention, dtype=np.float32)
    return ((attention - low) / (high - low)).astype(np.float32)


def _mock_attention_map(image: np.ndarray) -> np.ndarray:
    """Create a deterministic, visibly non-uniform simulated 14x14 map."""
    seed = int(np.sum(image[:: max(1, image.shape[0] // 20), :: max(1, image.shape[1] // 20)])) % 2**32
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:14, 0:14]
    center = np.exp(-((x - 6.5) ** 2 + (y - 6.5) ** 2) / 25)
    eyes = np.exp(-((x - 6.5) ** 2 + (y - 4.5) ** 2) / 8)
    mouth = np.exp(-((x - 6.5) ** 2 + (y - 9.5) ** 2) / 10)
    return _normalize_attention((0.45 * center + 0.35 * eyes + 0.2 * mouth) * rng.uniform(0.9, 1.1, (14, 14)))


def _encode_heatmaps(image_bgr: np.ndarray, attention: np.ndarray) -> tuple[str, str]:
    height, width = image_bgr.shape[:2]
    resized = cv2.resize(attention, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_TURBO)
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
            "highest_activation_zone": (
                "Highest ViT attention patch" if mode == "production_vit" else "Simulated facial-feature focus"
            ),
        },
    }


def _run_mock(image: Image.Image) -> dict[str, Any]:
    started = time.perf_counter()
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    sample = image_bgr[:: max(1, image_bgr.shape[0] // 64), :: max(1, image_bgr.shape[1] // 64), 0]
    image_score = int(np.sum(sample, dtype=np.uint64) % 100)
    is_synthetic = image_score >= 50
    confidence = 0.75 + (image_score / 100) * 0.2
    return _make_result(
        image,
        _mock_attention_map(image_bgr),
        is_synthetic,
        confidence,
        (time.perf_counter() - started) * 1000,
        "mock",
        "vit_base_patch16_224 (simulated)",
    )


def _run_vit(image: Image.Image) -> dict[str, Any]:
    import torch

    model = load_vit_model()
    input_tensor = _vit_transform(image).unsqueeze(0).to(_device)
    attention_weights = []

    def capture_attention(_module, inputs, _output):
        attention_weights.append(inputs[0].detach())

    hook = model.blocks[-1].attn.attn_drop.register_forward_hook(capture_attention)
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            logits = model(input_tensor)
    finally:
        hook.remove()
    elapsed_ms = (time.perf_counter() - started) * 1000

    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise RuntimeError("The configured ViT checkpoint must produce two class logits.")
    probabilities = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
    synthetic_index = 1
    is_synthetic = bool(np.argmax(probabilities) == synthetic_index)
    confidence = float(probabilities[synthetic_index if is_synthetic else 1 - synthetic_index])
    if not attention_weights:
        raise RuntimeError("The ViT model did not expose attention weights.")
    cls_attention = attention_weights[0][0, :, 0, 1:].mean(dim=0).float().cpu().numpy()
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

    return _run_mock(image) if USE_MOCK_MODEL else _run_vit(image)
