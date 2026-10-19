"""Initialize a compatible VeriFace ViT checkpoint."""

import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger("veriface.model_init")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

MODEL_NAME = "vit_base_patch16_224"
DEFAULT_HF_MODEL_REPO = "prithivMLmods/Deep-Fake-Detector-v2-Model"
IMAGENET_PREPROCESSING = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def _state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise ValueError("The checkpoint must contain a model state dictionary.")


def _clean_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith(("module.", "model.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def _class_row_indices(config: dict[str, Any]) -> tuple[int, int]:
    def is_real(label: str) -> bool:
        return any(
            term in label for term in ("real", "authentic", "genuine", "original")
        )

    def is_synthetic(label: str) -> bool:
        return any(
            term in label
            for term in ("synthetic", "deepfake", "fake", "generated", "manipulated")
        )

    labels = config.get("id2label", {})
    if isinstance(labels, list):
        labels = dict(enumerate(labels))
    labels = {int(index): str(label).lower() for index, label in labels.items()}
    real_index = next((index for index, label in labels.items() if is_real(label)), None)
    synthetic_index = next(
        (index for index, label in labels.items() if is_synthetic(label)), None
    )
    if real_index is None or synthetic_index is None:
        label_to_index = config.get("label2id", {})
        if isinstance(label_to_index, dict):
            labels = {
                int(index): str(label).lower()
                for label, index in label_to_index.items()
            }
            real_index = next(
                (index for index, label in labels.items() if is_real(label)), None
            )
            synthetic_index = next(
                (index for index, label in labels.items() if is_synthetic(label)),
                None,
            )
    if real_index is None or synthetic_index is None or real_index == synthetic_index:
        source_order = os.getenv("HF_CLASS_ORDER", "").strip().lower().split(",")
        if (
            len(source_order) == 2
            and is_real(source_order[0])
            and is_synthetic(source_order[1])
        ):
            real_index, synthetic_index = 0, 1
        elif (
            len(source_order) == 2
            and is_synthetic(source_order[0])
            and is_real(source_order[1])
        ):
            real_index, synthetic_index = 1, 0
        else:
            raise ValueError(
                "Could not identify the Hugging Face class order from config.json. "
                "Set HF_CLASS_ORDER to the source row order, e.g. 'real,synthetic'."
            )
    return real_index, synthetic_index


def _convert_hf_vit_state_dict(
    source: dict[str, Any],
    config: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, Any]:
    import torch

    source = _clean_keys(source)

    def find(*keys: str):
        for key in keys:
            if key in source:
                return source[key]
        raise ValueError(f"The Hugging Face ViT checkpoint is missing {keys[0]}.")

    converted = {
        "cls_token": find("vit.embeddings.cls_token"),
        "pos_embed": find("vit.embeddings.position_embeddings"),
        "patch_embed.proj.weight": find(
            "vit.embeddings.patch_embeddings.projection.weight"
        ),
        "patch_embed.proj.bias": find(
            "vit.embeddings.patch_embeddings.projection.bias"
        ),
        "norm.weight": find("vit.layernorm.weight", "vit.layer_norm.weight"),
        "norm.bias": find("vit.layernorm.bias", "vit.layer_norm.bias"),
    }

    for index in range(12):
        source_prefix = f"vit.encoder.layer.{index}."
        target_prefix = f"blocks.{index}."
        converted[target_prefix + "norm1.weight"] = find(
            source_prefix + "layernorm_before.weight"
        )
        converted[target_prefix + "norm1.bias"] = find(
            source_prefix + "layernorm_before.bias"
        )
        for parameter in ("weight", "bias"):
            converted[target_prefix + "attn.qkv." + parameter] = torch.cat(
                [
                    find(
                        source_prefix
                        + f"attention.attention.{projection}.{parameter}"
                    )
                    for projection in ("query", "key", "value")
                ],
                dim=0,
            )
            converted[target_prefix + "attn.proj." + parameter] = find(
                source_prefix + f"attention.output.dense.{parameter}"
            )
            converted[target_prefix + "norm2." + parameter] = find(
                source_prefix + f"layernorm_after.{parameter}"
            )
            converted[target_prefix + "mlp.fc1." + parameter] = find(
                source_prefix + f"intermediate.dense.{parameter}"
            )
            converted[target_prefix + "mlp.fc2." + parameter] = find(
                source_prefix + f"output.dense.{parameter}"
            )

    classifier_weight = find("classifier.weight", "head.weight")
    classifier_bias = find("classifier.bias", "head.bias")
    real_index, synthetic_index = _class_row_indices(config)
    order = torch.tensor([real_index, synthetic_index], dtype=torch.long)
    converted["head.weight"] = classifier_weight.index_select(0, order)
    converted["head.bias"] = classifier_bias.index_select(0, order)

    missing = set(target) - set(converted)
    unexpected = set(converted) - set(target)
    if missing or unexpected:
        raise ValueError(
            "The Hugging Face model does not match vit_base_patch16_224 "
            f"(missing keys: {sorted(missing)}, unexpected keys: {sorted(unexpected)})."
        )
    return converted


def _download_hf_checkpoint(repo_id: str, destination: Path) -> dict[str, Any]:
    import torch

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        try:
            processor_path = hf_hub_download(
                repo_id=repo_id, filename="preprocessor_config.json"
            )
        except EntryNotFoundError:
            preprocessing = IMAGENET_PREPROCESSING
        else:
            with open(processor_path, encoding="utf-8") as processor_file:
                processor = json.load(processor_file)
            preprocessing = {
                "mean": processor.get("image_mean", IMAGENET_PREPROCESSING["mean"]),
                "std": processor.get("image_std", IMAGENET_PREPROCESSING["std"]),
            }
        preprocessing["layer_norm_eps"] = config.get("layer_norm_eps", 1e-6)
        if config.get("model_type") != "vit":
            raise ValueError(f"{repo_id} is not a Hugging Face ViT model.")
        if (
            config.get("hidden_size") != 768
            or config.get("num_hidden_layers") != 12
            or config.get("patch_size") != 16
            or config.get("image_size") != 224
        ):
            raise ValueError(f"{repo_id} is not a ViT-Base patch16 224 checkpoint.")
        try:
            weights_path = hf_hub_download(
                repo_id=repo_id, filename="model.safetensors"
            )
            from safetensors.torch import load_file

            source_state = load_file(weights_path, device="cpu")
        except EntryNotFoundError:
            weights_path = hf_hub_download(
                repo_id=repo_id, filename="pytorch_model.bin"
            )
            import torch

            source_state = torch.load(
                weights_path, map_location="cpu", weights_only=True
            )

        import timm

        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
        converted = _convert_hf_vit_state_dict(
            _state_dict(source_state), config, model.state_dict()
        )
        model.load_state_dict(converted, strict=True)
        torch_state = model.state_dict()
        torch.save(
            {"state_dict": torch_state, "preprocessing": preprocessing},
            destination,
        )
        logger.info(
            "Downloaded and converted Hugging Face checkpoint %s to %s",
            repo_id,
            destination,
        )
        return torch_state
    except Exception:
        logger.exception("Could not initialize weights from Hugging Face repo %s", repo_id)
        raise


def _download_url(url: str, destination: Path) -> None:
    import torch
    import timm

    suffix = (
        ".safetensors"
        if url.lower().split("?", 1)[0].endswith(".safetensors")
        else ".pth"
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        urllib.request.urlretrieve(url, temporary_path)
        if suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(temporary_path), device="cpu")
        else:
            checkpoint = torch.load(
                temporary_path, map_location="cpu", weights_only=True
            )
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
        state = _clean_keys(_state_dict(checkpoint))
        model.load_state_dict(state, strict=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "preprocessing": {
                    **IMAGENET_PREPROCESSING,
                    "layer_norm_eps": 1e-6,
                },
            },
            destination,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_checkpoint(path: Path) -> None:
    import torch
    import timm

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(_clean_keys(_state_dict(checkpoint)), strict=True)


def initialize_checkpoint() -> Path | None:
    if os.getenv("INSTALL_VIT_MODEL", "true").strip().lower() not in {
        "true",
        "1",
        "yes",
    }:
        logger.info("INSTALL_VIT_MODEL is disabled; skipping model initialization.")
        return None

    checkpoint = Path(
        os.getenv(
            "MODEL_CHECKPOINT",
            str(Path(__file__).resolve().parents[1] / "models" / "vit_deepfake.pth"),
        )
    ).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (Path.cwd() / checkpoint).resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint.is_file():
        _validate_checkpoint(checkpoint)
        logger.info("Validated existing ViT checkpoint at %s", checkpoint)
        return checkpoint

    model_url = os.getenv("MODEL_URL", "").strip()
    hf_model_repo = os.getenv("HF_MODEL_REPO", DEFAULT_HF_MODEL_REPO).strip()
    if model_url:
        _download_url(model_url, checkpoint)
    elif hf_model_repo:
        _download_hf_checkpoint(hf_model_repo, checkpoint)
    else:
        import torch
        import timm

        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "preprocessing": {
                    **IMAGENET_PREPROCESSING,
                    "layer_norm_eps": 1e-6,
                },
            },
            checkpoint,
        )
        logger.warning(
            "Created an ImageNet-initialized binary ViT checkpoint. It is suitable "
            "for inference smoke tests, not deepfake detection; use fine-tuned weights."
        )

    _validate_checkpoint(checkpoint)
    return checkpoint


if __name__ == "__main__":
    initialize_checkpoint()
