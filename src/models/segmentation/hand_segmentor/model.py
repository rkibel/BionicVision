"""Hand-segmentor model construction and tensor helpers."""

from __future__ import annotations

import torch
from torch import nn


MODEL_BUILDERS = {
    "smp-unetpp-efficientnet-b4": ("UnetPlusPlus", "efficientnet-b4"),
    "smp-unetpp-resnet101": ("UnetPlusPlus", "resnet101"),
    "smp-deeplabv3plus-resnet101": ("DeepLabV3Plus", "resnet101"),
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(
    model_name: str,
    device: str | torch.device,
    *,
    encoder_weights: str | None,
) -> nn.Module:
    """Build a supported binary hand-segmentation model."""

    import segmentation_models_pytorch as smp

    try:
        constructor_name, encoder_name = MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported hand-segmentor model: {model_name}") from exc
    constructor = getattr(smp, constructor_name)
    return constructor(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=1).to(device)


def model_logits(model: nn.Module, images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    logits = model(images)
    if isinstance(logits, dict):
        logits = logits["out"]
    if logits.shape[-2:] != size:
        logits = torch.nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """Normalize RGB tensors in `[0, 1]` using ImageNet statistics."""

    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std
