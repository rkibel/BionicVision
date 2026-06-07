"""Model factory for Scheme 3 hand segmentation checkpoints."""

from __future__ import annotations

import torch
from torch import nn


def build_model(model_name: str, device: str | torch.device, *, encoder_weights: str | None) -> nn.Module:
    import segmentation_models_pytorch as smp

    if model_name == "smp-unetpp-efficientnet-b4":
        model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=encoder_weights, in_channels=3, classes=1)
    elif model_name == "smp-unetpp-resnet101":
        model = smp.UnetPlusPlus(encoder_name="resnet101", encoder_weights=encoder_weights, in_channels=3, classes=1)
    elif model_name == "smp-deeplabv3plus-resnet101":
        model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=encoder_weights, in_channels=3, classes=1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)
