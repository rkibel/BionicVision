"""Scheme 3 dense interacting-object segmentation conditioned on hands."""

from .adapter import DEFAULT_CHECKPOINT, Scheme3DenseSegmentor
from .model import DenseModelConfig, HandPrior, build_model

__all__ = ["DEFAULT_CHECKPOINT", "DenseModelConfig", "HandPrior", "Scheme3DenseSegmentor", "build_model"]
