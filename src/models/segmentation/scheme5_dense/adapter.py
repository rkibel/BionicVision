"""Scheme 5 GLC-saliency-prior dense segmentor."""

from __future__ import annotations

from pathlib import Path

from models.segmentation.scheme3_dense.adapter import Scheme3DenseSegmentor


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = ROOT / "external/model_weights/scheme5.pt"


class Scheme5DenseSegmentor(Scheme3DenseSegmentor):
    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, **kwargs) -> None:
        super().__init__(checkpoint, **kwargs)
        if self.config.image_feature_mode != "glc_gaze":
            raise ValueError(f"Scheme 5 expects image_feature_mode='glc_gaze', got {self.config.image_feature_mode!r}")
