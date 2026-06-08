"""Scheme 4 TCMonoDepth-prior dense segmentor."""

from __future__ import annotations

from pathlib import Path

from models.segmentation.scheme3_dense.adapter import Scheme3DenseSegmentor


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = ROOT / "external/model_weights/scheme4.pt"


class Scheme4DenseSegmentor(Scheme3DenseSegmentor):
    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT, **kwargs) -> None:
        super().__init__(checkpoint, **kwargs)
        if self.config.image_feature_mode != "tc_monodepth":
            raise ValueError(f"Scheme 4 expects image_feature_mode='tc_monodepth', got {self.config.image_feature_mode!r}")
