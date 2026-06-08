"""Train Scheme 5: dense objects with hand and GLC saliency priors."""

from __future__ import annotations

from pathlib import Path

from models.segmentation.scheme3_dense.adapter import DEFAULT_CHECKPOINT as SCHEME3_DEFAULT_CHECKPOINT
from models.segmentation.scheme3_dense.train import main as train_dense


ROOT = Path(__file__).resolve().parents[4]


if __name__ == "__main__":
    train_dense(
        {
            "--image-feature-mode": "glc_gaze",
            "--init-checkpoint": str(SCHEME3_DEFAULT_CHECKPOINT),
            "--output": str(ROOT / "outputs/models/scheme5_dense/best.pt"),
            "--lr": "1e-6",
            "--flow-weight": "0.25",
            "--epochs": "20",
        }
    )
