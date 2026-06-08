"""Evaluate the Scheme 5 GLC-saliency-prior dense checkpoint."""

from __future__ import annotations

from pathlib import Path

from evaluation.dense_schemes.run_evaluation import main
from models.segmentation.scheme5_dense.adapter import DEFAULT_CHECKPOINT


ROOT = Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    main(
        {
            "--checkpoint": str(DEFAULT_CHECKPOINT),
            "--output": str(ROOT / "outputs/evaluation/scheme5_dense/results.json"),
        }
    )
