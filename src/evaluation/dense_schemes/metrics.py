"""Dense binary-mask metrics for schemes 3, 4, and 5."""

from __future__ import annotations

import numpy as np


def evaluate_supervised_masks(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, float | int]:
    if len(predictions) != len(targets):
        raise ValueError(f"Prediction/target count mismatch: {len(predictions)} != {len(targets)}")
    rows = [mask_metrics(prediction, target) for prediction, target in zip(predictions, targets) if np.any(target)]
    return {
        "frames": len(targets),
        "positive_frames": len(rows),
        "mean_iou": mean([row["iou"] for row in rows]),
        "mean_precision": mean([row["precision"] for row in rows]),
        "mean_recall": mean([row["recall"] for row in rows]),
        "mean_dice": mean([row["dice"] for row in rows]),
        "mean_prediction_area": mean([float(np.mean(row)) for row in predictions]),
        "mean_target_area": mean([float(np.mean(row)) for row in targets]),
    }


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction, target = prediction.astype(bool), target.astype(bool)
    true_positive = int(np.count_nonzero(prediction & target))
    false_positive = int(np.count_nonzero(prediction & ~target))
    false_negative = int(np.count_nonzero(~prediction & target))
    union = true_positive + false_positive + false_negative
    return {
        "iou": true_positive / union if union else 0.0,
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "dice": 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1),
    }


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
