"""Shared hand-segmentation evaluation metrics and runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from models.segmentation.hand_segmentor.adapter import HandSegmentor


def evaluate_items(
    segmentor: HandSegmentor,
    items: list[Any],
    load_sample: Callable[[Any], tuple[np.ndarray, np.ndarray]],
    thresholds: list[float],
    *,
    batch_size: int,
    write_dir: Path | None = None,
    group_key: str | None = None,
) -> dict[str, Any]:
    """Evaluate a hand segmentor over dataset items."""

    records_by_threshold = {threshold: [] for threshold in thresholds}
    for start in range(0, len(items), batch_size):
        batch_items = items[start : start + batch_size]
        samples = [load_sample(item) for item in batch_items]
        probs = segmentor.predict_batch([image for image, _ in samples])
        for item, (_, ground_truth), prob in zip(batch_items, samples, probs):
            for threshold in thresholds:
                prediction = prob >= threshold
                record = frame_record(item, prediction, ground_truth)
                records_by_threshold[threshold].append(record)
                if write_dir is not None:
                    threshold_dir = write_dir / f"threshold_{threshold:.2f}"
                    threshold_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(threshold_dir / f"{item.identifier}.png"), prediction.astype(np.uint8) * 255)

    rows = []
    for threshold, records in records_by_threshold.items():
        row = {
            "threshold": threshold,
            **summarize_records(records),
            "by_split": summarize_by(records, "split"),
        }
        if group_key is not None:
            row[f"by_{group_key}"] = summarize_by(records, group_key)
        rows.append(row)
    rows.sort(key=lambda row: (row["mean_iou"], row["mean_precision"]), reverse=True)
    return {"best": rows[0], "thresholds": rows}


def frame_record(item: Any, prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, Any]:
    record = {
        "identifier": item.identifier,
        "split": item.split,
        "metrics": mask_metrics(prediction, ground_truth),
        "prediction_pixels": int(np.count_nonzero(prediction)),
        "ground_truth_pixels": int(np.count_nonzero(ground_truth)),
        "detections": int(np.count_nonzero(prediction) > 0),
    }
    if hasattr(item, "video_id"):
        record["video_id"] = item.video_id
    return record


def mask_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float | None]:
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    intersection = int(np.count_nonzero(prediction & ground_truth))
    pred_count = int(np.count_nonzero(prediction))
    gt_count = int(np.count_nonzero(ground_truth))
    union = int(np.count_nonzero(prediction | ground_truth))
    return {
        "iou": intersection / union if union else None,
        "precision": intersection / pred_count if pred_count else (None if gt_count == 0 else 0.0),
        "recall": intersection / gt_count if gt_count else (None if pred_count == 0 else 0.0),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    metrics = [record["metrics"] for record in records]
    detections = [record["detections"] for record in records]
    return {
        "frames": len(records),
        "mean_iou": mean_metric(metrics, "iou"),
        "mean_precision": mean_metric(metrics, "precision"),
        "mean_recall": mean_metric(metrics, "recall"),
        "mean_detections": float(np.mean(detections)) if detections else 0.0,
        "empty_frames": sum(count == 0 for count in detections),
    }


def summarize_by(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    values = sorted({str(record[key]) for record in records})
    return {value: summarize_records([record for record in records if str(record[key]) == value]) for value in values}


def mean_metric(metrics: Iterable[dict[str, float | None]], key: str) -> float:
    values = [float(metric[key]) for metric in metrics if metric[key] is not None]
    return float(np.mean(values)) if values else 0.0


def parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("Expected at least one comma-separated value")
    return parsed


def parse_thresholds(value: str | None, default: float) -> list[float]:
    if value is None:
        return [default]
    thresholds = [float(part) for part in parse_csv(value)]
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("Thresholds must be between 0 and 1")
    return thresholds


def print_summary(summary: dict[str, Any]) -> None:
    best = summary["best"]
    print(
        f"best threshold={best['threshold']:.2f} "
        f"IoU={best['mean_iou']:.4f} "
        f"P={best['mean_precision']:.4f} "
        f"R={best['mean_recall']:.4f} "
        f"frames={best['frames']}"
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
