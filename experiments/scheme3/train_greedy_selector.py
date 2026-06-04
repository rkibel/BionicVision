#!/usr/bin/env python3
"""Train a selector from greedy oracle union-IoU labels over track caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from common import OUTPUT_DIR, write_json


POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-track-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--val-track-cache", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "track_score_model/greedy_selector.joblib")
    parser.add_argument("--summary-output", type=Path, default=OUTPUT_DIR / "track_score_model/summary_greedy_selector.json")
    parser.add_argument("--max-oracle-select", type=int, default=12)
    parser.add_argument("--threshold-grid", default="0.02:0.98:0.02")
    parser.add_argument("--temporal-weight", type=float, default=0.10)
    parser.add_argument("--count-penalty", type=float, default=0.015)
    args = parser.parse_args()

    train_caches = [TrackCache.load(path) for path in args.train_track_cache]
    val_caches = [TrackCache.load(path) for path in args.val_track_cache]
    for cache in train_caches + val_caches:
        cache.greedy_labels = greedy_oracle_labels(cache, args.max_oracle_select)

    train_features = np.concatenate([cache.augmented_features[cache.supervised_rows] for cache in train_caches])
    train_labels = np.concatenate([cache.greedy_labels[cache.supervised_rows] for cache in train_caches])
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=0.03,
        class_weight="balanced",
        random_state=17,
    )
    model.fit(train_features, train_labels)

    thresholds = parse_grid(args.threshold_grid)
    train_scores = [score_cache(model, cache) for cache in train_caches]
    best_train = best_threshold(train_caches, train_scores, thresholds, args.temporal_weight, args.count_penalty)
    summary = {
        "feature_names": list(train_caches[0].feature_names) + ["heuristic_object_score"],
        "train_track_caches": [str(path) for path in args.train_track_cache],
        "val_track_caches": [str(path) for path in args.val_track_cache],
        "max_oracle_select": args.max_oracle_select,
        "objective": "selected_union_iou + temporal_weight * selected_temporal_union_iou - count_penalty * selected_count",
        "temporal_weight": args.temporal_weight,
        "count_penalty": args.count_penalty,
        "threshold_grid": args.threshold_grid,
        "train": classifier_summary(train_labels, np.concatenate([scores[cache.supervised_rows] for cache, scores in zip(train_caches, train_scores)])),
        "best_train_threshold": best_train,
    }
    if val_caches:
        val_scores = [score_cache(model, cache) for cache in val_caches]
        summary["val_at_train_threshold"] = aggregate_metrics(
            [evaluate_cache(cache, scores, best_train["threshold"]) for cache, scores in zip(val_caches, val_scores)],
            args.temporal_weight,
            args.count_penalty,
        )
        summary["val_oracle_threshold"] = best_threshold(val_caches, val_scores, thresholds, args.temporal_weight, args.count_penalty)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_kind": "augmented_sklearn",
            "model": model,
            "feature_names": summary["feature_names"],
            "threshold": float(best_train["threshold"]),
            "summary": summary,
        },
        args.output,
    )
    write_json(args.summary_output, {"model": str(args.output), **summary})
    print(json.dumps({"model": str(args.output), **summary}, indent=2))


class TrackCache:
    def __init__(self, path: Path, payload) -> None:
        self.path = path
        self.features = payload["features"].astype(np.float32)
        self.object_scores = payload["object_scores"].astype(np.float32)
        self.frame_indices = payload["frame_indices"].astype(np.int32)
        self.track_ids = payload["track_ids"].astype(np.int32)
        self.pred_masks = payload["pred_masks"].astype(np.uint8)
        self.gt_unions = payload["gt_unions"].astype(np.uint8)
        gt_frame_indices = payload["gt_frame_indices"].astype(np.int32)
        self.feature_names = [str(name) for name in payload["feature_names"]]
        self.height = int(payload["height"][0])
        self.width = int(payload["width"][0])
        self.add_geometry_features_if_needed()
        self.add_color_features_if_needed(payload["original_indices"].astype(np.int32))
        self.augmented_features = np.concatenate([self.features, self.object_scores[:, None]], axis=1)
        self.gt_by_frame = {}
        for row, frame in enumerate(gt_frame_indices):
            gt_union = self.gt_unions[row]
            if packed_count(gt_union) > 0:
                self.gt_by_frame[int(frame)] = gt_union
        self.frames = sorted(self.gt_by_frame)
        self.rows_by_frame = {frame: np.flatnonzero(self.frame_indices == frame) for frame in self.frames}
        self.supervised_rows = np.concatenate([self.rows_by_frame[frame] for frame in self.frames]) if self.frames else np.asarray([], dtype=np.int64)

    @classmethod
    def load(cls, path: Path) -> "TrackCache":
        return cls(path, np.load(path, allow_pickle=True))

    def add_geometry_features_if_needed(self) -> None:
        extra_names = [
            "bbox_cx",
            "bbox_cy",
            "bbox_w",
            "bbox_h",
            "bbox_aspect",
            "border_touch",
            "track_progress",
            "track_span",
            "visible_fraction",
            "prev_iou",
            "next_iou",
            "centroid_motion",
        ]
        if all(name in self.feature_names for name in extra_names):
            return

        rows = self.pred_masks.shape[0]
        boxes = np.zeros((rows, 4), dtype=np.float32)
        centers = np.zeros((rows, 2), dtype=np.float32)
        border = np.zeros(rows, dtype=np.float32)
        for row in range(rows):
            mask = unpack_mask(self.pred_masks[row], self.height, self.width)
            ys, xs = np.nonzero(mask)
            if xs.size:
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                boxes[row] = (x1, y1, w, h)
                centers[row] = (x1 + 0.5 * w, y1 + 0.5 * h)
                border[row] = float(x1 <= 2 or y1 <= 2 or x1 + w >= self.width - 3 or y1 + h >= self.height - 3)

        track_rows: dict[int, list[int]] = {}
        for row, track_id in enumerate(self.track_ids):
            track_rows.setdefault(int(track_id), []).append(row)

        progress = np.zeros(rows, dtype=np.float32)
        span_feature = np.zeros(rows, dtype=np.float32)
        visible_fraction = np.zeros(rows, dtype=np.float32)
        prev_iou = np.zeros(rows, dtype=np.float32)
        next_iou = np.zeros(rows, dtype=np.float32)
        centroid_motion = np.zeros(rows, dtype=np.float32)
        diagonal = float(np.hypot(self.width, self.height))
        for row_ids in track_rows.values():
            row_ids = sorted(row_ids, key=lambda row: self.frame_indices[row])
            frames = self.frame_indices[row_ids]
            first = int(frames[0])
            last = int(frames[-1])
            span = max(last - first + 1, 1)
            for pos, row in enumerate(row_ids):
                progress[row] = (int(self.frame_indices[row]) - first) / max(last - first, 1)
                span_feature[row] = span / max(int(self.frame_indices.max()) + 1, 1)
                visible_fraction[row] = len(row_ids) / max(int(self.frame_indices.max()) + 1, 1)
                if pos > 0 and self.frame_indices[row] - self.frame_indices[row_ids[pos - 1]] <= 1:
                    prev_row = row_ids[pos - 1]
                    prev_iou[row] = packed_iou(self.pred_masks[row], self.pred_masks[prev_row])
                    centroid_motion[row] = min(float(np.linalg.norm(centers[row] - centers[prev_row]) / diagonal), 1.0)
                if pos + 1 < len(row_ids) and self.frame_indices[row_ids[pos + 1]] - self.frame_indices[row] <= 1:
                    next_iou[row] = packed_iou(self.pred_masks[row], self.pred_masks[row_ids[pos + 1]])

        bbox_w = boxes[:, 2] / max(self.width, 1)
        bbox_h = boxes[:, 3] / max(self.height, 1)
        extras = np.stack(
            [
                (boxes[:, 0] + 0.5 * boxes[:, 2]) / max(self.width, 1),
                (boxes[:, 1] + 0.5 * boxes[:, 3]) / max(self.height, 1),
                bbox_w,
                bbox_h,
                np.minimum(bbox_w / np.maximum(bbox_h, 1e-6), 8.0) / 8.0,
                border,
                progress,
                span_feature,
                visible_fraction,
                prev_iou,
                next_iou,
                centroid_motion,
            ],
            axis=1,
        ).astype(np.float32)
        self.features = np.concatenate([self.features, extras], axis=1)
        self.feature_names.extend(extra_names)

    def add_color_features_if_needed(self, original_indices: np.ndarray) -> None:
        color_names = ["rgb_r", "rgb_g", "rgb_b", "rgb_std", "saturation", "value", "colorfulness"]
        if all(name in self.feature_names for name in color_names):
            return
        manifest_path = self.path.parent / "manifest.json"
        if not manifest_path.exists():
            self.features = np.concatenate([self.features, np.zeros((self.features.shape[0], len(color_names)), dtype=np.float32)], axis=1)
            self.feature_names.extend(color_names)
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = load_video_frames(Path(manifest["input"]), original_indices, self.width, self.height)
        extras = np.zeros((self.pred_masks.shape[0], len(color_names)), dtype=np.float32)
        for row, frame_idx in enumerate(self.frame_indices):
            frame = frames.get(int(frame_idx))
            if frame is None:
                continue
            mask = unpack_mask(self.pred_masks[row], self.height, self.width)
            extras[row] = np.asarray(masked_color_features(frame, mask), dtype=np.float32)
        self.features = np.concatenate([self.features, extras], axis=1)
        self.feature_names.extend(color_names)


def greedy_oracle_labels(cache: TrackCache, max_select: int) -> np.ndarray:
    labels = np.zeros(cache.frame_indices.shape[0], dtype=np.int32)
    for frame in cache.frames:
        rows = cache.rows_by_frame[frame]
        gt = cache.gt_by_frame[frame]
        union = np.zeros(cache.pred_masks.shape[1], dtype=np.uint8)
        current = packed_iou(union, gt)
        selected: set[int] = set()
        while len(selected) < max_select:
            best = None
            for row in rows:
                row = int(row)
                if row in selected:
                    continue
                candidate = np.bitwise_or(union, cache.pred_masks[row])
                score = packed_iou(candidate, gt)
                if best is None or score > best[0]:
                    best = (score, row, candidate)
            if best is None or best[0] <= current + 1e-6:
                break
            current, row, union = best
            selected.add(row)
            labels[row] = 1
    return labels


def score_cache(model, cache: TrackCache) -> np.ndarray:
    return model.predict_proba(cache.augmented_features)[:, 1]


def classifier_summary(labels: np.ndarray, scores: np.ndarray) -> dict:
    positives = int(labels.sum())
    total = int(labels.size)
    try:
        auroc = float(roc_auc_score(labels, scores)) if positives and positives < total else None
    except ValueError:
        auroc = None
    try:
        ap = float(average_precision_score(labels, scores)) if positives else None
    except ValueError:
        ap = None
    return {
        "records": total,
        "positives": positives,
        "positive_rate": positives / total if total else 0.0,
        "auroc": auroc,
        "average_precision": ap,
    }


def best_threshold(caches: list[TrackCache], scores_by_cache: list[np.ndarray], thresholds: list[float], temporal_weight: float, count_penalty: float) -> dict:
    best = None
    for threshold in thresholds:
        row = {
            "threshold": float(threshold),
            **aggregate_metrics(
                [evaluate_cache(cache, scores, threshold) for cache, scores in zip(caches, scores_by_cache)],
                temporal_weight,
                count_penalty,
            ),
        }
        if best is None or row["objective"] > best["objective"]:
            best = row
    assert best is not None
    return best


def aggregate_metrics(items: list[dict], temporal_weight: float, count_penalty: float) -> dict:
    total_frames = sum(item["frames"] for item in items)
    if not total_frames:
        return {"objective": 0.0, "selected_union_mean_iou": 0.0, "selected_temporal_union_iou": 0.0, "selected_mean_count": 0.0, "frames": 0, "temporal_pairs": 0}
    mean_iou = sum(item["selected_union_mean_iou"] * item["frames"] for item in items) / total_frames
    mean_count = sum(item["selected_mean_count"] * item["frames"] for item in items) / total_frames
    temporal_pairs = sum(item["temporal_pairs"] for item in items)
    temporal = sum(item["selected_temporal_union_iou"] * item["temporal_pairs"] for item in items) / temporal_pairs if temporal_pairs else 0.0
    return {
        "objective": float(mean_iou + temporal_weight * temporal - count_penalty * mean_count),
        "selected_union_mean_iou": float(mean_iou),
        "selected_temporal_union_iou": float(temporal),
        "selected_mean_count": float(mean_count),
        "frames": int(total_frames),
        "temporal_pairs": int(temporal_pairs),
    }


def evaluate_cache(cache: TrackCache, scores: np.ndarray, threshold: float) -> dict:
    ious = []
    counts = []
    unions = []
    for frame in cache.frames:
        rows = cache.rows_by_frame[frame]
        selected = rows[scores[rows] >= threshold]
        union = np.bitwise_or.reduce(cache.pred_masks[selected], axis=0) if selected.size else np.zeros(cache.pred_masks.shape[1], dtype=np.uint8)
        ious.append(packed_iou(union, cache.gt_by_frame[frame]))
        counts.append(int(selected.size))
        unions.append(union)
    temporal = [packed_iou(a, b) for a, b in zip(unions[:-1], unions[1:]) if packed_count(np.bitwise_or(a, b)) > 0]
    return {
        "selected_union_mean_iou": float(np.mean(ious)) if ious else 0.0,
        "selected_temporal_union_iou": float(np.mean(temporal)) if temporal else 0.0,
        "selected_mean_count": float(np.mean(counts)) if counts else 0.0,
        "frames": len(cache.frames),
        "temporal_pairs": len(temporal),
    }


def packed_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = packed_count(np.bitwise_or(a, b))
    return packed_count(np.bitwise_and(a, b)) / union if union else 0.0


def packed_count(mask: np.ndarray) -> int:
    return int(POPCOUNT[mask].sum())


def unpack_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.unpackbits(mask, count=height * width).reshape(height, width).astype(bool)


def load_video_frames(input_path: Path, original_indices: np.ndarray, width: int, height: int) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        return {}
    frames = {}
    try:
        for sampled_idx, original_idx in enumerate(original_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(original_idx))
            ok, bgr = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames[int(sampled_idx)] = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    finally:
        capture.release()
    return frames


def masked_color_features(frame: np.ndarray, mask: np.ndarray) -> list[float]:
    pixels = frame[mask]
    if pixels.size == 0:
        return [0.0] * 7
    normalized = pixels.astype(np.float32) / 255.0
    mean = normalized.mean(axis=0)
    std = float(normalized.std())
    max_channel = normalized.max(axis=1)
    min_channel = normalized.min(axis=1)
    saturation = float(((max_channel - min_channel) / np.maximum(max_channel, 1e-6)).mean())
    value = float(max_channel.mean())
    rg = normalized[:, 0] - normalized[:, 1]
    yb = 0.5 * (normalized[:, 0] + normalized[:, 1]) - normalized[:, 2]
    colorfulness = float(np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return [float(mean[0]), float(mean[1]), float(mean[2]), std, saturation, value, colorfulness]


def parse_grid(value: str) -> list[float]:
    start, stop, step = (float(part) for part in value.split(":"))
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


if __name__ == "__main__":
    main()
