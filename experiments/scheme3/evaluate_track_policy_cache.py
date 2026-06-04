#!/usr/bin/env python3
"""Evaluate a saved track policy on a cached detector/SAM track cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_pipeline import FEATURE_NAMES, apply_score_model
from train_greedy_selector import TrackCache, packed_count, packed_iou


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-cache", type=Path, required=True)
    parser.add_argument("--score-model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cache = TrackCache.load(args.track_cache)
    per_frame_scores = build_per_frame_scores(cache)
    threshold = apply_score_model(args.score_model, [], per_frame_scores, args.device)
    if threshold is None:
        raise RuntimeError("Score model did not provide an object threshold")
    metrics = evaluate_cache_selection(cache, per_frame_scores, float(threshold))
    print(json.dumps({"threshold": float(threshold), **metrics}, indent=2))


def build_per_frame_scores(cache: TrackCache) -> dict[int, dict[int, dict]]:
    per_frame_scores: dict[int, dict[int, dict]] = {}
    feature_index = {name: index for index, name in enumerate(cache.feature_names)}
    for row, (frame_idx, track_id) in enumerate(zip(cache.frame_indices, cache.track_ids)):
        score = {name: float(cache.features[row, feature_index[name]]) for name in FEATURE_NAMES if name in feature_index}
        score["object_score"] = float(cache.object_scores[row])
        per_frame_scores.setdefault(int(frame_idx), {})[int(track_id)] = score
    return per_frame_scores


def evaluate_cache_selection(cache: TrackCache, per_frame_scores: dict[int, dict[int, dict]], threshold: float) -> dict:
    ious = []
    counts = []
    unions = []
    for frame in cache.frames:
        rows = cache.rows_by_frame[frame]
        selected = [
            int(row)
            for row in rows
            if per_frame_scores[int(frame)][int(cache.track_ids[row])]["object_score"] >= threshold
        ]
        union = np.bitwise_or.reduce(cache.pred_masks[selected], axis=0) if selected else np.zeros(cache.pred_masks.shape[1], dtype=np.uint8)
        ious.append(packed_iou(union, cache.gt_by_frame[frame]))
        counts.append(len(selected))
        unions.append(union)
    temporal = [packed_iou(a, b) for a, b in zip(unions[:-1], unions[1:]) if packed_count(np.bitwise_or(a, b)) > 0]
    return {
        "selected_union_mean_iou": float(np.mean(ious)) if ious else 0.0,
        "selected_temporal_union_iou": float(np.mean(temporal)) if temporal else 0.0,
        "selected_mean_count": float(np.mean(counts)) if counts else 0.0,
        "frames": len(cache.frames),
        "temporal_pairs": len(temporal),
    }


if __name__ == "__main__":
    main()
