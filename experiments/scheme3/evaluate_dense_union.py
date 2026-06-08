#!/usr/bin/env python3
"""Evaluate Scheme 3 masks and flow-aligned temporal consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dataset_loaders import DEFAULT_EGOEXO_CAMERA, EGOHOS_ROOT, EgoExoMaskDataset, EgoHOSMaskDataset, take_video_path
from config import (
    CURRENT_DENSE_CHECKPOINT,
    DEFAULT_BENCHMARK_CAMERA,
    DEFAULT_BENCHMARK_END,
    DEFAULT_BENCHMARK_START,
    DEFAULT_BENCHMARK_TAKE,
    HAND_CHECKPOINT,
    OUTPUT_DIR,
)
from evaluation.runtime import extract_frames, load_runtime, predict_probs, preprocess, transform_hand_prior
from evaluation.supervised import evaluate_supervised_dataset
from evaluation.video_temporal import evaluate_full_fps, evaluate_sparse, load_gt_by_frame, parse_ints, postprocess_probs, precompute_flow_warp_maps
from utils import parse_csv, parse_grid, parse_label_ids, write_json


def main() -> None:
    args = parse_args()
    model, hand_prior, cfg = load_runtime(args.checkpoint, args.hand_checkpoint, args.device)
    threshold = float(args.threshold_override if args.threshold_override is not None else cfg["threshold"])
    result: dict[str, Any] = base_result(args, cfg, threshold)

    if not args.skip_video_temporal:
        result["video_temporal"] = evaluate_video_temporal(model, hand_prior, cfg, args, threshold)

    result["egoexo_supervised"] = evaluate_supervised_dataset(
        model,
        hand_prior,
        EgoExoMaskDataset(
            args.eval_egoexo_split,
            args.camera,
            cfg["image_size"],
            args.eval_egoexo_samples,
            991,
            exclude_target_window=True,
            target_take=args.take,
            target_start=args.start_frame,
            target_end=args.benchmark_end_frame,
            preserve_order=True,
            image_feature_mode=cfg["image_feature_mode"],
            image_feature_cache=cfg["image_feature_cache"],
        ),
        cfg,
        args,
        threshold,
        f"egoexo:{args.eval_egoexo_split}",
    )
    result["egohos_supervised"] = evaluate_egohos_splits(model, hand_prior, cfg, args, threshold)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CURRENT_DENSE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "checkpoints/best_eval.json")
    parser.add_argument("--hand-checkpoint", type=Path, default=HAND_CHECKPOINT)
    parser.add_argument("--take", default=DEFAULT_BENCHMARK_TAKE)
    parser.add_argument("--camera", default=DEFAULT_BENCHMARK_CAMERA)
    parser.add_argument("--gt-split", default="val")
    parser.add_argument("--start-frame", type=int, default=DEFAULT_BENCHMARK_START)
    parser.add_argument("--benchmark-end-frame", type=int, default=DEFAULT_BENCHMARK_END)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--skip-video-temporal", action="store_true")
    parser.add_argument("--eval-egoexo-split", default="val")
    parser.add_argument("--eval-egoexo-samples", type=int, default=180)
    parser.add_argument("--egohos-root", type=Path, default=EGOHOS_ROOT)
    parser.add_argument("--eval-egohos-splits", default="val,test_indomain,test_outdomain")
    parser.add_argument("--eval-egohos-samples", type=int, default=240)
    parser.add_argument("--egohos-sources", default="")
    parser.add_argument("--egohos-object-ids", default="3,4,5,6,7,8")
    parser.add_argument("--threshold-override", type=float, default=None)
    parser.add_argument("--supervised-threshold-grid", default="0.02:0.98:0.02")
    parser.add_argument("--ema-alpha", type=float, default=0.55)
    parser.add_argument("--hysteresis-keep-threshold", type=float, default=0.35)
    parser.add_argument("--morph-close-kernel", type=int, default=3)
    parser.add_argument("--hand-prior-power", type=float, default=1.5)
    parser.add_argument("--flow-horizons", default="1,2,5,10,15,30")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def base_result(args, cfg: dict, threshold: float) -> dict[str, Any]:
    return {
        "checkpoint": str(args.checkpoint),
        "hand_checkpoint": str(args.hand_checkpoint),
        "threshold": threshold,
        "image_size": cfg["image_size"],
        "image_feature_mode": cfg["image_feature_mode"],
        "hand_input_mode": cfg["hand_input_mode"],
        "postprocess": {
            "ema_alpha": args.ema_alpha,
            "hysteresis_keep_threshold": args.hysteresis_keep_threshold,
            "morph_close_kernel": args.morph_close_kernel,
            "hand_prior_power": args.hand_prior_power,
        },
    }


def evaluate_video_temporal(model, hand_prior, cfg: dict, args, threshold: float) -> dict[str, Any]:
    frames, fps, indices = extract_frames(take_video_path(args.take, args.camera), args.start_frame, args.duration_seconds)
    probs = predict_probs(model, hand_prior, frames, indices, cfg, args)
    masks = postprocess_probs(probs, indices, threshold, args.ema_alpha, args.hysteresis_keep_threshold, args.morph_close_kernel)
    gt = load_gt_by_frame(args.gt_split, args.take, args.camera, args.start_frame, args.duration_seconds, cfg["image_size"])
    warp_maps = precompute_flow_warp_maps(frames, cfg["image_size"])
    positions = {frame_number: idx for idx, frame_number in enumerate(indices)}
    return {
        "take": args.take,
        "camera": args.camera,
        "gt_split": args.gt_split,
        "start_frame": args.start_frame,
        "duration_seconds": args.duration_seconds,
        "video_frames": len(frames),
        "fps": fps,
        "gt_frame_numbers": sorted(gt),
        "sparse_gt_frames": evaluate_sparse(masks, gt, warp_maps, positions),
        "full_fps": evaluate_full_fps(masks, indices, warp_maps, positions, parse_ints(args.flow_horizons)),
    }


def evaluate_egohos_splits(model, hand_prior, cfg: dict, args, threshold: float) -> dict[str, dict]:
    return {
        split: evaluate_supervised_dataset(
            model,
            hand_prior,
            EgoHOSMaskDataset(
                split=split,
                root=args.egohos_root,
                image_size=cfg["image_size"],
                max_samples=args.eval_egohos_samples,
                seed=1771 + idx * 101,
                sources=tuple(parse_csv(args.egohos_sources)),
                object_ids=parse_label_ids(args.egohos_object_ids),
                preserve_order=True,
                image_feature_mode=cfg["image_feature_mode"],
                image_feature_cache=cfg["image_feature_cache"],
            ),
            cfg,
            args,
            threshold,
            f"egohos:{split}",
        )
        for idx, split in enumerate(parse_csv(args.eval_egohos_splits))
    }


__all__ = [
    "extract_frames",
    "load_runtime",
    "parse_csv",
    "parse_grid",
    "parse_label_ids",
    "postprocess_probs",
    "predict_probs",
    "preprocess",
    "transform_hand_prior",
]


if __name__ == "__main__":
    main()
