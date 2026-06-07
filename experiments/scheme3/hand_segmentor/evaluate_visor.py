#!/usr/bin/env python3
"""Evaluate a Scheme 3 EgoHOS hand segmentor on EPIC-KITCHENS VISOR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

SCHEME_DIR = Path(__file__).resolve().parents[1]
if str(SCHEME_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEME_DIR))

import visor
from hand_segmentor.model import build_model
from train_hand_segmentor import MEAN, STD, parse_size, parse_thresholds, predict_probs


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "outputs/experiments/scheme3"
DEFAULT_CHECKPOINT = OUTPUT_DIR / "hand_segmentor/best.pt"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "hand_segmentor/visor_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", help="VISOR split(s), comma separated: train,val,test.")
    parser.add_argument("--model", default="smp-unetpp-efficientnet-b4", help="Fallback model name if checkpoint lacks metadata.")
    parser.add_argument("--thresholds", help="Comma-separated thresholds. Defaults to the checkpoint threshold.")
    parser.add_argument("--max-frames", type=int, help="Optional cap for quick/debug evaluation.")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for CLI symmetry; evaluation is framewise.")
    parser.add_argument("--workers", type=int, default=0, help="Reserved for CLI symmetry; evaluation is framewise.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--write-masks", action="store_true")
    args = parser.parse_args()

    del args.batch_size, args.workers
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_name = str(checkpoint.get("model_name", args.model))
    image_size = parse_size(str(checkpoint.get("image_size", "512x704")))
    checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
    thresholds = parse_thresholds(args.thresholds) if args.thresholds else [checkpoint_threshold]

    model = build_model(model_name, args.device, encoder_weights=None)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    items = visor_items(args.split, max_frames=args.max_frames)
    if not items:
        raise RuntimeError(f"No VISOR frames found for split(s): {args.split}")

    write_dir = args.output_dir / "masks" if args.write_masks else None
    summary = evaluate(model, items, image_size, args.device, thresholds, tta_flip=args.tta_flip, write_dir=write_dir)
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "model_name": model_name,
            "checkpoint_threshold": checkpoint_threshold,
            "checkpoint_image_size": f"{image_size[0]}x{image_size[1]}",
            "split": args.split,
            "frame_cap": args.max_frames,
            "tta_flip": args.tta_flip,
        }
    )
    out_path = args.output_dir / f"visor_{split_slug(args.split)}_summary.json"
    write_json(out_path, summary)
    print_summary(summary)
    print(out_path)


def visor_items(split_value: str, *, max_frames: int | None) -> list[tuple[Any, str]]:
    items = []
    seen = set()
    for split in [part.strip() for part in split_value.split(",") if part.strip()]:
        if split not in visor.SPLITS:
            raise ValueError(f"Unknown VISOR split: {split}")
        for video_id in visor.SPLITS[split]:
            if not visor.annotation_path(video_id).exists() or not visor.sparse_frame_dir(video_id).exists():
                continue
            frames_by_index = visor.load_frames_by_index(video_id)
            for path in sorted(visor.sparse_frame_dir(video_id).glob(f"{video_id}_frame_*.jpg")):
                key = visor.FrameKey.from_stem(path.stem)
                frame = frames_by_index.get(key.frame_index)
                if frame is None:
                    continue
                if not any(visor.is_hand_like_label(annotation.name) for annotation in frame.annotations):
                    continue
                ident = (key.video_id, key.frame_index)
                if ident in seen:
                    continue
                seen.add(ident)
                items.append((key, split))
                if max_frames is not None and len(items) >= max_frames:
                    return items
    return items


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    items: list[tuple[Any, str]],
    image_size: tuple[int, int],
    device: str,
    thresholds: list[float],
    *,
    tta_flip: bool,
    write_dir: Path | None,
) -> dict[str, Any]:
    records_by_threshold: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds}
    for key, split in items:
        image = visor.read_rgb(visor.sparse_frame_path(key))
        gt = visor.hand_ground_truth_mask(key.video_id, key.frame_index, image.shape[:2])
        image_small, gt_small = resize_pair(image, gt, image_size)
        image_t = image_to_tensor(image_small).to(device)
        probs = predict_probs(model, image_t, tta_flip=tta_flip)[0, 0].detach().cpu().numpy()
        for threshold in thresholds:
            pred = probs >= threshold
            records_by_threshold[threshold].append(frame_record(key, split, pred, gt_small))
            if write_dir is not None:
                threshold_dir = write_dir / f"threshold_{threshold:.2f}"
                threshold_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(threshold_dir / f"{key.stem}.png"), pred.astype(np.uint8) * 255)

    rows = [
        {
            "threshold": threshold,
            **summarize_records(records),
            "by_split": per_split(records),
            "per_video": per_video(records),
        }
        for threshold, records in records_by_threshold.items()
    ]
    rows.sort(key=lambda row: (row["mean_iou"], row["mean_precision"]), reverse=True)
    return {"best": rows[0], "thresholds": rows}


def resize_pair(image: np.ndarray, gt: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = size
    image_small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    gt_small = cv2.resize(gt.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return image_small, gt_small


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0)


def frame_record(key, split: str, pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    return {
        "video_id": key.video_id,
        "frame_index": key.frame_index,
        "split": split,
        "metrics": visor.mask_metrics(pred, gt),
        "prediction_pixels": int(np.count_nonzero(pred)),
        "ground_truth_pixels": int(np.count_nonzero(gt)),
        "detections": int(np.count_nonzero(pred) > 0),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    return visor.summarize_frame_metrics(records)


def per_split(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    return {
        split: summarize_records([record for record in records if record["split"] == split])
        for split in sorted({record["split"] for record in records})
    }


def per_video(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    return {
        video_id: summarize_records([record for record in records if record["video_id"] == video_id])
        for video_id in sorted({record["video_id"] for record in records})
    }


def split_slug(value: str) -> str:
    return "_".join(part.strip() for part in value.split(",") if part.strip())


def print_summary(summary: dict[str, Any]) -> None:
    best = summary["best"]
    print(
        f"best threshold={best['threshold']:.2f} "
        f"IoU={best['mean_iou']:.4f} "
        f"P={best['mean_precision']:.4f} "
        f"R={best['mean_recall']:.4f} "
        f"frames={best['frames']}"
    )
    print(json.dumps(best["by_split"], indent=2))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
