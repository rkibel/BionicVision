#!/usr/bin/env python3
"""Render Scheme 3 dense predictions as cyan hand and red object overlays."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from dataset_loaders import DEFAULT_EGOEXO_CAMERA, take_video_path
from evaluation.runtime import extract_frames, load_runtime, predict_probs, preprocess, transform_hand_prior
from evaluation.video_temporal import postprocess_probs


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs/experiments/scheme3"
HAND_CHECKPOINT = OUTPUT_DIR / "hand_segmentor/best.pt"
CURRENT_DENSE_CHECKPOINT = OUTPUT_DIR / "checkpoints/best.pt"
DEFAULT_BENCHMARK_TAKE = "sfu_cooking_008_3"
DEFAULT_BENCHMARK_CAMERA = DEFAULT_EGOEXO_CAMERA
DEFAULT_BENCHMARK_START = 3150


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CURRENT_DENSE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "qualitative_runs/best_target")
    parser.add_argument("--input-video", type=Path, default=None, help="Any RGB video path. If omitted, --take/--camera select an Ego-Exo4D video.")
    parser.add_argument("--label", default="", help="Optional run label stored in the manifest.")
    parser.add_argument("--take", default=DEFAULT_BENCHMARK_TAKE)
    parser.add_argument("--camera", default=DEFAULT_BENCHMARK_CAMERA)
    parser.add_argument("--start-frame", type=int, default=DEFAULT_BENCHMARK_START)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--threshold-override", type=float, default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.55)
    parser.add_argument("--hysteresis-keep-threshold", type=float, default=0.35)
    parser.add_argument("--morph-close-kernel", type=int, default=3)
    parser.add_argument("--hand-prior-power", type=float, default=1.5)
    parser.add_argument("--hand-checkpoint", type=Path, default=HAND_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--contact-sheet-samples", type=int, default=12)
    parser.add_argument("--write-object-masks", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, hand_prior, cfg = load_runtime(args.checkpoint, args.hand_checkpoint, args.device)
    threshold = float(args.threshold_override if args.threshold_override is not None else cfg["threshold"])
    video_path = args.input_video if args.input_video is not None else take_video_path(args.take, args.camera)
    frames, fps, indices = extract_frames(video_path, args.start_frame, args.duration_seconds)
    probs = predict_probs(model, hand_prior, frames, indices, cfg, args)
    object_masks = postprocess_probs(probs, indices, threshold, args.ema_alpha, args.hysteresis_keep_threshold, args.morph_close_kernel)
    hand_masks = predict_hand_masks(hand_prior, frames, cfg["image_size"], args)

    overlays = []
    mask_dir = args.output_dir / "object_masks" if args.write_object_masks else None
    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
    for frame, frame_number, hand_mask in zip(frames, indices, hand_masks):
        object_mask = cv2.resize(object_masks[frame_number].numpy().astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlays.append(render_frame(frame, hand_mask, object_mask, frame_number))
        if mask_dir is not None:
            cv2.imwrite(str(mask_dir / f"frame_{frame_number:010d}.png"), object_mask.astype(np.uint8) * 255)

    overlay_path = args.output_dir / "overlay.mp4"
    contact_sheet_path = args.output_dir / "contact_sheet.jpg"
    write_video(overlay_path, overlays, fps)
    transcode_for_browser(overlay_path)
    write_contact_sheet(contact_sheet_path, overlays, args.contact_sheet_samples)
    manifest = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "input_video": str(video_path),
        "video_source": "explicit" if args.input_video is not None else "egoexo_take",
        "take": args.take,
        "camera": args.camera,
        "start_frame": args.start_frame,
        "duration_seconds": args.duration_seconds,
        "frames": len(frames),
        "fps": fps,
        "threshold": threshold,
        "image_feature_mode": cfg["image_feature_mode"],
        "ema_alpha": args.ema_alpha,
        "hysteresis_keep_threshold": args.hysteresis_keep_threshold,
        "morph_close_kernel": args.morph_close_kernel,
        "hand_prior_power": args.hand_prior_power,
        "overlay_video": str(overlay_path.resolve()),
        "contact_sheet": str(contact_sheet_path.resolve()),
        "object_masks": str(mask_dir.resolve()) if mask_dir is not None else None,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


@torch.inference_mode()
def predict_hand_masks(hand_prior, frames: list[np.ndarray], image_size: int, args) -> list[np.ndarray]:
    masks = []
    for start in range(0, len(frames), args.batch_size):
        batch = frames[start : start + args.batch_size]
        images = preprocess(batch, image_size).to(args.device)
        probs = transform_hand_prior(hand_prior(images), args.hand_prior_power).detach().cpu().float().numpy()[:, 0]
        for frame, prob in zip(batch, probs):
            masks.append(cv2.resize(prob, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR) >= hand_prior.threshold)
    return masks


def render_frame(frame_rgb: np.ndarray, hand_mask: np.ndarray, object_mask: np.ndarray, frame_number: int) -> np.ndarray:
    overlay = frame_rgb.copy()
    overlay[hand_mask] = (0.35 * overlay[hand_mask] + 0.65 * np.array([0, 220, 255])).astype(np.uint8)
    overlay[object_mask] = (0.35 * overlay[object_mask] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
    cv2.putText(overlay, f"frame {frame_number}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError("No frames to render")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def transcode_for_browser(path: Path) -> None:
    tmp = path.with_suffix(".h264.mp4")
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(command, check=True)
        tmp.replace(path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        if tmp.exists():
            tmp.unlink()


def write_contact_sheet(path: Path, frames: list[np.ndarray], samples: int = 12) -> None:
    if not frames:
        return
    picks = np.linspace(0, len(frames) - 1, num=min(samples, len(frames)), dtype=int)
    thumbs = [cv2.resize(frames[idx], (224, 126), interpolation=cv2.INTER_AREA) for idx in picks]
    cols = min(4, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 126, cols * 224, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        sheet[row * 126 : (row + 1) * 126, col * 224 : (col + 1) * 224] = thumb
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
