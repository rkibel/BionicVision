"""Scheme 3: dense interacting-object segmentation plus retained hands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

import cv2
import numpy as np

from models.segmentation.hand_segmentor.adapter import DEFAULT_CHECKPOINT as DEFAULT_HAND_CHECKPOINT
from models.segmentation.scheme3_dense.adapter import DEFAULT_CHECKPOINT, Scheme3DenseSegmentor

from .han_fusion_baseline_models import ensure_cuda_if_requested, extract_video_frames, write_video
from .scheme1 import parse_clip_name


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP_DIR = ROOT / "data/epic_kitchens/continuous_segments"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/scheme3"


@dataclass(frozen=True)
class Scheme3Config:
    scheme_name: str = "scheme3"
    checkpoint: Path = DEFAULT_CHECKPOINT
    hand_checkpoint: Path = DEFAULT_HAND_CHECKPOINT
    target_fps: float = 10.0
    max_frames: int | None = None
    device: str = "cuda"
    batch_size: int = 8
    object_threshold: float | None = None
    hand_threshold: float | None = None
    sequence_feature_context: bool = False


def run_scheme3(
    clip_dir: Path = DEFAULT_CLIP_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config: Scheme3Config = Scheme3Config(),
) -> list[dict[str, str | int | float]]:
    validate_config(config)
    ensure_cuda_if_requested(config.device)
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No MP4 clips found in {clip_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    segmentor = Scheme3DenseSegmentor(
        config.checkpoint,
        hand_checkpoint=config.hand_checkpoint,
        device=config.device,
        sequence_feature_context=config.sequence_feature_context,
    )
    summaries = []
    for clip in clips:
        print(f"processing {clip.name}", flush=True)
        summaries.append(run_scheme3_on_clip(clip, output_root, config, segmentor=segmentor))
    return summaries


def run_scheme3_on_clip(
    clip_path: Path,
    output_root: Path,
    config: Scheme3Config = Scheme3Config(),
    *,
    segmentor: Scheme3DenseSegmentor | None = None,
) -> dict[str, str | int | float]:
    validate_config(config)
    parse_clip_name(clip_path.stem)
    clip_root = output_root / clip_path.stem
    frames = extract_video_frames(clip_path, clip_root / "frames", config.target_fps, config.max_frames)
    segmentor = segmentor or Scheme3DenseSegmentor(
        config.checkpoint,
        hand_checkpoint=config.hand_checkpoint,
        device=config.device,
        sequence_feature_context=config.sequence_feature_context,
    )
    segmentor.reset_sequence()
    object_threshold = segmentor.threshold if config.object_threshold is None else config.object_threshold
    hand_threshold = segmentor.hand_prior.threshold if config.hand_threshold is None else config.hand_threshold

    object_masks, hand_masks = predict_masks(
        frames,
        segmentor,
        clip_root / "object_segmentation_frames",
        clip_root / "hand_segmentation_frames",
        object_threshold,
        hand_threshold,
        config.batch_size,
    )
    combined = combine_masks(hand_masks, object_masks, clip_root / "segmentation_frames")
    overlays = build_overlays(frames, hand_masks, object_masks, clip_root / "overlay_frames")
    videos = clip_root / "videos"
    name = config.scheme_name
    write_video(combined, videos / f"{name}.mp4", config.target_fps, is_color=False)
    write_video(hand_masks, videos / f"{name}_hands.mp4", config.target_fps, is_color=False)
    write_video(object_masks, videos / f"{name}_dense_objects.mp4", config.target_fps, is_color=False)
    write_video(overlays, videos / f"{name}_overlay.mp4", config.target_fps, is_color=True)
    return {
        "clip": str(clip_path),
        "frames": len(frames),
        "object_frames": len(object_masks),
        "hand_frames": len(hand_masks),
        "segmentation_frames": len(combined),
        "object_threshold": float(object_threshold),
        "hand_threshold": float(hand_threshold),
        "output": str(clip_root),
    }


def predict_masks(
    frame_paths: list[Path],
    segmentor: Scheme3DenseSegmentor,
    object_dir: Path,
    hand_dir: Path,
    object_threshold: float,
    hand_threshold: float,
    batch_size: int,
) -> tuple[list[Path], list[Path]]:
    reset_directory(object_dir)
    reset_directory(hand_dir)
    object_paths, hand_paths = [], []
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        images = [read_color(path) for path in batch_paths]
        object_probs, hand_probs = segmentor.predict_batch(images)
        for path, object_prob, hand_prob in zip(batch_paths, object_probs, hand_probs):
            object_path, hand_path = object_dir / f"{path.stem}.png", hand_dir / f"{path.stem}.png"
            cv2.imwrite(str(object_path), (object_prob >= object_threshold).astype(np.uint8) * 255)
            cv2.imwrite(str(hand_path), (hand_prob >= hand_threshold).astype(np.uint8) * 255)
            object_paths.append(object_path)
            hand_paths.append(hand_path)
    return object_paths, hand_paths


def combine_masks(hand_paths: list[Path], object_paths: list[Path], output_dir: Path) -> list[Path]:
    if len(hand_paths) != len(object_paths):
        raise ValueError("Hand and object frame counts must match")
    reset_directory(output_dir)
    outputs = []
    for index, (hand_path, object_path) in enumerate(zip(hand_paths, object_paths)):
        output = output_dir / f"frame{index:05d}.png"
        cv2.imwrite(str(output), np.maximum(read_mask(hand_path), read_mask(object_path)))
        outputs.append(output)
    return outputs


def build_overlays(frame_paths: list[Path], hand_paths: list[Path], object_paths: list[Path], output_dir: Path) -> list[Path]:
    reset_directory(output_dir)
    outputs = []
    for index, (frame_path, hand_path, object_path) in enumerate(zip(frame_paths, hand_paths, object_paths)):
        overlay = read_color(frame_path)
        objects, hands = read_mask(object_path) > 0, read_mask(hand_path) > 0
        overlay[objects] = (0.35 * overlay[objects] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        overlay[hands] = (0.35 * overlay[hands] + 0.65 * np.array([255, 255, 0])).astype(np.uint8)
        output = output_dir / f"frame{index:05d}.png"
        cv2.imwrite(str(output), overlay)
        outputs.append(output)
    return outputs


def read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def validate_config(config: Scheme3Config) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not config.scheme_name:
        raise ValueError("scheme_name cannot be empty")
    for name, value in (("object_threshold", config.object_threshold), ("hand_threshold", config.hand_threshold)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hand-checkpoint", type=Path, default=DEFAULT_HAND_CHECKPOINT)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--object-threshold", type=float)
    parser.add_argument("--hand-threshold", type=float)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = Scheme3Config(
        scheme_name="scheme3",
        checkpoint=args.checkpoint,
        hand_checkpoint=args.hand_checkpoint,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        device=args.device,
        batch_size=args.batch_size,
        object_threshold=args.object_threshold,
        hand_threshold=args.hand_threshold,
        sequence_feature_context=False,
    )
    for summary in run_scheme3(args.clip_dir.resolve(), args.output_root.resolve(), config):
        print(summary)


if __name__ == "__main__":
    main()
