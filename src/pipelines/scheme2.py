"""Scheme 2: always retain hands alongside temporally tracked DEVA objects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import shutil

import cv2
import numpy as np

from models.segmentation.deva.run_manual import run_deva_manual
from models.segmentation.hand_segmentor.adapter import DEFAULT_CHECKPOINT, HandSegmentor

from .han_fusion_baseline_models import ensure_cuda_if_requested, extract_video_frames, write_video
from .scheme1 import PROMPT_PRESETS, Scheme1Config, build_scheme1_segmentation_frames, parse_clip_name


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP_DIR = ROOT / "data/epic_kitchens" / "continuous_segments"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/scheme2"


@dataclass(frozen=True)
class Scheme2Config:
    """Configuration for hand retention plus temporal DEVA segmentation."""

    scheme1: Scheme1Config = field(default_factory=lambda: Scheme1Config(prompt_preset="objects"))
    hand_checkpoint: Path = DEFAULT_CHECKPOINT
    hand_threshold: float | None = None
    hand_batch_size: int = 4


def run_scheme2(
    clip_dir: Path = DEFAULT_CLIP_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config: Scheme2Config = Scheme2Config(),
) -> list[dict[str, str | int | float]]:
    """Run Scheme 2 on every MP4 clip in a directory."""

    validate_config(config)
    ensure_cuda_if_requested(config.scheme1.device)
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No MP4 clips found in {clip_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip in clips:
        print(f"processing {clip.name}", flush=True)
        summaries.append(run_scheme2_on_clip(clip, output_root, config))
    return summaries


def run_scheme2_on_clip(
    clip_path: Path,
    output_root: Path,
    config: Scheme2Config = Scheme2Config(),
) -> dict[str, str | int | float]:
    """Run temporal DEVA and the hand segmentor on one EPIC-KITCHENS clip."""

    parse_clip_name(clip_path.stem)
    clip_root = output_root / clip_path.stem
    scheme1 = config.scheme1
    frames = extract_video_frames(clip_path, clip_root / "frames", scheme1.target_fps, scheme1.max_frames)

    deva_outputs = run_deva_manual(
        frames_dir=clip_root / "frames",
        output_dir=clip_root / "deva_raw",
        prompt_groups=PROMPT_PRESETS[scheme1.prompt_preset],
        size=scheme1.deva_size,
        detection_every=scheme1.deva_detection_every,
        memory_reset_interval=scheme1.deva_memory_reset_interval,
        dino_threshold=scheme1.dino_threshold,
        dino_nms_threshold=scheme1.dino_nms_threshold,
        sam_variant=scheme1.sam_variant,
    )
    semantic_annotations = sorted(deva_outputs.raw_annotation_dir.glob("*.png"))
    if len(semantic_annotations) != len(frames):
        raise RuntimeError(f"DEVA returned {len(semantic_annotations)} annotations for {len(frames)} frames")
    deva_masks = build_scheme1_segmentation_frames(semantic_annotations, clip_root / "deva_segmentation_frames")

    hand_segmentor = HandSegmentor(checkpoint=config.hand_checkpoint, device=scheme1.device)
    hand_threshold = hand_segmentor.threshold if config.hand_threshold is None else config.hand_threshold
    hand_masks = build_hand_segmentation_frames(
        frames,
        clip_root / "hand_segmentation_frames",
        hand_segmentor,
        threshold=hand_threshold,
        batch_size=config.hand_batch_size,
    )
    combined = combine_segmentation_frames(hand_masks, deva_masks, clip_root / "segmentation_frames")
    overlays = build_overlay_frames(frames, hand_masks, deva_masks, clip_root / "overlay_frames")

    videos_dir = clip_root / "videos"
    write_video(combined, videos_dir / "scheme2.mp4", scheme1.target_fps, is_color=False)
    write_video(hand_masks, videos_dir / "scheme2_hands.mp4", scheme1.target_fps, is_color=False)
    write_video(deva_masks, videos_dir / "scheme2_deva_objects.mp4", scheme1.target_fps, is_color=False)
    write_video(semantic_annotations, videos_dir / "scheme2_semantic_deva.mp4", scheme1.target_fps, is_color=True)
    write_video(overlays, videos_dir / "scheme2_overlay.mp4", scheme1.target_fps, is_color=True)

    return {
        "clip": str(clip_path),
        "frames": len(frames),
        "hand_frames": len(hand_masks),
        "deva_frames": len(deva_masks),
        "segmentation_frames": len(combined),
        "hand_threshold": float(hand_threshold),
        "output": str(clip_root),
    }


def build_hand_segmentation_frames(
    frame_paths: list[Path],
    output_dir: Path,
    segmentor: HandSegmentor,
    *,
    threshold: float,
    batch_size: int,
) -> list[Path]:
    """Predict and save binary hand masks for sampled video frames."""

    reset_directory(output_dir)
    output_paths = []
    for start in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[start : start + batch_size]
        images = [read_image(path) for path in batch_paths]
        for frame_path, probability in zip(batch_paths, segmentor.predict_batch(images)):
            output_path = output_dir / f"{frame_path.stem}.png"
            cv2.imwrite(str(output_path), (probability >= threshold).astype(np.uint8) * 255)
            output_paths.append(output_path)
    return output_paths


def combine_segmentation_frames(hand_paths: list[Path], deva_paths: list[Path], output_dir: Path) -> list[Path]:
    """Save the binary union that always includes predicted hands."""

    if len(hand_paths) != len(deva_paths):
        raise ValueError(f"Hand/DEVA frame count mismatch: {len(hand_paths)} != {len(deva_paths)}")
    reset_directory(output_dir)
    output_paths = []
    for index, (hand_path, deva_path) in enumerate(zip(hand_paths, deva_paths)):
        combined = np.maximum(read_mask(hand_path), read_mask(deva_path))
        output_path = output_dir / f"frame{index:05d}.png"
        cv2.imwrite(str(output_path), combined)
        output_paths.append(output_path)
    return output_paths


def build_overlay_frames(
    frame_paths: list[Path],
    hand_paths: list[Path],
    deva_paths: list[Path],
    output_dir: Path,
) -> list[Path]:
    """Overlay DEVA objects in red and retained hands in cyan."""

    if len(frame_paths) != len(hand_paths) or len(frame_paths) != len(deva_paths):
        raise ValueError("Source, hand, and DEVA frame counts must match")
    reset_directory(output_dir)
    output_paths = []
    for index, (frame_path, hand_path, deva_path) in enumerate(zip(frame_paths, hand_paths, deva_paths)):
        frame = read_image(frame_path)
        hand = read_mask(hand_path) > 0
        deva = read_mask(deva_path) > 0
        overlay = frame.copy()
        overlay[deva] = (0.35 * overlay[deva] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
        overlay[hand] = (0.35 * overlay[hand] + 0.65 * np.array([255, 255, 0])).astype(np.uint8)
        output_path = output_dir / f"frame{index:05d}.png"
        cv2.imwrite(str(output_path), overlay)
        output_paths.append(output_path)
    return output_paths


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def validate_config(config: Scheme2Config) -> None:
    if config.hand_batch_size <= 0:
        raise ValueError("hand_batch_size must be positive")
    if config.hand_threshold is not None and not 0.0 <= config.hand_threshold <= 1.0:
        raise ValueError("hand_threshold must be between 0 and 1")
    if config.scheme1.deva_detection_every <= 0:
        raise ValueError("deva_detection_every must be positive")
    if config.scheme1.deva_memory_reset_interval < 0:
        raise ValueError("deva_memory_reset_interval cannot be negative")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--deva-detection-every", type=int, default=5)
    parser.add_argument("--deva-memory-reset-interval", type=int, default=0)
    parser.add_argument("--deva-size", type=int, default=360)
    parser.add_argument("--dino-threshold", type=float, default=0.35)
    parser.add_argument("--dino-nms-threshold", type=float, default=0.8)
    parser.add_argument("--sam-variant", choices=["original", "mobile", "sam_hq", "sam_hq_light"], default="sam_hq_light")
    parser.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS), default="objects")
    parser.add_argument("--hand-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hand-threshold", type=float)
    parser.add_argument("--hand-batch-size", type=int, default=4)
    args = parser.parse_args()

    scheme1 = Scheme1Config(
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        device=args.device,
        deva_size=args.deva_size,
        deva_detection_every=args.deva_detection_every,
        deva_memory_reset_interval=args.deva_memory_reset_interval,
        dino_threshold=args.dino_threshold,
        dino_nms_threshold=args.dino_nms_threshold,
        sam_variant=args.sam_variant,
        prompt_preset=args.prompt_preset,
    )
    config = Scheme2Config(
        scheme1=scheme1,
        hand_checkpoint=args.hand_checkpoint,
        hand_threshold=args.hand_threshold,
        hand_batch_size=args.hand_batch_size,
    )
    for summary in run_scheme2(args.clip_dir.resolve(), args.output_root.resolve(), config):
        print(summary)


if __name__ == "__main__":
    main()
