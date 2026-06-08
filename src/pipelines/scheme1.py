"""Scheme 1: DEVA-only indoor kitchen segmentation with temporal tracking."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from models.segmentation.deva.run_manual import PromptGroup, run_deva_manual

from .han_fusion_baseline_models import (
    ensure_cuda_if_requested,
    extract_video_frames,
    write_video,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLIP_DIR = ROOT / "data" / "epic_kitchens" / "video_snippets" / "test_set" / "inputs"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "scheme1_test_set"

KITCHEN_OBJECT_PROMPTS = (
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "plate",
    "pan",
    "pot",
    "lid",
    "tray",
    "cutting board",
    "food",
    "banana",
    "apple",
    "orange",
    "broccoli",
    "carrot",
    "sandwich",
    "pizza",
    "cake",
    "toaster",
    "kettle",
    "book",
    "scissors",
)

KITCHEN_SCENE_PROMPTS = (
    "wall",
    "floor",
    "ceiling",
    "window",
    "cabinet",
    "door",
    "table",
    "shelf",
    "counter",
    "countertop",
    "sink",
    "stove",
    "oven",
    "microwave",
    "refrigerator",
    "dishwasher",
    "chair",
    "drawer",
)

SCHEME1_PROMPT_GROUPS = (
    PromptGroup("objects", KITCHEN_OBJECT_PROMPTS),
    PromptGroup("scenes", KITCHEN_SCENE_PROMPTS),
)

SCHEME1_OBJECT_PROMPT_GROUPS = (
    PromptGroup("objects", KITCHEN_OBJECT_PROMPTS),
)

PROMPT_PRESETS = {
    "objects": SCHEME1_OBJECT_PROMPT_GROUPS,
    "objects_scenes": SCHEME1_PROMPT_GROUPS,
}


@dataclass(frozen=True)
class Scheme1Config:
    target_fps: float = 10.0
    max_frames: int | None = None
    device: str = "cuda"
    deva_size: int = 360
    deva_detection_every: int = 5
    deva_memory_reset_interval: int = 0
    dino_threshold: float = 0.35
    dino_nms_threshold: float = 0.8
    sam_variant: str = "sam_hq_light"
    prompt_preset: str = "objects_scenes"


def run_scheme1(
    clip_dir: Path = DEFAULT_CLIP_DIR,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    config: Scheme1Config = Scheme1Config(),
) -> list[dict[str, str | int]]:
    """Run Scheme 1 on every MP4 clip in `clip_dir`."""

    ensure_cuda_if_requested(config.device)
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError(f"No MP4 clips found in {clip_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for clip_path in clips:
        print(f"processing {clip_path.name}", flush=True)
        summaries.append(run_scheme1_on_clip(clip_path, output_root, config))
    return summaries


def run_scheme1_on_clip(
    clip_path: Path,
    output_root: Path,
    config: Scheme1Config = Scheme1Config(),
) -> dict[str, str | int]:
    """Run DEVA-only indoor segmentation on one EPIC-KITCHENS clip."""

    parse_clip_name(clip_path.stem)
    clip_root = output_root / clip_path.stem
    frames = extract_video_frames(clip_path, clip_root / "frames", config.target_fps, config.max_frames)
    prompt_groups = PROMPT_PRESETS[config.prompt_preset]
    deva_outputs = run_deva_manual(
        frames_dir=clip_root / "frames",
        output_dir=clip_root / "deva_raw",
        prompt_groups=prompt_groups,
        size=config.deva_size,
        detection_every=config.deva_detection_every,
        memory_reset_interval=config.deva_memory_reset_interval,
        dino_threshold=config.dino_threshold,
        dino_nms_threshold=config.dino_nms_threshold,
        sam_variant=config.sam_variant,
    )
    semantic_annotations = sorted(deva_outputs.raw_annotation_dir.glob("*.png"))
    visualizations = sorted(deva_outputs.raw_visualization_dir.glob("*.png"))
    segmentation = build_scheme1_segmentation_frames(semantic_annotations, clip_root / "segmentation_frames")

    videos_dir = clip_root / "videos"
    write_video(segmentation, videos_dir / "scheme1.mp4", config.target_fps, is_color=False)
    write_video(semantic_annotations, videos_dir / "scheme1_semantic_deva.mp4", config.target_fps, is_color=True)
    write_video(visualizations, videos_dir / "scheme1_deva_overlay.mp4", config.target_fps, is_color=True)

    return {
        "clip": str(clip_path),
        "frames": len(frames),
        "segmentation_frames": len(segmentation),
        "semantic_annotation_frames": len(semantic_annotations),
        "visualization_frames": len(visualizations),
        "output": str(clip_root),
    }


def build_scheme1_segmentation_frames(
    semantic_annotation_paths: list[Path],
    output_dir: Path,
) -> list[Path]:
    """Render DEVA labels into one binary segmentation mask."""

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for index, annotation_path in enumerate(semantic_annotation_paths):
        annotation = cv2.imread(str(annotation_path), cv2.IMREAD_COLOR)
        if annotation is None:
            raise FileNotFoundError(annotation_path)
        mask = np.where(np.any(annotation > 0, axis=2), 255, 0).astype(np.uint8)
        output_path = output_dir / f"frame{index:05d}.png"
        cv2.imwrite(str(output_path), mask)
        output_paths.append(output_path)
    return output_paths


def parse_clip_name(name: str) -> tuple[str, int | None, int | None]:
    match = re.match(r"(?P<video>P\d+_\d+)_continuous_\d+_frames_(?P<start>\d+)_(?P<end>\d+)$", name)
    if match:
        return match.group("video"), int(match.group("start")), int(match.group("end"))
    match = re.match(r"(?P<video>P\d+_\d+)_frames_(?P<start>\d+)_(?P<end>\d+)$", name)
    if match:
        return match.group("video"), int(match.group("start")), int(match.group("end"))
    match = re.match(r"(?P<video>P\d+_\d+)_test_frames$", name)
    if not match:
        raise ValueError(f"Could not parse EPIC clip name: {name}")
    return match.group("video"), None, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Scheme 1 DEVA-only indoor segmentation.")
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--deva-detection-every", type=int, default=5)
    parser.add_argument("--deva-memory-reset-interval", type=int, default=0)
    parser.add_argument("--deva-size", type=int, default=360)
    parser.add_argument("--dino-threshold", type=float, default=0.35)
    parser.add_argument("--dino-nms-threshold", type=float, default=0.8)
    parser.add_argument("--sam-variant", choices=["original", "mobile", "sam_hq", "sam_hq_light"], default="sam_hq_light")
    parser.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS), default="objects_scenes")
    args = parser.parse_args()
    config = Scheme1Config(
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        device=args.device,
        deva_detection_every=args.deva_detection_every,
        deva_memory_reset_interval=args.deva_memory_reset_interval,
        deva_size=args.deva_size,
        dino_threshold=args.dino_threshold,
        dino_nms_threshold=args.dino_nms_threshold,
        sam_variant=args.sam_variant,
        prompt_preset=args.prompt_preset,
    )
    for summary in run_scheme1(args.clip_dir.resolve(), args.output_root.resolve(), config):
        print(summary)


if __name__ == "__main__":
    main()
