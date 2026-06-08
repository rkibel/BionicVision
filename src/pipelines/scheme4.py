"""Scheme 4: Scheme 3 dense masks with a TCMonoDepth prior."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from models.segmentation.hand_segmentor.adapter import DEFAULT_CHECKPOINT as DEFAULT_HAND_CHECKPOINT
from models.segmentation.scheme4_dense.adapter import DEFAULT_CHECKPOINT

from .scheme3 import DEFAULT_CLIP_DIR, Scheme3Config, run_scheme3


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/scheme4"


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
    config = replace(
        Scheme3Config(),
        scheme_name="scheme4",
        checkpoint=args.checkpoint,
        hand_checkpoint=args.hand_checkpoint,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        device=args.device,
        batch_size=args.batch_size,
        object_threshold=args.object_threshold,
        hand_threshold=args.hand_threshold,
    )
    for summary in run_scheme3(args.clip_dir.resolve(), args.output_root.resolve(), config):
        print(summary)


if __name__ == "__main__":
    main()
