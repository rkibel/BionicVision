"""Evaluate the reusable hand segmentor on EgoHOS."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets.egohos import DEFAULT_DATA_ROOT, HAND_LABEL_IDS, list_egohos_items, load_egohos_sample
from models.segmentation.hand_segmentor.adapter import DEFAULT_CHECKPOINT, HandSegmentor

from .common import evaluate_items, parse_csv, parse_thresholds, print_summary, write_json


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/hand_segmentor/egohos.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", default="val,test_indomain,test_outdomain")
    parser.add_argument("--thresholds", help="Comma-separated thresholds; defaults to the checkpoint threshold.")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--write-masks", action="store_true")
    args = parser.parse_args()

    segmentor = HandSegmentor(checkpoint=args.checkpoint, device=args.device)
    splits = parse_csv(args.splits)
    thresholds = parse_thresholds(args.thresholds, segmentor.threshold)
    items = list_egohos_items(splits, data_root=args.data_root, max_frames=args.max_frames)
    write_dir = args.output.parent / "egohos_masks" if args.write_masks else None
    summary = evaluate_items(segmentor, items, load_egohos_sample, thresholds, batch_size=args.batch_size, write_dir=write_dir)
    summary.update(
        {
            "dataset": "egohos",
            "checkpoint": str(args.checkpoint),
            "checkpoint_threshold": segmentor.threshold,
            "positive_labels": list(HAND_LABEL_IDS),
            "splits": list(splits),
            "frame_cap": args.max_frames,
        }
    )
    write_json(args.output, summary)
    print_summary(summary)
    print(args.output)


if __name__ == "__main__":
    main()
