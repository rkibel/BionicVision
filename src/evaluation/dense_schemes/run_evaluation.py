"""Evaluate dense Scheme 3/4/5 masks on supervised data and temporal video stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

from datasets.egoexo import DEFAULT_CAMERA, EgoExoObjectDataset, take_video_path
from datasets.egohos import DEFAULT_DATA_ROOT as EGOHOS_ROOT, EgoHOSObjectDataset
from models.segmentation.hand_segmentor.adapter import DEFAULT_CHECKPOINT as DEFAULT_HAND_CHECKPOINT
from models.segmentation.scheme3_dense.adapter import DEFAULT_CHECKPOINT, Scheme3DenseSegmentor

from .metrics import evaluate_supervised_masks
from .temporal import DEFAULT_HORIZONS, evaluate_temporal_masks


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "outputs/evaluation/scheme3_dense/results.json"


def main(defaults: dict[str, str | None] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hand-checkpoint", type=Path, default=DEFAULT_HAND_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--egoexo-split", default="val")
    parser.add_argument("--egoexo-camera", default=DEFAULT_CAMERA)
    parser.add_argument("--egoexo-samples", type=int, default=180)
    parser.add_argument("--egohos-root", type=Path, default=EGOHOS_ROOT)
    parser.add_argument("--egohos-splits", default="val,test_indomain,test_outdomain")
    parser.add_argument("--egohos-samples", type=int, default=240)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--egoexo-take", default="sfu_cooking_008_3")
    parser.add_argument("--start-frame", type=int, default=3150)
    parser.add_argument("--max-video-frames", type=int, default=900)
    parser.add_argument("--flow-horizons", default="1,2,5,10,15,30")
    parser.add_argument("--flow-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(injected_argv(defaults or {}, sys.argv[1:]))

    segmentor = Scheme3DenseSegmentor(args.checkpoint, hand_checkpoint=args.hand_checkpoint, device=args.device)
    threshold = segmentor.threshold if args.threshold is None else args.threshold
    result = {
        "checkpoint": str(args.checkpoint),
        "hand_checkpoint": str(args.hand_checkpoint),
        "threshold": threshold,
        "supervised_metric": "Predicted object mask compared with dataset ground-truth object mask.",
        "temporal_metric": "Unsupervised mask stability: predicted mask at t compared with predicted mask at t+h, before and after optical-flow correction.",
        "egoexo_supervised": evaluate_dataset(
            segmentor,
            EgoExoObjectDataset(
                args.egoexo_split,
                camera_name=args.egoexo_camera,
                image_size=segmentor.config.image_size,
                max_samples=args.egoexo_samples,
            ),
            threshold,
            args.batch_size,
        ),
        "egohos_supervised": {
            split: evaluate_dataset(
                segmentor,
                EgoHOSObjectDataset(
                    split,
                    data_root=args.egohos_root,
                    image_size=segmentor.config.image_size,
                    max_samples=args.egohos_samples,
                ),
                threshold,
                args.batch_size,
            )
            for split in parse_csv(args.egohos_splits)
        },
    }
    video = args.video or take_video_path(args.egoexo_take, args.egoexo_camera)
    frames = extract_video_frames(video, args.start_frame, args.max_video_frames)
    segmentor.sequence_feature_context = segmentor.config.image_feature_mode == "glc_gaze"
    segmentor.reset_sequence()
    object_masks = predict_masks(segmentor, frames, threshold, args.batch_size)
    result["temporal_unsupervised"] = {
        "video": str(video),
        "start_frame": args.start_frame,
        **evaluate_temporal_masks(frames, object_masks, horizons=parse_ints(args.flow_horizons), flow_size=args.flow_size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def evaluate_dataset(segmentor: Scheme3DenseSegmentor, dataset, threshold: float, batch_size: int) -> dict:
    predictions, targets, sources = [], [], []
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        frames = [normalized_tensor_to_bgr(sample["image"]) for sample in samples]
        probabilities, _ = segmentor.predict_batch(frames)
        predictions.extend(probability >= threshold for probability in probabilities)
        targets.extend(sample["target"].numpy().astype(bool) for sample in samples)
        sources.extend(str(sample.get("source", "")) for sample in samples)
    metrics = evaluate_supervised_masks(predictions, targets)
    metrics["by_source"] = {
        source: evaluate_supervised_masks(
            [prediction for prediction, row_source in zip(predictions, sources) if row_source == source],
            [target for target, row_source in zip(targets, sources) if row_source == source],
        )
        for source in sorted(set(sources))
    }
    return metrics


def normalized_tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb = ((image * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def extract_video_frames(path: Path, start_frame: int, max_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames


def predict_masks(segmentor, frames, threshold, batch_size) -> list[np.ndarray]:
    masks = []
    for start in range(0, len(frames), batch_size):
        probabilities, _ = segmentor.predict_batch(frames[start : start + batch_size])
        masks.extend(probability >= threshold for probability in probabilities)
    return masks


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in parse_csv(value))


def injected_argv(defaults: dict[str, str | None], argv: list[str]) -> list[str]:
    injected = []
    for option, value in defaults.items():
        if not any(arg == option or arg.startswith(f"{option}=") for arg in argv):
            injected.append(option)
            if value is not None:
                injected.append(value)
    return [*injected, *argv]


if __name__ == "__main__":
    main()
