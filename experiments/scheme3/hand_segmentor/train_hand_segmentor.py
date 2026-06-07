#!/usr/bin/env python3
"""Train/evaluate the EgoHOS hand segmentor used by Scheme 3."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCHEME_DIR = Path(__file__).resolve().parents[1]
if str(SCHEME_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEME_DIR))

from hand_segmentor.model import build_model


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "outputs/experiments/scheme3"
DEFAULT_DATA_ROOT = ROOT / "data/egohos/data"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "hand_segmentor"
DEFAULT_HAND_LABELS = "1,2"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to evaluate or resume from.")
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize training from a checkpoint without overwriting it.")
    parser.add_argument("--model", choices=("smp-unetpp-efficientnet-b4", "smp-unetpp-resnet101", "smp-deeplabv3plus-resnet101"), default="smp-unetpp-efficientnet-b4")
    parser.add_argument("--encoder-weights", default="imagenet", help="SMP encoder weights for new training; use 'none' to disable.")
    parser.add_argument("--image-size", default="512x704", help="HEIGHTxWIDTH input size.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tversky-beta", type=float, default=0.55, help="Higher values penalize false negatives more.")
    parser.add_argument("--thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75")
    parser.add_argument(
        "--positive-labels",
        default=DEFAULT_HAND_LABELS,
        help="Comma-separated EgoHOS label ids treated as positive. Default 1,2 = left/right hand.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast/mixed precision during training.")
    parser.add_argument("--tta-flip", action="store_true", help="Average predictions with a horizontal flip at evaluation time.")
    parser.add_argument("--write-masks", action="store_true", help="Write predicted masks during eval.")
    args = parser.parse_args()

    set_seed(args.seed)
    image_size = parse_size(args.image_size)
    thresholds = parse_thresholds(args.thresholds)
    positive_labels = parse_positive_labels(args.positive_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        train_items = dataset_items(args.data_root, args.train_split)
        val_items = dataset_items(args.data_root, args.val_split)
        write_json(args.output_dir / "split_summary.json", split_summary(args.train_split, train_items, args.val_split, val_items, positive_labels))
        model = build_model(args.model, args.device, encoder_weights=encoder_weights(args.encoder_weights))
        if args.init_checkpoint is not None:
            load_model_state(model, args.init_checkpoint, args.device)
        train_model(model, args, train_items, val_items, image_size, thresholds, positive_labels)
        return

    checkpoint_path = args.checkpoint or args.output_dir / "best.pt"
    eval_items = dataset_items(args.data_root, args.eval_split)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model_name = str(checkpoint.get("model_name", args.model))
    model = build_model(model_name, args.device, encoder_weights=None)
    model.load_state_dict(checkpoint["model"])
    checkpoint_size = parse_size(str(checkpoint.get("image_size", args.image_size)))
    threshold = float(checkpoint.get("threshold", thresholds[-1]))
    positive_labels = tuple(int(label) for label in checkpoint.get("positive_labels", positive_labels))
    write_dir = args.output_dir / args.eval_split / "masks" if args.write_masks else None
    summary = evaluate(model, eval_items, checkpoint_size, args.device, [threshold], args.workers, positive_labels, write_dir=write_dir, tta_flip=args.tta_flip)
    write_json(args.output_dir / f"{args.eval_split}_summary.json", summary)
    if args.eval_split == "val":
        write_json(args.output_dir / "dev_summary.json", summary)
    print_summary(summary)


def parse_size(value: str) -> tuple[int, int]:
    height, width = value.lower().split("x", maxsplit=1)
    size = int(height), int(width)
    if size[0] % 32 != 0 or size[1] % 32 != 0:
        raise ValueError(f"Image size must be divisible by 32 for the SMP encoders, got {value}")
    return size


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part) for part in value.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


def parse_positive_labels(value: str) -> tuple[int, ...]:
    labels = tuple(int(part) for part in value.split(",") if part.strip())
    if not labels:
        raise ValueError("At least one positive EgoHOS label is required")
    return labels


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def encoder_weights(value: str) -> str | None:
    if value.lower() in {"", "none", "null", "false"}:
        return None
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset_items(data_root: Path, split_value: str) -> list[tuple[Path, Path, str]]:
    items: list[tuple[Path, Path, str]] = []
    for split in [part.strip() for part in split_value.split(",") if part.strip()]:
        image_dir = data_root / split / "image"
        label_dir = data_root / split / "label"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing EgoHOS split directories: {image_dir} and {label_dir}")
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = label_dir / f"{image_path.stem}.png"
            if label_path.exists():
                items.append((image_path, label_path, split))
    if not items:
        raise RuntimeError(f"No EgoHOS image/label pairs found for split(s): {split_value}")
    return items


def split_summary(
    train_split: str,
    train_items: list[tuple[Path, Path, str]],
    val_split: str,
    val_items: list[tuple[Path, Path, str]],
    positive_labels: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "train_split": train_split,
        "train_items": len(train_items),
        "val_split": val_split,
        "val_items": len(val_items),
        "positive_labels": list(positive_labels),
        "target": "egohos_hand_only",
        "train_by_split": count_by_split(train_items),
        "val_by_split": count_by_split(val_items),
    }


def count_by_split(items: list[tuple[Path, Path, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, _, split in items:
        counts[split] = counts.get(split, 0) + 1
    return dict(sorted(counts.items()))


class EgoHOSDataset(Dataset):
    def __init__(self, items: list[tuple[Path, Path, str]], image_size: tuple[int, int], augment: bool, positive_labels: tuple[int, ...]):
        self.items = items
        self.image_size = image_size
        self.augment = augment
        self.positive_labels = positive_labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label_path, split = self.items[index]
        image = read_rgb(image_path)
        mask = read_mask(label_path, self.positive_labels)
        if self.augment:
            image, mask = augment_pair(image, mask)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - MEAN) / STD
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_t, mask_t, image_path.stem, split


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, positive_labels: tuple[int, ...]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    return np.isin(mask, np.asarray(positive_labels, dtype=mask.dtype))


def augment_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image, mask = random_zoom_pair(image, mask)
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < 0.85:
        alpha = random.uniform(0.75, 1.25)
        beta = random.uniform(-28, 28)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.35:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.8, 1.2), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if random.random() < 0.25:
        ksize = random.choice((3, 5))
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return image, mask


def random_zoom_pair(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = random.uniform(0.82, 1.18)
    new_height = max(8, int(round(height * scale)))
    new_width = max(8, int(round(width * scale)))
    image_scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    mask_scaled = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    if scale >= 1.0:
        top = random.randint(0, new_height - height)
        left = random.randint(0, new_width - width)
        return image_scaled[top : top + height, left : left + width], mask_scaled[top : top + height, left : left + width]
    top = random.randint(0, height - new_height)
    left = random.randint(0, width - new_width)
    image_out = np.zeros_like(image)
    mask_out = np.zeros_like(mask)
    image_out[top : top + new_height, left : left + new_width] = image_scaled
    mask_out[top : top + new_height, left : left + new_width] = mask_scaled
    return image_out, mask_out


def load_model_state(model: nn.Module, checkpoint_path: Path, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])


def train_model(
    model: nn.Module,
    args: argparse.Namespace,
    train_items: list[tuple[Path, Path, str]],
    val_items: list[tuple[Path, Path, str]],
    image_size: tuple[int, int],
    thresholds: list[float],
    positive_labels: tuple[int, ...],
) -> None:
    if len(train_items) < args.batch_size:
        raise ValueError(f"Not enough train frames for batch size {args.batch_size}: {len(train_items)}")
    train_loader = DataLoader(
        EgoHOSDataset(train_items, image_size, augment=True, positive_labels=positive_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device.startswith("cuda"))
    best = {"mean_iou": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, masks, _, _ in train_loader:
            images = images.to(args.device, non_blocking=True)
            masks = masks.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and args.device.startswith("cuda")):
                logits = model_logits(model, images, masks.shape[-2:])
                loss = segmentation_loss(logits, masks, beta=args.tversky_beta)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        summary = evaluate(model, val_items, image_size, args.device, thresholds, args.workers, positive_labels, write_dir=None, tta_flip=args.tta_flip)
        top = summary["best"]
        print(
            f"epoch {epoch} loss={np.mean(losses):.4f} "
            f"dev IoU={top['mean_iou']:.4f} P={top['mean_precision']:.4f} "
            f"R={top['mean_recall']:.4f} th={top['threshold']:.2f}",
            flush=True,
        )
        if top["mean_iou"] > best["mean_iou"]:
            best = top
            checkpoint = {
                "model_name": args.model,
                "model": model.state_dict(),
                "image_size": args.image_size,
                "positive_labels": list(positive_labels),
                "target": "egohos_hand_only",
                "threshold": float(top["threshold"]),
                "epoch": epoch,
                "val_summary": summary,
                "train_args": serializable_args(args),
            }
            torch.save(checkpoint, args.output_dir / "best.pt")
            write_json(args.output_dir / "dev_summary.json", summary)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = {}
    for key, value in vars(args).items():
        payload[key] = str(value) if isinstance(value, Path) else value
    return payload


def model_logits(model: nn.Module, images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    logits = model(images)
    if isinstance(logits, dict):
        logits = logits["out"]
    if logits.shape[-2:] != size:
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor, *, beta: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probs = torch.sigmoid(logits)
    true_pos = (probs * masks).sum(dim=(1, 2, 3))
    false_pos = (probs * (1.0 - masks)).sum(dim=(1, 2, 3))
    false_neg = ((1.0 - probs) * masks).sum(dim=(1, 2, 3))
    alpha = 1.0 - beta
    tversky = (true_pos + 1.0) / (true_pos + alpha * false_pos + beta * false_neg + 1.0)
    return bce + (1.0 - tversky.mean())


@torch.inference_mode()
def predict_probs(model: nn.Module, images: torch.Tensor, *, tta_flip: bool) -> torch.Tensor:
    logits = model_logits(model, images, images.shape[-2:])
    if tta_flip:
        flipped = torch.flip(images, dims=(-1,))
        flip_logits = torch.flip(model_logits(model, flipped, images.shape[-2:]), dims=(-1,))
        logits = 0.5 * (logits + flip_logits)
    return torch.sigmoid(logits)


def evaluate(
    model: nn.Module,
    items: list[tuple[Path, Path, str]],
    image_size: tuple[int, int],
    device: str,
    thresholds: list[float],
    workers: int,
    positive_labels: tuple[int, ...],
    *,
    write_dir: Path | None,
    tta_flip: bool = False,
) -> dict[str, Any]:
    loader = DataLoader(EgoHOSDataset(items, image_size, augment=False, positive_labels=positive_labels), batch_size=1, shuffle=False, num_workers=workers)
    model.eval()
    records_by_threshold: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds}
    for images, masks, stems, splits in loader:
        images = images.to(device, non_blocking=True)
        probs = predict_probs(model, images, tta_flip=tta_flip)[0, 0].detach().cpu().numpy()
        gt = masks[0, 0].numpy() > 0
        stem = str(stems[0])
        split = str(splits[0])
        for threshold in thresholds:
            pred = probs >= threshold
            records_by_threshold[threshold].append(frame_record(stem, split, pred, gt))
            if write_dir is not None:
                threshold_dir = write_dir / f"threshold_{threshold:.2f}"
                threshold_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(threshold_dir / f"{stem}.png"), pred.astype(np.uint8) * 255)
    rows = [{"threshold": threshold, **summarize_records(records), "by_split": per_split(records)} for threshold, records in records_by_threshold.items()]
    rows.sort(key=lambda row: (row["mean_iou"], row["mean_precision"]), reverse=True)
    return {"best": rows[0], "thresholds": rows, "positive_labels": list(positive_labels), "target": "egohos_hand_only"}


def frame_record(stem: str, split: str, pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    return {
        "stem": stem,
        "split": split,
        "metrics": mask_metrics(pred, gt),
        "prediction_pixels": int(np.count_nonzero(pred)),
        "ground_truth_pixels": int(np.count_nonzero(gt)),
        "detections": int(np.count_nonzero(pred) > 0),
    }


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | None]:
    intersection = int(np.count_nonzero(pred & gt))
    pred_count = int(np.count_nonzero(pred))
    gt_count = int(np.count_nonzero(gt))
    union = int(np.count_nonzero(pred | gt))
    return {
        "iou": intersection / union if union else None,
        "precision": intersection / pred_count if pred_count else (None if gt_count == 0 else 0.0),
        "recall": intersection / gt_count if gt_count else (None if pred_count == 0 else 0.0),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    metrics = [record["metrics"] for record in records]
    detections = [record["detections"] for record in records]
    return {
        "frames": len(records),
        "mean_iou": mean_metric(metrics, "iou"),
        "mean_precision": mean_metric(metrics, "precision"),
        "mean_recall": mean_metric(metrics, "recall"),
        "mean_detections": float(np.mean(detections)) if detections else 0.0,
        "empty_frames": sum(1 for count in detections if count == 0),
    }


def mean_metric(metrics: list[dict[str, float | None]], key: str) -> float:
    values = [float(metric[key]) for metric in metrics if metric[key] is not None]
    return float(np.mean(values)) if values else 0.0


def per_split(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    return {
        split: summarize_records([record for record in records if record["split"] == split])
        for split in sorted({record["split"] for record in records})
    }


def print_summary(summary: dict[str, Any]) -> None:
    best = summary["best"]
    print(
        f"best threshold={best['threshold']:.2f} "
        f"IoU={best['mean_iou']:.4f} "
        f"P={best['mean_precision']:.4f} "
        f"R={best['mean_recall']:.4f}"
    )
    print(json.dumps(best["by_split"], indent=2))


if __name__ == "__main__":
    main()
