"""Train the reusable hand segmentor on EgoHOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.egohos.evaluation import DEFAULT_DATA_ROOT, HAND_LABEL_IDS, list_egohos_items
from datasets.egohos.training import EgoHOSTrainingDataset
from models.segmentation.hand_segmentor.model import MODEL_BUILDERS, build_model, model_logits, normalize_images


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/models/hand_segmentor"
DEFAULT_THRESHOLDS = "0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="smp-unetpp-efficientnet-b4")
    parser.add_argument("--encoder-weights", default="imagenet", help="Use 'none' to disable pretrained encoder weights.")
    parser.add_argument("--image-size", default="512x704", help="HEIGHTxWIDTH, each divisible by 32.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tversky-beta", type=float, default=0.55)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-frames", type=int)
    parser.add_argument("--max-val-frames", type=int)
    parser.add_argument("--dev-run", action="store_true", help="Run one small epoch for integration testing.")
    args = parser.parse_args()

    if args.dev_run:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 2)
        args.workers = 0
        args.image_size = "64x64"
        args.max_train_frames = args.max_train_frames or 4
        args.max_val_frames = args.max_val_frames or 4
        args.thresholds = "0.5"
        if args.init_checkpoint is None:
            args.encoder_weights = "none"

    train(args)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    image_size = parse_size(args.image_size)
    thresholds = parse_thresholds(args.thresholds)
    train_items = list_egohos_items((args.train_split,), data_root=args.data_root, max_frames=args.max_train_frames)
    val_items = list_egohos_items((args.val_split,), data_root=args.data_root, max_frames=args.max_val_frames)
    if len(train_items) < args.batch_size:
        raise ValueError(f"Not enough training frames for batch size {args.batch_size}: {len(train_items)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "data_summary.json",
        {
            "train_split": args.train_split,
            "train_frames": len(train_items),
            "val_split": args.val_split,
            "val_frames": len(val_items),
            "positive_labels": list(HAND_LABEL_IDS),
        },
    )

    model = build_model(args.model, device, encoder_weights=parse_encoder_weights(args.encoder_weights))
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])

    train_loader = DataLoader(
        EgoHOSTrainingDataset(train_items, image_size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        EgoHOSTrainingDataset(val_items, image_size, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, amp_enabled, args.tversky_beta)
        validation = evaluate(model, val_loader, device, thresholds)
        top = validation["best"]
        row = {"epoch": epoch, "train_loss": train_loss, "validation": validation}
        history.append(row)
        print(
            f"epoch {epoch} loss={train_loss:.4f} "
            f"val IoU={top['mean_iou']:.4f} P={top['mean_precision']:.4f} "
            f"R={top['mean_recall']:.4f} threshold={top['threshold']:.2f}",
            flush=True,
        )
        if top["mean_iou"] > best_iou:
            best_iou = float(top["mean_iou"])
            torch.save(
                {
                    "model_name": args.model,
                    "model": model.state_dict(),
                    "image_size": args.image_size,
                    "positive_labels": list(HAND_LABEL_IDS),
                    "target": "egohos_hand_only",
                    "threshold": float(top["threshold"]),
                    "epoch": epoch,
                    "val_summary": validation,
                    "train_args": serializable_args(args),
                },
                args.output_dir / "best.pt",
            )
        write_json(args.output_dir / "summary.json", {"best_mean_iou": best_iou, "history": history})


def train_epoch(model, loader, optimizer, scaler, device, amp_enabled: bool, tversky_beta: float) -> float:
    model.train()
    losses = []
    for images, masks in loader:
        images = normalize_images(images.to(device, non_blocking=True))
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss = segmentation_loss(model_logits(model, images, masks.shape[-2:]), masks, beta=tversky_beta)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.inference_mode()
def evaluate(model, loader, device, thresholds: list[float]) -> dict[str, Any]:
    model.eval()
    metrics = {threshold: [] for threshold in thresholds}
    for images, masks in loader:
        images = normalize_images(images.to(device, non_blocking=True))
        probs = torch.sigmoid(model_logits(model, images, masks.shape[-2:])).cpu().numpy()[:, 0]
        ground_truths = masks.numpy()[:, 0].astype(bool)
        for threshold in thresholds:
            for prediction, ground_truth in zip(probs >= threshold, ground_truths):
                metrics[threshold].append(mask_metrics(prediction, ground_truth))
    rows = [{"threshold": threshold, **summarize(rows)} for threshold, rows in metrics.items()]
    rows.sort(key=lambda row: (row["mean_iou"], row["mean_precision"]), reverse=True)
    return {"best": rows[0], "thresholds": rows, "positive_labels": list(HAND_LABEL_IDS), "target": "egohos_hand_only"}


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor, *, beta: float) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
    probs = torch.sigmoid(logits)
    true_pos = (probs * masks).sum(dim=(1, 2, 3))
    false_pos = (probs * (1.0 - masks)).sum(dim=(1, 2, 3))
    false_neg = ((1.0 - probs) * masks).sum(dim=(1, 2, 3))
    tversky = (true_pos + 1.0) / (true_pos + (1.0 - beta) * false_pos + beta * false_neg + 1.0)
    return bce + (1.0 - tversky.mean())


def mask_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float | None]:
    intersection = int(np.count_nonzero(prediction & ground_truth))
    pred_count = int(np.count_nonzero(prediction))
    gt_count = int(np.count_nonzero(ground_truth))
    union = int(np.count_nonzero(prediction | ground_truth))
    return {
        "iou": intersection / union if union else None,
        "precision": intersection / pred_count if pred_count else (None if gt_count == 0 else 0.0),
        "recall": intersection / gt_count if gt_count else (None if pred_count == 0 else 0.0),
    }


def summarize(rows: list[dict[str, float | None]]) -> dict[str, float | int]:
    return {
        "frames": len(rows),
        "mean_iou": mean_metric(rows, "iou"),
        "mean_precision": mean_metric(rows, "precision"),
        "mean_recall": mean_metric(rows, "recall"),
    }


def mean_metric(rows: list[dict[str, float | None]], key: str) -> float:
    values = [float(row[key]) for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else 0.0


def parse_size(value: str) -> tuple[int, int]:
    height, width = (int(part) for part in value.lower().split("x", maxsplit=1))
    if height % 32 or width % 32:
        raise ValueError(f"Image size must be divisible by 32, got {value}")
    return height, width


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(part) for part in value.split(",") if part.strip()]
    if not thresholds or any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("Expected comma-separated thresholds between 0 and 1")
    return thresholds


def parse_encoder_weights(value: str) -> str | None:
    return None if value.lower() in {"", "none", "null", "false"} else value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
