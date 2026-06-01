#!/usr/bin/env python3
"""Train/evaluate a binary VISOR hand-glove segmentor."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
)

from common import (
    ROOT,
    FrameKey,
    load_frames_by_index,
    mask_metrics,
    rasterize_hand_mask,
    read_rgb,
    sparse_frame_path,
    split_frame_keys,
    summarize_frame_metrics,
    write_json,
)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--image-size", default="512x912")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/experiments/hand_segmentor/cache_512x912")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/hand_segmentor/deeplab_r50_512")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, help="Optional checkpoint to initialize training without overwriting it.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--model", choices=("mobilenetv3", "resnet50"), default="resnet50")
    parser.add_argument("--tversky-beta", type=float, default=0.5, help="Higher values penalize false negatives more.")
    parser.add_argument("--thresholds", default="0.35,0.4,0.45,0.5,0.55,0.6,0.65")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-empty-train-frames", action="store_true")
    parser.add_argument("--write-masks", action="store_true")
    parser.add_argument("--tta-flip", action="store_true", help="Average predictions with a horizontal flip at evaluation time.")
    args = parser.parse_args()

    set_seed(args.seed)
    size = parse_size(args.image_size)
    thresholds = [float(value) for value in args.thresholds.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        train_keys = frame_keys_for_splits(args.train_split, hand_only=not args.include_empty_train_frames)
        val_keys = frame_keys_for_splits(args.val_split, hand_only=True)
        ensure_cache(args.cache_dir, size, train_keys + val_keys)
        write_json(args.output_dir / "split_summary.json", split_summary(train_keys, val_keys, []))
        model = build_model(args.model, args.device)
        if args.init_checkpoint is not None:
            checkpoint = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
        train_model(model, args, train_keys, val_keys, thresholds)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in eval mode")
        eval_keys = frame_keys_for_splits(args.eval_split, hand_only=True)
        ensure_cache(args.cache_dir, size, eval_keys)
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model = build_model(args.model, args.device)
        model.load_state_dict(checkpoint["model"])
        threshold = float(checkpoint["threshold"])
        write_dir = args.output_dir / args.eval_split if args.write_masks else None
        summary = evaluate(model, args.cache_dir, eval_keys, args.device, [threshold], write_masks=write_dir, tta_flip=args.tta_flip)
        write_json(args.output_dir / f"{args.eval_split}_summary.json", summary)
        print_summary(summary)


def parse_size(value: str) -> tuple[int, int]:
    height, width = value.lower().split("x", maxsplit=1)
    return int(height), int(width)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def frame_keys_for_splits(value: str, *, hand_only: bool) -> list[FrameKey]:
    keys = []
    seen = set()
    for split in (part.strip() for part in value.split(",") if part.strip()):
        for key in split_frame_keys(split, hand_only=hand_only):
            ident = (key.video_id, key.frame_index)
            if ident not in seen:
                keys.append(key)
                seen.add(ident)
    return keys


def split_summary(train_keys: list[FrameKey], val_keys: list[FrameKey], test_keys: list[FrameKey]) -> dict:
    return {"train": count_by_video(train_keys), "val": count_by_video(val_keys), "test": count_by_video(test_keys)}


def count_by_video(keys: list[FrameKey]) -> dict[str, int]:
    counts = {}
    for key in keys:
        counts[key.video_id] = counts.get(key.video_id, 0) + 1
    return dict(sorted(counts.items()))


def cache_path(cache_dir: Path, key: FrameKey) -> Path:
    return cache_dir / f"{key.stem}.npz"


def ensure_cache(cache_dir: Path, size: tuple[int, int], frame_keys: list[FrameKey]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [key for key in frame_keys if not cache_path(cache_dir, key).exists()]
    if not missing:
        return
    current_video = None
    frames_by_index = None
    for offset, key in enumerate(sorted(missing, key=lambda item: (item.video_id, item.frame_index)), start=1):
        if key.video_id != current_video:
            current_video = key.video_id
            frames_by_index = load_frames_by_index(key.video_id)
        rgb = read_rgb(sparse_frame_path(key))
        mask = rasterize_hand_mask(frames_by_index[key.frame_index], rgb.shape[:2])
        image_small = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        np.savez_compressed(cache_path(cache_dir, key), image=image_small, mask=mask_small)
        if offset % 100 == 0:
            print(f"cached {offset}/{len(missing)} {key.stem}", flush=True)


class HandDataset(Dataset):
    def __init__(self, cache_dir: Path, frame_keys: list[FrameKey], augment: bool):
        self.cache_dir = cache_dir
        self.frame_keys = frame_keys
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame_keys)

    def __getitem__(self, index: int):
        key = self.frame_keys[index]
        data = np.load(cache_path(self.cache_dir, key))
        image = data["image"].copy()
        mask = data["mask"].copy()
        if self.augment:
            image, mask = augment_pair(image, mask)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - MEAN) / STD
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return image_t, mask_t, key.video_id, key.frame_index


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
    mask_scaled = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
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


def build_model(model_name: str, device: str) -> nn.Module:
    if model_name == "resnet50":
        model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT, aux_loss=True)
    else:
        model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(model.aux_classifier[-1].in_channels, 1, kernel_size=1)
    return model.to(device)


def model_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    logits = output.logits if hasattr(output, "logits") else output["out"]
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def aux_logits(output, images: torch.Tensor) -> torch.Tensor | None:
    if isinstance(output, dict) and "aux" in output:
        logits = output["aux"]
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits
    return None


def train_model(model: nn.Module, args, train_keys: list[FrameKey], val_keys: list[FrameKey], thresholds: list[float]) -> None:
    if len(train_keys) < args.batch_size:
        raise ValueError(f"Not enough train frames: {len(train_keys)}")
    train_loader = DataLoader(
        HandDataset(args.cache_dir, train_keys, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = {"mean_iou": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, masks, _, _ in train_loader:
            images = images.to(args.device, non_blocking=True)
            masks = masks.to(args.device, non_blocking=True)
            output = model(images)
            logits = output.logits if hasattr(output, "logits") else output["out"]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits, masks, beta=args.tversky_beta)
            aux = aux_logits(output, images)
            if aux is not None:
                loss = loss + 0.35 * segmentation_loss(aux, masks, beta=args.tversky_beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        summary = evaluate(model, args.cache_dir, val_keys, args.device, thresholds, write_masks=None, tta_flip=args.tta_flip)
        top = summary["best"]
        print(f"epoch {epoch} loss={np.mean(losses):.4f} val IoU={top['mean_iou']:.3f} P={top['mean_precision']:.3f} R={top['mean_recall']:.3f} th={top['threshold']}")
        if top["mean_iou"] > best["mean_iou"]:
            best = top
            torch.save({"model": model.state_dict(), "threshold": top["threshold"], "val_summary": summary}, args.output_dir / "best.pt")
            write_json(args.output_dir / "val_summary.json", summary)


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor, *, beta: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probs = torch.sigmoid(logits)
    true_pos = (probs * masks).sum(dim=(1, 2, 3))
    false_pos = (probs * (1.0 - masks)).sum(dim=(1, 2, 3))
    false_neg = ((1.0 - probs) * masks).sum(dim=(1, 2, 3))
    alpha = 1.0 - beta
    tversky = (true_pos + 1.0) / (true_pos + alpha * false_pos + beta * false_neg + 1.0)
    return bce + (1.0 - tversky.mean())


def predict_probs(model: nn.Module, images: torch.Tensor, *, tta_flip: bool) -> np.ndarray:
    logits = model_logits(model, images)
    if tta_flip:
        flipped = torch.flip(images, dims=(-1,))
        flip_logits = torch.flip(model_logits(model, flipped), dims=(-1,))
        logits = 0.5 * (logits + flip_logits)
    return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


def evaluate(model: nn.Module, cache_dir: Path, frame_keys: list[FrameKey], device: str, thresholds: list[float], write_masks: Path | None, tta_flip: bool = False) -> dict:
    loader = DataLoader(HandDataset(cache_dir, frame_keys, augment=False), batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    records_by_threshold = {threshold: [] for threshold in thresholds}
    with torch.inference_mode():
        for images, masks, video_ids, frame_indices in loader:
            probs = predict_probs(model, images.to(device), tta_flip=tta_flip)
            gt = masks[0, 0].numpy() > 0
            key = FrameKey(video_ids[0], int(frame_indices[0]))
            for threshold in thresholds:
                pred = probs >= threshold
                records_by_threshold[threshold].append(frame_record(key, pred, gt))
                if write_masks is not None:
                    out_dir = write_masks / f"threshold_{threshold:.2f}" / "masks"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_dir / f"{key.stem}.png"), pred.astype(np.uint8) * 255)
    rows = [{"threshold": threshold, **summarize_frame_metrics(records), "per_video": per_video(records)} for threshold, records in records_by_threshold.items()]
    rows.sort(key=lambda row: (row["mean_iou"], row["mean_precision"]), reverse=True)
    return {"best": rows[0], "thresholds": rows}


def frame_record(key: FrameKey, pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        "video_id": key.video_id,
        "frame_index": key.frame_index,
        "metrics": mask_metrics(pred, gt),
        "prediction_pixels": int(np.count_nonzero(pred)),
        "ground_truth_pixels": int(np.count_nonzero(gt)),
        "detections": int(np.count_nonzero(pred) > 0),
    }


def per_video(records: list[dict]) -> dict[str, dict]:
    return {
        video_id: summarize_frame_metrics([record for record in records if record["video_id"] == video_id])
        for video_id in sorted({record["video_id"] for record in records})
    }


def print_summary(summary: dict) -> None:
    best = summary["best"]
    print(f"best threshold={best['threshold']} IoU={best['mean_iou']:.3f} P={best['mean_precision']:.3f} R={best['mean_recall']:.3f}")
    print(json.dumps(best["per_video"], indent=2))


if __name__ == "__main__":
    main()
