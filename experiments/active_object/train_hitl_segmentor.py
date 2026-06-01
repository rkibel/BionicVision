#!/usr/bin/env python3
"""Train/evaluate an active-object segmentor on HITL masks."""

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
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

from common import ROOT, FrameKey, read_rgb, sparse_frame_path

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
HITL_ROOT = ROOT / "data/epic_kitchens/HITL/active_objects"
HAND_CKPT = ROOT / "outputs/experiments/hand_segmentor/deeplab_r50_512/best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/active_object/hitl_deeplab_r50_hand_prior")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/experiments/active_object/hitl_cache_384")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image-size", default="384x684")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.75, help="Tversky beta; higher penalizes false negatives.")
    parser.add_argument("--thresholds", default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    size = parse_size(args.image_size)
    thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]

    if args.mode == "train":
        train_items = hitl_items("train")
        eval_items = hitl_items("eval")
        ensure_cache(args.cache_dir, size, train_items + eval_items, args.device)
        write_json(args.output_dir / "split_summary.json", split_summary(train_items, eval_items, hitl_items("test")))
        model = build_active_model(args.device)
        train(model, args, train_items, eval_items, thresholds)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in eval mode")
        items = hitl_items(args.eval_split)
        ensure_cache(args.cache_dir, size, items, args.device)
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model = build_active_model(args.device, pretrained=False)
        model.load_state_dict(checkpoint["model"])
        threshold = float(checkpoint["threshold"])
        summary = evaluate(model, args.cache_dir, items, args.device, [threshold])
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


def hitl_items(split: str) -> list[dict]:
    items = []
    for frame_dir in sorted((HITL_ROOT / split).glob("*")):
        if not frame_dir.is_dir():
            continue
        key = FrameKey.from_stem(frame_dir.name)
        items.append({"split": split, "key": key, "mask_path": frame_dir / "active_object_mask.png"})
    return items


def split_summary(train_items: list[dict], eval_items: list[dict], test_items: list[dict]) -> dict:
    return {"train": count_items(train_items), "eval": count_items(eval_items), "test": count_items(test_items)}


def count_items(items: list[dict]) -> dict:
    counts = {}
    positives = 0
    for item in items:
        counts[item["key"].video_id] = counts.get(item["key"].video_id, 0) + 1
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        positives += int(mask is not None and np.any(mask))
    return {"frames": len(items), "positive": positives, "empty": len(items) - positives, "by_video": dict(sorted(counts.items()))}


def cache_path(cache_dir: Path, item: dict) -> Path:
    return cache_dir / item["split"] / f"{item['key'].stem}.npz"


def ensure_cache(cache_dir: Path, size: tuple[int, int], items: list[dict], device: str) -> None:
    missing = [item for item in items if not cache_path(cache_dir, item).exists()]
    if not missing:
        return
    hand_model = load_hand_model(device)
    for i, item in enumerate(missing, start=1):
        key = item["key"]
        rgb = read_rgb(sparse_frame_path(key))
        mask = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Missing HITL mask: {item['mask_path']}")
        image = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        target = cv2.resize((mask > 0).astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        hand_prob = predict_hand_prob(hand_model, image, device)
        path = cache_path(cache_dir, item)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, image=image, mask=target, hand=hand_prob.astype(np.float16))
        if i % 50 == 0:
            print(f"cached {i}/{len(missing)}", flush=True)


def load_hand_model(device: str) -> nn.Module:
    checkpoint = torch.load(HAND_CKPT, map_location=device, weights_only=False)
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(model.aux_classifier[-1].in_channels, 1, kernel_size=1)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict_hand_prob(model: nn.Module, image: np.ndarray, device: str) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = ((tensor - MEAN) / STD).unsqueeze(0).to(device)
    logits = model(tensor)["out"]
    if logits.shape[-2:] != tensor.shape[-2:]:
        logits = F.interpolate(logits, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
    return torch.sigmoid(logits[0, 0]).cpu().numpy()


class ActiveDataset(Dataset):
    def __init__(self, cache_dir: Path, items: list[dict], augment: bool):
        self.cache_dir = cache_dir
        self.items = items
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        data = np.load(cache_path(self.cache_dir, item))
        image = data["image"].copy()
        mask = data["mask"].copy()
        hand = data["hand"].astype(np.float32).copy()
        if self.augment:
            image, mask, hand = augment_triplet(image, mask, hand)
        image = hand_overlay(image, hand)
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_t = (image_t - MEAN) / STD
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        return image_t, mask_t, item["key"].video_id, item["key"].frame_index


def hand_overlay(image: np.ndarray, hand: np.ndarray) -> np.ndarray:
    out = image.astype(np.float32)
    out[..., 1] = np.maximum(out[..., 1], hand * 255.0)
    out[..., 0] *= 1.0 - 0.25 * hand
    out[..., 2] *= 1.0 - 0.25 * hand
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_triplet(image: np.ndarray, mask: np.ndarray, hand: np.ndarray):
    if random.random() < 0.5:
        image, mask, hand = random_zoom(image, mask, hand)
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
        hand = np.ascontiguousarray(hand[:, ::-1])
    if random.random() < 0.85:
        image = np.clip(image.astype(np.float32) * random.uniform(0.75, 1.25) + random.uniform(-25, 25), 0, 255).astype(np.uint8)
    return image, mask, hand


def random_zoom(image: np.ndarray, mask: np.ndarray, hand: np.ndarray):
    h, w = mask.shape
    scale = random.uniform(0.82, 1.18)
    nh, nw = max(8, round(h * scale)), max(8, round(w * scale))
    image_s = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    hand_s = cv2.resize(hand, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if scale >= 1:
        top, left = random.randint(0, nh - h), random.randint(0, nw - w)
        return image_s[top : top + h, left : left + w], mask_s[top : top + h, left : left + w], hand_s[top : top + h, left : left + w]
    top, left = random.randint(0, h - nh), random.randint(0, w - nw)
    image_o, mask_o, hand_o = np.zeros_like(image), np.zeros_like(mask), np.zeros_like(hand)
    image_o[top : top + nh, left : left + nw] = image_s
    mask_o[top : top + nh, left : left + nw] = mask_s
    hand_o[top : top + nh, left : left + nw] = hand_s
    return image_o, mask_o, hand_o


def build_active_model(device: str, *, pretrained: bool = True) -> nn.Module:
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(model.aux_classifier[-1].in_channels, 1, kernel_size=1)
    return model.to(device)


def train(model: nn.Module, args, train_items: list[dict], eval_items: list[dict], thresholds: list[float]) -> None:
    loader = DataLoader(ActiveDataset(args.cache_dir, train_items, True), batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = {"score": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, masks, _, _ in loader:
            images, masks = images.to(args.device, non_blocking=True), masks.to(args.device, non_blocking=True)
            out = model(images)
            logits = out["out"]
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = segmentation_loss(logits, masks, args.beta)
            if "aux" in out:
                aux = F.interpolate(out["aux"], size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = loss + 0.4 * segmentation_loss(aux, masks, args.beta)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        summary = evaluate(model, args.cache_dir, eval_items, args.device, thresholds)
        chosen = max(summary["thresholds"], key=lambda row: (row["macro_positive"]["recall"], row["macro_positive"]["iou"]))
        score = chosen["macro_positive"]["recall"] + chosen["macro_positive"]["iou"]
        print(f"epoch {epoch:02d} loss {np.mean(losses):.4f} thr {chosen['threshold']:.2f} pos_iou {chosen['macro_positive']['iou']:.3f} pos_p {chosen['macro_positive']['precision']:.3f} pos_r {chosen['macro_positive']['recall']:.3f}", flush=True)
        if score > best["score"]:
            best = {"score": score, "threshold": chosen["threshold"], "summary": summary}
            torch.save({"model": model.state_dict(), "threshold": chosen["threshold"], "eval_summary": summary}, args.output_dir / "best.pt")
            write_json(args.output_dir / "eval_summary.json", summary)
    print("best", best["threshold"], best["summary"]["best"])


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor, beta: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (probs * masks).sum(dims)
    fp = (probs * (1 - masks)).sum(dims)
    fn = ((1 - probs) * masks).sum(dims)
    tversky = (tp + 1.0) / (tp + (1 - beta) * fp + beta * fn + 1.0)
    return bce + (1.0 - tversky).mean()


@torch.no_grad()
def evaluate(model: nn.Module, cache_dir: Path, items: list[dict], device: str, thresholds: list[float]) -> dict:
    model.eval()
    loader = DataLoader(ActiveDataset(cache_dir, items, False), batch_size=1, shuffle=False, num_workers=2)
    probs, targets, meta = [], [], []
    for images, masks, video_ids, frame_indices in loader:
        images = images.to(device)
        logits = model(images)["out"]
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        probs.append(torch.sigmoid(logits[0, 0]).cpu().numpy())
        targets.append(masks[0, 0].numpy().astype(bool))
        meta.append((video_ids[0], int(frame_indices[0])))
    rows = []
    for threshold in thresholds:
        records = [metric_record(prob >= threshold, gt, video_id, frame_index) for prob, gt, (video_id, frame_index) in zip(probs, targets, meta)]
        rows.append({"threshold": threshold, **summarize(records)})
    best = max(rows, key=lambda row: (row["macro_positive"]["recall"], row["macro_positive"]["iou"]))
    return {"frames": len(items), "thresholds": rows, "best": best}


def metric_record(pred: np.ndarray, gt: np.ndarray, video_id: str, frame_index: int) -> dict:
    inter = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    pc = int(np.count_nonzero(pred))
    gc = int(np.count_nonzero(gt))
    return {"video_id": video_id, "frame_index": frame_index, "iou": inter / union if union else 1.0, "precision": inter / pc if pc else float(gc == 0), "recall": inter / gc if gc else float(pc == 0), "gt_pixels": gc, "pred_pixels": pc}


def summarize(records: list[dict]) -> dict:
    pos = [r for r in records if r["gt_pixels"] > 0]
    empty = [r for r in records if r["gt_pixels"] == 0]
    return {"macro_all": average(records), "macro_positive": average(pos), "empty_frames": len(empty), "positive_frames": len(pos)}


def average(records: list[dict]) -> dict:
    if not records:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0}
    return {key: float(np.mean([r[key] for r in records])) for key in ("iou", "precision", "recall")}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def print_summary(summary: dict) -> None:
    print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":
    main()
