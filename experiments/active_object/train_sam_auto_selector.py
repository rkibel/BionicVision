#!/usr/bin/env python3
"""Train a lightweight selector over SAM automatic active-object proposals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch import nn

from common import ROOT, FrameKey, read_rgb, sparse_frame_path
from models.depth.tc_monodepth.adapter import TCMonoDepthEstimator
from train_hitl_segmentor import MEAN, STD, build_active_model, hand_overlay, load_hand_model, predict_hand_prob

HITL_ROOT = ROOT / "data/epic_kitchens/HITL/active_objects"
HAND_CACHE = ROOT / "outputs/experiments/active_object/hitl_cache_384"
DEPTH_CACHE = ROOT / "outputs/experiments/active_object/depth_cache_tcmono"
PROPOSAL_CACHE = ROOT / "outputs/experiments/active_object/sam_auto_proposal_cache"
SAM_CKPT = ROOT / "external/model_sources/segmentation/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/active_object/sam_auto_selector")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--points-per-side", type=int, default=24)
    parser.add_argument("--active-checkpoint", type=Path, default=ROOT / "outputs/experiments/active_object/hitl_deeplab_r50_hand_prior/best.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--depth-features", action="store_true")
    parser.add_argument("--proposal-cache-dir", type=Path, default=PROPOSAL_CACHE)
    parser.add_argument("--target", choices=("precision", "oracle"), default="precision")
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sam = build_sam_generator(args)
    hand_model = load_hand_model(args.device)
    active_model = load_active_model(args.active_checkpoint, args.device)
    depth_model = TCMonoDepthEstimator(device=args.device) if args.depth_features else None
    train_items = hitl_items("train", args.train_limit)
    eval_items = hitl_items("eval", args.eval_limit)
    print(f"train frames={len(train_items)} eval frames={len(eval_items)}", flush=True)

    x_train, y_train = collect_training_rows(sam, hand_model, active_model, depth_model, train_items, args)
    normalizer = {"mean": x_train.mean(axis=0).tolist(), "std": np.maximum(x_train.std(axis=0), 1e-6).tolist()}
    model = Selector(x_train.shape[1]).to(args.device)
    train_selector(model, x_train, y_train, normalizer, args)

    thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
    topks = [1, 2, 3, 5, 8]
    eval_predictions = score_eval_candidates(sam, hand_model, active_model, depth_model, model, normalizer, eval_items, args)
    summaries = []
    for threshold in thresholds:
        for topk in topks:
            rows = evaluate_scored(eval_predictions, threshold, topk)
            summary = {"threshold": threshold, "topk": topk, **summarize(rows)}
            summaries.append(summary)
            mp = summary["macro_positive"]
            print(f"thr={threshold:.2f} topk={topk} iou={mp['iou']:.3f} p={mp['precision']:.3f} r={mp['recall']:.3f}", flush=True)
    summaries.sort(key=lambda row: (row["macro_positive"]["recall"], row["macro_positive"]["iou"]), reverse=True)
    result = {"train_frames": len(train_items), "eval_frames": len(eval_items), "summaries": summaries, "best": summaries[0], "normalizer": normalizer}
    torch.save({"model": model.state_dict(), "normalizer": normalizer, "best": summaries[0]}, args.output_dir / "best.pt")
    (args.output_dir / "eval_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["best"], indent=2))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hitl_items(split: str, limit: int | None = None) -> list[dict]:
    items = [{"split": split, "key": FrameKey.from_stem(p.name), "mask_path": p / "active_object_mask.png"} for p in sorted((HITL_ROOT / split).glob("*")) if p.is_dir()]
    return items[:limit] if limit else items


def build_sam_generator(args) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    sam.to(device=args.device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
        crop_n_layers=0,
        min_mask_region_area=100,
    )


def load_active_model(path: Path, device: str) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_active_model(device, pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def collect_training_rows(generator, hand_model, active_model, depth_model, items: list[dict], args) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i, item in enumerate(items, start=1):
        image, gt, hand, active, depth = load_frame(item, hand_model, active_model, depth_model, args.device)
        proposals = load_or_generate_proposals(generator, image, item, args)
        masks = [proposal["segmentation"].astype(bool) for proposal in proposals]
        oracle_indices = greedy_oracle_indices(masks, gt, max_masks=8) if args.target == "oracle" else set()
        for proposal, mask_index in zip(proposals, range(len(masks))):
            mask = masks[mask_index]
            mask = proposal["segmentation"].astype(bool)
            row = features(mask, proposal, hand, active, depth)
            xs.append(row)
            ys.append(float(mask_index in oracle_indices) if args.target == "oracle" else mask_precision(mask, gt))
        if i % 25 == 0:
            print(f"train proposals {i}/{len(items)} rows={len(xs)}", flush=True)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def load_frame(item: dict, hand_model, active_model, depth_model, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    gt = cv2.imread(str(item["mask_path"]), cv2.IMREAD_GRAYSCALE) > 0
    image = read_rgb(sparse_frame_path(item["key"]))
    if image.shape[:2] != gt.shape:
        image = cv2.resize(image, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
    cache_path = HAND_CACHE / item["split"] / f"{item['key'].stem}.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        hand = cache["hand"].astype(np.float32)
        hand = cv2.resize(hand, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        active = predict_active_prob(active_model, cache["image"], cache["hand"].astype(np.float32), gt.shape, device)
    else:
        hand = predict_hand_prob(hand_model, image, device)
        active = predict_active_prob(active_model, image, hand, gt.shape, device)
    depth = predict_depth(item, image, gt.shape, depth_model) if depth_model is not None else None
    return image, gt, hand, active, depth


def load_or_generate_proposals(generator, image: np.ndarray, item: dict, args) -> list[dict]:
    cache_path = args.proposal_cache_dir / f"pps{args.points_per_side}" / item["split"] / f"{item['key'].stem}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        shape = tuple(int(x) for x in data["shape"])
        packed = data["masks"]
        masks = np.unpackbits(packed, axis=1, count=shape[1] * shape[2]).reshape(shape).astype(bool)
        return [
            {
                "segmentation": masks[index],
                "predicted_iou": float(data["predicted_iou"][index]),
                "stability_score": float(data["stability_score"][index]),
            }
            for index in range(shape[0])
        ]
    proposals = generator.generate(image)
    masks = np.asarray([p["segmentation"].astype(bool) for p in proposals], dtype=bool)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if len(masks):
        packed = np.packbits(masks.reshape(len(masks), -1), axis=1)
        predicted_iou = np.asarray([p.get("predicted_iou", 0.0) for p in proposals], dtype=np.float32)
        stability = np.asarray([p.get("stability_score", 0.0) for p in proposals], dtype=np.float32)
        np.savez_compressed(cache_path, masks=packed, shape=np.asarray(masks.shape), predicted_iou=predicted_iou, stability_score=stability)
    else:
        np.savez_compressed(cache_path, masks=np.zeros((0, 0), dtype=np.uint8), shape=np.asarray((0, *image.shape[:2])), predicted_iou=[], stability_score=[])
    return proposals


def predict_depth(item: dict, image_rgb: np.ndarray, out_shape: tuple[int, int], depth_model) -> np.ndarray:
    cache_path = DEPTH_CACHE / item["split"] / f"{item['key'].stem}.npz"
    if cache_path.exists():
        depth = np.load(cache_path)["depth"].astype(np.float32)
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        depth = depth_model.predict(image_bgr).astype(np.float32)
        np.savez_compressed(cache_path, depth=depth)
    if depth.shape != out_shape:
        depth = cv2.resize(depth, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth


@torch.no_grad()
def predict_active_prob(model, image: np.ndarray, hand: np.ndarray, out_shape: tuple[int, int], device: str) -> np.ndarray:
    overlay = hand_overlay(image.copy(), hand)
    tensor = torch.from_numpy(overlay).permute(2, 0, 1).float() / 255.0
    tensor = ((tensor - MEAN) / STD).unsqueeze(0).to(device)
    logits = model(tensor)["out"]
    logits = nn.functional.interpolate(logits, size=out_shape, mode="bilinear", align_corners=False)
    return torch.sigmoid(logits[0, 0]).cpu().numpy()


def features(mask: np.ndarray, proposal: dict, hand_prob: np.ndarray, active_prob: np.ndarray, depth: np.ndarray | None = None) -> list[float]:
    h, w = mask.shape
    area = max(1, int(mask.sum()))
    hand = hand_prob >= 0.45
    d8, d24, d64 = dilate(hand, 8), dilate(hand, 24), dilate(hand, 64)
    ys, xs = np.nonzero(mask)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    row = [
        area / (h * w),
        (x2 - x1) / w,
        (y2 - y1) / h,
        ((x1 + x2) / 2) / w,
        ((y1 + y2) / 2) / h,
        float(proposal.get("predicted_iou", 0.0)),
        float(proposal.get("stability_score", 0.0)),
        float((mask & hand).sum() / area),
        float((mask & d8).sum() / area),
        float((mask & d24).sum() / area),
        float((mask & d64).sum() / area),
        float((mask & d8 & ~hand).sum() / area),
        float((mask & d24 & ~hand).sum() / area),
        float(np.count_nonzero(mask[0]) + np.count_nonzero(mask[-1]) + np.count_nonzero(mask[:, 0]) + np.count_nonzero(mask[:, -1])) / area,
        float(active_prob[mask].mean()),
        float(np.percentile(active_prob[mask], 75)),
        float(np.percentile(active_prob[mask], 90)),
        float(active_prob[mask].max()),
    ]
    if depth is not None:
        hand_depth = depth[hand]
        hand_median = float(np.median(hand_depth)) if hand_depth.size else 0.5
        mask_depth = depth[mask]
        mask_median = float(np.median(mask_depth))
        contact = mask & d24
        contact_depth = depth[contact]
        contact_median = float(np.median(contact_depth)) if contact_depth.size else mask_median
        diff = np.abs(mask_depth - hand_median)
        row.extend(
            [
                float(mask_depth.mean()),
                mask_median,
                float(mask_depth.std()),
                float(np.percentile(mask_depth, 25)),
                float(np.percentile(mask_depth, 75)),
                hand_median,
                abs(mask_median - hand_median),
                abs(contact_median - hand_median),
                float((diff <= 0.03).mean()),
                float((diff <= 0.06).mean()),
                float((diff <= 0.10).mean()),
            ]
        )
    return row


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


class Selector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_selector(model: nn.Module, x: np.ndarray, y: np.ndarray, normalizer: dict, args) -> None:
    x_t = normalize(torch.from_numpy(x), normalizer).to(args.device)
    y_t = torch.from_numpy(y).to(args.device)
    target = (y_t >= 0.50).float()
    weight = torch.where(target > 0, torch.full_like(target, 12.0), torch.ones_like(target))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for _ in range(args.epochs):
        model.train()
        logits = model(x_t)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weight)
        mse = nn.functional.mse_loss(torch.sigmoid(logits), y_t)
        loss = bce + mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def score_eval_candidates(generator, hand_model, active_model, depth_model, model, normalizer: dict, items: list[dict], args) -> list[dict]:
    predictions = []
    model.eval()
    for index, item in enumerate(items, start=1):
        image, gt, hand, active, depth = load_frame(item, hand_model, active_model, depth_model, args.device)
        proposals = load_or_generate_proposals(generator, image, item, args)
        scored = []
        for proposal in proposals:
            mask = proposal["segmentation"].astype(bool)
            row = features(mask, proposal, hand, active, depth)
            x = normalize(torch.tensor([row], dtype=torch.float32), normalizer).to(args.device)
            score = float(torch.sigmoid(model(x))[0].cpu())
            scored.append((score, mask))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        predictions.append({"gt": gt, "scored": scored})
        if index % 10 == 0:
            print(f"eval proposals {index}/{len(items)}", flush=True)
    return predictions


def evaluate_scored(predictions: list[dict], threshold: float, topk: int) -> list[dict]:
    rows = []
    for item in predictions:
        gt = item["gt"]
        pred = np.zeros(gt.shape, dtype=bool)
        selected = [(score, mask) for score, mask in item["scored"] if score >= threshold][:topk]
        for _, mask in selected:
            pred |= mask
        rows.append(metric(pred, item["gt"]))
    return rows


def normalize(x: torch.Tensor, normalizer: dict) -> torch.Tensor:
    mean = torch.tensor(normalizer["mean"], dtype=x.dtype, device=x.device)
    std = torch.tensor(normalizer["std"], dtype=x.dtype, device=x.device)
    return (x - mean) / std


def mask_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    count = int(np.count_nonzero(pred))
    return int(np.count_nonzero(pred & gt)) / count if count else float(not np.any(gt))


def greedy_oracle_indices(masks: list[np.ndarray], gt: np.ndarray, max_masks: int) -> set[int]:
    pred = np.zeros(gt.shape, dtype=bool)
    remaining = set(range(len(masks)))
    selected: set[int] = set()
    best_score = metric(pred, gt)["iou"]
    for _ in range(max_masks):
        best_index, best_candidate = None, None
        for index in remaining:
            candidate = pred | masks[index]
            score = metric(candidate, gt)["iou"]
            if score > best_score + 1e-6:
                best_score, best_index, best_candidate = score, index, candidate
        if best_index is None:
            break
        pred = best_candidate
        remaining.remove(best_index)
        selected.add(best_index)
    return selected


def metric(pred: np.ndarray, gt: np.ndarray) -> dict:
    inter, union = int(np.count_nonzero(pred & gt)), int(np.count_nonzero(pred | gt))
    pc, gc = int(np.count_nonzero(pred)), int(np.count_nonzero(gt))
    return {"iou": inter / union if union else 1.0, "precision": inter / pc if pc else float(gc == 0), "recall": inter / gc if gc else float(pc == 0), "gt_pixels": gc}


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["gt_pixels"] > 0]
    return {"macro_all": avg(rows), "macro_positive": avg(pos), "positive_frames": len(pos), "empty_frames": len(rows) - len(pos)}


def avg(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) if rows else 0.0 for k in ("iou", "precision", "recall")}


if __name__ == "__main__":
    main()
