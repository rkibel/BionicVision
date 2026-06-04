#!/usr/bin/env python3
"""Train a frame-level set selector with a differentiable union-IoU loss."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import OUTPUT_DIR, write_json
from train_greedy_selector import TrackCache, aggregate_metrics, best_threshold, evaluate_cache, greedy_oracle_labels, unpack_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-track-cache", type=Path, nargs="+", required=True)
    parser.add_argument("--val-track-cache", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "track_score_model/set_selector.pt")
    parser.add_argument("--summary-output", type=Path, default=OUTPUT_DIR / "track_score_model/summary_set_selector.json")
    parser.add_argument("--threshold-grid", default="0.02:0.98:0.02")
    parser.add_argument("--max-oracle-select", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--mask-size", type=int, default=112)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--bce-weight", type=float, default=0.08)
    parser.add_argument("--count-weight", type=float, default=0.035)
    parser.add_argument("--temporal-weight", type=float, default=0.10)
    parser.add_argument("--count-penalty", type=float, default=0.015)
    parser.add_argument("--use-dinov2", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    train_caches = [TrackCache.load(path) for path in args.train_track_cache]
    val_caches = [TrackCache.load(path) for path in args.val_track_cache]
    for cache in train_caches + val_caches:
        cache.greedy_labels = greedy_oracle_labels(cache, args.max_oracle_select)

    feature_blocks = [cache_features(cache, args.use_dinov2) for cache in train_caches]
    mean, std = fit_scaler(feature_blocks)
    train_samples = [
        sample
        for cache, features in zip(train_caches, feature_blocks)
        for sample in build_samples(cache, features, mean, std, args.mask_size)
    ]
    val_feature_blocks = [cache_features(cache, args.use_dinov2) for cache in val_caches]
    val_samples = [
        sample
        for cache, features in zip(val_caches, val_feature_blocks)
        for sample in build_samples(cache, features, mean, std, args.mask_size)
    ]

    model = SetSelector(input_dim=train_samples[0].features.shape[1], hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state = None
    best_train = None
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        losses = []
        model.train()
        for sample in train_samples:
            loss = frame_loss(model, sample, args.device, args.bce_weight, args.count_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            train_scores = score_caches(model, train_caches, [cache_features(cache, args.use_dinov2) for cache in train_caches], mean, std, args.device)
            thresholds = parse_grid(args.threshold_grid)
            train_metrics = best_threshold(train_caches, train_scores, thresholds, args.temporal_weight, args.count_penalty)
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": float(np.mean(losses)) if losses else 0.0,
                        "train_iou": train_metrics["selected_union_mean_iou"],
                        "train_temporal": train_metrics["selected_temporal_union_iou"],
                        "threshold": train_metrics["threshold"],
                    }
                ),
                flush=True,
            )
            if best_train is None or train_metrics["objective"] > best_train["objective"]:
                best_train = train_metrics
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    assert best_state is not None and best_train is not None
    model.load_state_dict(best_state)
    thresholds = parse_grid(args.threshold_grid)
    train_scores = score_caches(model, train_caches, [cache_features(cache, args.use_dinov2) for cache in train_caches], mean, std, args.device)
    best_train = best_threshold(train_caches, train_scores, thresholds, args.temporal_weight, args.count_penalty)

    summary = {
        "model": str(args.output),
        "model_kind": "set_selector_soft_union_iou",
        "train_track_caches": [str(path) for path in args.train_track_cache],
        "val_track_caches": [str(path) for path in args.val_track_cache],
        "feature_dim": int(train_samples[0].features.shape[1]),
        "use_dinov2": bool(args.use_dinov2),
        "train_frames": len(train_samples),
        "val_frames": len(val_samples),
        "epochs": args.epochs,
        "mask_size": args.mask_size,
        "loss": "soft_union_iou + bce_weight * greedy_bce + count_weight * count_mse",
        "bce_weight": args.bce_weight,
        "count_weight": args.count_weight,
        "best_train_threshold": best_train,
    }
    if val_caches:
        val_scores = score_caches(model, val_caches, val_feature_blocks, mean, std, args.device)
        summary["val_at_train_threshold"] = aggregate_metrics(
            [evaluate_cache(cache, scores, best_train["threshold"]) for cache, scores in zip(val_caches, val_scores)],
            args.temporal_weight,
            args.count_penalty,
        )
        summary["val_oracle_threshold"] = best_threshold(val_caches, val_scores, thresholds, args.temporal_weight, args.count_penalty)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": "set_selector_soft_union_iou",
            "state_dict": best_state,
            "input_dim": train_samples[0].features.shape[1],
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "feature_mean": mean,
            "feature_std": std,
            "use_dinov2": bool(args.use_dinov2),
            "threshold": float(best_train["threshold"]),
            "summary": summary,
        },
        args.output,
    )
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))


@dataclass
class FrameSample:
    features: np.ndarray
    masks: np.ndarray
    gt: np.ndarray
    labels: np.ndarray


class SetSelector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 3,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.input(features)
        x = self.encoder(x.unsqueeze(0)).squeeze(0)
        return self.output(x).squeeze(-1)


def frame_loss(model: SetSelector, sample: FrameSample, device: str, bce_weight: float, count_weight: float) -> torch.Tensor:
    features = torch.as_tensor(sample.features, dtype=torch.float32, device=device)
    masks = torch.as_tensor(sample.masks, dtype=torch.float32, device=device)
    gt = torch.as_tensor(sample.gt, dtype=torch.float32, device=device)
    labels = torch.as_tensor(sample.labels, dtype=torch.float32, device=device)
    logits = model(features)
    probs = torch.sigmoid(logits)
    soft_union = 1.0 - torch.prod(1.0 - probs[:, None] * masks, dim=0)
    intersection = torch.sum(soft_union * gt)
    union = torch.sum(soft_union + gt - soft_union * gt).clamp_min(1e-6)
    iou_loss = 1.0 - intersection / union
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    target_count = labels.sum().clamp_min(1.0)
    count_loss = ((probs.sum() - target_count) / target_count).pow(2)
    return iou_loss + bce_weight * bce + count_weight * count_loss


def cache_features(cache: TrackCache, use_dinov2: bool) -> np.ndarray:
    features = [cache.augmented_features]
    if use_dinov2:
        path = cache.path.parent / "dinov2_vits14_embeddings.npz"
        if path.exists():
            features.append(np.load(path)["embeddings"].astype(np.float32))
    return np.concatenate(features, axis=1).astype(np.float32)


def fit_scaler(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate(blocks, axis=0)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def build_samples(cache: TrackCache, features: np.ndarray, mean: np.ndarray, std: np.ndarray, mask_size: int) -> list[FrameSample]:
    scaled = (features - mean) / std
    samples = []
    for frame in cache.frames:
        rows = cache.rows_by_frame[frame]
        if rows.size == 0:
            continue
        masks = np.stack([downsample_mask(unpack_mask(cache.pred_masks[row], cache.height, cache.width), mask_size) for row in rows])
        gt = downsample_mask(unpack_mask(cache.gt_by_frame[frame], cache.height, cache.width), mask_size)
        labels = cache.greedy_labels[rows].astype(np.float32)
        samples.append(FrameSample(scaled[rows].astype(np.float32), masks.astype(np.float32), gt.astype(np.float32), labels))
    return samples


def downsample_mask(mask: np.ndarray, size: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(resized, 0.0, 1.0).reshape(-1)


def score_caches(
    model: SetSelector,
    caches: list[TrackCache],
    feature_blocks: list[np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> list[np.ndarray]:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for cache, features in zip(caches, feature_blocks):
            scaled = (features - mean) / std
            scores = np.zeros(cache.frame_indices.shape[0], dtype=np.float32)
            for frame in sorted(np.unique(cache.frame_indices)):
                rows = np.flatnonzero(cache.frame_indices == frame)
                if rows.size == 0:
                    continue
                tensor = torch.as_tensor(scaled[rows], dtype=torch.float32, device=device)
                scores[rows] = torch.sigmoid(model(tensor)).detach().cpu().numpy()
            outputs.append(scores)
    return outputs


def parse_grid(value: str) -> list[float]:
    start, stop, step = (float(part) for part in value.split(":"))
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
