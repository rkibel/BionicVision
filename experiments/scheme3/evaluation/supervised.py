#!/usr/bin/env python3
"""Supervised dense-mask evaluation helpers for Scheme 3."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_loaders import collate_samples, target_union
from utils import mean, parse_grid
from evaluation.runtime import transform_hand_prior
from models.dense import model_input_tensor_from_raw_hand
from evaluation.metrics import binary_iou


def evaluate_supervised_dataset(model, hand_prior: HandPrior, dataset, cfg: dict, args, threshold: float, name: str) -> dict:
    if len(dataset) == 0:
        return empty_supervised_result(name, threshold)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_samples)
    probs_rows, target_rows, sources = [], [], []
    for batch in loader:
        images = batch["images"].to(args.device)
        targets = target_union(batch["target_masks"], args.device).bool()
        raw_hand = transform_hand_prior(hand_prior(images), args.hand_prior_power)
        features = batch.get("image_features")
        probs = predict_image_tensor_probs(model, images, raw_hand, cfg, args, features.to(args.device) if features is not None else None)
        probs_rows.append(probs.detach().cpu())
        target_rows.append(targets.detach().cpu())
        sources.extend(batch.get("sources", [""] * int(targets.shape[0])))

    probs = torch.cat(probs_rows) if probs_rows else torch.empty((0, cfg["image_size"], cfg["image_size"]))
    targets = torch.cat(target_rows) if target_rows else torch.empty((0, cfg["image_size"], cfg["image_size"]), dtype=torch.bool)
    metrics = evaluate_supervised_probs(probs, targets, threshold, args.morph_close_kernel)
    grid = parse_grid(args.supervised_threshold_grid)
    sweep = [evaluate_supervised_probs(probs, targets, row, args.morph_close_kernel) for row in grid]
    by_source = evaluate_supervised_by_source(probs, targets, sources, threshold, args.morph_close_kernel, grid)
    return {
        "dataset": name,
        **metrics,
        "best_threshold_metrics": best_threshold_metrics(sweep, threshold),
        "by_source": by_source,
        **source_summary_metrics(by_source),
    }


def predict_image_tensor_probs(model, images: torch.Tensor, raw_hand: torch.Tensor, cfg: dict, args, image_features: torch.Tensor | None = None) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
        model_input = model_input_tensor_from_raw_hand(
            images,
            raw_hand,
            1.0,
            cfg["image_feature_mode"],
            cfg["hand_input_mode"],
            cfg["hand_kernel_size"],
            image_features,
        )
        return torch.sigmoid(model(model_input).squeeze(1)).detach().float()


def empty_supervised_result(name: str, threshold: float) -> dict:
    return {
        "dataset": name,
        "selected_union_mean_iou": 0.0,
        "selected_precision": 0.0,
        "selected_recall": 0.0,
        "selected_dice": 0.0,
        "selected_mean_area": 0.0,
        "target_mean_area": 0.0,
        "frames": 0,
        "positive_frames": 0,
        "threshold": threshold,
        "best_threshold_metrics": {},
        "by_source": {},
        "source_balanced_mean_iou": 0.0,
        "source_min_iou": 0.0,
    }


def evaluate_supervised_probs(probs: torch.Tensor, targets: torch.Tensor, threshold: float, close_kernel: int) -> dict:
    pred = postprocess_prob_tensor(probs, threshold, close_kernel)
    ious, precisions, recalls, dices = [], [], [], []
    areas = pred.flatten(1).float().mean(dim=1).tolist() if pred.numel() else []
    target_areas = targets.flatten(1).float().mean(dim=1).tolist() if targets.numel() else []
    positive_frames = 0
    for pred_mask, target_mask in zip(pred, targets):
        if target_mask.any():
            positive_frames += 1
            ious.append(binary_iou(pred_mask, target_mask))
            pred_bool = pred_mask.bool()
            target_bool = target_mask.bool()
            tp = torch.logical_and(pred_bool, target_bool).sum().item()
            fp = torch.logical_and(pred_bool, ~target_bool).sum().item()
            fn = torch.logical_and(~pred_bool, target_bool).sum().item()
            precisions.append(float(tp / max(tp + fp, 1)))
            recalls.append(float(tp / max(tp + fn, 1)))
            dices.append(float((2 * tp) / max(2 * tp + fp + fn, 1)))
    return {
        "selected_union_mean_iou": mean(ious),
        "selected_precision": mean(precisions),
        "selected_recall": mean(recalls),
        "selected_dice": mean(dices),
        "selected_mean_area": mean(areas),
        "target_mean_area": mean(target_areas),
        "frames": int(pred.shape[0]) if pred.ndim == 3 else 0,
        "positive_frames": positive_frames,
        "threshold": threshold,
    }


def postprocess_prob_tensor(probs: torch.Tensor, threshold: float, close_kernel: int) -> torch.Tensor:
    pred = probs >= threshold
    if close_kernel > 1 and pred.numel():
        pred = morph_close_tensor(pred, close_kernel).cpu()
    return pred.cpu()


def evaluate_supervised_by_source(probs: torch.Tensor, targets: torch.Tensor, sources: list[str], threshold: float, close_kernel: int, grid: list[float]) -> dict[str, dict]:
    if probs.numel() == 0 or not sources:
        return {}
    rows = {}
    for source in sorted(set(sources)):
        indices = [idx for idx, row in enumerate(sources) if row == source]
        index = torch.tensor(indices, dtype=torch.long)
        source_probs = probs.index_select(0, index)
        source_targets = targets.index_select(0, index)
        sweep = [evaluate_supervised_probs(source_probs, source_targets, row, close_kernel) for row in grid]
        rows[source or "unknown"] = {
            **evaluate_supervised_probs(source_probs, source_targets, threshold, close_kernel),
            "best_threshold_metrics": best_threshold_metrics(sweep, threshold),
        }
    return rows


def best_threshold_metrics(sweep: list[dict], reference_threshold: float) -> dict:
    if not sweep:
        return {}
    return max(sweep, key=lambda item: (item["selected_union_mean_iou"], -abs(item["threshold"] - reference_threshold)))


def source_summary_metrics(by_source: dict[str, dict]) -> dict[str, float]:
    ious = [float(row["selected_union_mean_iou"]) for row in by_source.values() if row.get("frames", 0) > 0]
    return {
        "source_balanced_mean_iou": mean(ious),
        "source_min_iou": min(ious) if ious else 0.0,
    }


def morph_close_tensor(masks: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    rows = []
    for mask in masks.detach().cpu().numpy().astype(np.uint8):
        rows.append(torch.from_numpy(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)))
    return torch.stack(rows).to(masks.device)
