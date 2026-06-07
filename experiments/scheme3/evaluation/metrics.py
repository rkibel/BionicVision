#!/usr/bin/env python3
"""Pure mask metrics shared by training-time validation and evaluation."""

from __future__ import annotations

import torch


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_bool = pred.bool()
    target_bool = target.bool()
    union = torch.logical_or(pred_bool, target_bool).sum().item()
    if union == 0:
        return 0.0
    inter = torch.logical_and(pred_bool, target_bool).sum().item()
    return float(inter / union)


def evaluate_union_logits(union_logits: torch.Tensor, targets: list[torch.Tensor], threshold: float) -> dict:
    probs = union_logits.sigmoid()
    target_unions = target_union_stack(targets, probs.shape[-2:], probs.device)
    return evaluate_union_probs(probs, target_unions, threshold)


def target_union_stack(targets: list[torch.Tensor], shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    rows = []
    for target_masks in targets:
        if target_masks.numel():
            rows.append((target_masks.to(device) >= 0.5).any(dim=0))
        else:
            rows.append(torch.zeros(shape, dtype=torch.bool, device=device))
    if not rows:
        return torch.empty((0, *shape), dtype=torch.bool, device=device)
    return torch.stack(rows)


def evaluate_union_probs(probs: torch.Tensor, target_unions: torch.Tensor, threshold: float) -> dict:
    if probs.numel() == 0 or target_unions.numel() == 0:
        return {
            "selected_union_mean_iou": 0.0,
            "selected_temporal_union_iou": 0.0,
            "selected_mean_area": 0.0,
            "frames": 0,
            "temporal_pairs": 0,
        }
    pred_unions = probs >= threshold
    pred_flat = pred_unions.flatten(1)
    target_flat = target_unions.flatten(1)
    inter = torch.logical_and(pred_flat, target_flat).sum(dim=1).float()
    union = torch.logical_or(pred_flat, target_flat).sum(dim=1).float()
    ious = torch.where(union > 0, inter / union.clamp_min(1.0), torch.zeros_like(union))
    counts = pred_flat.float().mean(dim=1)
    temporal_union = torch.logical_or(pred_unions[:-1], pred_unions[1:]).flatten(1).sum(dim=1).float()
    temporal_inter = torch.logical_and(pred_unions[:-1], pred_unions[1:]).flatten(1).sum(dim=1).float()
    valid_temporal = temporal_union > 0
    temporal_ious = temporal_inter[valid_temporal] / temporal_union[valid_temporal].clamp_min(1.0)
    return {
        "selected_union_mean_iou": float(ious.mean().item()),
        "selected_temporal_union_iou": float(temporal_ious.mean().item()) if temporal_ious.numel() else 0.0,
        "selected_mean_area": float(counts.mean().item()),
        "frames": int(probs.shape[0]),
        "temporal_pairs": int(temporal_ious.numel()),
    }


def evaluate_union_logits_sweep(union_logits: torch.Tensor, targets: list[torch.Tensor], thresholds: list[float]) -> list[dict]:
    probs = union_logits.sigmoid()
    target_unions = target_union_stack(targets, probs.shape[-2:], probs.device)
    rows = []
    for threshold in thresholds:
        metrics = evaluate_union_probs(probs, target_unions, threshold)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return rows
