#!/usr/bin/env python3
"""Training loop, losses, and validation for Scheme 3 dense masks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_loaders import target_union
from models.dense import model_input_tensor_from_raw_hand
from utils import mean, parse_csv, parse_label_ids, parse_offsets, parse_weight_map
from training.flow_loss import cached_flow_pair_grids, unsupervised_flow_loss
from training.data import sample_loader
from evaluation.metrics import evaluate_union_logits_sweep


def evaluate_epoch(model, hand_prior: HandPrior, datasets: dict[str, Any], target_loader: DataLoader, args, thresholds: list[float]) -> tuple[dict, dict[str, dict], float]:
    named_pred = collect_named_predictions(model, hand_prior, datasets, target_loader, args)
    val_pred = combine_prediction_bundles([named_pred["egoexo_val"], named_pred["egohos_val"]])
    if args.threshold_selection == "val_objective":
        val = evaluate_prediction_bundle(val_pred, thresholds)
        named = evaluate_named_predictions(named_pred, val["threshold"])
    else:
        threshold = select_supervised_threshold(named_pred, thresholds, args)
        val = evaluate_prediction_bundle(val_pred, [threshold])
        named = evaluate_named_predictions(named_pred, threshold)
    score = checkpoint_selection_score(val, named, args.save_selection, args.egohos_selection_stat, args.egohos_selection_min_source_frames)
    return val, named, score


def collect_named_predictions(model, hand_prior: HandPrior, datasets: dict[str, Any], target_loader: DataLoader, args) -> dict[str, Any]:
    egohos_split_predictions = {
        split: collect_loader_predictions(model, hand_prior, sample_loader(dataset, args), args)
        for split, dataset in datasets.get("egohos_val_splits", {}).items()
    }
    rows = {
        "egoexo_val": collect_loader_predictions(model, hand_prior, sample_loader(datasets["egoexo_val"], args), args),
        "egohos_val": combine_prediction_bundles(list(egohos_split_predictions.values())),
        "egohos_val_splits": egohos_split_predictions,
        "target": collect_loader_predictions(model, hand_prior, target_loader, args),
    }
    if datasets.get("train_eval") is not None:
        rows["train_eval"] = collect_loader_predictions(model, hand_prior, sample_loader(datasets["train_eval"], args), args)
    return rows


def combine_prediction_bundles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    bundles = [bundle for bundle in bundles if bundle.get("logits") is not None]
    logits = [bundle["logits"] for bundle in bundles if bundle["logits"].numel()]
    return {
        "logits": torch.cat(logits) if logits else torch.empty((0, 0, 0)),
        "targets": [target for bundle in bundles for target in bundle.get("targets", [])],
        "sources": [source for bundle in bundles for source in bundle.get("sources", [])],
    }


def evaluate_named_predictions(predictions: dict[str, Any], threshold: float) -> dict[str, dict]:
    rows = {
        "egoexo_val": evaluate_prediction_bundle(predictions["egoexo_val"], [threshold]),
        "egohos_val": evaluate_prediction_bundle(predictions["egohos_val"], [threshold]),
        "egohos_val_splits": {
            split: evaluate_prediction_bundle(prediction, [threshold])
            for split, prediction in predictions.get("egohos_val_splits", {}).items()
        },
        "target": evaluate_prediction_bundle(predictions["target"], [threshold]),
    }
    if "train_eval" in predictions:
        rows["train_eval"] = evaluate_prediction_bundle(predictions["train_eval"], [threshold])
    return rows


def select_supervised_threshold(predictions: dict[str, Any], thresholds: list[float], args) -> float:
    best_threshold = thresholds[0]
    best_score = -float("inf")
    for threshold in thresholds:
        named = evaluate_named_predictions(predictions, threshold)
        score = checkpoint_selection_score({"objective": 0.0}, named, "best_min_supervised", args.egohos_selection_stat, args.egohos_selection_min_source_frames)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return float(best_threshold)


def evaluate_named_splits(model, hand_prior: HandPrior, datasets: dict[str, Any], target_loader: DataLoader, args, threshold: float) -> dict[str, dict]:
    return evaluate_named_predictions(collect_named_predictions(model, hand_prior, datasets, target_loader, args), threshold)


def collect_loader_predictions(model, hand_prior: HandPrior, loader: DataLoader, args) -> dict[str, Any]:
    model.eval()
    logits_rows, targets, sources = [], [], []
    for batch in loader:
        images = batch["images"].to(args.device)
        logits_rows.append(predict_logits(model, hand_prior, images, args).squeeze(1).detach().cpu())
        targets.extend([target.detach().cpu() for target in batch["target_masks"]])
        sources.extend(batch.get("sources", [""] * len(batch["target_masks"])))
    logits = torch.cat(logits_rows) if logits_rows else torch.empty((0, args.image_size, args.image_size))
    return {"logits": logits, "targets": targets, "sources": sources}


def train_epoch(model, hand_prior: HandPrior, optimizer, scaler, train_loader: DataLoader, flow_loader: DataLoader | None, args) -> tuple[float, float]:
    model.train()
    losses, flow_losses = [], []
    flow_iter = iter(flow_loader) if flow_loader is not None else None
    for batch in train_loader:
        losses.append(train_supervised_step(model, hand_prior, optimizer, scaler, batch, args))
        if flow_iter is not None and (args.flow_steps_per_epoch <= 0 or len(flow_losses) < args.flow_steps_per_epoch):
            try:
                flow_batch = next(flow_iter)
            except StopIteration:
                flow_iter = iter(flow_loader)
                flow_batch = next(flow_iter)
            flow_losses.append(train_flow_step(model, hand_prior, optimizer, scaler, flow_batch, args))
    return mean(losses), mean(flow_losses)


def train_supervised_step(model, hand_prior: HandPrior, optimizer, scaler, batch: dict, args) -> float:
    images = batch["images"].to(args.device)
    target = target_union(batch["target_masks"], args.device)
    sample_weights = sample_loss_weights(batch.get("datasets", []), batch.get("sources", []), args, args.device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
        raw_hand = hand_prior(images)
        model_input = model_input_tensor_from_raw_hand(images, raw_hand, args.hand_prior_power, args.image_feature_mode, args.hand_input_mode, args.hand_kernel_size)
        logits = model(model_input).squeeze(1)
        loss = dense_loss(logits, target, args, sample_weights)
    scaler.scale(loss).backward()
    step_optimizer(optimizer, scaler, model)
    return float(loss.detach().cpu())


def train_flow_step(model, hand_prior: HandPrior, optimizer, scaler, batch: dict, args) -> float:
    left = batch["images_left"].to(args.device)
    right = batch["images_right"].to(args.device)
    target_left = target_union(batch["target_masks_left"], args.device)
    sample_weights = sample_loss_weights(batch.get("datasets_left", []), batch.get("sources_left", []), args, args.device)
    flow_grids = cached_flow_pair_grids(batch, left, right, args) if args.flow_cache else None
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
        images = torch.cat([left, right], dim=0)
        raw_hand = hand_prior(images)
        model_input = model_input_tensor_from_raw_hand(images, raw_hand, args.hand_prior_power, args.image_feature_mode, args.hand_input_mode, args.hand_kernel_size)
        logits_left, logits_right = model(model_input).squeeze(1).chunk(2, dim=0)
        supervised = dense_loss(logits_left, target_left, args, sample_weights)
        temporal = unsupervised_flow_loss(logits_left, logits_right, left, right, args.flow_pair_bg_weight, args.flow_pair_min_prob, flow_grids)
        loss = supervised + args.flow_pair_weight * temporal
    scaler.scale(loss).backward()
    step_optimizer(optimizer, scaler, model)
    return float(loss.detach().cpu())


def sample_loss_weights(datasets: list[str], sources: list[str], args, device: str) -> torch.Tensor | None:
    source_weights = parse_weight_map(args.source_loss_weights)
    has_dataset_weights = abs(args.egoexo_loss_weight - 1.0) >= 1e-8 or abs(args.egohos_loss_weight - 1.0) >= 1e-8
    if not datasets or (not has_dataset_weights and not source_weights):
        return None
    if len(sources) != len(datasets):
        sources = [""] * len(datasets)
    rows = []
    for dataset, source in zip(datasets, sources):
        weight = args.egohos_loss_weight if dataset == "egohos" else args.egoexo_loss_weight
        weight *= source_weights.get(str(source).lower(), 1.0)
        rows.append(weight)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def dense_loss(logits: torch.Tensor, target: torch.Tensor, args, sample_weights: torch.Tensor | None = None) -> torch.Tensor:
    bce = weighted_bce_loss(logits, target, args.positive_bce_weight)
    prob = logits.sigmoid()
    if args.supervised_loss == "bce_dice":
        rows = bce + args.dice_weight * dice_loss(prob, target, reduction="none")
    else:
        rows = bce + args.dice_weight * tversky_loss(prob, target, args.tversky_alpha, args.tversky_beta)
    if args.boundary_weight > 0:
        rows = rows + args.boundary_weight * boundary_loss(prob, target, args.boundary_kernel_size, reduction="none")
    return weighted_mean(rows, sample_weights)


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, positive_weight: float) -> torch.Tensor:
    reduce_dims = tuple(range(1, logits.ndim))
    if positive_weight <= 1.0:
        return F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=reduce_dims)
    weights = torch.where(target > 0.5, torch.full_like(target, positive_weight), torch.ones_like(target))
    return F.binary_cross_entropy_with_logits(logits, target, weight=weights, reduction="none").mean(dim=reduce_dims)


def weighted_mean(rows: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return rows.mean()
    weights = sample_weights.to(rows.dtype)
    return (rows * weights).sum() / weights.sum().clamp_min(1e-6)


def dice_loss(prob: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    inter = (prob * target).flatten(1).sum(dim=1)
    denom = prob.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0))
    return dice.mean() if reduction == "mean" else dice


def tversky_loss(prob: torch.Tensor, target: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    flat_prob = prob.flatten(1)
    flat_target = target.flatten(1)
    tp = (flat_prob * flat_target).sum(dim=1)
    fp = (flat_prob * (1.0 - flat_target)).sum(dim=1)
    fn = ((1.0 - flat_prob) * flat_target).sum(dim=1)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return 1.0 - score


def boundary_loss(prob: torch.Tensor, target: torch.Tensor, kernel_size: int, reduction: str = "mean") -> torch.Tensor:
    kernel_size = max(int(kernel_size), 3)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return dice_loss(soft_boundary(prob, kernel_size), soft_boundary(target, kernel_size), reduction=reduction)


def soft_boundary(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    padding = kernel_size // 2
    rows = mask.unsqueeze(1)
    dilated = F.max_pool2d(rows, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-rows, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).squeeze(1).clamp(0.0, 1.0)


def step_optimizer(optimizer, scaler, model) -> None:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    scaler.step(optimizer)
    scaler.update()


def predict_logits(model, hand_prior: HandPrior, images: torch.Tensor, args) -> torch.Tensor:
    raw_hand = hand_prior(images)
    model_input = model_input_tensor_from_raw_hand(images, raw_hand, args.hand_prior_power, args.image_feature_mode, args.hand_input_mode, args.hand_kernel_size)
    return model(model_input)


def evaluate_prediction_bundle(prediction: dict[str, Any], thresholds: list[float]) -> dict:
    logits = prediction["logits"]
    targets = prediction["targets"]
    sources = prediction["sources"]
    if not targets:
        return {"threshold": thresholds[0], "objective": 0.0, "selected_union_mean_iou": 0.0, "selected_temporal_union_iou": 0.0, "selected_mean_area": 0.0, "frames": 0}
    best = None
    for metrics in evaluate_union_logits_sweep(logits, targets, thresholds):
        metrics["objective"] = float(metrics["selected_union_mean_iou"] + 0.10 * metrics["selected_temporal_union_iou"] - 0.10 * metrics["selected_mean_area"])
        if best is None or metrics["objective"] > best["objective"]:
            best = metrics
    assert best is not None
    best.update(evaluate_loader_source_metrics(logits, targets, sources, best["threshold"]))
    return best


def evaluate_loader_source_metrics(logits: torch.Tensor, targets: list[torch.Tensor], sources: list[str], threshold: float) -> dict:
    if len(sources) != len(targets):
        return {}
    by_source = {}
    for source in sorted({str(row) for row in sources}):
        indices = [idx for idx, row in enumerate(sources) if str(row) == source]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        source_metrics = evaluate_union_logits_sweep(logits.index_select(0, index_tensor), [targets[idx] for idx in indices], [threshold])[0]
        source_metrics.pop("threshold", None)
        by_source[source or "unknown"] = source_metrics
    ious = [float(row["selected_union_mean_iou"]) for row in by_source.values() if row.get("frames", 0) > 0]
    return {
        "by_source": by_source,
        "source_balanced_mean_iou": mean(ious),
        "source_min_iou": min(ious) if ious else 0.0,
    }


def epoch_summary(epoch: int, train_loss: float, flow_loss: float, val: dict, named: dict[str, dict], selection_score: float, save_selection: str) -> dict:
    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "flow_pair_loss": flow_loss,
        "save_selection": save_selection,
        "selection_score": selection_score,
        "val": val,
        "egoexo_val_at_val_threshold": named["egoexo_val"],
        "egohos_val_at_val_threshold": named["egohos_val"],
        "egohos_val_splits_at_val_threshold": named.get("egohos_val_splits", {}),
        "target_at_val_threshold": named["target"],
    }
    if "train_eval" in named:
        row["train_eval_at_val_threshold"] = named["train_eval"]
    return row


def checkpoint_selection_score(val: dict, named: dict[str, dict], mode: str, egohos_statistic: str = "source_min", min_source_frames: int = 16) -> float:
    if mode == "best_val":
        return float(val["objective"])
    if mode == "best_egohos":
        return egohos_selection_iou(named, egohos_statistic, min_source_frames)
    if mode == "best_min_supervised":
        return min(
            float(named["egoexo_val"]["selected_union_mean_iou"]),
            egohos_selection_iou(named, egohos_statistic, min_source_frames),
            float(named["target"]["selected_union_mean_iou"]),
        )
    if mode == "last":
        return float(val["objective"])
    raise ValueError(f"Unsupported save selection mode: {mode}")


def egohos_selection_iou(named: dict[str, dict], statistic: str = "source_min", min_source_frames: int = 16) -> float:
    split_rows = named.get("egohos_val_splits") or {}
    split_ious = [selection_iou(row, statistic, min_source_frames) for row in split_rows.values() if row.get("frames", 0) > 0]
    if split_ious:
        return min(split_ious)
    return selection_iou(named["egohos_val"], statistic, min_source_frames)


def selection_iou(row: dict, statistic: str, min_source_frames: int = 16) -> float:
    if statistic == "aggregate":
        return float(row["selected_union_mean_iou"])
    eligible_source_ious = [
        float(source_row["selected_union_mean_iou"])
        for source_row in (row.get("by_source") or {}).values()
        if source_row.get("frames", 0) >= min_source_frames
    ]
    if statistic == "source_balanced":
        return mean(eligible_source_ious) if eligible_source_ious else float(row.get("source_balanced_mean_iou", row["selected_union_mean_iou"]))
    if statistic == "source_min":
        return min(eligible_source_ious) if eligible_source_ious else float(row.get("source_min_iou", row.get("source_balanced_mean_iou", row["selected_union_mean_iou"])))
    raise ValueError(f"Unsupported EgoHOS selection statistic: {statistic}")


def build_summary(args, init_metadata: dict | None, datasets: dict[str, Any], best_val: dict, final_named: dict[str, dict], history: list[dict], best_score: float) -> dict:
    return {
        "model_kind": "scheme3_dense_union_unetpp",
        "checkpoint": str(args.output),
        "dev_run": args.dev_run,
        "encoder": args.encoder,
        "encoder_weights": args.encoder_weights,
        "image_size": args.image_size,
        "image_feature_mode": args.image_feature_mode,
        "hand_input_mode": args.hand_input_mode,
        "hand_kernel_size": args.hand_kernel_size,
        "init_checkpoint": init_metadata,
        "train_samples": len(datasets["train"]),
        "egoexo_train_samples": args.train_samples,
        "egohos_train_samples": args.egohos_train_samples,
        "egohos_root": str(args.egohos_root),
        "egohos_train_splits": args.egohos_train_splits,
        "egohos_val_splits": parse_csv(args.egohos_val_splits),
        "egohos_sources": parse_csv(args.egohos_sources),
        "egohos_object_ids": parse_label_ids(args.egohos_object_ids),
        "egohos_balance_sources": args.egohos_balance_sources,
        "egohos_balance_val_sources": args.egohos_balance_val_sources,
        "augment_supervised": args.augment_supervised,
        "egoexo_loss_weight": args.egoexo_loss_weight,
        "egohos_loss_weight": args.egohos_loss_weight,
        "source_loss_weights": parse_weight_map(args.source_loss_weights),
        "train_eval_samples": args.train_eval_samples,
        "supervised_loss": args.supervised_loss,
        "positive_bce_weight": args.positive_bce_weight,
        "dice_weight": args.dice_weight,
        "boundary_weight": args.boundary_weight,
        "boundary_kernel_size": args.boundary_kernel_size,
        "tversky_alpha": args.tversky_alpha,
        "tversky_beta": args.tversky_beta,
        "flow_pair_samples": len(datasets["flow"]) if datasets["flow"] is not None else 0,
        "flow_steps_per_epoch": args.flow_steps_per_epoch,
        "flow_pair_weight": args.flow_pair_weight,
        "flow_pair_offsets": parse_offsets(args.flow_pair_offsets),
        "flow_pair_bg_weight": args.flow_pair_bg_weight,
        "flow_pair_min_prob": args.flow_pair_min_prob,
        "flow_cache": args.flow_cache,
        "flow_cache_dir": str(args.flow_cache_dir),
        "save_selection": args.save_selection,
        "threshold_selection": args.threshold_selection,
        "egohos_selection_stat": args.egohos_selection_stat,
        "egohos_selection_min_source_frames": args.egohos_selection_min_source_frames,
        "best_selection_score": best_score,
        "held_out_target": {"take": args.benchmark_take, "camera": args.egoexo_camera, "frames": [args.benchmark_start_frame, args.benchmark_end_frame]},
        "best_val": best_val,
        "egoexo_val_at_best_val_threshold": final_named["egoexo_val"],
        "egohos_val_at_best_val_threshold": final_named["egohos_val"],
        "egohos_val_splits_at_best_val_threshold": final_named.get("egohos_val_splits", {}),
        "target_at_best_val_threshold": final_named["target"],
        "train_eval_at_best_val_threshold": final_named.get("train_eval"),
        "history": history,
    }
