#!/usr/bin/env python3
"""Train Scheme 3 dense masks with supervised IoU and flow consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset_loaders import collate_flow_pairs, collate_samples
from config import (
    CURRENT_DENSE_CHECKPOINT,
    DEFAULT_BENCHMARK_CAMERA,
    DEFAULT_BENCHMARK_END,
    DEFAULT_BENCHMARK_START,
    DEFAULT_BENCHMARK_TAKE,
    DEFAULT_FLOW_CACHE_DIR,
    HAND_CHECKPOINT,
    NEXT_DENSE_CHECKPOINT,
    SUPERVISED_LOSSES,
)
from models.dense import build_model, load_init_checkpoint, model_input_channels
from training.data import build_datasets, sample_loader
from training.loop import build_summary, epoch_summary, evaluate_epoch, evaluate_named_splits, train_epoch
from models.hand_prior import HandPrior
from utils import parse_grid, set_seed, write_json


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    datasets = build_datasets(args)
    train_loader = DataLoader(datasets["train"], args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_samples)
    flow_loader = make_flow_loader(datasets["flow"], args)
    target_loader = sample_loader(datasets["target"], args)

    model = build_model(args.encoder, args.encoder_weights, model_input_channels(args)).to(args.device)
    init_metadata = load_init_checkpoint(model, args.init_checkpoint, args.device)
    hand_prior = HandPrior(args.hand_checkpoint, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    thresholds = parse_grid(args.threshold_grid)

    best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    best_val, best_named, best_score = evaluate_epoch(model, hand_prior, datasets, target_loader, args, thresholds)
    history = [epoch_summary(0, 0.0, 0.0, best_val, best_named, best_score, args.save_selection)]
    if init_metadata:
        history[0]["init_checkpoint"] = init_metadata["checkpoint"]
    print(json.dumps(history[-1]), flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, flow_loss = train_epoch(model, hand_prior, optimizer, scaler, train_loader, flow_loader, args)
        val, named, selection_score = evaluate_epoch(model, hand_prior, datasets, target_loader, args, thresholds)
        history.append(epoch_summary(epoch, train_loss, flow_loss, val, named, selection_score, args.save_selection))
        print(json.dumps(history[-1]), flush=True)
        if args.save_selection == "last" or selection_score >= best_score:
            best_val, best_named, best_score = val, named, selection_score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    final_named = evaluate_named_splits(model, hand_prior, datasets, target_loader, args, best_val["threshold"])
    summary = build_summary(args, init_metadata, datasets, best_val, final_named, history, best_score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "args": vars(args), "summary": summary, "threshold": float(best_val["threshold"])}, args.output)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2))


def make_flow_loader(dataset, args) -> DataLoader | None:
    if dataset is None or not len(dataset):
        return None
    return DataLoader(dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_flow_pairs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-run", action="store_true", help="Use a deterministic small-sample preset.")
    parser.add_argument("--output", type=Path, default=NEXT_DENSE_CHECKPOINT)
    parser.add_argument("--summary-output", type=Path, default=NEXT_DENSE_CHECKPOINT.with_name(f"{NEXT_DENSE_CHECKPOINT.stem}_summary.json"))
    parser.add_argument("--init-checkpoint", type=Path, default=CURRENT_DENSE_CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=1200)
    parser.add_argument("--val-samples", type=int, default=180)
    parser.add_argument("--train-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--calib-split", default="val")
    parser.add_argument("--egoexo-camera", default=DEFAULT_BENCHMARK_CAMERA)
    parser.add_argument("--benchmark-take", default=DEFAULT_BENCHMARK_TAKE)
    parser.add_argument("--benchmark-start-frame", type=int, default=DEFAULT_BENCHMARK_START)
    parser.add_argument("--benchmark-end-frame", type=int, default=DEFAULT_BENCHMARK_END)
    parser.add_argument("--egohos-root", type=Path, default=None)
    parser.add_argument("--egohos-train-splits", nargs="+", default=["train", "val", "test_indomain"])
    parser.add_argument("--egohos-val-splits", default="val,test_indomain,test_outdomain")
    parser.add_argument("--egohos-train-samples", type=int, default=2400)
    parser.add_argument("--egohos-val-samples", type=int, default=240)
    parser.add_argument("--egohos-sources", default="")
    parser.add_argument("--egohos-object-ids", default="3,4,5,6,7,8")
    parser.add_argument("--egohos-balance-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--egohos-balance-val-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment-supervised", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--egoexo-loss-weight", type=float, default=1.0)
    parser.add_argument("--egohos-loss-weight", type=float, default=1.0)
    parser.add_argument("--source-loss-weights", default="")
    parser.add_argument("--train-eval-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--supervised-loss", choices=SUPERVISED_LOSSES, default="bce_tversky")
    parser.add_argument("--positive-bce-weight", type=float, default=1.25)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--boundary-weight", type=float, default=0.15)
    parser.add_argument("--boundary-kernel-size", type=int, default=5)
    parser.add_argument("--tversky-alpha", type=float, default=0.35)
    parser.add_argument("--tversky-beta", type=float, default=0.65)
    parser.add_argument("--flow-pair-weight", type=float, default=0.10)
    parser.add_argument("--flow-pair-samples", type=int, default=1200)
    parser.add_argument("--flow-steps-per-epoch", type=int, default=0)
    parser.add_argument("--flow-pair-offsets", default="1,-1,2,-2,5,-5,10,-10")
    parser.add_argument("--flow-pair-bg-weight", type=float, default=0.02)
    parser.add_argument("--flow-pair-min-prob", type=float, default=0.20)
    parser.add_argument("--flow-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flow-cache-dir", type=Path, default=DEFAULT_FLOW_CACHE_DIR)
    parser.add_argument("--threshold-grid", default="0.02:0.98:0.02")
    parser.add_argument("--encoder", default="efficientnet-b4")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--hand-checkpoint", type=Path, default=HAND_CHECKPOINT)
    parser.add_argument("--hand-prior-power", type=float, default=1.5)
    parser.add_argument("--hand-input-mode", default="raw_ring_outer_distance")
    parser.add_argument("--hand-kernel-size", type=int, default=15)
    parser.add_argument("--image-feature-mode", choices=("none",), default="none")
    parser.add_argument("--save-selection", choices=("best_val", "best_egohos", "best_min_supervised", "last"), default="best_min_supervised")
    parser.add_argument("--threshold-selection", choices=("best_min_supervised", "val_objective"), default="best_min_supervised")
    parser.add_argument("--egohos-selection-stat", choices=("aggregate", "source_balanced", "source_min"), default="source_min")
    parser.add_argument("--egohos-selection-min-source-frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    normalize_defaults(args)
    apply_dev_run_preset(args)
    return args


def normalize_defaults(args) -> None:
    if args.egohos_root is None:
        from dataset_loaders import EGOHOS_ROOT

        args.egohos_root = EGOHOS_ROOT


def apply_dev_run_preset(args) -> None:
    if not args.dev_run:
        return
    args.epochs = min(args.epochs, 1)
    args.train_samples = min(args.train_samples, 96)
    args.egohos_train_samples = min(args.egohos_train_samples, 192)
    args.val_samples = min(args.val_samples, 32)
    args.egohos_val_samples = min(args.egohos_val_samples, 64)
    args.train_eval_samples = min(args.train_eval_samples, 32) if args.train_eval_samples > 0 else 32
    args.flow_pair_samples = min(args.flow_pair_samples, 32)
    args.flow_steps_per_epoch = min(args.flow_steps_per_epoch, 8) if args.flow_steps_per_epoch > 0 else 8
    args.threshold_grid = "0.08:0.56:0.04"


if __name__ == "__main__":
    main()
