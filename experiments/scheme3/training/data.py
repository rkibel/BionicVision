#!/usr/bin/env python3
"""Dataset assembly for Scheme 3 dense training."""

from __future__ import annotations

from typing import Any

from torch.utils.data import ConcatDataset, DataLoader

from dataset_loaders import EgoExoFlowPairDataset, EgoExoMaskDataset, EgoHOSMaskDataset, collate_samples
from utils import parse_csv, parse_label_ids, parse_offsets


def build_datasets(args) -> dict[str, Any]:
    train_ds = build_supervised_dataset(args)
    egoexo_val_ds = EgoExoMaskDataset(
        args.calib_split,
        args.egoexo_camera,
        args.image_size,
        args.val_samples,
        args.seed + 991,
        exclude_target_window=True,
        target_take=args.benchmark_take,
        target_start=args.benchmark_start_frame,
        target_end=args.benchmark_end_frame,
        preserve_order=True,
    )
    egohos_val_by_split = {
        split: build_egohos_dataset(split, args.image_size, args.egohos_val_samples, args.seed + 1991 + idx * 101, args, preserve_order=True, balance_sources=args.egohos_balance_val_sources)
        for idx, split in enumerate(parse_csv(args.egohos_val_splits))
    }
    target_ds = EgoExoMaskDataset(
        "val",
        args.egoexo_camera,
        args.image_size,
        0,
        args.seed,
        exclude_target_window=False,
        target_only=True,
        target_take=args.benchmark_take,
        target_start=args.benchmark_start_frame,
        target_end=args.benchmark_end_frame,
        shuffle=False,
    )
    flow_ds = build_flow_dataset(args.train_splits, args.image_size, args.flow_pair_samples, args.seed + 1707, parse_offsets(args.flow_pair_offsets), args) if args.flow_pair_weight > 0 else None
    train_eval_ds = build_supervised_dataset(args, samples_override=args.train_eval_samples, augment=False) if args.train_eval_samples > 0 else None
    return {
        "train": train_ds,
        "train_eval": train_eval_ds,
        "egoexo_val": egoexo_val_ds,
        "egohos_val_splits": egohos_val_by_split,
        "target": target_ds,
        "flow": flow_ds,
    }


def sample_loader(dataset, args) -> DataLoader:
    return DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_samples)


def build_supervised_dataset(args, samples_override: int | None = None, augment: bool | None = None):
    augment = args.augment_supervised if augment is None else augment
    datasets = []
    egoexo_samples = args.train_samples if samples_override is None else samples_override
    datasets.append(build_mask_dataset(args.train_splits, args.image_size, egoexo_samples, args.seed, args, augment=augment))
    egohos_samples = args.egohos_train_samples if samples_override is None else samples_override
    per_split = max(egohos_samples // max(len(args.egohos_train_splits), 1), 1) if egohos_samples > 0 else 0
    datasets.extend(
        build_egohos_dataset(
            split,
            args.image_size,
            per_split,
            args.seed + 3000 + idx * 101,
            args,
            balance_sources=args.egohos_balance_sources,
            augment=augment,
        )
        for idx, split in enumerate(args.egohos_train_splits)
    )
    return concat_nonempty(datasets)


def build_mask_dataset(splits: list[str], image_size: int, samples: int, seed: int, args, augment: bool = False):
    per_split = max(samples // max(len(splits), 1), 1) if samples > 0 else 0
    datasets = [
        EgoExoMaskDataset(
            split,
            args.egoexo_camera,
            image_size,
            per_split,
            seed + idx * 101,
            exclude_target_window=True,
            target_take=args.benchmark_take,
            target_start=args.benchmark_start_frame,
            target_end=args.benchmark_end_frame,
            augment=augment,
        )
        for idx, split in enumerate(splits)
    ]
    return concat_nonempty(datasets)


def build_egohos_dataset(split: str, image_size: int, samples: int, seed: int, args, preserve_order: bool = False, balance_sources: bool = False, augment: bool | None = None):
    augment = args.augment_supervised if augment is None else augment
    return EgoHOSMaskDataset(
        split=split,
        root=args.egohos_root,
        image_size=image_size,
        max_samples=samples,
        seed=seed,
        sources=tuple(parse_csv(args.egohos_sources)),
        object_ids=parse_label_ids(args.egohos_object_ids),
        balance_sources=balance_sources,
        preserve_order=preserve_order,
        augment=augment and not preserve_order,
    )


def build_flow_dataset(splits: list[str], image_size: int, samples: int, seed: int, offsets: tuple[int, ...], args):
    per_split = max(samples // max(len(splits), 1), 1) if samples > 0 else 0
    datasets = [
        EgoExoFlowPairDataset(
            split,
            args.egoexo_camera,
            image_size,
            per_split,
            seed + idx * 101,
            target_take=args.benchmark_take,
            target_start=args.benchmark_start_frame,
            target_end=args.benchmark_end_frame,
            frame_offsets=offsets,
        )
        for idx, split in enumerate(splits)
    ]
    return concat_nonempty(datasets)


def concat_nonempty(datasets: list):
    datasets = [dataset for dataset in datasets if dataset is not None and len(dataset) > 0]
    if not datasets:
        raise RuntimeError("No samples found for the requested dataset configuration")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
