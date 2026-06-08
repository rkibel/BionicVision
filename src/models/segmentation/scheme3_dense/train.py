"""Train Scheme 3 dense object segmentation on EgoExo and EgoHOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from datasets.egoexo import EgoExoFlowPairDataset, EgoExoObjectDataset
from datasets.egohos import DEFAULT_DATA_ROOT as EGOHOS_ROOT, EgoHOSObjectDataset
from models.segmentation.hand_segmentor.adapter import DEFAULT_CHECKPOINT as DEFAULT_HAND_CHECKPOINT

from .model import DenseModelConfig, HandPrior, IMAGENET_MEAN, IMAGENET_STD, build_model, model_input_tensor


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "outputs/models/scheme3_dense/best.pt"


def main(defaults: dict[str, str | None] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hand-checkpoint", type=Path, default=DEFAULT_HAND_CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder", default="efficientnet-b4")
    parser.add_argument("--encoder-weights", default="imagenet")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--image-feature-mode", choices=("none", "tc_monodepth", "glc_gaze"), default="none")
    parser.add_argument("--egoexo-split", default="train")
    parser.add_argument("--egohos-root", type=Path, default=EGOHOS_ROOT)
    parser.add_argument("--egohos-splits", default="train,val,test_indomain")
    parser.add_argument("--egoexo-samples", type=int, default=1200)
    parser.add_argument("--egohos-samples", type=int, default=2400)
    parser.add_argument("--flow-samples", type=int, default=1200)
    parser.add_argument("--flow-offsets", default="1,-1,2,-2,5,-5,10,-10")
    parser.add_argument("--flow-weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--thresholds", default="0.10,0.14,0.18,0.22,0.26,0.30,0.40,0.50")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dev-run", action="store_true")
    args = parser.parse_args(injected_argv(defaults or {}, sys.argv[1:]))
    if args.dev_run:
        args.epochs, args.workers = 1, 0
        args.egoexo_samples = min(args.egoexo_samples, 8)
        args.egohos_samples = min(args.egohos_samples, 8)
        args.flow_samples = min(args.flow_samples, 4)
        args.image_size = 64
        args.encoder_weights = "none"
    train(args)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    config = DenseModelConfig(
        image_size=args.image_size,
        encoder=args.encoder,
        encoder_weights=parse_weights(args.encoder_weights),
        image_feature_mode=args.image_feature_mode,
    )
    train_dataset = build_supervised_dataset(args)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_dense)
    flow_dataset = EgoExoFlowPairDataset(
        args.egoexo_split,
        image_size=args.image_size,
        max_samples=args.flow_samples,
        offsets=parse_ints(args.flow_offsets),
    )
    flow_loader = DataLoader(flow_dataset, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_flow)
    model = build_model(config).to(device)
    if args.init_checkpoint is not None:
        load_init_checkpoint(model, args.init_checkpoint, device)
    hand_prior = HandPrior(args.hand_checkpoint, device)
    feature_prior = build_feature_prior(args.image_feature_mode, device.type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    flow_iter = iter(flow_loader)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            try:
                flow_batch = next(flow_iter)
            except StopIteration:
                flow_iter = iter(flow_loader)
                flow_batch = next(flow_iter)
            loss = train_step(model, hand_prior, feature_prior, optimizer, scaler, batch, flow_batch, config, args.flow_weight, device)
            losses.append(loss)
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(history[-1], flush=True)
    threshold = select_threshold(model, hand_prior, feature_prior, train_loader, config, parse_floats(args.thresholds), device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save({"state_dict": model.state_dict(), "args": serializable_args, "threshold": threshold, "history": history}, args.output)
    args.output.with_suffix(".json").write_text(json.dumps({"threshold": threshold, "history": history}, indent=2) + "\n")


def build_supervised_dataset(args) -> ConcatDataset:
    datasets = [EgoExoObjectDataset(args.egoexo_split, image_size=args.image_size, max_samples=args.egoexo_samples, shuffle=True)]
    splits = tuple(part.strip() for part in args.egohos_splits.split(",") if part.strip())
    per_split = max(args.egohos_samples // max(len(splits), 1), 1)
    datasets.extend(EgoHOSObjectDataset(split, data_root=args.egohos_root, image_size=args.image_size, max_samples=per_split, shuffle=True) for split in splits)
    return ConcatDataset(datasets)


def train_step(model, hand_prior, feature_prior, optimizer, scaler, batch, flow_batch, config, flow_weight, device) -> float:
    images, target = batch["images"].to(device), batch["targets"].to(device)
    left, right = flow_batch["left"].to(device), flow_batch["right"].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(model_input_tensor(images, hand_prior, config, predict_features(feature_prior, config.image_feature_mode, images))).squeeze(1)
        supervised = segmentation_loss(logits, target)
        pair_images = torch.cat([left, right])
        pair_logits = model(model_input_tensor(pair_images, hand_prior, config, predict_features(feature_prior, config.image_feature_mode, pair_images))).squeeze(1)
        left_logits, right_logits = pair_logits.chunk(2)
        temporal = flow_consistency_loss(left_logits, right_logits, left, right)
        loss = supervised + flow_weight * temporal
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().cpu())


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = logits.sigmoid()
    intersection = (probs * targets).flatten(1).sum(1)
    denominator = probs.flatten(1).sum(1) + targets.flatten(1).sum(1)
    return bce + (1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))).mean()


def flow_consistency_loss(left_logits, right_logits, left_images, right_images) -> torch.Tensor:
    left_prob, right_prob = left_logits.sigmoid(), right_logits.sigmoid()
    right_grid = flow_grids(left_images, right_images)
    left_grid = flow_grids(right_images, left_images)
    warped_right = torch.nn.functional.grid_sample(left_prob.unsqueeze(1), right_grid, align_corners=True).squeeze(1)
    warped_left = torch.nn.functional.grid_sample(right_prob.unsqueeze(1), left_grid, align_corners=True).squeeze(1)
    return 0.5 * (
        torch.nn.functional.l1_loss(warped_right, right_prob)
        + torch.nn.functional.l1_loss(warped_left, left_prob)
    )


def flow_grids(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_gray, target_gray = normalized_grays(source), normalized_grays(target)
    height, width = source_gray.shape[-2:]
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    rows = []
    for source_frame, target_frame in zip(source_gray, target_gray):
        flow = cv2.calcOpticalFlowFarneback(target_frame, source_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        rows.append(np.stack([(2 * (x + flow[..., 0]) / max(width - 1, 1)) - 1, (2 * (y + flow[..., 1]) / max(height - 1, 1)) - 1], axis=-1))
    return torch.from_numpy(np.stack(rows).astype(np.float32)).to(source.device, source.dtype)


def normalized_grays(images: torch.Tensor) -> np.ndarray:
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = ((images.detach() * std + mean).clamp(0, 1) * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
    return np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb])


@torch.inference_mode()
def select_threshold(model, hand_prior, feature_prior, loader, config, thresholds, device) -> float:
    model.eval()
    best = (thresholds[0], -1.0)
    for threshold in thresholds:
        rows = []
        for batch in loader:
            images, targets = batch["images"].to(device), batch["targets"].to(device).bool()
            pred = model(model_input_tensor(images, hand_prior, config, predict_features(feature_prior, config.image_feature_mode, images))).squeeze(1).sigmoid() >= threshold
            intersection = (pred & targets).flatten(1).sum(1).float()
            union = (pred | targets).flatten(1).sum(1).float()
            rows.extend(torch.where(union > 0, intersection / union, torch.zeros_like(union)).cpu().tolist())
        score = float(np.mean(rows))
        if score > best[1]:
            best = (threshold, score)
    return float(best[0])


def collate_dense(samples):
    return {"images": torch.stack([row["image"] for row in samples]), "targets": torch.stack([row["target"] for row in samples])}


def collate_flow(samples):
    return {
        "left": torch.stack([row["left_image"] for row in samples]),
        "right": torch.stack([row["right_image"] for row in samples]),
        "targets": torch.stack([row["target"] for row in samples]),
    }


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_weights(value: str) -> str | None:
    return None if value.lower() in {"", "none", "null"} else value


def injected_argv(defaults: dict[str, str | None], argv: list[str]) -> list[str]:
    injected = []
    for option, value in defaults.items():
        if not any(arg == option or arg.startswith(f"{option}=") for arg in argv):
            injected.append(option)
            if value is not None:
                injected.append(value)
    return [*injected, *argv]


def load_init_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    source_state = checkpoint["state_dict"]
    target_state = model.state_dict()
    adapted = dict(source_state)
    for key, target_weight in target_state.items():
        source_weight = adapted.get(key)
        if source_weight is None or source_weight.shape == target_weight.shape:
            continue
        if source_weight.ndim == 4 and target_weight.ndim == 4 and source_weight.shape[0] == target_weight.shape[0] and source_weight.shape[2:] == target_weight.shape[2:]:
            row = target_weight.clone()
            copy_channels = min(source_weight.shape[1], target_weight.shape[1])
            row[:, :copy_channels] = source_weight[:, :copy_channels].to(row.device, row.dtype)
            if target_weight.shape[1] > source_weight.shape[1]:
                row[:, source_weight.shape[1] :] = source_weight[:, : min(3, source_weight.shape[1])].mean(dim=1, keepdim=True).to(row.device, row.dtype) * 0.1
            adapted[key] = row
    model.load_state_dict(adapted, strict=False)


def build_feature_prior(mode: str, device: str):
    if mode == "none":
        return None
    if mode == "tc_monodepth":
        from models.depth.tc_monodepth.adapter import TCMonoDepthEstimator

        return TCMonoDepthEstimator(device=device)
    if mode == "glc_gaze":
        from models.saliency.glc.adapter import GLCGazeEstimator

        return GLCGazeEstimator(device=device)
    raise ValueError(f"Unsupported image feature mode: {mode}")


def predict_features(prior, mode: str, images: torch.Tensor) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode == "tc_monodepth":
        rgb_255 = ((images * IMAGENET_STD.to(images.device) + IMAGENET_MEAN.to(images.device)).clamp(0.0, 1.0) * 255.0).float()
        return prior.predict_tensor(rgb_255, output_size=images.shape[-2:]).to(images.device, images.dtype)
    if mode == "glc_gaze":
        rgb = ((images.detach() * IMAGENET_STD.to(images.device) + IMAGENET_MEAN.to(images.device)).clamp(0.0, 1.0) * 255.0).byte().cpu().permute(0, 2, 3, 1).numpy()
        maps = [prior.predict_image(frame) for frame in rgb]
        return torch.from_numpy(np.stack(maps)).unsqueeze(1).to(images.device, images.dtype)
    raise ValueError(f"Unsupported image feature mode: {mode}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
