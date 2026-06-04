# Scheme 3

Scheme 3 is the interaction-aware scene simplification pipeline:

```text
video
  -> EgoHOS hand segmentor
  -> non-oracle object proposals and SAM2 track cache
  -> learned track-set selector
  -> calibrated one-score threshold
  -> cyan hands + red selected object tracks
```

This replaces the separate `hand_segmentor` and `object_track_selector`
experiments with one minimal runnable experiment.

## Current Best Result

Held-out Ego-Exo4D clip:

```text
take: sfu_cooking_008_3
camera: aria01_214-1
frames: 3150-4050
duration: 30 seconds
sampled FPS: 6
```

Metrics:

```text
selected_union_mean_iou: 0.7190
selected_temporal_union_iou: 0.4663
selected_mean_count: 5.43 red tracks/frame
proposal recall@0.5: 0.3621
medium-object recall@0.5: 0.7700
large-object recall@0.5: 1.0000
```

The score crosses `0.7` using the best cached non-oracle detector/SAM2 proposal
volume. Fresh proposal regeneration is still the main bottleneck, especially
for small 448px objects.

## Models

Hand model:

```text
outputs/experiments/scheme3/hand_segmentor/best.pt
architecture: U-Net++ EfficientNet-B4
input size: 512x704
EgoHOS dev IoU: 0.9234
threshold: 0.65
```

Track selector:

```text
outputs/experiments/scheme3/track_score_model/set_selector_dense2_to_denseval.pt
```

Calibrated policy:

```text
outputs/experiments/scheme3/track_score_model/calibrated_track_policy_dense2_static_gate.joblib
threshold: 0.52
set score weight: 0.475
raw heuristic weight: 0.525
temporal smoothing alpha: 0.96
track score quantile: 0.80
```

## Run Best Pipeline

```bash
.venv-models/bin/python experiments/scheme3/run_pipeline.py \
  --input data/egoexo4d/takes/sfu_cooking_008_3/frame_aligned_videos/downscaled/448/aria01_214-1.mp4 \
  --output-dir outputs/experiments/scheme3/unsupervised_30s_runs \
  --run-name replay_val_sfu008_3150_4050_cache_policy_render \
  --start-frame 3150 \
  --duration-seconds 30 \
  --stride 5 \
  --width 448 \
  --height 448 \
  --load-track-cache outputs/experiments/scheme3/unsupervised_30s_runs/val_sfu008_3150_4050_dense_s64_c15_cache/track_cache.npz \
  --score-model outputs/experiments/scheme3/track_score_model/calibrated_track_policy_dense2_static_gate.joblib \
  --take-uid 44d647ce-72d2-4312-b80c-99faea2d017d \
  --camera-name aria01_214-1 \
  --gt-window-start 3150 \
  --gt-window-end 4050 \
  --overwrite \
  --device cuda
```

Rendered artifacts:

```text
outputs/experiments/scheme3/unsupervised_30s_runs/replay_val_sfu008_3150_4050_cache_policy_render/overlay.mp4
outputs/experiments/scheme3/unsupervised_30s_runs/replay_val_sfu008_3150_4050_cache_policy_render/contact_sheet.jpg
```

Colors:

```text
cyan = hands
red  = object track whose calibrated score >= threshold
```

## Train Hand Segmentor

```bash
.venv-models/bin/python experiments/scheme3/train_hand_segmentor.py \
  --data-root data/egohos/data \
  --output-dir outputs/experiments/scheme3/hand_segmentor \
  --model smp-unetpp-efficientnet-b4 \
  --image-size 512x704 \
  --epochs 80 \
  --batch-size 4 \
  --amp \
  --device cuda
```

The saved `best.pt` checkpoint includes the `model_name`, `image_size`,
`threshold`, and model state expected by `run_pipeline.py`.

## Evaluate Cached Policy

```bash
.venv-models/bin/python experiments/scheme3/evaluate_track_policy_cache.py \
  --track-cache outputs/experiments/scheme3/unsupervised_30s_runs/val_sfu008_3150_4050_dense_s64_c15_cache/track_cache.npz \
  --score-model outputs/experiments/scheme3/track_score_model/calibrated_track_policy_dense2_static_gate.joblib \
  --device cuda
```

## Retained Files

Code:

```text
common.py
run_pipeline.py
evaluate_track_policy_cache.py
train_set_selector.py
train_greedy_selector.py
train_hand_segmentor.py
README.md
EXPERIMENT_LOG.md
```

Outputs:

```text
hand_segmentor/
  best.pt
  dev_summary.json
  split_summary.json

track_score_model/
  set_selector_dense2_to_denseval.pt
  calibrated_track_policy_dense2_static_gate.joblib
  summary_set_selector_dense2_to_denseval.json
  summary_calibrated_track_policy_dense2_static_gate.json

unsupervised_30s_runs/
  val_sfu008_3150_4050_dense_s64_c15_cache/
    track_cache.npz
    manifest.json
  replay_val_sfu008_3150_4050_cache_policy_render/
    overlay.mp4
    contact_sheet.jpg
    manifest.json
```
