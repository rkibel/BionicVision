# Scheme 3 Experiment Log

## 2026-06-04: Consolidation

Combined the previously separate hand segmentation and object track selection
experiments into `experiments/scheme3`.

Removed as standalone experiments after consolidation:

```text
experiments/hand_segmentor
experiments/object_track_selector
outputs/experiments/hand_segmentor
outputs/experiments/object_track_selector
```

Kept only the artifacts needed to run and inspect Scheme 3:

```text
hand model checkpoint
calibrated track policy
set-selector checkpoint
best cached non-oracle track volume
best rendered overlay and contact sheet
small JSON summaries/manifests
```

## Hand Segmentor

Selected model:

```text
architecture: U-Net++ with EfficientNet-B4 encoder
training data: EgoHOS
target: binary hand mask, left OR right hand
input size: 512x704
checkpoint: outputs/experiments/scheme3/hand_segmentor/best.pt
```

Selected dev result:

```text
IoU: 0.9234
precision: 0.9572
recall: 0.9630
threshold: 0.65
```

Reason this model was kept:

```text
The 640x896 comparison reached 0.9276 IoU, but it was slower and improved IoU
by only 0.0042. The 512x704 model is the better speed/quality point for the
full Scheme 3 pipeline.
```

Deleted hand outputs:

```text
continuous EPIC hand overlay videos
contact sheets
older VISOR hand models
training caches
```

## Object Track Selector

Best selector stack:

```text
proposal source:
  cached non-oracle detector/SAM2 dense track cache

selector:
  set_selector_dense2_to_denseval.pt

policy:
  calibrated_track_policy_dense2_static_gate.joblib
```

Policy:

```text
set_weight: 0.475
raw_weight: 0.525
normalization: per-frame minmax
smoothing_alpha: 0.96
track_quantile: 0.80
threshold: 0.52
```

Held-out cached-pipeline result:

```text
track_count: 380
selected_union_mean_iou: 0.7190
selected_temporal_union_iou: 0.4663
selected_mean_count: 5.43 selected tracks/frame
proposal recall@0.5: 0.3621
small recall@0.5: 0.1899
medium recall@0.5: 0.7700
large recall@0.5: 1.0000
```

Important limitation:

```text
The selector crosses 0.7 IoU on the best cached non-oracle proposal volume.
Fresh detector/SAM2 proposal regeneration has been less stable and remains
below this level. Small objects at 448px are still the biggest proposal
bottleneck.
```

## Verification Commands

Compile:

```bash
.venv-models/bin/python -m py_compile experiments/scheme3/*.py
```

Evaluate cached policy:

```bash
.venv-models/bin/python experiments/scheme3/evaluate_track_policy_cache.py \
  --track-cache outputs/experiments/scheme3/unsupervised_30s_runs/val_sfu008_3150_4050_dense_s64_c15_cache/track_cache.npz \
  --score-model outputs/experiments/scheme3/track_score_model/calibrated_track_policy_dense2_static_gate.joblib \
  --device cuda
```

Expected result:

```text
selected_union_mean_iou: 0.7190
selected_temporal_union_iou: about 0.465-0.466
selected_mean_count: 5.43
```

