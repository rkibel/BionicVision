# Scheme 3

Scheme 3 trains a dense U-Net++ object mask from two signals:

- supervised object-union masks from Ego-Exo4D and EgoHOS
- unsupervised optical-flow consistency between nearby Ego-Exo video frames

The hand segmentor is used only as an input prior. The dense model predicts the
object/interaction mask.

## Retained Checkpoints

Current best broad-object model:

```
outputs/experiments/scheme3/checkpoints/best.pt
```

Hand-prior checkpoint required by both:

```
outputs/experiments/scheme3/hand_segmentor/best.pt
```

See `OUTPUTS.md` for the retained artifact list and reference metrics.

## Files

```
dataset_loaders/           Ego-Exo, Ego-Exo flow-pair, and EgoHOS datasets
training/                  dataset assembly, losses, train loop, and flow loss
evaluation/                metric primitives, runtime loading, supervised IoU, and flow evaluation
models/                    dense relevance model and hand-prior model wrapper
config.py                  shared paths and defaults
utils.py                   parsing, JSON, seeding, and small numeric helpers
train_dense_union.py       training CLI
evaluate_dense_union.py evaluation CLI
render_dense_union.py      render any checkpoint on an Ego-Exo take or video path
hand_segmentor/            hand-prior training/evaluation code
```

Historical logs remain in `LOG_V*.md`, but current code paths are concentrated
in the files above.

## Train

Default training starts from `best.pt` and writes a candidate checkpoint:

```bash
.venv-models/bin/python experiments/scheme3/train_dense_union.py \
  --output outputs/experiments/scheme3/checkpoints/candidate.pt \
  --summary-output outputs/experiments/scheme3/checkpoints/candidate_summary.json
```

A quick smoke run uses the reduced deterministic preset:

```bash
.venv-models/bin/python experiments/scheme3/train_dense_union.py --dev-run \
  --output /tmp/scheme3_dev.pt \
  --summary-output /tmp/scheme3_dev_summary.json
```

Useful knobs that remain:

```
--source-loss-weights ego4d:2.25,youtube:1.25
--flow-pair-weight 0.10
--flow-pair-offsets 1,-1,2,-2,5,-5,10,-10
--save-selection best_min_supervised
--egohos-selection-stat source_min
```

Removed trial paths include EPIC active-object supervision, VISOR hand-object
supervision, distillation, hard-sample weighting, small-target reweighting,
Sobel image channels, frozen encoder training, Lovasz/focal experiments, and
diagnostic contact-sheet export.

## Evaluate

Evaluate the retained best model with postprocessed supervised and temporal metrics:

```bash
.venv-models/bin/python experiments/scheme3/evaluate_dense_union.py \
  --checkpoint outputs/experiments/scheme3/checkpoints/best.pt \
  --output outputs/experiments/scheme3/checkpoints/best_eval.json
```

The evaluator reports:

```
egoexo_supervised
egohos_supervised
video_temporal.sparse_gt_frames
video_temporal.full_fps.full_fps_flow_temporal_by_horizon
```

The temporal metric is unsupervised: predictions are postprocessed first, then
warped with Farneback flow and compared across frame horizons.

## Render

Render an Ego-Exo take:

```bash
.venv-models/bin/python experiments/scheme3/render_dense_union.py \
  --output-dir outputs/experiments/scheme3/qualitative_runs/best_target
```

Render an arbitrary video:

```bash
.venv-models/bin/python experiments/scheme3/render_dense_union.py \
  --input-video /path/to/video.mp4 \
  --start-frame 0 \
  --duration-seconds 30 \
  --output-dir outputs/experiments/scheme3/qualitative_runs/best_custom_video
```

Rendered videos and masks are treated as disposable qualitative outputs and are
not retained in the cleaned `outputs/experiments/scheme3` tree.
