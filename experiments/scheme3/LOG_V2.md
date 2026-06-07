# Scheme 3 v2 Experiment Log

## 2026-06-04: Goal

Objective:

```text
Improve the legitimate non-oracle full pipeline toward selected-object mean IoU
>= 0.7 without using cached replay as the headline result.
```

Starting baseline from Scheme 3 full-FPS fresh run:

```text
frames: 900
sampled_fps: 30
stride: 1
track_count: 293
selected_union_mean_iou: 0.4732
selected_temporal_union_iou: 0.4231
proposal recall@0.5: 0.2414
small recall@0.5: 0.0534
medium recall@0.5: 0.6700
large recall@0.5: 1.0000
```

Diagnosis:

```text
The low selected IoU is mostly proposal/tracking failure. The inherited
selector cannot choose objects that never become usable tracks.
```

## Intervention 1: More Frequent Fresh Object Seeding

Plan:

```text
run 30 FPS fresh detector/SAM2 proposal generation
reduce chunk_frames from 75 to 30
keep dense SAM points and broad cooking DINO prompt
save track_cache for analysis
render overlay only if the result improves materially
```

Command summary:

```text
run_pipeline.py
  input: Ego-Exo4D sfu_cooking_008_3 aria01_214-1
  window: frames 3150..4050, 30 seconds, stride 1
  chunk_frames: 30
  max_seeds: 64
  seed_mode: hybrid
  SAM2: sam2.1_hiera_small
  DINO prompt: broad cooking-object list
  score_model: inherited calibrated set selector from Scheme 3
  render: skipped
  track cache: saved
```

Result:

```text
track_count: 594
proposal mean_best_iou: 0.2799
proposal recall@0.3: 0.3082
proposal recall@0.5: 0.2522
small recall@0.5: 0.0504
medium recall@0.5: 0.7300
large recall@0.5: 1.0000
selected_union_mean_iou: 0.5691
selected_temporal_union_iou: 0.4417
selected_mean_count: 3.9000
```

Interpretation:

```text
This is a real improvement over the Scheme 3 fresh full-FPS baseline
(selected_union_mean_iou 0.4732 -> 0.5691), but it is not enough.
The same proposals have enough coverage to do much better if selection is
solved: greedy oracle selection over this cache reaches selected_union_mean_iou
0.7905. Therefore the next bottleneck is selector training/calibration on
fresh v2-style track distributions, not simply adding more proposal seeds.
```

## Intervention 2: Fresh Train Caches and Selector Retraining

Train caches generated:

```text
train_sfu005_f3930_chunk30_s64
  split: Ego-Exo4D train
  proposal recall@0.5: 0.4791
  greedy oracle same-proposal selected_union_mean_iou: 0.9215

train_sfu003_f5460_chunk30_s64
  split: Ego-Exo4D train
  proposal recall@0.5: 0.8145
  greedy oracle same-proposal selected_union_mean_iou: 0.8982

train_nus007_f3030_chunk30_s64
  split: Ego-Exo4D train
  proposal recall@0.5: 0.3167
  greedy oracle same-proposal selected_union_mean_iou: 0.7306
```

Selector attempts:

```text
set_selector_v2_sfu005_to_sfu008
  features: hand/proximity/geometry/color
  train selected_union_mean_iou: 0.9118
  val selected_union_mean_iou at train threshold: 0.0049
  val selected_union_mean_iou with threshold sweep: 0.0725

set_selector_v2_dinov2_sfu005_to_sfu008
  features: base features + cached DINOv2 crop embeddings
  train selected_union_mean_iou: 0.9193
  val selected_union_mean_iou at train threshold: 0.3140
  val selected_union_mean_iou with threshold sweep: 0.4358

set_selector_v2_dinov2_2train_to_sfu008
  features: base features + cached DINOv2 crop embeddings
  train selected_union_mean_iou: 0.9088
  val selected_union_mean_iou at train threshold: 0.1604
  val selected_union_mean_iou with threshold sweep: 0.3849

set_selector_v2_3train_to_sfu008
  features: deployable base features only
  train selected_union_mean_iou: 0.8472
  val selected_union_mean_iou at train threshold: 0.3371
  val selected_union_mean_iou with threshold sweep: 0.3870

greedy_selector_v2_3train_to_sfu008
  model: HistGradientBoostingClassifier on greedy union-oracle labels
  train AUROC / AP: 0.9995 / 0.9962
  val selected_union_mean_iou at train threshold: 0.0710
  val selected_union_mean_iou with threshold sweep: 0.2586

logistic_selector_v2_3train_to_sfu008
  model: heavily regularized linear classifier on greedy union-oracle labels
  best val selected_union_mean_iou at train threshold: 0.1116

union_linear_v2_random250_3train
  model: constrained random linear policy search on train objective
  train selected_union_mean_iou: 0.3824
  val selected_union_mean_iou at train threshold: 0.1405
```

Interpretation:

```text
The selector problem did not improve by adding small amounts of supervised
train data. High-capacity selectors memorize train windows. Regularized and
linear selectors underfit the union-selection behavior. Cached DINOv2 crop
features helped one-window generalization somewhat, but the current pipeline
cannot use those features online, and the result still remained below the
incumbent temporal policy.

The best legitimate fresh non-oracle v2 result is still Intervention 1:
selected_union_mean_iou 0.5691 on the held-out sfu_cooking_008_3 full-FPS
30-second window.
```

## Intervention 3: Alternative Proposal / Semantic Models

YOLO-World probe:

```text
package installed in .venv-models: ultralytics 8.4.60
model downloaded: yolov8s-world.pt
test: custom cooking classes on held-out annotated frames
mean union IoU of all YOLO boxes: 0.1397
mean boxes per annotated frame: 16.1
```

CLIP crop semantic probe:

```text
model: OpenAI CLIP ViT-B/32
test: proposal crops scored against cooking-object prompts vs background/hand prompts
val selected_union_mean_iou with threshold sweep: 0.0900
```

SAM2.1 base-plus probe:

```text
checkpoint downloaded: sam2.1_hiera_base_plus.pt
run: fresh full-FPS held-out window, same seeds, incumbent temporal policy
track_count: 577
proposal recall@0.5: 0.2845
small recall@0.5: 0.1009
medium recall@0.5: 0.7100
large recall@0.5: 1.0000
selected_union_mean_iou: 0.4261
selected_temporal_union_iou: 0.4392
```

Broad-selection diagnostic:

```text
small SAM2 cache, union all proposals:
  selected_union_mean_iou: 0.1480
  selected_mean_count: 57.9

base-plus SAM2 cache, union all proposals:
  selected_union_mean_iou: 0.1354
  selected_mean_count: 58.9
```

Interpretation:

```text
YOLO-World and CLIP do not directly solve semantic relevance for this GT.
SAM2.1 base-plus is not an upgrade for this pipeline; it is worse than the
small checkpoint on the held-out full-FPS run. Selecting broadly is also not
viable because background/incorrect proposals swamp the object union.

Current best model stack remains:
  hand segmentor: outputs/experiments/scheme3_v2/hand_segmentor/best.pt
  object tracker/proposals: SAM2.1 small + GroundingDINO prompt seeds
  selector: calibrated_track_policy_dense2_static_gate.joblib

Current best fresh held-out metric:
  selected_union_mean_iou: 0.5691
  selected_temporal_union_iou: 0.4417

This is not solved. The next serious intervention should be an online
track-level architecture with persistent object memory and semantic crop
features, not more threshold tuning over the current hand/proximity features.
```
