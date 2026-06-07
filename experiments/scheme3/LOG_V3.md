# Scheme 3 v3 Experiment Log

## 2026-06-04: Start

Objective:

```text
Build and iterate a new Scheme 3 v3 experiment until it reaches legitimate
non-oracle 0.7 IoU on the target 30-second evaluation, with documented methods,
cleaned outputs, and no shortcut evaluation.
```

Why v3:

```text
Scheme 3 v2 reached selected_union_mean_iou 0.5691 on the full-FPS held-out
30-second Ego-Exo4D window. The same cached proposals had oracle headroom near
0.79, but learned selectors transferred poorly and temporal consistency remained
weak. More threshold tuning over fixed proposals is unlikely to solve this.
```

Architecture plan:

```text
Use a DETR-style query mask decoder with a pretrained ResNet encoder, a
hand-conditioning channel, query-level masks, and query-level relevance scores.
Train against Ego-Exo4D object masks with Hungarian matching. Evaluate by
thresholding learned scores and computing union IoU over held-out relation-mask
frames inside the 30-second video window.
```

Non-oracle rules:

```text
No GT threshold calibration on the held-out target.
No greedy/oracle selector for headline metrics.
No reporting on a compressed-framerate substitute for the full 30-second target
when claiming target-window performance.
```

## Attempt 1: Query Mask Decoder Smoke

Model:

```text
ResNet18 encoder
learned query embeddings
Transformer decoder
query mask logits + query relevance logits
hand prior: zero for smoke only
train samples: 8
image size: 128
```

Result:

```text
target selected_union_mean_iou: 0.2684
target selected_temporal_union_iou: 0.4596
selected_mean_count: 9.63
```

Interpretation:

```text
The training/evaluation/checkpoint path worked, but this smoke run was not a
meaningful model.
```

## Attempt 2: Query Mask Decoder With Frozen Hand Prior

Model:

```text
ResNet18 query mask decoder
image size: 192
queries: 24
train samples: 240
val samples: 60
epochs: 3
hand prior: Scheme 3 v2 U-Net++ hand segmentor
```

Result:

```text
best validation selected_union_mean_iou: 0.2887
target selected_union_mean_iou at validation threshold: 0.2196
target selected_temporal_union_iou: 0.2437
selected_mean_count: 14.93
```

Interpretation:

```text
The model over-selected query masks and did not learn transferable relevance.
The failure mode was not the hand prior; it was weak query score calibration and
weak object-mask quality.
```

## Attempt 3: Stronger Score/Union Loss For Query Masks

Change:

```text
Added stronger no-object scoring pressure, count calibration, and a
score-weighted soft union-IoU loss.
```

First stronger loss collapsed to empty predictions:

```text
train samples: 600
val samples: 100
epoch 3 val selected_union_mean_iou: 0.0000
epoch 3 target selected_union_mean_iou: 0.0000
```

Softer loss improved calibration but still failed:

```text
best target selected_union_mean_iou: 0.2324
best target selected_temporal_union_iou: 0.2614
```

Interpretation:

```text
The query-only route was not sufficient. Penalizing false positives could make
the model select nothing, while softening the loss kept IoU far below the v2
SAM-selector baseline.
```

## Attempt 4: Add Dense Union/Relevance Head

Change:

```text
Kept the query mask decoder, but added a dense pixel-decoder union head trained
directly against the Ego-Exo4D object-mask union.
```

Result:

```text
train samples: 600
val samples: 100
epochs: 5
best validation dense_union selected_union_mean_iou: 0.2641
target dense_union selected_union_mean_iou at validation threshold: 0.2881
final epoch target dense_union selected_union_mean_iou: 0.3183
```

Interpretation:

```text
The dense head was a real improvement over the query-only v3 attempts, but the
ResNet18 pixel decoder was still much weaker than the v2 SAM-based pipeline.
```

## Attempt 5: Strong Dense Relevance Baseline

Model:

```text
U-Net++ decoder
EfficientNet-B4 encoder
ImageNet pretrained encoder
input: RGB + frozen Scheme 3 v2 hand prior
output: dense object/relevance logit map
loss: BCEWithLogits + Dice over Ego-Exo4D object-mask union
```

Run A, official train split only:

```text
train samples: 1200
val samples: 180
epochs: 6
best validation selected_union_mean_iou: 0.3697
target selected_union_mean_iou at best validation threshold: 0.2744
best observed target during training: 0.3109
```

Interpretation:

```text
The stronger segmentation backbone helped validation, but target transfer was
still poor. This indicated the held-out salad window needed better scene/domain
coverage, not only a better architecture.
```

## Attempt 6: Train On Non-Target Train+Val Frames

Change:

```text
Train on Ego-Exo4D train and val frames, while explicitly excluding the target
sfu_cooking_008_3 frames 3150..4050 from every training dataset.
Calibration still used non-target validation frames; the target threshold was
not calibrated on the target window.
```

Result as originally summarized:

```text
checkpoint: outputs/experiments/scheme3_v3/checkpoints/dense_union_unetpp_b4_trainval.pt
train samples: 1600
train splits: train + val
calibration split: val
epochs: 6
validation threshold: 0.70
validation selected_union_mean_iou: 0.8522
validation selected_temporal_union_iou: 0.1300
target selected_union_mean_iou: 0.8084
target selected_temporal_union_iou: 0.3255
target selected_mean_area: 0.0738
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_target_30s/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_target_30s/contact_sheet.jpg
```

Interpretation:

```text
This is the first v3 result above the requested 0.7 target-window IoU. The win
is legitimate for dense non-oracle relevance masking: the target 30-second
window is held out, the threshold comes from non-target validation frames, and
the hand prior is the saved hand segmentor, not GT.

Important caveat: this is not yet instance-level object tracking. It outputs a
dense red relevance/object region, not stable per-object IDs. It proves that a
learned dense hand-conditioned relevance mask can beat the 0.7 IoU target when
the training distribution includes enough similar non-target cooking frames.
The next architecture step should distill this dense head into instance masks
or combine it with SAM2/Mask2Former queries for stable object identity over
30-second videos.
```

## Temporal Metric Audit

Problem:

```text
EgoExoMaskDataset shuffled entries by default, including target_only=True.
Mean IoU was unaffected because predictions and GT stayed paired per sample,
but temporal IoU was computed across shuffled target frames. The original
target selected_temporal_union_iou 0.3255 was therefore not a trustworthy
chronological stability number.
```

Fix:

```text
Added shuffle_entries and preserve_order_after_sample controls.
Target evaluation now uses chronological frames:
3150, 3180, 3210, ..., 4020.
Added evaluate_dense_temporal.py for full-window chronological evaluation.
```

Corrected raw full-window result:

```text
checkpoint: outputs/experiments/scheme3_v3/checkpoints/dense_union_unetpp_b4_trainval.pt
threshold: 0.70
ema_alpha: 0.0
hysteresis_keep_threshold: 0.0
selected_union_mean_iou: 0.8083
selected_temporal_union_iou: 0.4913
gt_temporal_union_iou: 0.4674
selected_mean_area: 0.0738
```

Interpretation:

```text
The model's chronological prediction-to-prediction temporal IoU is now slightly
higher than the GT union's own temporal IoU between adjacent 1 FPS annotation
frames. The temporal score is therefore not as weak as the original shuffled
metric suggested.
```

Temporal smoothing tests:

```text
causal EMA alpha 0.15:
  selected_union_mean_iou: 0.8080
  selected_temporal_union_iou: 0.4917

causal EMA alpha 0.30:
  selected_union_mean_iou: 0.8029
  selected_temporal_union_iou: 0.4925

causal EMA alpha 0.65:
  selected_union_mean_iou: 0.7762
  selected_temporal_union_iou: 0.4935
```

Interpretation:

```text
EMA smoothing raises temporal IoU only slightly and costs mean IoU, so it is
not a good default.
```

Best exploratory hysteresis result:

```text
on threshold: 0.70
keep threshold: 0.55
selected_union_mean_iou: 0.8102
selected_temporal_union_iou: 0.4944
gt_temporal_union_iou: 0.4674
selected_mean_area: 0.0744
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_target_30s_hyst055/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_target_30s_hyst055/contact_sheet.jpg
```

Caveat:

```text
The hysteresis keep threshold was explored on the target window. It improves
both target mean IoU and temporal IoU, but it should be selected on non-target
validation windows before becoming the official headline number.
```

## Hand Prior Ablation

Question:

```text
Should the relevance model receive only a dilated hand prior, or should it see
the exact hand mask separately from the near-hand interaction neighborhood?
```

Implementation:

```text
Added hand_input_mode to train_dense_union.py, render_dense_union.py, and
evaluate_dense_temporal.py.

modes:
  raw: RGB + raw hand probability
  dilated: RGB + dilated hand/proximity probability
  raw_ring: RGB + raw hand probability + proximity ring
  raw_dilated: RGB + raw hand probability + dilated proximity probability
  raw_distance: RGB + raw hand probability + smooth distance-decay proximity
  raw_ring_distance: RGB + raw hand probability + proximity ring + smooth distance-decay proximity
```

Baseline, original dilated input:

```text
checkpoint: dense_union_unetpp_b4_trainval.pt
hand_input_mode: dilated
threshold: 0.70
chronological target mean IoU: 0.8083
chronological target temporal IoU: 0.4913

with target-exploratory hysteresis keep 0.55:
  mean IoU: 0.8102
  temporal IoU: 0.4944
```

Ablation 1, raw hand + ring:

```text
checkpoint: dense_union_unetpp_b4_raw_ring.pt
hand_input_mode: raw_ring
threshold: 0.72
chronological target mean IoU: 0.8099
chronological target temporal IoU: 0.4859

with target-exploratory hysteresis keep 0.55:
  mean IoU: 0.8109
  temporal IoU: 0.4890
```

Interpretation:

```text
Separating raw hand from the ring did not beat the old dilated model's temporal
score. It slightly improved mean IoU, but the tradeoff was not compelling.
```

Ablation 2, raw hand + full dilated proximity:

```text
checkpoint: dense_union_unetpp_b4_raw_dilated.pt
hand_input_mode: raw_dilated
threshold: 0.54
chronological target mean IoU: 0.8207
chronological target temporal IoU: 0.4923

with target-exploratory hysteresis keep 0.45:
  mean IoU: 0.8209
  temporal IoU: 0.4943

with target-exploratory hysteresis keep 0.35:
  mean IoU: 0.8210
  temporal IoU: 0.4964
```

Interpretation:

```text
raw_dilated is the current best hand-prior design. Giving the network exact hand
pixels plus a separate proximity channel improves mean IoU by about +0.012 over
the original dilated-only model and preserves/slightly improves temporal IoU.

This does not get us to 0.9. The contact sheet still shows occasional broad
workspace/static-object relevance. The next likely bottleneck is no longer the
hand prior alone; it is the dense union target and lack of explicit object/track
identity.
```

Current best qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_target_30s_hyst035/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_target_30s_hyst035/contact_sheet.jpg
```

## Distance Prior Ablation

Motivation:

```text
A binary dilation/ring is coarse. A smooth distance prior should let the network
learn a graded interaction field around the hands: touching, near, and far.
```

Implementation:

```text
Added distance_proximity() in hand_prior.py. It computes an OpenCV L2 distance
transform from the frozen hand mask and converts it into an exponential decay
map. This is intentionally detached/non-differentiable because the hand
segmentor is frozen.
```

raw_ring_distance:

```text
checkpoint trained then removed because it was not best:
  dense_union_unetpp_b4_raw_ring_distance.pt

best validation checkpoint target:
  mean IoU: 0.7930
  temporal IoU: 0.4860

best temporal observed during training:
  epoch 2 target mean IoU: 0.7527
  epoch 2 target temporal IoU: 0.5027
```

raw_distance:

```text
checkpoint trained then removed because it was not best:
  dense_union_unetpp_b4_raw_distance.pt

best validation checkpoint target:
  mean IoU: 0.8063
  temporal IoU: 0.4890

best temporal observed during training:
  epoch 1 target mean IoU: 0.7078
  epoch 1 target temporal IoU: 0.5181
```

Interpretation:

```text
Distance priors helped early temporal stability but did not improve the final
mean-IoU model. The distance channel may be too smooth and encourages stable
but under-complete relevance regions. The best current tradeoff remains
raw_dilated + hysteresis:

  checkpoint: dense_union_unetpp_b4_raw_dilated.pt
  raw chronological mean IoU: 0.8207
  raw chronological temporal IoU: 0.4923
  hysteresis keep 0.35 mean IoU: 0.8210
  hysteresis keep 0.35 temporal IoU: 0.4964

The path to 0.9 likely needs more than hand-prior engineering: object/track
identity supervision, temporal training clips, or stronger target-specific
object-mask supervision.
```

## Combined Hand/Distance/Ring Prior Ablation

Question:

```text
What if the relevance model receives all hand-derived priors separately:
raw hand, full dilated hand neighborhood, outside-hand ring, and a smooth
distance/proximity field?
```

Implementation:

```text
Added hand_input_mode:
  raw_dilated_ring_distance

Input channels:
  RGB
  raw frozen hand probability
  dilated hand probability
  dilated-minus-raw proximity ring
  smooth hand proximity field
```

Training/evaluation speed note:

```text
The first run used OpenCV distanceTransform for the distance prior and was
CPU-bound. Replaced CUDA distance prior generation with a separable Gaussian
proximity map in hand_prior.py. This keeps the prior on-device during training.

Also vectorized dense-union threshold evaluation in metrics.py so validation
does not rebuild GT unions and compute per-frame IoU in Python for every
threshold.
```

Training recipe:

```text
checkpoint: dense_union_unetpp_b4_raw_dilated_ring_distance.pt
encoder: EfficientNet-B4 U-Net++
image size: 256
train samples: 1600 from train+val, excluding target window
validation samples: 180 non-target validation frames
epochs: 6
seed: 17
```

Best validation-selected result:

```text
threshold: 0.50
target mean IoU from training target loader: 0.8004
target temporal IoU from training target loader: 0.4958
```

Official full-window target evaluation:

```text
raw:
  mean IoU: 0.8111
  temporal IoU: 0.4958
  gt temporal IoU: 0.4788

hysteresis keep 0.20:
  mean IoU: 0.8082
  temporal IoU: 0.5017
  gt temporal IoU: 0.4788
```

Interpretation:

```text
The all-priors model did not beat raw_dilated. It slightly improves temporal
stability under hysteresis, but loses too much mean IoU. The extra priors also
make the logits more threshold-sensitive, suggesting redundant/correlated hand
channels rather than new object-relevance information.

Conclusion: keep raw_dilated as the best model. Distance/ring channels are not
the main bottleneck. The next serious improvement should target object identity,
track-aware temporal training, or better relevance supervision rather than more
hand-prior variants.
```

## Non-Target-Tuned Temporal/Spatial Post-Processing

Motivation:

```text
Target-window hysteresis was useful but not a valid headline number. Tune
threshold, keep threshold, and small spatial morphology on non-target windows,
then apply the chosen setting once to the held-out 30-second target window.
```

Initial non-target selection, superseded by the corrected expanded tuner:

```text
checkpoint: dense_union_unetpp_b4_raw_dilated.pt
on threshold: 0.50
keep threshold: 0.20
morphology: close, kernel 5
```

Initial non-target calibration aggregate:

```text
mean IoU: 0.8505
temporal IoU: 0.5375
mean area: 0.0951
```

Issue found after the first run:

```text
The initial tuner applied the selected target threshold and keep threshold in
the wrong argument order for the held-out target only. Calibration rows were
valid, but target_at_best_keep was not. Fixed eval_single_window call sites and
renamed the function arguments to make the ordering explicit.
```

Expanded corrected tuning:

```text
Added connected-component cleanup knobs:
  min_component_area_frac
  max_components

Also cached base temporal masks per (window, on threshold, keep threshold), so
component/morphology sweeps do not recompute full 900-frame hysteresis masks for
every candidate.
```

Corrected selection by six non-target validation windows:

```text
checkpoint: dense_union_unetpp_b4_raw_dilated.pt
on threshold: 0.50
keep threshold: 0.25
morphology: close, kernel 5
min component area fraction: 0.0
max components: 0
```

Interpretation:

```text
Component pruning was tested but not selected. Limiting to one or two connected
components often raised temporal stability but hurt mean IoU, because the
official union frequently contains multiple spatially separated object regions.
The best corrected setting is still a global dense relevance mask with small
closing and hysteresis.
```

Non-target calibration aggregate:

```text
mean IoU: 0.8349
temporal IoU: 0.4871
mean area: 0.0774
```

Held-out target result:

```text
mean IoU: 0.8296
temporal IoU: 0.5013
gt temporal IoU: 0.4788
mean area: 0.0778
```

Independent evaluator check:

```text
The corrected post-process result was re-evaluated with
evaluate_dense_temporal.py using explicit overrides:
  threshold override: 0.50
  hysteresis keep threshold: 0.25
  morphology: close, kernel 5

It matched the tuner target result exactly:
  mean IoU: 0.8296
  temporal IoU: 0.5013
```

Current mean-best raw checkpoint result:

```text
threshold: 0.54
mean IoU: 0.8308
temporal IoU: 0.4923
gt temporal IoU: 0.4788
mean area: 0.0760
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_target_30s_valtuned_corrected_close5_keep025/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_target_30s_valtuned_corrected_close5_keep025/contact_sheet.jpg
```

Visual read:

```text
Cyan hands are crisp because the renderer uses the raw hand segmentor output,
not the dilated conditioning channel. Red relevance is coherent and temporally
steadier than the raw model, but still often covers broad workspace/object
unions rather than clean object instances. This is better than Scheme 3 v2, but
not a solved 0.9-IoU tracker.
```

## EMA And Temporal Vote Ablations

Motivation:

```text
Hysteresis helps a little, but temporal IoU remains far below the mean IoU.
Try probability-level EMA and binary-mask temporal voting as no-target-tuned
post-processing. Selection still uses only non-target validation windows.
```

Implementation:

```text
Added --ema-alphas to tune_temporal_postprocess.py and --ema-alpha to
render_dense_union.py.

Added centered binary temporal voting:
  evaluate_dense_temporal.py:
    --temporal-window
    --temporal-min-vote-frac
  tune_temporal_postprocess.py:
    --temporal-windows
    --temporal-min-vote-fracs
  render_dense_union.py:
    same temporal vote options

For tuning/evaluation, temporal voting is computed sparsely at GT frames, using
neighboring masks from the full 30 FPS sequence. Rendering still applies voting
to all frames because the video needs every frame.
```

EMA tuning:

```text
output: dense_union_raw_dilated_ema_tuning.json
alphas tested: 0.0, 0.25, 0.50, 0.70, 0.85
objective: mean IoU + 1.0 * temporal IoU - 0.05 * area

selected:
  ema_alpha: 0.0
  on threshold: 0.50
  keep threshold: 0.25
  morphology: close, kernel 5
  non-target mean IoU: 0.8349
  non-target temporal IoU: 0.4871

best per alpha:
  alpha 0.00: mean 0.8349, temporal 0.4871
  alpha 0.25: mean 0.8334, temporal 0.4883
  alpha 0.50: mean 0.8251, temporal 0.4898
  alpha 0.70: mean 0.7893, temporal 0.4967
  alpha 0.85: mean 0.7276, temporal 0.5014
```

EMA interpretation:

```text
EMA does raise temporal IoU as alpha increases, but it trades away too much
mean IoU. Under the non-target objective, no EMA is selected.
```

Temporal vote tuning:

```text
output: dense_union_raw_dilated_temporal_vote_tuning.json
windows tested: 1, 3, 5
vote fraction: 0.50
objective: mean IoU + 1.0 * temporal IoU - 0.05 * area

selected:
  temporal window: 1
  temporal vote: disabled
  on threshold: 0.50
  keep threshold: 0.25
  morphology: close, kernel 5
  non-target mean IoU: 0.8349
  non-target temporal IoU: 0.4871

best per window:
  window 1: mean 0.8349, temporal 0.4871
  window 3: mean 0.8310, temporal 0.4882
  window 5: mean 0.8207, temporal 0.4889
```

Temporal vote interpretation:

```text
Centered voting slightly improves temporal IoU but loses more mean IoU than it
gains. The official selected post-process remains:

  ema_alpha: 0.0
  temporal_window: 1
  threshold: 0.50
  keep threshold: 0.25
  morphology: close, kernel 5

The remaining bottleneck is not simple temporal smoothing. It is likely the
dense union formulation itself: the model can find a broad relevant workspace,
but it has no explicit object identity, track persistence, or instance-level
selection head.
```

## Diagnostic Target Postprocess Ceiling

Motivation:

```text
Before spending more time on post-processing, estimate whether threshold,
hysteresis, EMA, temporal voting, morphology, or component cleanup have enough
hidden headroom to reach 0.9 mean IoU.
```

Important warning:

```text
This diagnostic uses the held-out target labels to select post-process settings.
It is invalid as a reported non-oracle result. It is only an upper-bound sanity
check for whether post-processing can plausibly close the remaining gap.
```

Grid:

```text
output: dense_union_target_postprocess_ceiling_diagnostic.json
thresholds: 0.40..0.72 step 0.02
keep thresholds: 0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40
EMA alphas: 0, 0.25, 0.50
temporal vote windows: 1, 3, 5
morphology: none, close, open with kernels 3/5 where applicable
component filtering: disabled
```

Diagnostic target-oracle ceiling:

```text
best by mean IoU:
  mean IoU: 0.8312
  temporal IoU: 0.4964
  ema_alpha: 0.0
  threshold: 0.54
  keep threshold: 0.35
  temporal vote: disabled
  morphology: none

best by temporal-weighted objective:
  mean IoU: 0.8209
  temporal IoU: 0.5136
  ema_alpha: 0.25
  threshold: 0.40
  keep threshold: 0.15
  temporal vote window: 3
  morphology: close, kernel 5

best by temporal IoU alone:
  mean IoU: 0.8076
  temporal IoU: 0.5241
  ema_alpha: 0.50
  threshold: 0.40
  keep threshold: 0.15
  temporal vote window: 5
  morphology: close, kernel 5
```

Interpretation:

```text
Even with target-oracle post-process selection, mean IoU only reaches 0.8312.
The official non-target mean-best setting is 0.8308. That means post-processing
has essentially no remaining mean-IoU headroom on this checkpoint. To approach
0.9, the model output itself must improve.

Next intervention: train a higher-resolution raw_dilated dense model.
```

## Higher-Resolution Dense Model Ablation

Motivation:

```text
Since target-oracle post-processing cannot push the 256px checkpoint above
0.8312 mean IoU, test whether higher spatial resolution improves the dense
mask output itself.
```

Training recipe:

```text
checkpoint trained then removed because it was not best:
  dense_union_unetpp_b4_raw_dilated_384.pt

encoder: EfficientNet-B4 U-Net++
image size: 384
hand input mode: raw_dilated
train samples: 1600 from train+val, excluding target window
validation samples: 180 non-target validation frames
epochs: 6
batch size: 2
seed: 17
```

Training observations:

```text
epoch 5 target_at_val_threshold:
  mean IoU: 0.8340
  temporal IoU: 0.4833

epoch 6 was selected by non-target validation:
  validation mean IoU: 0.8467
  validation temporal IoU: 0.3227
  selected threshold: 0.42
```

Official full-window evaluation of the saved validation-selected checkpoint:

```text
output: dense_union_raw_dilated_384_temporal_eval_raw.json
mean IoU: 0.8110
temporal IoU: 0.4869
gt temporal IoU: 0.4787
threshold: 0.42
```

Interpretation:

```text
Higher input resolution did not improve the official target result. It briefly
looked better on target at epoch 5, but that checkpoint was not selected by
non-target validation, so it cannot be used as a valid non-oracle improvement.
The saved validation-selected 384px checkpoint generalizes worse than the
current 256px raw_dilated model.

Conclusion: keep dense_union_unetpp_b4_raw_dilated.pt as the only model
checkpoint. Resolution alone is not the route to 0.9.
```

## Focal/Tversky Loss Ablation

Motivation:

```text
The best 256px BCE+Dice model often predicts a broad relevant workspace.
Try a false-positive-weighted focal/Tversky loss to sharpen masks and reduce
over-selection without changing the architecture, hand prior, or data split.
```

Implementation:

```text
Added --loss-mode to train_dense_union.py:
  bce_dice
  focal_tversky

Focal/Tversky settings:
  focal_gamma: 2.0
  tversky_alpha: 0.70  # false-positive weight
  tversky_beta: 0.30   # false-negative weight
```

Training recipe:

```text
checkpoint trained then removed because it was not best:
  dense_union_unetpp_b4_raw_dilated_focal_tversky.pt

encoder: EfficientNet-B4 U-Net++
image size: 256
hand input mode: raw_dilated
train samples: 1600 from train+val, excluding target window
validation samples: 180 non-target validation frames
epochs: 6
seed: 17
```

Training-selected result:

```text
best non-target validation epoch: 6
validation mean IoU: 0.8425
validation temporal IoU: 0.3259
selected threshold: 0.26
target_at_val_threshold mean IoU: 0.7870
target_at_val_threshold temporal IoU: 0.4706
```

Official full-window evaluation of the saved validation-selected checkpoint:

```text
output: dense_union_raw_dilated_focal_tversky_temporal_eval_raw.json
mean IoU: 0.7902
temporal IoU: 0.4705
gt temporal IoU: 0.4788
threshold: 0.26
```

Interpretation:

```text
Focal/Tversky did not improve the held-out target. The false-positive-heavy
loss likely reduced broad over-selection, but it lost too much recall on the
official target union. The current BCE+Dice raw_dilated checkpoint remains much
better at 0.8308 mean IoU.

Conclusion: keep BCE+Dice as the current best loss. Simple loss sharpening is
not enough to reach 0.9.
```

## Full Local Train+Val Data Run

Motivation:

```text
The 1600-sample train+val run used only part of the locally available
non-target Ego-Exo4D relation-mask frames. Increase the data volume before
changing the architecture again.
```

Local data count:

```text
train split, target excluded: 2697 frames
val split, target excluded: 1431 frames
total non-target train+val entries used: 4128 frames
previous 1600-frame run used about 39% of this local pool
```

Training recipe:

```text
checkpoint:
  dense_union_unetpp_b4_raw_dilated_full_local.pt

encoder: EfficientNet-B4 U-Net++
image size: 256
hand input mode: raw_dilated
loss: BCEWithLogits + Dice
train splits: train + val, excluding target window
train samples requested: 6000
actual train samples: 4128
validation samples: 180 non-target validation frames
epochs: 6
batch size: 4
num workers: 2
```

Training-selected result:

```text
best non-target validation epoch: 5
validation mean IoU: 0.8676
validation temporal IoU: 0.3283
selected threshold: 0.62
target_at_val_threshold mean IoU: 0.8208
target_at_val_threshold temporal IoU: 0.4852
```

Official full-window raw target evaluation:

```text
output: dense_union_raw_dilated_full_local_temporal_eval_raw.json
threshold: 0.62
mean IoU: 0.8311
temporal IoU: 0.4853
gt temporal IoU: 0.4788
```

Non-target postprocess tuning:

```text
output: dense_union_raw_dilated_full_local_postprocess_tuning.json
selected on threshold: 0.58
selected keep threshold: 0.30
morphology: close, kernel 5
EMA: disabled
temporal vote: disabled
component filtering: disabled

non-target aggregate:
  mean IoU: 0.8551
  temporal IoU: 0.4823
```

Held-out target result with selected postprocess:

```text
output: dense_union_raw_dilated_full_local_temporal_eval_corrected_close5_keep030.json
mean IoU: 0.8325
temporal IoU: 0.4943
gt temporal IoU: 0.4788
mean area: 0.0755
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_full_local_target_30s_valtuned_close5_keep030/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_full_local_target_30s_valtuned_close5_keep030/contact_sheet.jpg
```

Interpretation:

```text
Using all local non-target train+val frames gives a small valid mean-IoU gain
over the previous raw_dilated result, but not a step change:

  previous raw mean: 0.8308
  full-local postprocessed mean: 0.8325

The temporal-best result is still the earlier raw_dilated checkpoint with
non-target-tuned close5/keep0.25:

  mean IoU: 0.8296
  temporal IoU: 0.5013

Conclusion: more local frame coverage helps a little, but the bottleneck is
not only data volume. The dense union target still encourages broad workspace
regions instead of clean object instances.
```

## Raw + Dilated + Distance Prior Ablation

Question:

```text
Would passing raw hand, dilated hand neighborhood, and a smooth distance prior
as separate channels improve over raw+dilated alone, without the extra redundant
ring channel from raw_dilated_ring_distance?
```

Implementation:

```text
Added hand_input_mode values:
  raw_dilated_ring
  raw_dilated_distance

raw_dilated_distance input:
  RGB
  raw frozen hand probability
  dilated hand probability
  smooth hand proximity field
```

Training recipe:

```text
checkpoint trained then removed because it was not best:
  dense_union_unetpp_b4_raw_dilated_distance_full_local.pt

summary kept:
  dense_union_unetpp_b4_raw_dilated_distance_full_local_summary.json

encoder: EfficientNet-B4 U-Net++
image size: 256
hand input mode: raw_dilated_distance
loss: BCEWithLogits + Dice
train samples: 4128 local non-target train+val frames
validation samples: 180 non-target validation frames
epochs: 6
```

Training-selected result:

```text
best non-target validation epoch: 6
validation mean IoU: 0.8754
validation temporal IoU: 0.3293
selected threshold: 0.58
target_at_val_threshold mean IoU: 0.8175
target_at_val_threshold temporal IoU: 0.4766
```

Official full-window target evaluation:

```text
output: dense_union_raw_dilated_distance_full_local_temporal_eval_raw.json
mean IoU: 0.8263
temporal IoU: 0.4766
gt temporal IoU: 0.4788
threshold: 0.58
```

Interpretation:

```text
This ablation improved non-target validation but transferred worse than
raw_dilated on the target window. The distance prior appears to add a stable
near-hand geometry cue, but naive concatenation makes the model more
domain-sensitive and does not improve object completeness.

Conclusion: raw+dilated remains the best hand-prior input. If distance is used
again, it should be gated or regularized rather than simply concatenated.
```

## Adjacent-Frame Temporal Pair Training

Motivation:

```text
Post-processing could not raise mean IoU beyond about 0.83, and simple EMA or
temporal voting only traded mean IoU for small temporal gains. The dense model
was still trained frame-by-frame, so it never received a direct training signal
for how masks should change between adjacent annotated frames.
```

Implementation:

```text
Added EgoExoMaskPairDataset in data.py.

Pair construction:
  use only annotated frames with object-mask supervision
  group by take/camera
  pair adjacent annotated frames when frame gap <= 30
  exclude the held-out target window exactly like the frame dataset

Available local non-target pairs:
  4091 adjacent annotated pairs
```

Training loss:

```text
main frame batch:
  BCEWithLogits + Dice over object-mask union

temporal pair batch:
  supervised BCEWithLogits + Dice on both frames
  + 0.05 * target-aware temporal loss

target-aware temporal loss:
  stable pixels, where GT union does not change:
    penalize |p_t - p_{t+1}|
  changing pixels, where GT union changes:
    penalize mismatch between predicted change magnitude and GT change magnitude

This allows real object changes while discouraging flicker in unchanged regions.
```

Training recipe:

```text
checkpoint:
  dense_union_unetpp_b4_raw_dilated_full_local_temporal005.pt

encoder: EfficientNet-B4 U-Net++
image size: 256
hand input mode: raw_dilated
loss: BCEWithLogits + Dice
temporal pair weight: 0.05
temporal change weight: 0.50
train samples: 4128 local non-target train+val frames
temporal pair samples: 4091 local non-target adjacent pairs
validation samples: 180 non-target validation frames
epochs: 6
```

Training-selected result:

```text
best non-target validation epoch: 6
validation mean IoU: 0.9129
validation temporal IoU: 0.3254
selected threshold: 0.46
target_at_val_threshold mean IoU: 0.8568
target_at_val_threshold temporal IoU: 0.4787
```

Official full-window raw target evaluation:

```text
output: dense_union_raw_dilated_full_local_temporal005_temporal_eval_raw.json
threshold: 0.46
mean IoU: 0.8568
temporal IoU: 0.4787
gt temporal IoU: 0.4788
```

Non-target postprocess tuning:

```text
output: dense_union_raw_dilated_full_local_temporal005_postprocess_tuning.json
selected on threshold: 0.38
selected keep threshold: 0.30
EMA: disabled
temporal vote: disabled
morphology: none
component filtering: disabled

non-target aggregate:
  mean IoU: 0.9074
  temporal IoU: 0.4682
```

Held-out target result with selected postprocess:

```text
output: dense_union_raw_dilated_full_local_temporal005_temporal_eval_hyst030.json
mean IoU: 0.8586
temporal IoU: 0.4816
gt temporal IoU: 0.4788
mean area: 0.0742
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_full_local_temporal005_target_30s_valtuned_hyst030/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_full_local_temporal005_target_30s_valtuned_hyst030/contact_sheet.jpg
```

Interpretation:

```text
This is the largest valid mean-IoU improvement so far:

  previous mean-best: 0.8325
  temporal-pair mean-best: 0.8586
  improvement: +0.0261

The temporal metric did not improve:

  previous temporal-best: 0.5013
  temporal-pair selected temporal: 0.4816

The pair loss seems to improve object-union completeness and transfer, but it
does not yet solve temporal consistency. The likely reason is that adjacent
annotated supervision is still sparse at about 1 FPS, while the rendered video
is 30 FPS. The next intervention should target dense temporal memory/tracking
or train on short full-frame clips with pseudo-label propagation, rather than
only adjacent sparse annotated pairs.
```

## Stronger Temporal Pair Weight Ablation

Question:

```text
Does increasing the adjacent-frame temporal pair loss from 0.05 to 0.10 improve
temporal IoU without losing the mean-IoU gain?
```

Training recipe:

```text
checkpoint trained then removed because it was not best:
  dense_union_unetpp_b4_raw_dilated_full_local_temporal010.pt

summary kept:
  dense_union_unetpp_b4_raw_dilated_full_local_temporal010_summary.json

encoder: EfficientNet-B4 U-Net++
image size: 256
hand input mode: raw_dilated
loss: BCEWithLogits + Dice
temporal pair weight: 0.10
temporal change weight: 0.50
train samples: 4128 local non-target train+val frames
temporal pair samples: 4091 local non-target adjacent pairs
validation samples: 180 non-target validation frames
epochs: 6
```

Training-selected result:

```text
best non-target validation epoch: 5
validation mean IoU: 0.9101
validation temporal IoU: 0.3267
selected threshold: 0.46
target_at_val_threshold mean IoU: 0.8482
target_at_val_threshold temporal IoU: 0.4761
```

Official full-window target evaluation:

```text
output: dense_union_raw_dilated_full_local_temporal010_temporal_eval_raw.json
threshold: 0.46
mean IoU: 0.8482
temporal IoU: 0.4761
gt temporal IoU: 0.4788
```

Interpretation:

```text
Increasing the pair loss weight hurt both the selected mean and temporal result
relative to the 0.05 model:

  temporal 0.05 selected: mean 0.8586, temporal 0.4816
  temporal 0.10 selected: mean 0.8482, temporal 0.4761

The epoch history briefly reached target temporal 0.4924 at epoch 2, but that
checkpoint was not selected by non-target validation and had target mean only
0.8242. This suggests the pair-loss strength controls a real mean/temporal
tradeoff, but simply increasing it is not the path to a better final model.
```

## Current Checkpoint Postprocess Ceiling Diagnostic

Motivation:

```text
Estimate whether threshold, hysteresis, EMA, temporal voting, and morphology
still have enough hidden headroom on the current temporal005 checkpoint to
reach 0.9 mean IoU.
```

Important warning:

```text
This diagnostic uses held-out target labels to select postprocess settings.
It is not valid as a headline result.
```

Diagnostic result:

```text
output: dense_union_full_local_temporal005_target_postprocess_ceiling_diagnostic.json

best by target mean IoU:
  mean IoU: 0.8601
  temporal IoU: 0.4859
  on threshold: 0.24
  keep threshold: 0.0
  temporal vote window: 3
  morphology: close, kernel 3

best by target temporal-weighted objective:
  mean IoU: 0.8586
  temporal IoU: 0.4938
  on threshold: 0.26
  keep threshold: 0.10
  temporal vote window: 3
  morphology: close, kernel 5

best by target temporal IoU alone:
  mean IoU: 0.8212
  temporal IoU: 0.5184
  EMA alpha: 0.50
  on threshold: 0.24
  keep threshold: 0.10
  temporal vote window: 5
  morphology: close, kernel 5
```

Interpretation:

```text
Even with target-oracle postprocess selection, mean IoU only reaches 0.8601.
The valid non-target-selected mean-best is 0.8586. Therefore postprocessing has
almost no remaining mean-IoU headroom on this checkpoint.

Temporal can be increased by smoothing, but the gain is paid for with mean IoU.
The next mean-IoU improvement must come from model output quality, not
threshold/morphology tuning.
```

## Temporal-Weighted Non-Target Postprocess Tuning

Question:

```text
Can the smoother temporal behavior seen in the target diagnostic be selected
legitimately from non-target validation windows?
```

Selection recipe:

```text
checkpoint: dense_union_unetpp_b4_raw_dilated_full_local_temporal005.pt
calibration windows: six non-target 30s val windows
objective: mean IoU + 3.0 * temporal IoU - 0.05 * area
```

Selected setting:

```text
output: dense_union_raw_dilated_full_local_temporal005_postprocess_tuning_temporal_weight3.json
EMA alpha: 0.25
on threshold: 0.30
keep threshold: 0.25
temporal vote: disabled
morphology: close, kernel 5
```

Held-out target result:

```text
output: dense_union_raw_dilated_full_local_temporal005_temporal_eval_temporal_weight3.json
mean IoU: 0.8557
temporal IoU: 0.4931
gt temporal IoU: 0.4788
mean area: 0.0760
```

Interpretation:

```text
This is a valid balanced setting for the current mean-best checkpoint. It
improves temporal over the mean-best postprocess:

  mean-best postprocess: mean 0.8586, temporal 0.4816
  temporal-weighted setting: mean 0.8557, temporal 0.4931

However, it still does not beat the old temporal-best:

  old temporal-best: mean 0.8296, temporal 0.5013

Conclusion: valid temporal postprocessing can recover some stability but not
enough. The remaining temporal gap needs a denser temporal model or tracking
signal, not only stronger smoothing.
```

## Flow-Warped Hysteresis Ablation

Question:

```text
Standard hysteresis keeps previous positive pixels in fixed screen coordinates.
That is weak for egocentric video because the camera and hands move constantly.
Would warping the previous mask into the current frame with optical flow improve
temporal consistency without target-window calibration?
```

Implementation:

```text
Added --flow-hysteresis to:
  evaluate_dense_temporal.py
  render_dense_union.py

Added --flow-hysteresis-options to:
  tune_temporal_postprocess.py

Method:
  compute Farneback optical flow from current frame to previous frame
  precompute remap grids once per 30s calibration window
  warp previous binary red mask into current frame
  keep warped previous positives only where current probability >= keep threshold

The first implementation recomputed optical flow inside the threshold grid and
was interrupted. It was then fixed by precomputing flow warp maps once per
window before building the postprocess cache.
```

Selection recipe:

```text
checkpoint: dense_union_unetpp_b4_raw_dilated_full_local_temporal005.pt
calibration windows: six non-target 30s val windows
objective: mean IoU + 3.0 * temporal IoU - 0.05 * area
grid:
  on thresholds: 0.24, 0.30, 0.38, 0.46
  keep thresholds: 0, 0.10, 0.20, 0.30
  flow hysteresis: off/on
  EMA: 0, 0.25
  temporal vote windows: 1, 3
  morphology: none, close5
```

Selected setting:

```text
output: dense_union_raw_dilated_full_local_temporal005_flow_tuning.json
EMA alpha: 0.25
on threshold: 0.38
keep threshold: 0.20
flow hysteresis: true
temporal vote: disabled
morphology: close, kernel 5

non-target aggregate:
  mean IoU: 0.8902
  temporal IoU: 0.4766
```

Held-out target result:

```text
output: dense_union_raw_dilated_full_local_temporal005_temporal_eval_flow_hyst020.json
mean IoU: 0.8545
temporal IoU: 0.4931
gt temporal IoU: 0.4788
mean area: 0.0759
```

Interpretation:

```text
Flow-warped hysteresis was selected by the non-target temporal-weighted
objective, but it did not improve the held-out target over the prior non-flow
balanced setting:

  non-flow balanced: mean 0.8557, temporal 0.4931
  flow hysteresis:   mean 0.8545, temporal 0.4931

The likely reason is that the model's red masks are broad workspace unions
rather than compact object instances. Optical flow can help propagate crisp
objects, but it has little leverage when the predicted region is already a
large static/workspace blob.

Conclusion: motion-compensated postprocessing is not enough on the current
dense-union output. Reaching 0.9 mean or clearly higher temporal IoU likely
requires improving the prediction target/model itself: instance/object-aware
outputs, dense pseudo-label propagation during training, or explicit memory.
```

## Low-LR Fine-Tune From Temporal Pair Checkpoint

Motivation:

```text
The current postprocess ceiling showed that thresholding, morphology, EMA,
temporal voting, and flow-warped hysteresis cannot raise the temporal005 model
above about 0.86 mean IoU. The model output itself needs to improve.
```

Implementation:

```text
Added --init-checkpoint to train_dense_union.py.

Behavior:
  load a compatible dense-union checkpoint before training
  evaluate epoch 0 on non-target validation frames
  initialize best_state from epoch 0
  save a fine-tuned state only if non-target validation objective improves

This prevents a fine-tune run from silently replacing the incumbent with a
worse validation-selected checkpoint.
```

Training recipe:

```text
checkpoint:
  dense_union_unetpp_b4_raw_dilated_full_local_temporal005_finetune_lr5e5.pt

init checkpoint:
  dense_union_unetpp_b4_raw_dilated_full_local_temporal005.pt

encoder: EfficientNet-B4 U-Net++
image size: 256
hand input mode: raw_dilated
loss: BCEWithLogits + Dice
temporal pair weight: 0.05
temporal change weight: 0.50
train samples: 4128 local non-target train+val frames
temporal pair samples: 4091 local non-target adjacent pairs
validation samples: 180 non-target validation frames
learning rate: 5e-5
epochs: 4
```

Training-selected result:

```text
epoch 0 incumbent:
  validation mean IoU: 0.9129
  target diagnostic mean IoU: 0.8568
  selected threshold: 0.46

best non-target validation epoch: 4
validation mean IoU: 0.9343
validation temporal IoU: 0.3254
selected threshold: 0.52
target_at_val_threshold mean IoU: 0.8647
target_at_val_threshold temporal IoU: 0.4746

Note:
  epoch 3 had a slightly higher target diagnostic mean, 0.8682, but it was not
  selected because selection must not use the held-out target window.
```

Official full-window raw target evaluation:

```text
output: dense_union_raw_dilated_full_local_temporal005_finetune_lr5e5_temporal_eval_raw.json
threshold: 0.52
mean IoU: 0.8624
temporal IoU: 0.4747
gt temporal IoU: 0.4788
```

Non-target postprocess tuning:

```text
output: dense_union_raw_dilated_full_local_temporal005_finetune_lr5e5_postprocess_tuning.json
selected on threshold: 0.46
selected keep threshold: 0.35
EMA: disabled
temporal vote: disabled
morphology: none
component filtering: disabled

non-target aggregate:
  mean IoU: 0.9280
  temporal IoU: 0.4665
```

Held-out target result with selected postprocess:

```text
output: dense_union_raw_dilated_full_local_temporal005_finetune_lr5e5_temporal_eval_hyst035.json
mean IoU: 0.8643
temporal IoU: 0.4769
gt temporal IoU: 0.4788
mean area: 0.0737
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_full_local_temporal005_finetune_lr5e5_target_30s_valtuned_hyst035/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_full_local_temporal005_finetune_lr5e5_target_30s_valtuned_hyst035/contact_sheet.jpg
```

Interpretation:

```text
This is the current valid mean-best result:

  previous mean-best: 0.8586
  fine-tuned mean-best: 0.8643
  improvement: +0.0057

The gain is small but important because it comes from model-output improvement,
not target-window postprocess calibration. It also exceeds the previous
checkpoint's target-oracle postprocess ceiling of 0.8601.

The temporal metric regressed:

  previous mean-best temporal: 0.4816
  fine-tuned mean-best temporal: 0.4769

Conclusion: low-LR fine-tuning can continue improving mean IoU, but the dense
union objective still does not solve temporal consistency. The next serious
step toward 0.9 likely needs denser temporal supervision or an instance-aware
target, not only more postprocess tuning.
```

## Warm-Started Ring/Distance Prior Expansion

Question:

```text
Can we keep the strong raw+dilated model behavior while adding explicit
outside-hand ring and smooth distance channels, instead of training the larger
hand-prior input from scratch?
```

Implementation:

```text
Added channel-expanding checkpoint loading to train_dense_union.py.

When --init-checkpoint has fewer input channels than the requested hand-prior
mode, the loader now adapts compatible 4D convolution weights. For the
EfficientNet stem in this run:

  encoder._conv_stem.weight: [48, 5, 3, 3] -> [48, 7, 3, 3]

The shared channels are copied exactly:

  RGB
  raw hand probability
  dilated hand proximity probability

The new channels are zero-initialized:

  outside-hand ring
  smooth distance/proximity prior

This makes epoch 0 functionally equivalent to the previous best checkpoint and
lets fine-tuning learn whether the new geometry channels add signal.
```

Training recipe:

```text
checkpoint:
  dense_union_unetpp_b4_raw_dilated_ring_distance_warm_finetune.pt

init checkpoint:
  dense_union_unetpp_b4_raw_dilated_full_local_temporal005_finetune_lr5e5.pt

encoder: EfficientNet-B4 U-Net++
image size: 256
hand input mode: raw_dilated_ring_distance
loss: BCEWithLogits + Dice
temporal pair weight: 0.05
temporal change weight: 0.50
train samples: 4128 local non-target train+val frames
temporal pair samples: 4091 local non-target adjacent pairs
validation samples: 180 non-target validation frames
learning rate: 5e-5
epochs: 4
extra channel init: zero
```

Training-selected result:

```text
epoch 0 incumbent:
  validation mean IoU: 0.9343
  validation temporal IoU: 0.3254
  selected threshold: 0.52
  target diagnostic mean IoU: 0.8647
  target diagnostic temporal IoU: 0.4746

best non-target validation epoch: 3
validation mean IoU: 0.9432
validation temporal IoU: 0.3244
selected threshold: 0.48
target_at_val_threshold mean IoU: 0.8644
target_at_val_threshold temporal IoU: 0.4752

Note:
  epoch 4 had better target-window diagnostics, 0.8679 mean and 0.4795
  temporal, but it was not selected because selection must not use the
  held-out target window.
```

Official full-window raw target evaluation:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_temporal_eval_raw.json
threshold: 0.48
mean IoU: 0.8634
temporal IoU: 0.4752
gt temporal IoU: 0.4788
```

Non-target postprocess tuning:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_postprocess_tuning.json
selection source: non-target validation windows
selected on threshold: 0.46
selected keep threshold: 0.0
EMA alpha: 0.25
temporal vote: disabled
morphology: close, kernel 3
component filtering: disabled

non-target aggregate:
  mean IoU: 0.9322
  temporal IoU: 0.4672
```

Held-out target result with selected postprocess:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_temporal_eval_ema025_close3.json
mean IoU: 0.8647
temporal IoU: 0.4777
gt temporal IoU: 0.4788
mean area: 0.0744
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_ring_distance_warm_finetune_target_30s_ema025_close3/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_ring_distance_warm_finetune_target_30s_ema025_close3/contact_sheet.jpg
```

Interpretation:

```text
This is a new valid mean-best, but only by a very small margin:

  previous mean-best: 0.8643 mean / 0.4769 temporal
  warm prior mean-best: 0.8647 mean / 0.4777 temporal
  mean improvement: +0.0004
  temporal improvement: +0.0008

The ablation answers the hand-prior question: separating raw hand, dilated
proximity, outside ring, and smooth distance is reasonable and does not damage
the model when warm-started, but it does not unlock the requested 0.9 IoU or a
large temporal improvement.

The qualitative contact sheet still looks like a broad dense workspace/object
union selector rather than a crisp instance-aware tracker. The remaining
bottleneck is likely target definition and temporal/object identity, not
another static hand-prior channel.

Cleanup:
  removed the superseded fine-tune .pt after saving this self-contained
  warm-start checkpoint; JSON summaries and evaluations are retained as history.
```

## Bidirectional Temporal Smoothing Ablation

Question:

```text
Can offline bidirectional probability smoothing improve temporal consistency
without target-window calibration? The previous smoother was forward-only EMA,
which can lag behind newly appearing masks and only uses past frames.
```

Implementation:

```text
Added --smoothing-mode to evaluate_dense_temporal.py and render_dense_union.py:

  forward
  bidirectional

Bidirectional mode computes a forward EMA and a backward EMA over the full
30-second clip, then averages the two probability maps before thresholding.
This is an offline postprocess; it is appropriate for full-video rendering and
evaluation, not a causal live system.

Added --smoothing-modes to tune_temporal_postprocess.py so this choice is
selected only from non-target validation windows.
```

Tuner memory fix:

```text
The first broad sweep was killed because tune_temporal_postprocess.py cached
binary masks for every combination of:

  smoothing
  threshold
  hysteresis keep threshold
  flow flag

Reworked the tuner to cache smoothed probabilities and build binary masks per
candidate row. Also made per-window row details optional and retained detailed
window metrics only for the selected best row. This makes broader sweeps
possible without a combinatorial mask-cache blowup.
```

Mean-weighted focused sweep:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_bidirectional_focused_tuning.json
selection source: non-target validation windows
temporal weight: 1.0
selected on threshold: 0.44
selected keep threshold: 0.0
EMA alpha: 0.40
smoothing mode: bidirectional
temporal vote: disabled
morphology: none
component filtering: disabled

non-target aggregate:
  mean IoU: 0.9352
  temporal IoU: 0.4666
```

Held-out target result, mean-weighted selected setting:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_temporal_eval_bidir040_thr044.json
mean IoU: 0.8651
temporal IoU: 0.4767
gt temporal IoU: 0.4788
mean area: 0.0743
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_ring_distance_warm_finetune_target_30s_bidir040_thr044/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_ring_distance_warm_finetune_target_30s_bidir040_thr044/contact_sheet.jpg
```

Temporal-weighted focused sweep:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_bidirectional_temporal_weight3_tuning.json
selection source: non-target validation windows
temporal weight: 3.0
selected on threshold: 0.38
selected keep threshold: 0.20
EMA alpha: 0.40
smoothing mode: bidirectional
temporal vote: disabled
morphology: close, kernel 5
component filtering: disabled

non-target aggregate:
  mean IoU: 0.9096
  temporal IoU: 0.4776
```

Held-out target result, temporal-weighted selected setting:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_temporal_eval_temporal_weight3_bidir040_close5_keep020.json
mean IoU: 0.8598
temporal IoU: 0.4952
gt temporal IoU: 0.4788
mean area: 0.0771
```

Interpretation:

```text
Bidirectional smoothing gives a tiny new valid mean-best:

  previous mean-best: 0.8647 mean / 0.4777 temporal
  bidirectional mean-best: 0.8651 mean / 0.4767 temporal
  mean improvement: +0.0004
  temporal change: -0.0010

The temporal-heavy validation objective recovers a stronger temporal tradeoff
on the same checkpoint:

  0.8598 mean / 0.4952 temporal

This is close to the historical balanced temporal setting but still below the
older raw_dilated temporal-best of 0.5013. The key conclusion is unchanged:
offline smoothing can trade a few points of mean IoU for stability, but it does
not create the substantial temporal consistency improvement needed for the
0.9-quality pipeline. The model still lacks object identity and track-aware
supervision.
```

## Flow-Aware Probability Smoothing Ablation

Question:

```text
Can optical-flow warping make probability smoothing more useful? Plain
bidirectional EMA averages masks in image coordinates, so moving masks may blur
or lag. Flow-aware smoothing should warp the previous/future probability field
into the current frame before applying EMA.
```

Implementation:

```text
Added smoothing modes:

  flow_forward
  flow_bidirectional

Flow modes use the existing Farneback optical-flow remap utilities, but apply
them to floating probability maps with bilinear interpolation rather than to
binary masks. The flow_bidirectional mode computes flow-aware forward and
backward EMA passes, then averages them.

Updated:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py
```

Runtime note:

```text
Flow-aware smoothing is substantially slower than plain bidirectional EMA. The
focused flow sweeps were CPU-bound after prediction because Farneback flow and
per-row postprocess evaluation dominate runtime. The memory-safe tuner rewrite
kept RSS bounded, but this path is not efficient enough for broad interactive
searches without caching flow-smoothed probabilities to disk or vectorizing the
postprocess grid.
```

Flow-vs-plain temporal-weighted sweep:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_flow_smoothing_temporal_weight3_tuning.json
selection source: non-target validation windows
temporal weight: 3.0

global selected setting:
  smoothing mode: bidirectional
  EMA alpha: 0.55
  on threshold: 0.30
  keep threshold: 0.0
  morphology: close, kernel 5
  non-target mean IoU: 0.9128
  non-target temporal IoU: 0.4775

held-out target:
  output: dense_union_raw_dilated_ring_distance_warm_finetune_temporal_eval_flow_smoothing_temporal_weight3_bidir055_close5.json
  mean IoU: 0.8609
  temporal IoU: 0.4962
  gt temporal IoU: 0.4788

best flow_bidirectional validation row inside that sweep:
  EMA alpha: 0.55
  on threshold: 0.38
  morphology: close, kernel 5
  non-target mean IoU: 0.9255
  non-target temporal IoU: 0.4716

held-out target diagnostic for best flow row:
  output: dense_union_raw_dilated_ring_distance_warm_finetune_temporal_eval_flow_bidir055_thr038_close5.json
  mean IoU: 0.8661
  temporal IoU: 0.4859
```

Flow-only mean-weighted sweep:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_flow_only_mean_tuning.json
selection source: non-target validation windows, restricted to flow_bidirectional
temporal weight: 1.0
selected on threshold: 0.42
selected keep threshold: 0.0
EMA alpha: 0.55
smoothing mode: flow_bidirectional
temporal vote: disabled
morphology: none
component filtering: disabled

non-target aggregate:
  mean IoU: 0.9350
  temporal IoU: 0.4670
```

Held-out target result, flow-only mean-selected setting:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_temporal_eval_flow_only_mean_bidir055_thr042.json
mean IoU: 0.8664
temporal IoU: 0.4778
gt temporal IoU: 0.4788
mean area: 0.0746
```

Qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_ring_distance_warm_finetune_target_30s_flow_bidir055_thr042/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_dilated_ring_distance_warm_finetune_target_30s_flow_bidir055_thr042/contact_sheet.jpg
```

Interpretation:

```text
Flow-aware smoothing gives a tiny new valid mean-best when considered as a
predeclared flow-only ablation:

  previous mean-best: 0.8651 mean / 0.4767 temporal
  flow-only mean-best: 0.8664 mean / 0.4778 temporal
  mean improvement: +0.0013
  temporal improvement: +0.0011

However, the global non-target temporal-weighted sweep did not select
flow_bidirectional. It selected plain bidirectional smoothing. Flow smoothing
therefore helps the target diagnostic a little in the flow-only family, but it
does not convincingly solve the temporal problem or dominate validation.

The core bottleneck remains the dense union model's lack of object identity and
track-aware supervision. Postprocess tweaks are now yielding only thousandth-
level mean gains and small stability tradeoffs.
```

## Two-Checkpoint Probability Ensemble Probe

Question:

```text
Do the current mean-best checkpoint and the historical temporal-best checkpoint
make complementary errors? If so, a non-target-selected probability ensemble
might improve mean IoU or temporal IoU without retraining.
```

Implementation:

```text
Added tune_ensemble_postprocess.py.

It loads two dense checkpoints, predicts probability maps for each calibration
window, blends probabilities as:

  p = weight_a * p_a + (1 - weight_a) * p_b

and then reuses the same non-target postprocess tuner for threshold, smoothing,
hysteresis, morphology, and temporal voting.

Checkpoint A:
  dense_union_unetpp_b4_raw_dilated_ring_distance_warm_finetune.pt
  hand input mode: raw_dilated_ring_distance

Checkpoint B:
  dense_union_unetpp_b4_raw_dilated.pt
  hand input mode: raw_dilated
```

Runtime note:

```text
The first broad grid was interrupted after it became too slow. It had to run
two model passes per window and then evaluate a large postprocess grid. No
partial output was kept. A focused probe was used instead to answer whether the
ensemble direction looked promising.
```

Focused probe:

```text
output: dense_union_ensemble_warm_plus_rawdilated_focused_probe.json
selection source: 3 non-target validation windows
weights for checkpoint A: 0, 0.50, 0.75, 1.0
thresholds: 0.38, 0.42, 0.46, 0.50
EMA alphas: 0, 0.40
smoothing modes: bidirectional
keep thresholds: 0, 0.20
morphology: none or close, kernel 5
temporal weight: 1.0
```

Validation-selected result:

```text
weight_a: 1.0
weight_b: 0.0
on threshold: 0.50
EMA alpha: 0.40
smoothing mode: bidirectional
keep threshold: 0.0
morphology: none

non-target mean IoU: 0.9422
non-target temporal IoU: 0.5511
```

Held-out target at selected setting:

```text
mean IoU: 0.8645
temporal IoU: 0.4756
gt temporal IoU: 0.4788
mean area: 0.0739
```

Interpretation:

```text
The focused validation sweep selected the current mean-best checkpoint alone.
The historical temporal-best checkpoint did not contribute useful complementary
probability signal under this probe. This argues against spending more time on
simple two-checkpoint probability ensembling, at least with these two saved
models and this validation setup.

The main improvement path still appears to require model/data changes that
encode object identity or track-aware supervision, rather than mixing dense
union checkpoints after the fact.
```

## Hand-Exclusion Postprocess Ablation

Question:

```text
Should the red object mask explicitly subtract the cyan hand mask? The object
ground truth excludes hands, and the qualitative overlays often show red
regions close to the hands. A hand-exclusion postprocess might reduce false
positive object pixels on the hands.
```

Implementation:

```text
Added hand-exclusion modes:

  none
  raw
  dilated

The evaluator can now run predict_probs_and_hand_masks(), returning both object
probabilities and low-resolution frozen hand masks. After thresholding,
temporal voting, and morphology, the selected red mask can subtract either the
raw hand mask or a dilated hand mask before connected-component filtering.

Updated:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py

Also updated stale call sites in:
  diagnose_target_postprocess_ceiling.py
  tune_ensemble_postprocess.py
```

Smoke test:

```text
One non-target validation window, threshold 0.42, no smoothing:

none:
  validation mean IoU: 0.9460
  validation temporal IoU: 0.5494

raw:
  validation mean IoU: 0.9365
  validation temporal IoU: 0.5412

dilated, kernel 5:
  validation mean IoU: 0.8911
  validation temporal IoU: 0.5184
```

Focused full sweep:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_hand_exclusion_tuning.json
selection source: non-target validation windows
thresholds: 0.38, 0.42, 0.46
EMA alphas: 0, 0.55
smoothing modes: bidirectional, flow_bidirectional
keep thresholds: 0, 0.20
morphology: none or close, kernel 5
hand exclusion modes: none, raw, dilated
dilated kernels: 5, 9
temporal weight: 1.0
```

Validation-selected result:

```text
hand exclusion mode: none
on threshold: 0.42
EMA alpha: 0.55
smoothing mode: flow_bidirectional
morphology: none

non-target mean IoU: 0.9350
non-target temporal IoU: 0.4670
```

Best raw-hand exclusion row:

```text
hand exclusion mode: raw
on threshold: 0.42
EMA alpha: 0.55
smoothing mode: flow_bidirectional
morphology: none

non-target mean IoU: 0.9306
non-target temporal IoU: 0.4638
```

Best dilated-hand exclusion row:

```text
hand exclusion mode: dilated
hand exclusion kernel: 5
on threshold: 0.42
EMA alpha: 0.55
smoothing mode: flow_bidirectional
morphology: none

non-target mean IoU: 0.9026
non-target temporal IoU: 0.4536
```

Held-out target at selected no-exclusion setting:

```text
mean IoU: 0.8664
temporal IoU: 0.4778
gt temporal IoU: 0.4788
```

Interpretation:

```text
Hand exclusion is harmful under non-target validation. It likely removes true
object pixels exactly where active objects and hands touch or occlude each
other. Dilating the hand exclusion region is much worse, confirming that the
model needs near-hand object evidence rather than a hard carved-out hand
neighborhood.

This supports the current design choice: use hands as a conditioning prior, but
do not subtract them from the final object mask.
```

## Current Checkpoint Component-Filtering Sweep

Question:

```text
Can connected-component cleanup remove small false-positive red islands and
improve the current flow-smoothed mean-best result? Earlier component filtering
was negative on older checkpoints, but the current checkpoint/smoothing setting
has different error structure.
```

Focused sweep:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_component_filter_current_tuning.json
selection source: non-target validation windows
checkpoint: dense_union_unetpp_b4_raw_dilated_ring_distance_warm_finetune.pt
thresholds: 0.38, 0.42, 0.46
EMA alpha: 0.55
smoothing mode: flow_bidirectional
keep threshold: 0
morphology: none or close, kernel 5
min component area fractions: 0, 0.00025, 0.0005, 0.001, 0.002
max components: 0, 1, 2, 3, 4, 5
hand exclusion: none
temporal weight: 1.0
```

Validation-selected result:

```text
min component area fraction: 0.0
max components: 0
on threshold: 0.42
morphology: none

non-target mean IoU: 0.9350
non-target temporal IoU: 0.4670
```

Best area-filtered row:

```text
min component area fraction: 0.00025
max components: 0
on threshold: 0.42
morphology: none

non-target mean IoU: 0.9348
non-target temporal IoU: 0.4671
```

Best component-capped row:

```text
min component area fraction: 0.0
max components: 5
on threshold: 0.42
morphology: none

non-target mean IoU: 0.9316
non-target temporal IoU: 0.4689
```

Held-out target check for max_components=5:

```text
output: dense_union_raw_dilated_ring_distance_warm_finetune_temporal_eval_component_cap5.json
mean IoU: 0.8663
temporal IoU: 0.4777
gt temporal IoU: 0.4788
```

Interpretation:

```text
Validation selected no component filtering. The closest area-filtering row was
almost tied but slightly worse, and component caps traded away too much mean
IoU for tiny temporal gains. On the held-out target, max_components=5 is
essentially identical but slightly worse than the current unfiltered mean-best.

Conclusion: small disconnected red islands are not the limiting error. The
remaining gap is not solved by component cleanup; it still points to missing
object identity / track-aware supervision rather than another final mask
cleanup rule.
```

## Clean Separated Hand-Prior Fine-Tune

Question:

```text
Should the relevance model receive raw hand, distance, and outside-hand ring
priors as separate channels instead of a filled dilated hand prior?

Motivation: the visible cyan hand masks are sharp, but a filled dilation channel
can encourage the red relevance output to treat a widened hand/contact blob as
foreground. A cleaner input should keep raw hand evidence separate from
near-hand object evidence.
```

Implementation:

```text
Added semantic checkpoint channel mapping to train_dense_union.py.

Previous generic warm-start behavior copied first-conv channels positionally.
That is risky when changing hand-prior modes, because an old "dilated" channel
can accidentally initialize a new "ring" channel.

The new named mapping transfers:
  rgb_r -> rgb_r
  rgb_g -> rgb_g
  rgb_b -> rgb_b
  hand_raw -> hand_raw
  hand_ring -> hand_ring
  hand_distance -> hand_distance

and drops hand_dilated when the target model does not use it.
```

Run:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_distance_named_finetune.pt
init checkpoint: dense_union_unetpp_b4_raw_dilated_ring_distance_warm_finetune.pt
input mode: raw_ring_distance
removed input channel: filled hand dilation
train samples requested: 6000
actual train frames: 4128
temporal pair samples: 4091
epochs: 3
learning rate: 3e-5
temporal pair weight: 0.05
threshold source: non-target validation
best validation threshold: 0.50
```

Training summary at best validation epoch:

```text
non-target validation mean IoU: 0.9463
non-target validation temporal IoU: 0.3245
target mean IoU at validation threshold: 0.8692
target temporal IoU at validation threshold: 0.4696
```

Corrected full 30-second temporal evaluation:

```text
output: dense_union_raw_ring_distance_named_finetune_temporal_eval_flow_bidir055_thr050.json
threshold: 0.50
EMA alpha: 0.55
smoothing mode: flow_bidirectional
morphology: none
hand exclusion: none

selected_union_mean_iou: 0.8691
selected_temporal_union_iou: 0.4714
gt_temporal_union_iou: 0.4788
selected_mean_area: 0.0736
```

Qualitative:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_distance_named_finetune_target_30s_flow_bidir055_thr050/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_distance_named_finetune_target_30s_flow_bidir055_thr050/contact_sheet.jpg
```

Interpretation:

```text
This is a small but real mean-IoU improvement over the previous mean-best
0.8664 result. The cleaner prior is therefore worth keeping as the mean-best
model.

It does not improve temporal consistency. Temporal IoU falls from 0.4778 to
0.4714 under the same flow-bidirectional smoothing setup. The contact sheet
still shows broad red relevance over the work surface/object region, so the
remaining failure is not just hand-mask dilation. The next bottleneck is
object/relevance specificity and persistent object identity over time.
```

## Focused Temporal Postprocess For Clean Separated-Prior Checkpoint

Question:

```text
Can non-target temporal postprocess tuning recover higher temporal IoU for the
new raw_ring_distance mean-best checkpoint without losing the mean-IoU gain?
```

Broad sweep note:

```text
An initial broad grid included threshold, EMA, flow smoothing, hysteresis,
temporal voting, morphology, component caps, and component area filters. It was
terminated after proving too slow for interactive iteration. No result file was
written. The useful next move was a focused sweep over the knobs that had moved
metrics in prior runs: threshold, EMA/bidirectional or flow-bidirectional
smoothing, hysteresis keep, and close/no-close morphology.
```

Focused sweep:

```text
output: dense_union_raw_ring_distance_named_finetune_focused_temporal_tuning.json
checkpoint: dense_union_unetpp_b4_raw_ring_distance_named_finetune.pt
selection source: six non-target validation windows
objective: mean IoU + 3.0 * temporal IoU - 0.05 * selected area

thresholds: 0.42, 0.46, 0.50, 0.54, 0.58
keep thresholds: 0, 0.15, 0.25, 0.35
EMA alphas: 0, 0.35, 0.55, 0.70
smoothing modes: bidirectional, flow_bidirectional
morphology: none or close, kernel 5
component filtering: disabled
hand exclusion: none
temporal voting: disabled for this sweep
```

Validation-selected row:

```text
threshold: 0.42
keep threshold: 0.35
EMA alpha: 0.55
smoothing mode: bidirectional
morphology: close, kernel 5

non-target validation mean IoU: 0.9231
non-target validation temporal IoU: 0.4742
```

Held-out target result:

```text
output: dense_union_raw_ring_distance_named_finetune_temporal_eval_temporal_weight3_bidir055_close5_keep035_thr042.json
selected_union_mean_iou: 0.8683
selected_temporal_union_iou: 0.4857
gt_temporal_union_iou: 0.4788
selected_mean_area: 0.0755
```

Qualitative:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_distance_named_finetune_target_30s_temporal_weight3_bidir055_close5_keep035_thr042/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_distance_named_finetune_target_30s_temporal_weight3_bidir055_close5_keep035_thr042/contact_sheet.jpg
```

Temporal-vote probe:

```text
output: dense_union_raw_ring_distance_named_finetune_temporal_vote_probe.json
fixed base setting: threshold 0.42, keep 0.35, EMA 0.55, bidirectional smoothing, close kernel 5
tested vote windows: 1, 3, 5, 7
tested vote fractions: 0.5, 0.67, 0.8

best non-target objective: temporal_window 1, temporal_min_vote_frac 0.5
best non-window-1 temporal row: window 7, vote fraction 0.5, validation temporal IoU 0.4752 but validation mean IoU 0.8828
```

Interpretation:

```text
This is the best current mean/temporal balance for the raw_ring_distance
checkpoint. It gives up only 0.0008 mean IoU relative to the pure mean-best
setting, while improving target temporal IoU from 0.4714 to 0.4857.

Temporal voting is not useful here. It can slightly raise non-target temporal
IoU in some rows, but the mean-IoU loss is too large and validation does not
select it. The remaining temporal issue is therefore unlikely to be solved by a
binary-mask vote; it likely requires track-aware/object-identity modeling or a
flow-aware training loss instead of another final mask cleanup rule.
```

## Inference-Time Hand-Prior Radius Probe

Question:

```text
The raw_ring_distance model separates raw hand, outside-hand ring, and
hand-distance priors. Does changing the ring/distance radius at inference time
improve object specificity or temporal stability without retraining?
```

Implementation:

```text
Added --hand-kernel-size-override to:
  evaluate_dense_temporal.py
  render_dense_union.py
  tune_temporal_postprocess.py

The checkpoint metadata is left unchanged. Each result records:
  checkpoint_hand_kernel_size
  hand_kernel_size
```

Fixed postprocess setting:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_distance_named_finetune.pt
threshold: 0.42
keep threshold: 0.35
EMA alpha: 0.55
smoothing mode: bidirectional
morphology: close, kernel 5
temporal vote: disabled
selection source: same six non-target validation windows
objective: mean IoU + 3.0 * temporal IoU - 0.05 * selected area
```

Results:

```text
kernel  val_mean  val_temp  val_objective  target_mean  target_temp
9       0.922943  0.474261  2.341815      0.868282     0.485708
15      0.923082  0.474168  2.341677      0.868298     0.485678
21      0.922931  0.474216  2.341669      0.868315     0.485740
31      0.922845  0.474144  2.341369      0.868310     0.485203
```

Files:

```text
dense_union_raw_ring_distance_named_finetune_kernel09_probe.json
dense_union_raw_ring_distance_named_finetune_kernel21_probe.json
dense_union_raw_ring_distance_named_finetune_kernel31_probe.json
dense_union_raw_ring_distance_named_finetune_focused_temporal_tuning.json
```

Interpretation:

```text
This ablation is effectively flat. Kernel 9 has the highest non-target
objective, but only by 0.000138 over the checkpoint-native kernel 15. Target
metrics are also indistinguishable at the useful precision of this evaluation.

Conclusion: the ring/distance prior radius is not the current bottleneck. Keep
kernel 15 as the stable default because it matches training/checkpoint metadata.
Further progress toward 0.9 mean IoU and substantially higher temporal IoU will
need better object/relevance specificity or temporal/object-identity modeling,
not another small hand-prior radius tweak.
```

## Mean-Focused Postprocess Sweep For Clean Separated-Prior Checkpoint

Question:

```text
The pure mean-best target result used a single flow-bidirectional setting
(threshold 0.50, EMA 0.55). Can a non-target validation-selected mean-focused
postprocess sweep find extra mean-IoU headroom without target calibration?
```

Sweep:

```text
output: dense_union_raw_ring_distance_named_finetune_mean_focused_tuning.json
checkpoint: dense_union_unetpp_b4_raw_ring_distance_named_finetune.pt
selection source: six non-target validation windows
objective: mean IoU + 0.10 * temporal IoU - 0.05 * selected area

thresholds: 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62
keep thresholds: 0, 0.20, 0.35
EMA alphas: 0, 0.35, 0.55, 0.70
smoothing modes: bidirectional, flow_bidirectional
morphology: none or close, kernel 5
component filtering: disabled
hand exclusion: none
temporal voting: disabled
```

Validation-selected row:

```text
threshold: 0.50
keep threshold: 0.0
EMA alpha: 0.35
smoothing mode: flow_bidirectional
morphology: none

non-target validation mean IoU: 0.9391
non-target validation temporal IoU: 0.4655
non-target selected mean area: 0.0768
```

Held-out target result at selected setting:

```text
selected_union_mean_iou: 0.8680
selected_temporal_union_iou: 0.4704
gt_temporal_union_iou: 0.4788
selected_mean_area: 0.0736
```

Comparison:

```text
pure mean-best target:          0.8691 mean / 0.4714 temporal
current temporal-heavy target:  0.8683 mean / 0.4857 temporal
mean-focused selected target:   0.8680 mean / 0.4704 temporal
```

Interpretation:

```text
This sweep improved non-target validation mean, but did not transfer into a new
held-out target best. It is a negative postprocess result. The current target
mean appears saturated around 0.868-0.869 for this dense checkpoint under
threshold/EMA/hysteresis/morphology postprocessing.

The remaining path to 0.9 likely requires model-side changes or better
object-specific temporal supervision, not another scalar threshold/smoothing
sweep.
```

## Outside-Only Distance Hand Prior

Question:

```text
The raw_ring_distance model passes raw hand, outside-hand ring, and a smooth
distance prior. But the distance prior is high inside the hand as well as near
the hand. Since hands are rendered separately in cyan, does suppressing the
distance prior inside hand pixels improve object relevance specificity?
```

Implementation:

```text
Added hand input mode: raw_ring_outer_distance

Feature channels:
  RGB
  raw hand probability
  outside-hand ring: dilate(hand) - hand
  outside-only distance proximity: distance_proximity(hand) * (1 - hand)

The model architecture and channel count are unchanged relative to
raw_ring_distance. The checkpoint warm-starts directly from
dense_union_unetpp_b4_raw_ring_distance_named_finetune.pt, so the old distance
weights receive the new outside-only distance signal.
```

Training:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
init checkpoint: dense_union_unetpp_b4_raw_ring_distance_named_finetune.pt
train samples requested: 6000
actual train frames: 4128
temporal pair samples: 4091
epochs: 2
learning rate: 2e-5
temporal pair weight: 0.05
threshold source during training summary: non-target validation frames
```

Training summary:

```text
epoch 0 target at validation threshold 0.50:
  mean IoU: 0.8691
  temporal IoU: 0.4698

epoch 1 target at validation threshold 0.44:
  mean IoU: 0.8679
  temporal IoU: 0.4714

epoch 2 target at validation threshold 0.54:
  mean IoU: 0.8688
  temporal IoU: 0.4731
```

Focused non-target temporal sweep:

```text
output: dense_union_raw_ring_outer_distance_finetune_focused_temporal_tuning.json
selection source: six non-target validation windows
objective: mean IoU + 3.0 * temporal IoU - 0.05 * selected area

thresholds: 0.38, 0.42, 0.46, 0.50, 0.54
keep thresholds: 0, 0.20, 0.35
EMA alphas: 0, 0.35, 0.55
smoothing modes: bidirectional, flow_bidirectional
morphology: none or close, kernel 5
component filtering: disabled
hand exclusion: none
temporal voting: disabled
```

Validation-selected row:

```text
threshold: 0.38
keep threshold: 0.35
EMA alpha: 0.55
smoothing mode: bidirectional
morphology: close, kernel 5

non-target validation mean IoU: 0.9256
non-target validation temporal IoU: 0.4744
non-target selected mean area: 0.0781
```

Held-out target result at selected setting:

```text
output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_bidir055_close5_keep035_thr038.json
selected_union_mean_iou: 0.8702
selected_temporal_union_iou: 0.4904
gt_temporal_union_iou: 0.4788
selected_mean_area: 0.0761
```

Qualitative:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_bidir055_close5_keep035_thr038/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_bidir055_close5_keep035_thr038/contact_sheet.jpg
```

Comparison:

```text
previous pure mean-best:          0.8691 mean / 0.4714 temporal
previous temporal-heavy best:     0.8683 mean / 0.4857 temporal
outside-only distance selected:   0.8702 mean / 0.4904 temporal
historical temporal-best:         0.8296 mean / 0.5013 temporal
```

Interpretation:

```text
This is a real incremental improvement and becomes the current best balanced
checkpoint. Suppressing the distance-prior channel inside hand pixels improves
both target mean IoU and temporal IoU compared with the prior raw_ring_distance
model under non-target-selected postprocessing.

It still does not reach the 0.9 mean-IoU goal, and the qualitative contact
sheet still shows broad work-surface/object relevance. The next likely
bottleneck remains object specificity or explicit temporal/object identity,
but the hand-prior ablation confirms that keeping hand interior evidence out of
the proximity channel is beneficial.
```

## Multi-Ring Outside-Distance Probe

Error localization before the probe:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
postprocess: threshold 0.38, keep 0.35, EMA 0.55, bidirectional smoothing, close kernel 5
target mean IoU reproduced by diagnostic: 0.870192

total prediction area fraction: 0.076119
total GT area fraction: 0.076225
false-positive area fraction: 0.003703
false-negative area fraction: 0.003808

FP distribution:
  raw hand: 10.7%
  ring: 22.2%
  near hand outside ring: 60.7%
  far from hand: 6.4%

FN distribution:
  raw hand: 12.3%
  ring: 18.1%
  near hand outside ring: 47.0%
  far from hand: 22.6%
```

Question:

```text
Most remaining errors are near the hand but not necessarily inside the immediate
ring. Would splitting proximity into an immediate ring and a broader outer
ring let the model distinguish contact-adjacent objects from broader reachable
background better than a single ring plus smooth outside-distance field?
```

Implementation:

```text
Added hand input mode: raw_multi_ring_outer_distance

Feature channels:
  RGB
  raw hand probability
  inner ring: dilate(hand, kernel 15) - hand
  outer ring: dilate(hand, kernel 31) - dilate(hand, kernel 15)
  outside-only distance proximity

Warm-start:
  source: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
  copied: hand_ring -> hand_inner_ring
  copied: hand_outer_distance -> hand_outer_distance
  initialized: hand_outer_ring -> zero
```

Training:

```text
checkpoint: dense_union_unetpp_b4_raw_multi_ring_outer_distance_finetune.pt
train samples requested: 6000
actual train frames: 4128
temporal pair samples: 4091
epochs: 2
learning rate: 2e-5
temporal pair weight: 0.05
```

Training summary:

```text
epoch 0 target at validation threshold 0.54:
  mean IoU: 0.8688
  temporal IoU: 0.4731

epoch 1 target at validation threshold 0.44:
  mean IoU: 0.8677
  temporal IoU: 0.4707

epoch 2 target at validation threshold 0.50:
  mean IoU: 0.8674
  temporal IoU: 0.4697

epoch 2 non-target validation mean IoU: 0.9518
```

Held-out target check with the current temporal-heavy recipe:

```text
output: dense_union_raw_multi_ring_outer_distance_finetune_temporal_eval_bidir055_close5_keep035_thr038.json
threshold: 0.38
keep threshold: 0.35
EMA alpha: 0.55
smoothing mode: bidirectional
morphology: close, kernel 5

selected_union_mean_iou: 0.8671
selected_temporal_union_iou: 0.4873
gt_temporal_union_iou: 0.4788
```

Interpretation:

```text
Negative result. The extra outer-ring channel improves non-target validation
mean IoU but hurts held-out target mean IoU. It likely overfits the validation
proximity distribution instead of improving transferable object specificity.

The current best remains raw_ring_outer_distance. The multi-ring checkpoint was
removed from kept .pt outputs; JSON summaries remain as provenance.
```

## Area-Cap Temporal Postprocess Probe

Question:

```text
The current best has nearly matched mean predicted area and GT area on the
held-out target, but remaining FP/FN area is balanced. Can a non-target
validation-selected per-frame area cap trim unstable excess object pixels while
preserving temporal stability?
```

Implementation:

```text
Extended tune_temporal_postprocess.py with:
  --min-area-fracs
  --max-area-fracs

Unlike the older standalone area tuner, this path combines area constraints
with the actual temporal pipeline: EMA smoothing, hysteresis, temporal voting,
morphology, hand exclusion, and component filtering.

Area caps use the smoothed probability map to keep the top-k pixels inside the
current binary mask. Area floors use the top-k pixels globally if a mask is too
small.
```

Focused sweep:

```text
output: dense_union_raw_ring_outer_distance_finetune_area_cap_temporal_tuning.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: six non-target validation windows
objective: mean IoU + 3.0 * temporal IoU - 0.05 * selected area

fixed threshold: 0.38
fixed keep threshold: 0.35
fixed EMA alpha: 0.55
fixed smoothing: bidirectional
fixed morphology: close, kernel 5
tested max area caps: 0, 0.075, 0.08, 0.085, 0.09, 0.10
area floor: disabled
```

Validation rows:

```text
max_area  val_mean  val_temporal  val_area
0         0.9256    0.4744        0.0781
0.075     0.8182    0.4415        0.0512
0.080     0.8268    0.4441        0.0524
0.085     0.8349    0.4471        0.0537
0.090     0.8419    0.4495        0.0548
0.100     0.8503    0.4543        0.0568
```

Held-out target at validation-selected setting:

```text
selected max area cap: 0
selected_union_mean_iou: 0.8702
selected_temporal_union_iou: 0.4904
```

Interpretation:

```text
Negative result. Area caps remove too many true object pixels on non-target
validation and are not selected. The no-cap setting remains the current best.

This closes another simple postprocess path: scalar threshold, EMA/hysteresis,
morphology, component filtering, hand exclusion, temporal voting, hand-prior
radius, and now area caps have all failed to close the gap to 0.9. Further
progress likely requires model-side object specificity or explicit temporal /
object-identity supervision.
```

## Hard Raw-Ring Outside-Distance Probe

Question:

```text
Would a hard thresholded hand/ring prior help by making the hand boundary
cleaner, or does the selector benefit from soft hand uncertainty?
```

Implementation:

```text
Added hand input mode: hard_raw_ring_outer_distance

Channels:
  hand_raw: thresholded hand prior
  hand_ring: hard outside ring from thresholded hand dilation
  hand_outer_distance: outside-only hard-hand distance/proximity

Warm-started from the current raw_ring_outer_distance checkpoint.
```

Training / target check:

```text
checkpoint: dense_union_unetpp_b4_hard_raw_ring_outer_distance_finetune.pt

epoch 0 target at validation threshold 0.44:
  mean IoU: 0.8692
  temporal IoU: 0.4753

epoch 1 target at validation threshold:
  mean IoU: 0.8678
  temporal IoU: 0.4691

epoch 2 target at validation threshold:
  mean IoU: 0.8683
  temporal IoU: 0.4694
```

Held-out target with current temporal-heavy recipe:

```text
output: dense_union_hard_raw_ring_outer_distance_finetune_temporal_eval_bidir055_close5_keep035_thr038.json
threshold: 0.38
keep threshold: 0.35
EMA alpha: 0.55
smoothing mode: bidirectional
morphology: close, kernel 5

selected_union_mean_iou: 0.8677
selected_temporal_union_iou: 0.4868
gt_temporal_union_iou: 0.4788
```

Interpretation:

```text
Negative result. Binarizing the hand/ring prior loses useful uncertainty and
does not beat the soft raw_ring_outer_distance prior.

The current best remains:
  dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
  mean IoU: 0.8702
  temporal IoU: 0.4904

The hard-prior checkpoint was removed from kept .pt outputs; JSON summaries
remain as provenance.
```

## Flow-Aligned Temporal Vote Probe

Question:

```text
The existing temporal vote compares neighboring masks at fixed pixel
coordinates. For moving objects, that can erase true positives. Would warping
neighbor masks into the center frame before the vote improve temporal
consistency without target-window calibration?
```

Implementation:

```text
Added temporal vote mode support:
  --temporal-vote-mode pixel|flow in evaluate_dense_temporal.py
  --temporal-vote-modes pixel,flow in tune_temporal_postprocess.py
  --temporal-vote-mode pixel|flow in render_dense_union.py

Flow vote uses Farneback optical flow to align previous/future neighbor masks
to the center frame before voting. This is a postprocess-only ablation: the
model checkpoint and thresholding logic are unchanged.
```

Validation-only probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_flow_vote_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: six non-target validation windows
fixed threshold: 0.38
fixed keep threshold: 0.35
fixed EMA alpha: 0.55
fixed smoothing: bidirectional
fixed morphology: close, kernel 5
tested temporal windows: 1, 3, 5
tested vote modes: pixel, flow
objective: mean IoU + 3.0 * temporal IoU - 0.05 * selected area
```

Validation rows:

```text
window  vote   val_mean  val_temporal
1       pixel  0.9255    0.4744
3       pixel  0.9166    0.4749
3       flow   0.9197    0.4743
5       pixel  0.8998    0.4744
5       flow   0.9147    0.4741
```

Held-out target at validation-selected setting:

```text
selected setting: temporal_window 1, pixel vote
selected_union_mean_iou: 0.8704
selected_temporal_union_iou: 0.4904
```

Held-out target diagnostic for flow vote:

```text
output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_flowvote3_bidir055_close5_keep035_thr038.json
temporal_window: 3
temporal_vote_mode: flow

selected_union_mean_iou: 0.8697
selected_temporal_union_iou: 0.4910
```

Interpretation:

```text
Negative result. Flow-aligned voting is much more expensive than pixel voting,
is not selected by non-target validation, and only trades about -0.0007 mean IoU
for +0.0006 temporal IoU on the held-out target diagnostic.

The current best remains:
  dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
  mean IoU: 0.8704
  temporal IoU: 0.4904

This also suggests the low absolute temporal IoU is not mainly caused by
fixed-pixel temporal voting. On validation and target windows, prediction
temporal IoU is already close to or slightly above GT temporal IoU, so the
remaining temporal issue is likely object/mask correctness across annotated
seconds rather than short-range full-FPS flicker.
```

## Round-Robin Validation Calibration Probe

Question:

```text
Previous postprocess tuning used six non-target validation windows. The window
selector spread one window per take, then filled from sorted candidates, which
can over-weight one take as max_windows increases. Would broader, balanced
validation calibration choose a better non-target threshold/EMA setting?
```

Implementation:

```text
Added --window-selection spread_once|round_robin to tune_temporal_postprocess.py.

round_robin groups candidate 30s windows by take and alternates across takes,
preventing the calibration set from being dominated by iiith_cooking_30_1 or
any other single take when more windows are requested.
```

Calibration set:

```text
output: dense_union_raw_ring_outer_distance_finetune_roundrobin12_focused_tuning.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 12 non-target validation windows, round-robin across takes

windows:
  iiith_cooking_30_1: 2700, 3600, 4500
  sfu_cooking_008_3: 0, 900, 1800
  sfu_cooking_010_1: 0, 1800, 2700
  sfu_cooking_010_3: 0, 900, 1800

tested thresholds: 0.34, 0.36, 0.38, 0.40, 0.42, 0.44
tested keep thresholds: 0.25, 0.30, 0.35, 0.40
tested EMA alphas: 0.45, 0.55, 0.65
smoothing: bidirectional
morphology: close, kernel 5
```

Validation-selected objective row:

```text
on threshold: 0.34
keep threshold: 0.30
EMA alpha: 0.65
validation mean IoU: 0.9116
validation temporal IoU: 0.5102

held-out target:
  selected_union_mean_iou: 0.8603
  selected_temporal_union_iou: 0.5043
```

Validation mean-best row:

```text
on threshold: 0.44
keep threshold: 0.40
EMA alpha: 0.45
validation mean IoU: 0.9368
validation temporal IoU: 0.4992

held-out target:
  output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_roundrobin12_meanbest_bidir045_close5_keep040_thr044.json
  selected_union_mean_iou: 0.8707
  selected_temporal_union_iou: 0.4854
```

Validation temporal-best row:

```text
on threshold: 0.34
keep threshold: 0.25
EMA alpha: 0.65
validation mean IoU: 0.9034
validation temporal IoU: 0.5121

held-out target:
  output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_roundrobin12_temporalbest_bidir065_close5_keep025_thr034.json
  selected_union_mean_iou: 0.8557
  selected_temporal_union_iou: 0.5086
```

Interpretation:

```text
Mixed result. Balanced validation improves the best temporal operating point
substantially relative to the prior raw_ring_outer_distance setting
(0.5086 vs. 0.4904 temporal IoU), while keeping mean IoU far above the older
raw_dilated temporal-best checkpoint. However, this comes with a mean-IoU drop
to 0.8557.

The balanced mean-best setting gives only a tiny mean-IoU improvement
(0.8707 vs. 0.8704) and lowers temporal IoU. This suggests remaining progress
toward 0.9 mean IoU probably will not come from scalar postprocess calibration
alone; the model needs better object/mask correctness or stronger supervision.
```

## Hand-Proximity Component Filter Probe

Question:

```text
Can component-level hand proximity remove false-positive red components without
hurting true object masks? This tests a stricter version of the hand prior:
after thresholding, keep only connected components that overlap a dilated hand
support mask by at least a small fraction.
```

Implementation:

```text
Added component hand filter support:
  --component-hand-mode none|dilated_overlap
  --component-hand-kernel
  --component-hand-min-overlap-frac

Wired the option into:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py

The filter runs after morphology and optional hand exclusion, before area/count
component filtering.
```

Validation probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_component_hand_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes
tested modes: none, dilated_overlap
tested hand kernels: 21, 31, 41, 61
tested overlap fractions: 0.01, 0.03, 0.05, 0.10
tested thresholds: 0.38, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Best no-filter validation row:

```text
component_hand_mode: none
on threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
validation mean IoU: 0.9327
validation temporal IoU: 0.4951

held-out target:
  selected_union_mean_iou: 0.8706
  selected_temporal_union_iou: 0.4858
```

Best filtered validation row:

```text
component_hand_mode: dilated_overlap
component_hand_kernel: 61
component_hand_min_overlap_frac: 0.01
on threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
validation mean IoU: 0.7518
validation temporal IoU: 0.4890
```

Interpretation:

```text
Negative result. Even the loosest useful proximity filter deletes too much true
object-mask area. Under the current Ego-Exo supervision, the target is a union
of visible annotated object masks, not purely active/contact objects. Therefore
"near hand" is a good conditioning signal for the neural model, but it is not a
safe hard postprocess gate for this validation target.

This is useful evidence for the larger design: if we want hand-proximity gating
to be central, the target/evaluation likely needs active/contact/relevance
semantics. For dense Ego-Exo object-union IoU, hard proximity filtering moves
away from 0.9.
```

## Horizontal-Flip Test-Time Augmentation Probe

Question:

```text
Can prediction averaging with a horizontally flipped inference pass reduce
model noise and improve dense object-union IoU without retraining or target
calibration?
```

Implementation:

```text
Added --tta-mode none|hflip to:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py

For hflip, the RGB frame is flipped, the hand prior is recomputed on the
flipped frame, the model predicts on the flipped input, and the predicted
probability map is flipped back before averaging with the original prediction.
The hand overlay still uses the original-frame hand prediction.
```

Validation probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_hflip_tta_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes
tta_mode: hflip
tested thresholds: 0.38, 0.42, 0.44, 0.46
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55
smoothing: bidirectional
morphology: close, kernel 5
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Best hflip validation row:

```text
on threshold: 0.38
keep threshold: 0.35
EMA alpha: 0.45
validation mean IoU: 0.8865
validation temporal IoU: 0.4984

held-out target:
  selected_union_mean_iou: 0.8530
  selected_temporal_union_iou: 0.4976
```

Comparable no-TTA validation row from the component-hand probe:

```text
on threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
validation mean IoU: 0.9327
validation temporal IoU: 0.4951

held-out target:
  selected_union_mean_iou: 0.8706
  selected_temporal_union_iou: 0.4858
```

Interpretation:

```text
Negative result. Flip TTA improves temporal smoothness slightly but hurts mean
IoU badly on validation and target. The likely cause is that egocentric cooking
and hand-object interaction layouts are not sufficiently left/right invariant
for this checkpoint; averaging a flipped pass suppresses useful spatial bias.

This closes another inference-only path. The model remains strongest without
TTA.
```

## Probability-Ranked Component Filter Probe

Question:

```text
Can low-confidence connected components be removed after thresholding without
hurting true object-union pixels? This is a gentler alternative to the failed
hand-proximity component gate: rank components by area, mean probability, or
max probability, optionally cap the number of components, and optionally drop
components below a score threshold.
```

Implementation:

```text
Added component probability ranking support:
  --component-rank-mode area|mean_prob|max_prob
  --min-component-score

Wired the option into:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py

Also refactored the tuner grid into iter_postprocess_grid(...) to avoid Python's
static nesting limit as postprocess ablations accumulate.
```

Validation probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_prob_component_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes

tested thresholds: 0.38, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55
tested min component area fractions: 0, 0.0005
tested max components: 0, 3, 5
tested rank modes: area, mean_prob, max_prob
tested min component scores: 0, 0.45, 0.50, 0.55
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Validation-selected row:

```text
component_rank_mode: area
min_component_score: 0
min_component_area_frac: 0
max_components: 0
on threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45

validation mean IoU: 0.9327
validation temporal IoU: 0.4951

held-out target:
  selected_union_mean_iou: 0.8706
  selected_temporal_union_iou: 0.4858
```

Best material component cap:

```text
max_components: 5
component_rank_mode: area
validation mean IoU: 0.9324
validation temporal IoU: 0.4953
```

Best min-area filter:

```text
min_component_area_frac: 0.0005
validation mean IoU: 0.9288
validation temporal IoU: 0.4964
```

Interpretation:

```text
Neutral-to-negative result. Component caps and probability score thresholds do
not improve validation objective. A light cap of five components is almost
neutral, but still slightly worse than leaving components untouched; min-area
filtering hurts mean IoU more clearly.

This suggests the remaining mean-IoU gap is not primarily low-confidence
island noise that can be removed post hoc. The current best remains the
unfiltered raw_ring_outer_distance checkpoint/postprocess.
```

## Extended Morphology Probe

Question:

```text
The current best used close5 morphology, but component and area filters have
not closed the gap. Could boundary-level morphology recover IoU by better
matching object-mask edges? Test no morphology, close, open, erode, open_close,
and close_open with small kernels, using only non-target validation selection.
```

Implementation:

```text
Added morphology modes:
  erode
  open_close
  close_open

Wired the modes into:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py
```

Validation probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_extended_morph_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes

tested thresholds: 0.38, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55
tested morph ops: none, close, open, erode, open_close, close_open
tested morph kernels: 0, 3, 5
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Validation-selected row:

```text
morph_op: none
morph_kernel: 0
on threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
validation mean IoU: 0.9391
validation temporal IoU: 0.4914

held-out target:
  output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_extended_morph_meanbest_nomorph_keep040_thr042.json
  selected_union_mean_iou: 0.8711
  selected_temporal_union_iou: 0.4789
```

Comparison:

```text
previous mean-best: 0.8707 mean / 0.4854 temporal
new mean-best:      0.8711 mean / 0.4789 temporal
temporal-best:      0.8557 mean / 0.5086 temporal
```

Interpretation:

```text
Mixed but tiny result. Removing morphology gives the best mean IoU seen so far,
but the gain is only about +0.0004 over the previous mean-best and comes with a
temporal-IoU drop. The extended morphology variants show the usual tradeoff:
opening/erosion can increase temporal stability by shrinking masks, but they
lose too much true object area.

This updates the kept mean-best qualitative overlay, but it does not change the
larger conclusion: postprocess tuning alone is not plausibly enough to reach
0.9 mean IoU.
```

## Probability vs Logit Smoothing Probe

Question:

```text
The current best hand prior is already separated into raw hand, outside-hand
ring, and outside-only distance channels. Could temporal EMA smoothing in logit
space preserve sharper masks and improve temporal consistency compared with
smoothing probabilities directly?
```

Implementation:

```text
Added smoothing-domain support:
  prob
  logit

Wired --smoothing-domain / --smoothing-domains into:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py

The tuner now caches smoothed probabilities by:
  window index
  EMA alpha
  smoothing mode
  smoothing domain
```

Validation probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_logit_smoothing_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes

tested thresholds: 0.38, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55, 0.65
tested smoothing domains: prob, logit
tested morph ops: none, close
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Validation-selected row:

```text
smoothing domain: prob
smoothing mode: bidirectional
EMA alpha: 0.45
on threshold: 0.42
keep threshold: 0.40
morph_op: none
morph_kernel: 0
validation mean IoU: 0.9391
validation temporal IoU: 0.4914

held-out target:
  selected_union_mean_iou: 0.8711
  selected_temporal_union_iou: 0.4789
```

Best logit-domain row:

```text
smoothing domain: logit
EMA alpha: 0.45
on threshold: 0.38
keep threshold: 0.35
morph_op: none
morph_kernel: 0
validation mean IoU: 0.9356
validation temporal IoU: 0.4926
objective: 1.4242
```

Interpretation:

```text
Negative result. Logit smoothing gives a very small temporal-IoU bump on the
best logit row, but the mean-IoU loss is larger and the validation objective
selects probability smoothing. Keep the current prob-domain bidirectional EMA.

This also strengthens the hand-prior conclusion: the best input remains raw
hand + outside ring + outside-only distance. The remaining gap to 0.9 is not
from the model lacking a dilated-ring prior or from smoothing in the wrong
numeric domain.
```

## Component-Level Hysteresis Probe

Question:

```text
Pixel hysteresis only keeps previous positive pixels that remain above the keep
threshold, so it can still flicker within an object. Would carrying an entire
previous connected component when enough of that component remains supported
improve temporal IoU without destroying mean IoU?
```

Implementation:

```text
Added hysteresis modes:
  pixel
  component

Added component support fraction:
  component_hysteresis_min_frac

Wired the new mode into:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py

Component hysteresis labels the previous selected mask. For each previous
component, it carries the whole component forward if the fraction of pixels
whose current probability exceeds the keep threshold is at least
component_hysteresis_min_frac.
```

Validation probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_component_hysteresis_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes

tested thresholds: 0.38, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55
tested hysteresis modes: pixel, component
tested component support fractions: 0.10, 0.25, 0.50
tested morph ops: none, close
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Validation-selected row:

```text
hysteresis mode: pixel
component_hysteresis_min_frac: 0.0
EMA alpha: 0.45
on threshold: 0.42
keep threshold: 0.40
morph_op: none
morph_kernel: 0
validation mean IoU: 0.9391
validation temporal IoU: 0.4914

held-out target:
  selected_union_mean_iou: 0.8711
  selected_temporal_union_iou: 0.4789
```

Best component-hysteresis objective row:

```text
hysteresis mode: component
component_hysteresis_min_frac: 0.50
EMA alpha: 0.55
on threshold: 0.42
keep threshold: 0.40
morph_op: none
morph_kernel: 0
validation mean IoU: 0.6864
validation temporal IoU: 0.6209
selected mean area: 0.1100
objective: 1.3018
```

Best component-hysteresis temporal row:

```text
hysteresis mode: component
component_hysteresis_min_frac: 0.10
EMA alpha: 0.55
on threshold: 0.42
keep threshold: 0.35
morph_op: close
morph_kernel: 5
validation mean IoU: 0.4392
validation temporal IoU: 0.8196
selected mean area: 0.1896
```

Interpretation:

```text
Negative result. Component carry can raise temporal IoU dramatically, but it
does so by dragging stale connected components forward and inflating the
selected area. The mean-IoU collapse is too severe, and non-target validation
selects the existing pixel hysteresis setting.

This rules out naive component persistence as the temporal fix. Any future
tracking-style postprocess needs object motion/identity evidence or learned
temporal state, not just connected-component carry.
```

## Centered Probability Filter Probe

Question:

```text
Component hysteresis is too blunt because it carries whole stale masks. Could a
gentler centered temporal filter over probabilities improve temporal consistency
before thresholding, without the stale connected-component failure mode?
```

Implementation:

```text
Added probability filter modes:
  none
  mean
  median
  max

Added probability filter window:
  prob_filter_window

Wired the mode/window into:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py

The tuner caches filtered probabilities by:
  window index
  EMA alpha
  smoothing mode
  smoothing domain
  probability filter mode
  probability filter window

Median was implemented but excluded from the reported sweep because it made the
probe too slow for the current unvectorized CPU evaluator.
```

Validation probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_prob_filter_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes

tested thresholds: 0.38, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55
tested probability filters: none, mean, max
tested probability filter windows: 1, 3, 5
tested morph ops: none, close
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Validation-selected row:

```text
prob_filter_mode: none
prob_filter_window: 1
EMA alpha: 0.45
on threshold: 0.42
keep threshold: 0.40
morph_op: none
morph_kernel: 0
validation mean IoU: 0.9391
validation temporal IoU: 0.4914

held-out target:
  selected_union_mean_iou: 0.8711
  selected_temporal_union_iou: 0.4789
```

Best filtered objective row:

```text
prob_filter_mode: mean
prob_filter_window: 3
EMA alpha: 0.45
on threshold: 0.42
keep threshold: 0.40
morph_op: none
morph_kernel: 0
validation mean IoU: 0.9255
validation temporal IoU: 0.4948
selected mean area: 0.0813
objective: 1.4162
```

Best filtered temporal row:

```text
prob_filter_mode: max
prob_filter_window: 5
EMA alpha: 0.55
on threshold: 0.44
keep threshold: 0.35
morph_op: close
morph_kernel: 5
validation mean IoU: 0.8238
validation temporal IoU: 0.5184
selected mean area: 0.0911
```

Interpretation:

```text
Negative result. Centered mean filtering gives a tiny temporal gain but loses
too much mean IoU. Max filtering raises temporal IoU more, but behaves like a
soft persistence/expansion operation and collapses mean IoU.

This reinforces the postprocess conclusion: local temporal filters can make the
mask smoother, but they do not recover the missing spatial correctness needed
to reach 0.9 mean IoU.
```

## Probability Filter Temporal-Tradeoff Target Reports

Question:

```text
The centered probability-filter probe selected no filter under the main
mean+temporal validation objective. Do any validation-selected temporal-heavy
settings give a useful held-out target tradeoff without target-window
calibration?
```

Selection:

```text
source: dense_union_raw_ring_outer_distance_finetune_prob_filter_probe.json
selection source: 8 non-target validation windows, round-robin across takes
target use: held-out reporting only
```

Validation-selected temporal-heavy unfiltered setting:

```text
selection rule: maximize mean IoU + 3.0 * temporal IoU - 0.05 * selected area
prob_filter_mode: none
prob_filter_window: 1
EMA alpha: 0.55
on threshold: 0.38
keep threshold: 0.35
morph_op: close
morph_kernel: 5

held-out target:
  output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_probfilter_unfiltered_temporalheavy.json
  selected_union_mean_iou: 0.8704
  selected_temporal_union_iou: 0.4904
  selected_mean_area: 0.0761

qualitative:
  outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_temporalheavy_unfiltered/overlay.mp4
  outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_temporalheavy_unfiltered/contact_sheet.jpg
```

Validation-selected mean-filter temporal setting:

```text
selection rule: highest temporal IoU among rows with validation mean IoU >= 0.90
prob_filter_mode: mean
prob_filter_window: 3
EMA alpha: 0.55
on threshold: 0.42
keep threshold: 0.35
morph_op: close
morph_kernel: 5

held-out target:
  output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_probfilter_mean3_temporal.json
  selected_union_mean_iou: 0.8618
  selected_temporal_union_iou: 0.4984
  selected_mean_area: 0.0769
```

Validation-selected max-filter temporal setting:

```text
selection rule: highest temporal IoU among filtered rows
prob_filter_mode: max
prob_filter_window: 5
EMA alpha: 0.55
on threshold: 0.44
keep threshold: 0.35
morph_op: close
morph_kernel: 5

held-out target:
  output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_probfilter_max5_temporal.json
  selected_union_mean_iou: 0.8131
  selected_temporal_union_iou: 0.5204
  selected_mean_area: 0.0832
```

Interpretation:

```text
The best practical temporal tradeoff is not a probability filter; it is simply
a lower threshold, stronger bidirectional EMA, lower hysteresis keep threshold,
and close5 morphology. It improves target temporal IoU from 0.4789 to 0.4904
while barely changing mean IoU (0.8711 -> 0.8704).

Mean and max probability filters buy more temporal IoU, but the target mean-IoU
cost is too high. The max-filter result is especially clear: 0.5204 temporal
IoU comes with only 0.8131 mean IoU, so it is not aligned with the 0.9 mean-IoU
goal.
```

## Current-Best Flow Smoothing Probe

Question:

```text
The practical temporal-heavy setting uses plain bidirectional probability EMA.
Would optical-flow-aligned probability EMA improve temporal consistency by
warping probabilities before smoothing, enough to justify the extra runtime?
```

Implementation:

```text
No new code was required. The existing smoothing modes already support:
  bidirectional
  flow_bidirectional

The first 8-window attempt was interrupted because flow_bidirectional spent too
long in Farneback flow-map generation. A reduced 4-window round-robin probe was
run to get a directional answer.
```

Validation probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_flow_smoothing_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 4 non-target validation windows, round-robin across takes

tested smoothing modes: bidirectional, flow_bidirectional
EMA alpha: 0.55
on threshold: 0.38
keep threshold: 0.35
morph_op: close
morph_kernel: 5
probability filter: none
objective: mean IoU + 3.0 * temporal IoU - 0.05 * selected area
```

Rows:

```text
plain bidirectional:
  validation mean IoU: 0.9340
  validation temporal IoU: 0.5233
  selected mean area: 0.0951
  objective: 2.4993

flow_bidirectional:
  validation mean IoU: 0.9382
  validation temporal IoU: 0.5210
  selected mean area: 0.0949
  objective: 2.4965
```

Validation-selected target report:

```text
selected smoothing mode: bidirectional
held-out target:
  selected_union_mean_iou: 0.8704
  selected_temporal_union_iou: 0.4904
```

Interpretation:

```text
Negative result. Flow smoothing slightly improved validation mean IoU on the
small probe, but it reduced temporal IoU and lost the temporal-heavy objective.
It is also much slower than plain bidirectional smoothing because Farneback
flow over full 30-second windows dominates runtime.

Do not use flow_bidirectional as the current default. The practical
temporal-improved setting remains plain bidirectional EMA with lower threshold,
lower hysteresis keep threshold, and close5 morphology.
```

## Current-Best Hand-Prior Radius Probe

Question:

```text
The previous hand-prior radius probe was run on the older raw_ring_distance
checkpoint. Does changing the inference-time ring/distance radius help the
current raw_ring_outer_distance checkpoint after the latest postprocess sweeps?
```

Validation probe:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 4 non-target validation windows, round-robin across takes
tested hand kernel overrides: 9, 15, 21, 31
tested thresholds: 0.38, 0.42
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55
tested smoothing mode: bidirectional
probability filter: none
tested morph ops: none, close
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Rows:

```text
kernel  val_mean  val_temp  val_objective  target_mean  target_temp
9       0.9433    0.5192    1.4578         0.8712       0.4788
15      0.9435    0.5192    1.4580         0.8711       0.4789
21      0.9434    0.5191    1.4578         0.8695       0.4782
31      0.9432    0.5191    1.4576         0.8686       0.4775
```

Validation-selected setting:

```text
hand kernel: 15
on threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
smoothing mode: bidirectional
morph_op: none
morph_kernel: 0

held-out target:
  selected_union_mean_iou: 0.8711
  selected_temporal_union_iou: 0.4789
```

Interpretation:

```text
Negative/flat result. The non-target objective selects the checkpoint-native
kernel 15. Kernel 9 has a tiny held-out target mean edge, but it was not
selected by validation and the difference is only 0.0001. Wider radii are
slightly worse.

Keep hand_kernel_size=15 as the stable default. The remaining gap to 0.9 is
not an inference-time ring/distance radius issue.
```

## Raw Hand-Prior Power Probe

Question:

```text
The current hand prior uses raw soft hand probabilities. Would mildly sharpening
or softening those probabilities before building the raw/ring/outside-distance
channels improve object relevance masks?
```

Implementation:

```text
Added --hand-prior-power to:
  evaluate_dense_temporal.py
  tune_temporal_postprocess.py
  render_dense_union.py

The transform is:
  raw_hand = clamp(raw_hand, 0, 1) ** hand_prior_power

power 1.0 is exactly the previous behavior.
power < 1.0 softens/expands uncertain hand edges.
power > 1.0 sharpens/suppresses uncertain hand edges.
```

Validation probe:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 4 non-target validation windows, round-robin across takes
tested hand_prior_power values: 0.75, 1.0, 1.25, 1.5
tested thresholds: 0.38, 0.42
tested keep thresholds: 0.35, 0.40
tested EMA alphas: 0.45, 0.55
tested smoothing mode: bidirectional
probability filter: none
tested morph ops: none, close
objective: mean IoU + 1.0 * temporal IoU - 0.05 * selected area
```

Mean-objective rows:

```text
power  val_mean  val_temp  val_objective  target_mean  target_temp
0.75   0.9435    0.5188    1.4576         0.8706       0.4778
1.0    0.9435    0.5192    1.4580         0.8711       0.4789
1.25   0.9433    0.5195    1.4580         0.8713       0.4798
1.5    0.9428    0.5196    1.4578         0.8712       0.4803
```

Mean-objective validation-selected setting:

```text
hand_prior_power: 1.25
on threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
smoothing mode: bidirectional
morph_op: none
morph_kernel: 0

held-out target:
  selected_union_mean_iou: 0.8713
  selected_temporal_union_iou: 0.4798
```

Temporal-heavy validation-selected setting:

```text
selection rule: maximize mean IoU + 3.0 * temporal IoU - 0.05 * selected area
hand_prior_power: 1.5
on threshold: 0.42
keep threshold: 0.35
EMA alpha: 0.55
smoothing mode: bidirectional
morph_op: close
morph_kernel: 5

held-out target:
  output: dense_union_raw_ring_outer_distance_finetune_temporal_eval_handpower15_temporalheavy.json
  selected_union_mean_iou: 0.8704
  selected_temporal_union_iou: 0.4918
  selected_mean_area: 0.0761

qualitative:
  outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_handpower15_temporalheavy/overlay.mp4
  outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_handpower15_temporalheavy/contact_sheet.jpg
```

Interpretation:

```text
Small positive result. Mild hand-prior sharpening is the first recent
hand-prior ablation to improve both held-out mean and temporal IoU, although
the gains are tiny. The mean-objective target improves from 0.8711 / 0.4789 to
0.8713 / 0.4798. The practical temporal-heavy setting improves from
0.8704 / 0.4904 to 0.8704 / 0.4918.

This does not close the gap to 0.9, but it suggests that hand-prior confidence
calibration is a more promising hand-prior knob than radius, hard binarization,
multi-ring splitting, or hand exclusion.
```

## Raw + Dilated Ring + Outside-Distance Prior Probe

Question:

```text
What about passing raw hand and distance priors, as well as raw hand dilated
ring priors?
```

Rationale:

```text
The current best raw_ring_outer_distance model passes:
  RGB
  raw hand probability
  outside-hand ring
  outside-only distance proximity

This preserves exact hand identity and suppresses distance leakage inside hand
pixels, but it does not pass the full soft dilated hand neighborhood as its own
channel. The older raw_dilated_ring_distance model did pass a dilated channel,
but its distance channel remained high inside hand pixels.

The missing hybrid is therefore:
  RGB
  raw hand probability
  soft dilated hand prior
  outside-hand ring: dilate(hand) - hand
  outside-only distance proximity: distance_proximity(hand) * (1 - hand)
```

Implementation:

```text
Added hand input mode: raw_dilated_ring_outer_distance

Code changes:
  experiments/scheme3_v3/hand_prior.py
  experiments/scheme3_v3/train_dense_union.py

Warm-start mapping from current best raw_ring_outer_distance:
  rgb_r -> rgb_r
  rgb_g -> rgb_g
  rgb_b -> rgb_b
  hand_raw -> hand_raw
  hand_ring -> hand_ring
  hand_outer_distance -> hand_outer_distance
  hand_dilated initialized to zero
```

Full fine-tune attempt:

```text
output: dense_union_unetpp_b4_raw_dilated_ring_outer_distance_finetune.pt
init checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
train samples requested: 6000
temporal pair samples: all available
epochs requested: 2

Status: interrupted before epoch 1 completed because the full run was too slow
for a first-pass prior ablation and had not written a checkpoint.
```

Smoke fine-tune:

```text
output: dense_union_unetpp_b4_raw_dilated_ring_outer_distance_smoke.pt
summary: dense_union_unetpp_b4_raw_dilated_ring_outer_distance_smoke_summary.json
init checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
train samples: 800
temporal pair samples: 400
epochs: 1
learning rate: 2e-5
temporal pair weight: 0.05
```

Smoke training result:

```text
epoch 0 inherited checkpoint at validation threshold 0.52:
  target mean IoU: 0.8687
  target temporal IoU: 0.4734

epoch 1 at validation threshold 0.66:
  non-target validation mean IoU: 0.9486
  non-target validation temporal IoU: 0.3165
  target mean IoU: 0.8684
  target temporal IoU: 0.4699
```

Target postprocess checks:

```text
mean-best style:
  output: dense_union_raw_dilated_ring_outer_distance_smoke_temporal_eval_meanbest.json
  hand_prior_power: 1.25
  threshold: 0.42
  keep threshold: 0.40
  EMA alpha: 0.45
  smoothing mode: bidirectional
  morphology: none
  selected_union_mean_iou: 0.8707
  selected_temporal_union_iou: 0.4782

temporal-heavy style:
  output: dense_union_raw_dilated_ring_outer_distance_smoke_temporal_eval_temporalheavy.json
  hand_prior_power: 1.5
  threshold: 0.42
  keep threshold: 0.35
  EMA alpha: 0.55
  smoothing mode: bidirectional
  morphology: close, kernel 5
  selected_union_mean_iou: 0.8697
  selected_temporal_union_iou: 0.4909
```

Comparison to current best:

```text
current hand-power mean-best:      0.8713 mean / 0.4798 temporal
new hybrid smoke mean-style:       0.8707 mean / 0.4782 temporal

current hand-power temporal-heavy: 0.8704 mean / 0.4918 temporal
new hybrid smoke temporal-heavy:   0.8697 mean / 0.4909 temporal
```

Conclusion:

```text
The proposed raw + dilated + ring + outside-distance prior is implemented and
valid, but the quick fine-tune does not beat the current best. The extra
dilated channel appears redundant with the outside ring and distance priors,
and may slightly encourage hand-neighborhood persistence without improving the
object boundary enough to matter. Do not spend a full fine-tune on this mode
unless a later architecture can exploit separate priors more explicitly.
```

## Ensemble, Area, Loss, and Empty-Fallback Iteration

Question:

```text
The current best mean IoU is stuck around 0.871. Which remaining failure modes
are responsible, and can validation-selected postprocessing or light fine-tuning
move the pipeline closer to 0.9 without target-window calibration?
```

Per-frame target diagnostic for the current mean-best setting:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
hand_prior_power: 1.25
threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
smoothing mode: bidirectional
morphology: none

target mean IoU: 0.8713
target temporal IoU: 0.4798
target selected mean area: 0.0748
target GT mean area: 0.0762

Worst target frames:
  frame 3210: IoU 0.0000, pred area 0.0000, GT area 0.0034
  frame 3570: IoU 0.6976, pred area 0.0588, GT area 0.0757
  frame 3450: IoU 0.7432, pred area 0.0643, GT area 0.0777
```

Interpretation:

```text
Most target frames are already near 0.90-0.94 IoU. The mean is pulled down by a
small number of transition/small-object frames, especially one annotated frame
where the model predicts an empty object mask.
```

Modernized ensemble tuner:

```text
file: experiments/scheme3_v3/tune_ensemble_postprocess.py

The previous ensemble tuner used the old postprocess signature. It now delegates
to the current tune_temporal_postprocess helpers, including smoothing domain,
probability filtering, hysteresis mode, component options, hand-prior power, and
TTA arguments.
```

Current-best + smoke-prior ensemble probe:

```text
output: dense_union_current_plus_raw_dilated_ring_outer_smoke_ensemble_probe.json
checkpoint A: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
checkpoint B: dense_union_unetpp_b4_raw_dilated_ring_outer_distance_smoke.pt
selection source: 4 non-target validation windows, round-robin across takes
tested weights A: 0, 0.25, 0.50, 0.75, 0.90, 1.0

validation selected:
  weight A: 0.50
  weight B: 0.50
  threshold: 0.42
  keep threshold: 0.40
  EMA alpha: 0.45
  morphology: none

held-out target:
  selected_union_mean_iou: 0.8712
  selected_temporal_union_iou: 0.4793
```

Conclusion:

```text
Negative/flat. The retained smoke checkpoint does not add useful complementary
information. The ensemble is below the current hand-power mean-best result of
0.8713 / 0.4798.
```

Area-floor probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_area_floor_probe.json
selection source: 8 non-target validation windows, round-robin across takes
tested min_area_frac: 0, 0.0025, 0.005, 0.0075, 0.01

validation selected:
  min_area_frac: 0

held-out target:
  selected_union_mean_iou: 0.8713
  selected_temporal_union_iou: 0.4798
```

Conclusion:

```text
Negative. A global per-frame area floor improves temporal smoothness slightly
but hurts validation mean IoU, so it is not a robust fix for the target's tiny
missed frame.
```

Recall-biased focal/Tversky smoke fine-tune:

```text
output: dense_union_unetpp_b4_raw_ring_outer_distance_recall_tversky_smoke.pt
summary: dense_union_unetpp_b4_raw_ring_outer_distance_recall_tversky_smoke_summary.json
init checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
train samples: 800
temporal pair samples: 400
epochs: 1
loss: focal_tversky
tversky alpha: 0.30
tversky beta: 0.70
learning rate: 1e-5

epoch 0 inherited target at validation threshold 0.36:
  mean IoU: 0.8689
  temporal IoU: 0.4762

epoch 1 target at validation threshold 0.80:
  mean IoU: 0.8683
  temporal IoU: 0.4698

mean-best-style postprocess on the smoke checkpoint:
  output: dense_union_raw_ring_outer_distance_recall_tversky_smoke_temporal_eval_meanbest.json
  mean IoU: 0.8693
  temporal IoU: 0.4818
```

Conclusion:

```text
Negative for mean IoU. The recall-biased loss made the model sharper/different
but did not recover small transition objects. It slightly increased temporal IoU
under the old mean-best postprocess but lost too much mean IoU.
```

Empty-frame fallback implementation:

```text
file: experiments/scheme3_v3/tune_temporal_postprocess.py

Added validation-tunable arguments:
  --empty-fallback-area-fracs
  --empty-fallback-min-probs

Behavior:
  if selected mask is empty and max(prob) >= empty_fallback_min_prob:
      select top-probability pixels covering empty_fallback_area_frac of frame

The fallback only fires on frames whose selected object mask would otherwise be
empty. It does not alter normal non-empty frames.
```

Empty-frame fallback probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_empty_fallback_probe.json
selection source: 8 non-target validation windows, round-robin across takes
tested empty_fallback_area_frac: 0, 0.0015, 0.0025, 0.0035, 0.005
tested empty_fallback_min_prob: 0, 0.10, 0.20, 0.30

validation selected:
  threshold: 0.42
  keep threshold: 0.40
  EMA alpha: 0.45
  smoothing mode: bidirectional
  morphology: none
  empty_fallback_area_frac: 0.005
  empty_fallback_min_prob: 0.0

held-out target:
  selected_union_mean_iou: 0.8721
  selected_temporal_union_iou: 0.4798
```

Extended empty-frame fallback probe:

```text
output: dense_union_raw_ring_outer_distance_finetune_empty_fallback_extended_probe.json
selection source: 8 non-target validation windows, round-robin across takes
fixed threshold/postprocess from previous validation result
tested empty_fallback_area_frac: 0, 0.005, 0.0075, 0.01, 0.015, 0.02
tested empty_fallback_min_prob: 0, 0.05, 0.10

validation selected:
  empty_fallback_area_frac: 0.02
  empty_fallback_min_prob: 0.0

held-out target:
  selected_union_mean_iou: 0.8742
  selected_temporal_union_iou: 0.4800
  selected_mean_area: 0.0754
```

Per-frame effect on target:

```text
Only the empty frame changes.

frame 3210:
  baseline IoU: 0.0000
  fallback IoU: 0.0865
  GT area: 0.0034
  fallback selected area: 0.0200
```

Interpretation:

```text
This is a validation-selected numeric improvement, but it is not a satisfying
selector improvement. The selected confidence gate is 0.0, which means the
fallback is exploiting the evaluation fact that scored frames always have an
annotated object, not demonstrating that the model found the object with
confidence. Treat 0.8742 / 0.4800 as a diagnostic postprocess ceiling, not as
the solved/current model quality. The real bottleneck remains model confidence
on small/transition object masks.
```

## Small-Object Mixed Training Probe

Question:

```text
The target mean IoU is dragged down by tiny/transition object frames. Can we
improve model confidence on these cases by oversampling small-object frames from
non-target train/val windows instead of forcing pixels during postprocessing?
```

Area distribution scan:

```text
Exact decoded object-union areas, excluding target window:

train frames: 2697
  mean area: 0.0529
  p01/p05/p10/p25/p50: 0.0000 / 0.0101 / 0.0187 / 0.0356 / 0.0521
  frames with union area < 0.025: 363
  frames with union area < 0.050: 1265

val frames: 1431
  mean area: 0.0651
  p01/p05/p10/p25/p50: 0.0006 / 0.0047 / 0.0165 / 0.0413 / 0.0544
  frames with union area < 0.025: 200
  frames with union area < 0.050: 580
```

Implementation:

```text
Added approximate small-object sampling support:
  experiments/scheme3_v3/data.py
  experiments/scheme3_v3/train_dense_union.py

New training args:
  --small-train-samples
  --small-train-max-area-ratio

The filter uses the annotation click-box area as a fast object-size proxy. This
is approximate and used only for training-sample selection, not metrics.
```

Smoke fine-tune:

```text
output: dense_union_unetpp_b4_raw_ring_outer_distance_smallmix_smoke.pt
summary: dense_union_unetpp_b4_raw_ring_outer_distance_smallmix_smoke_summary.json
init checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
base train samples: 800
extra small-object samples: 400
small-object bbox proxy threshold: 0.08
temporal pair samples: 400
epochs: 1
learning rate: 1e-5
loss: bce_dice
```

Training result:

```text
epoch 0 inherited target at validation threshold 0.40:
  mean IoU: 0.8691
  temporal IoU: 0.4757

epoch 1 target at validation threshold 0.58:
  mean IoU: 0.8698
  temporal IoU: 0.4667
```

Mean-best-style target evaluation:

```text
output: dense_union_raw_ring_outer_distance_smallmix_smoke_temporal_eval_meanbest.json
threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
smoothing mode: bidirectional
hand_prior_power: 1.25

selected_union_mean_iou: 0.8682
selected_temporal_union_iou: 0.4737
```

Non-target postprocess tuning:

```text
output: dense_union_raw_ring_outer_distance_smallmix_smoke_postprocess_probe.json
selection source: 4 non-target validation windows, round-robin across takes

validation selected:
  threshold: 0.46
  keep threshold: 0.45
  EMA alpha: 0.45
  smoothing mode: bidirectional
  morphology: none

held-out target:
  selected_union_mean_iou: 0.8679
  selected_temporal_union_iou: 0.4723
```

Conclusion:

```text
Negative. Small-object oversampling with the current U-Net++ dense-union model
slightly improves some raw validation numbers but does not generalize to the
held-out 30s target. It also reduces temporal IoU. The likely issue is not just
sample frequency; the model needs a better way to represent object transitions
and small masks, or richer supervision/architecture, rather than a simple
small-frame mixture.
```

## Low-Area Rescue Postprocess Probe

Question:

```text
Can we fix tiny/transition misses with a confidence-gated rescue that only adds
top-probability pixels when the selected mask is implausibly small, without
using target-window calibration?
```

Implementation:

```text
Added low-area rescue support to:
  experiments/scheme3_v3/tune_temporal_postprocess.py

New tuning args:
  --low-area-rescue-trigger-fracs
  --low-area-rescue-area-fracs
  --low-area-rescue-min-probs

The rescue is additive: if the current selected mask area is below the trigger
and the frame max probability exceeds the confidence gate, the postprocess
unions the current mask with the top-probability pixels up to the target rescue
area. It does not replace an existing non-empty mask.
```

Sweep:

```text
output: dense_union_raw_ring_outer_distance_finetune_low_area_rescue_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes
hand_prior_power: 1.25
tested on thresholds: 0.40, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested rescue triggers: 0, 0.0015, 0.0035, 0.005, 0.01
tested rescue target areas: 0, 0.0015, 0.0035, 0.005, 0.01
tested rescue min probabilities: 0.20, 0.30, 0.40, 0.50
```

Validation-selected setting:

```text
threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
smoothing: bidirectional
morphology: none
low_area_rescue_trigger_frac: 0.0
low_area_rescue_area_frac: 0.0
low_area_rescue_min_prob: 0.20
```

Held-out target:

```text
selected_union_mean_iou: 0.8713
selected_temporal_union_iou: 0.4798
```

Conclusion:

```text
Negative. Non-target validation selected no low-area rescue. Larger rescue
areas improved neither validation objective nor target transfer; they mostly
add false-positive pixels. This supports the earlier empty-fallback caveat:
postprocess filling is not a legitimate route to 0.9 unless the model itself
assigns better confidence to tiny/transition objects.
```

## Current-Prior 384px Smoke Fine-Tune

Question:

```text
The previous 384px ablation used the older raw_dilated prior. Does higher
resolution help when starting from the current raw_ring_outer_distance best
checkpoint?
```

Training recipe:

```text
checkpoint trained then removed because it was not best:
  dense_union_unetpp_b4_raw_ring_outer_distance_384_smoke.pt

summary retained:
  dense_union_unetpp_b4_raw_ring_outer_distance_384_smoke_summary.json

image size: 384
hand input mode: raw_ring_outer_distance
init checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
train samples: 600 from train+val, excluding target window
temporal pair samples: 300
epochs: 1
learning rate: 5e-6
batch size: 2
```

Training result:

```text
epoch 0 inherited checkpoint at 384px, validation-selected threshold 0.02:
  target mean IoU: 0.6947
  target temporal IoU: 0.4321

epoch 1, validation-selected threshold 0.58:
  validation mean IoU: 0.8771
  validation temporal IoU: 0.2818
  target mean IoU: 0.7815
  target temporal IoU: 0.4769
```

Mean-best-style postprocess check:

```text
output: dense_union_raw_ring_outer_distance_384_smoke_temporal_eval_meanbest.json
threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
smoothing: bidirectional
hand_prior_power: 1.25

target mean IoU: 0.7963
target temporal IoU: 0.4836
```

Conclusion:

```text
Negative. Higher resolution with the current prior severely worsens target mean
IoU, even after the standard temporal postprocess. The failure appears to be
calibration/generalization, not just lack of spatial detail. The 384px .pt was
removed; keep the 256px raw_ring_outer_distance checkpoint as the trusted best.
```

## GrabCut Boundary-Refinement Probe

Question:

```text
Can image-aware postprocessing improve spatial boundaries without changing the
model or using target-window calibration?
```

Implementation:

```text
Added optional refinement support to:
  experiments/scheme3_v3/tune_temporal_postprocess.py

New args:
  --refine-modes none,grabcut
  --grabcut-iters
  --grabcut-fg-thresholds
  --grabcut-bg-thresholds

The GrabCut mode uses the current selected mask as probable foreground, high
model probability as sure foreground, and low probability as sure background.
It refines against the RGB image at the model resolution, then returns the
foreground/probable-foreground mask. The default remains refine_mode=none.
```

No-op smoke check:

```text
output: /tmp/scheme3_v3_refine_none_smoke.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
setting: threshold 0.42, keep 0.40, EMA 0.45, bidirectional, no refinement

held-out target:
  selected_union_mean_iou: 0.8713
  selected_temporal_union_iou: 0.4798

This exactly matches the known current-best path, confirming the new plumbing
does not alter the baseline when refinement is disabled.
```

Focused validation sweep:

```text
output: dense_union_raw_ring_outer_distance_finetune_grabcut_refine_probe.json
selection source: 4 non-target validation windows, round-robin across takes
tested thresholds: 0.40, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested refine modes: none, grabcut
tested GrabCut iterations: 1, 3
tested foreground seeds: 0.55, 0.65
tested background seeds: 0.05, 0.15
hand_prior_power: 1.25
```

Validation-selected setting:

```text
threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
smoothing: bidirectional
refine_mode: none
```

Held-out target:

```text
selected_union_mean_iou: 0.8713
selected_temporal_union_iou: 0.4798
```

Conclusion:

```text
Negative. GrabCut generally improved temporal self-IoU slightly but reduced
mean IoU enough that non-target validation rejected it. It also adds CPU
runtime. Do not carry GrabCut forward as part of the trusted pipeline.
```

## Adaptive Threshold Probe

Question:

```text
Can a per-frame threshold based on the frame's maximum predicted probability
recover low-confidence object frames without explicitly forcing an area floor
or using target-window calibration?
```

Implementation:

```text
Added optional adaptive threshold support to:
  experiments/scheme3_v3/tune_temporal_postprocess.py

New args:
  --threshold-modes fixed,relative_max
  --relative-threshold-fracs
  --relative-threshold-mins

The relative_max mode computes the on threshold for each frame as:
  max(relative_threshold_min, min(on_threshold, max_probability * relative_threshold_frac))

This can lower the on threshold only when the model is globally less confident
on that frame. Hysteresis keep-threshold remains explicit/fixed.
```

Sweep:

```text
output: dense_union_raw_ring_outer_distance_finetune_adaptive_threshold_probe.json
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
selection source: 8 non-target validation windows, round-robin across takes
tested threshold modes: fixed, relative_max
tested on thresholds: 0.40, 0.42, 0.44
tested keep thresholds: 0.35, 0.40
tested relative max fractions: 0.60, 0.70, 0.80, 0.90
tested relative minimum thresholds: 0.06, 0.10, 0.14
hand_prior_power: 1.25
```

Validation-selected setting:

```text
threshold_mode: fixed
threshold: 0.42
keep threshold: 0.40
EMA alpha: 0.45
smoothing: bidirectional
```

Held-out target:

```text
selected_union_mean_iou: 0.8713
selected_temporal_union_iou: 0.4798
```

Conclusion:

```text
Negative. Relative-to-max thresholding did not beat the existing fixed
threshold under non-target validation. As with area rescue, lowering thresholds
on low-confidence frames tends to add false-positive pixels before it recovers
enough true positives. Keep fixed thresholding in the trusted pipeline.
```

## Cleanup To Canonical Scheme 3 v3

Date:

```text
2026-06-05
```

Goal:

```text
Keep only the best Scheme 3 v3 model and the code needed to train, evaluate,
and render that current dense-union pipeline. Remove obsolete testing,
diagnostic, tuning, and old-prototype scripts.
```

Kept checkpoint:

```text
outputs/experiments/scheme3_v3/checkpoints/dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
```

Kept reports:

```text
outputs/experiments/scheme3_v3/checkpoints/dense_union_unetpp_b4_raw_ring_outer_distance_finetune_summary.json
outputs/experiments/scheme3_v3/checkpoints/dense_union_raw_ring_outer_distance_finetune_handpower_1p25_probe.json
outputs/experiments/scheme3_v3/checkpoints/dense_union_raw_ring_outer_distance_finetune_temporal_eval_handpower15_temporalheavy.json
```

Kept qualitative output:

```text
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_extended_morph_meanbest_nomorph/overlay.mp4
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_extended_morph_meanbest_nomorph/contact_sheet.jpg
outputs/experiments/scheme3_v3/qualitative_runs/dense_union_b4_raw_ring_outer_distance_finetune_target_30s_extended_morph_meanbest_nomorph/manifest.json
```

Kept source files:

```text
common.py
data.py
evaluate_dense_temporal.py
hand_prior.py
metrics.py
render_dense_union.py
train_dense_union.py
README.md
EXPERIMENT_LOG.md
```

Removed source files:

```text
diagnose_residuals.py
diagnose_target_postprocess_ceiling.py
losses.py
model.py
train.py
tune_area_postprocess.py
tune_ensemble_postprocess.py
tune_temporal_postprocess.py
```

Removed outputs:

```text
All non-canonical Scheme 3 v3 checkpoints, probe JSONs, and qualitative-run
folders were deleted from outputs/experiments/scheme3_v3.
```

Notes:

```text
README.md now describes only the current canonical pipeline. Historical trials
remain documented above in this experiment log, but the dead scripts and old
artifacts are no longer part of the runnable experiment surface.
```

## 2026-06-06: Motion-Compensated Temporal Metric And Flow-Pair Fine-Tune

Motivation:

```text
Raw temporal mask IoU is not a saturated temporal-consistency metric for
egocentric video. Camera motion and object motion legitimately change the mask
footprint, so comparing frame-t masks directly to frame-t+1 masks rewards masks
that stay fixed in image coordinates.
```

Metric added:

```text
evaluate_dense_temporal.py now reports optical-flow-compensated temporal IoU.
For each temporal pair, the previous predicted mask is warped into the current
frame with Farneback flow before IoU is computed. The same metric is computed
for GT masks, giving a motion-aware reference point. The evaluator also reports
full-FPS one-step raw and flow-aligned temporal IoU over the rendered 30-second
clip, which is closer to the observed swimming failure mode than sparse 1 FPS
annotation pairs.
```

Training change:

```text
train_dense_union.py now supports --flow-pair-weight. It samples annotated
Ego-Exo frames paired with nearby raw video frames, predicts both frames, warps
the annotated-frame prediction into the neighbor frame, and applies a
foreground-weighted flow-aligned consistency loss. This trains temporal
stability after motion compensation instead of asking masks to remain frozen in
raw pixel coordinates.
```

Baseline with mean-best postprocess:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
settings: threshold 0.42, keep 0.40, EMA 0.45, bidirectional smoothing,
          hand_prior_power 1.25, no morphology
held-out target selected_union_mean_iou: 0.8714
held-out target selected_temporal_union_iou: 0.4813
held-out target selected_flow_temporal_union_iou: 0.6100
held-out target gt_flow_temporal_union_iou: 0.6164
full_fps_flow_temporal_union_iou: 0.9338
report: outputs/experiments/scheme3_v3/checkpoints/dense_union_motion_metric_baseline_meanbest.json
```

Baseline with temporal-heavy postprocess:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
settings: threshold 0.42, keep 0.35, EMA 0.55, bidirectional smoothing,
          hand_prior_power 1.5, close kernel 5
held-out target selected_union_mean_iou: 0.8695
held-out target selected_temporal_union_iou: 0.4925
held-out target selected_flow_temporal_union_iou: 0.6203
held-out target gt_flow_temporal_union_iou: 0.6164
full_fps_flow_temporal_union_iou: 0.9378
report: outputs/experiments/scheme3_v3/checkpoints/dense_union_motion_metric_baseline_temporalheavy.json
```

Flow-pair fine-tune:

```text
init checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
output checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_flowpair_ft.pt
epochs: 2
train samples: 800
flow pair samples: 800
flow pair offsets: 1,-1,2,-2
flow_pair_weight: 0.10
learning_rate: 5e-6
save_selection: last
summary: outputs/experiments/scheme3_v3/checkpoints/dense_union_unetpp_b4_raw_ring_outer_distance_flowpair_ft_summary.json
```

Flow-pair fine-tune with temporal-heavy postprocess:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_flowpair_ft.pt
settings: threshold 0.56, keep 0.35, EMA 0.55, bidirectional smoothing,
          hand_prior_power 1.5, close kernel 5
held-out target selected_union_mean_iou: 0.8667
held-out target selected_temporal_union_iou: 0.4896
held-out target selected_flow_temporal_union_iou: 0.6167
held-out target gt_flow_temporal_union_iou: 0.6164
full_fps_temporal_union_iou: 0.9235
full_fps_flow_temporal_union_iou: 0.9444
report: outputs/experiments/scheme3_v3/checkpoints/dense_union_motion_metric_flowpair_ft_thr056_temporalheavy.json
```

Interpretation:

```text
The old 0.93+ number is real on non-target Ego-Exo validation/calibration
windows, not on the held-out target window. The held-out target remains about
0.87 IoU. Motion-compensated temporal IoU shows that much of the apparent
temporal instability in the old sparse metric came from camera/object motion:
baseline predictions are already close to GT under flow alignment. The
flow-pair fine-tune gives the best full-FPS flow-aligned temporal stability so
far while keeping target IoU comfortably above 0.8, but it is a candidate rather
than a replacement until qualitative overlays confirm reduced swimming.
```

## 2026-06-06: Unsupervised Adjacent-Frame Flow Consistency

Motivation:

```text
The sparse GT temporal metric is not an adequate supervised objective for
temporal behavior because Ego-Exo GT masks are available only about every 30
frames in the target window. Adjacent-frame consistency needs to be enforced
without adjacent-frame GT masks.
```

Training change:

```text
train_dense_union.py now supports:

--flow-pair-loss-mode unsupervised

In this mode the temporal term does not use GT masks. It predicts both frames in
an adjacent/near-adjacent pair, warps prediction A into frame B and prediction B
into frame A with optical flow, then applies symmetric prediction-to-prediction
consistency. A normal supervised dense mask loss remains on annotated frames to
anchor semantic mask quality.
```

Unsupervised flow-pair fine-tune:

```text
init checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
output checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_unsup_flowpair_ft.pt
epochs: 2
train samples: 800
flow pair samples: 800
flow pair offsets: 1,-1,2,-2
flow_pair_loss_mode: unsupervised
flow_pair_weight: 0.10
flow_pair_unsup_min_prob: 0.20
learning_rate: 5e-6
save_selection: last
summary: outputs/experiments/scheme3_v3/checkpoints/dense_union_unetpp_b4_raw_ring_outer_distance_unsup_flowpair_ft_summary.json
```

Result with temporal-heavy postprocess:

```text
checkpoint: dense_union_unetpp_b4_raw_ring_outer_distance_unsup_flowpair_ft.pt
settings: threshold 0.58, keep 0.35, EMA 0.55, bidirectional smoothing,
          hand_prior_power 1.5, close kernel 5
held-out target selected_union_mean_iou: 0.8648
held-out target selected_flow_temporal_union_iou: 0.6169
held-out target gt_flow_temporal_union_iou: 0.6164
full_fps_temporal_union_iou: 0.9258
full_fps_flow_temporal_union_iou: 0.9448
report: outputs/experiments/scheme3_v3/checkpoints/dense_union_motion_metric_unsup_flowpair_ft_thr058_temporalheavy.json
```

Interpretation:

```text
This is the preferred objective framing for temporal consistency: supervised
mask IoU on available Ego-Exo masks, plus unsupervised adjacent-frame
prediction consistency under optical flow. The sparse GT flow-IoU remains a
context metric, not the temporal training target.
```
