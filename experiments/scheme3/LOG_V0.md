# Hand Segmentor Experiment Log

This document records the model-design path that led to the current
`experiments/hand_segmentor` implementation. It is meant to be a paper-facing
working log: quantitative results are included wherever they are still available
from retained artifacts or from the chat record.

## Provenance And Caution

The most reliable numbers in this file are those backed by retained artifacts:

- `outputs/experiments/hand_segmentor/deeplab_r50_512/val_summary.json`
- `outputs/experiments/hand_segmentor/deeplab_r50_512/test_summary.json`
- `outputs/evaluation/han_baseline_p06_10s/summary.json`
- `outputs/evaluation/combination1_p06_10s/summary.json`

Earlier Grounded-SAM2 threshold sweeps and candidate-cache experiments were
cleaned up once the project moved to the supervised VISOR-trained segmentor.
For those, this document records the quantitative values that survived in the
chat history and clearly marks them as `chat-recorded / artifact removed`.

Important validity note: some early iterations tuned too aggressively on what
later became a hold-out set. Those results are documented for scientific
honesty and engineering context, but they should not be cited as clean
generalization evidence. The clean audited result is the final `test` split
reported below. That test split should now be considered spent for future model
changes.

## Task Definition

Goal: segment egocentric hands, gloves, and visible arms in EPIC-KITCHENS VISOR
frames and videos.

Target class:

```text
positive = hand OR glove OR visible arm attached to the wearer
negative = all other pixels
```

The motivation is scene simplification for visually impaired users. Hands and
active objects are central in egocentric video; the hand segmentor is intended as
a cleaner primitive than generic saliency for locating embodied interaction.

## Metric Definitions

The supervised segmentor reports per-frame metrics and then averages over
frames:

```text
IoU       = |prediction AND ground_truth| / |prediction OR ground_truth|
precision = |prediction AND ground_truth| / |prediction|
recall    = |prediction AND ground_truth| / |ground_truth|
```

The final supervised summaries also report:

```text
mean_detections = count of non-empty predicted masks per frame
empty_frames    = frames with no predicted positive pixels
```

For the older full scene-simplification pipeline evaluation, additional metrics
come from `src/evaluation/metrics.py`, including:

```text
foreground_recall / background_recall
target_pixel_precision / target_pixel_recall / target_pixel_jaccard
track_dropout_rate / object_frame_miss_rate / track_fragmentation_rate
output_load / output_active_area / flow_compensated_flicker
```

Those scene-level metrics are not the same as binary hand-mask IoU.

## Dataset And Splits

Final split naming is intentionally simple:

```text
train = model weight updates
val   = checkpoint and threshold selection
test  = clean audit only
```

All splits are whole-video disjoint.

| Split | Videos | Hand/glove frames |
|---|---:|---:|
| train | 36 | 1,845 |
| val | 5 | 729 |
| test | 4 | 1,171 |

### Train Videos

| Video | Frames |
|---|---:|
| P01_03 | 42 |
| P01_07 | 87 |
| P01_103 | 138 |
| P01_104 | 110 |
| P01_107 | 85 |
| P02_01 | 17 |
| P02_07 | 56 |
| P02_107 | 55 |
| P02_128 | 76 |
| P03_03 | 34 |
| P03_10 | 36 |
| P03_11 | 26 |
| P03_13 | 26 |
| P03_17 | 32 |
| P03_22 | 10 |
| P04_12 | 23 |
| P04_13 | 57 |
| P04_21 | 21 |
| P04_26 | 4 |
| P04_33 | 30 |
| P06_01 | 35 |
| P06_110 | 16 |
| P06_12 | 42 |
| P06_14 | 43 |
| P07_08 | 37 |
| P07_101 | 100 |
| P07_103 | 22 |
| P07_110 | 106 |
| P09_02 | 91 |
| P09_07 | 31 |
| P09_103 | 62 |
| P09_104 | 119 |
| P09_106 | 29 |
| P18_01 | 55 |
| P18_02 | 30 |
| P18_07 | 62 |

### Validation Videos

| Video | Frames |
|---|---:|
| P04_06 | 132 |
| P06_03 | 172 |
| P06_10 | 80 |
| P25_09 | 168 |
| P37_102 | 177 |

### Clean Test Videos

| Video | Frames |
|---|---:|
| P03_120 | 454 |
| P06_108 | 218 |
| P08_17 | 175 |
| P22_107 | 324 |

## Continuous Test Snippets

The sparse test-frame videos are useful for inspecting ground truth, but they
are not temporally smooth. To inspect temporal behavior, 12 continuous 30-second
snippets were generated, three per test video. The snippet manifest records the
source frame span and which VISOR annotated frames fall inside the snippet:

```text
data/epic_kitchens/video_snippets/test_set/continuous_segments/manifest.json
```

The hand segmentor was run on all 12 snippets:

```text
outputs/experiments/hand_segmentor/continuous_segment_runs/
```

Each output folder contains:

```text
overlay.mp4
contact_sheet.png
manifest.json
```

The continuous video run used the final checkpoint threshold `0.55` and
`--tta-flip`. It is a visual inspection run only; no dense ground truth exists
for the intermediate frames, so no IoU/precision/recall is reported for these
continuous clips.

| Video | Snippet | Source frames | VISOR annotated frames inside | Output frames |
|---|---:|---:|---:|---:|
| P03_120 | 1 | 7447-8946 | 38 | 1547 |
| P03_120 | 2 | 18367-19866 | 60 | 1517 |
| P03_120 | 3 | 35264-36763 | 14 | 1514 |
| P06_108 | 1 | 10312-11811 | 12 | 1512 |
| P06_108 | 2 | 21992-23491 | 18 | 1542 |
| P06_108 | 3 | 30026-31525 | 17 | 1526 |
| P08_17 | 1 | 4323-5822 | 4 | 1500 |
| P08_17 | 2 | 21233-22732 | 8 | 1500 |
| P08_17 | 3 | 31143-32642 | 15 | 1500 |
| P22_107 | 1 | 3893-5392 | 24 | 1543 |
| P22_107 | 2 | 12646-14145 | 14 | 1546 |
| P22_107 | 3 | 21898-23397 | 24 | 1548 |

Total continuous-segment inference frames: `18,295`.

`P08_17` had no public MP4 endpoint. For that video, the public RGB-frame tar
was streamed once, the selected snippet frames were extracted temporarily,
encoded to MP4, and the temporary JPEGs were deleted. The P08 source resolution
is `456x256`; the public MP4 snippets are `1920x1080`.

## Iteration History

### 2026-05-31 Follow-Up: 0.95 IoU Attempt

The retained DeepLabV3-ResNet50 hand segmentor remains the best model.

Current retained checkpoint:

```text
outputs/experiments/hand_segmentor/deeplab_r50_512/best.pt
```

Retained validation result:

```text
threshold 0.55 with horizontal-flip TTA
mean IoU       0.9175
mean precision 0.9515
mean recall    0.9612
```

Retained test result:

```text
threshold 0.55 with horizontal-flip TTA
mean IoU       0.9074
mean precision 0.9553
mean recall    0.9480
```

Per-video test result:

```text
P03_120  IoU 0.9094
P06_108  IoU 0.9466
P08_17   IoU 0.9293
P22_107  IoU 0.8664
```

The main held-out weakness is `P22_107`, where both precision and recall are
lower than the other test videos.

#### Threshold And Connected-Component Postprocessing

A sweep over thresholds, minimum component area, top-k components, and small
morphological opening was run on the retained DeepLab model.

Best validation postprocess setting:

```text
threshold 0.55
min_area 64
topk 3
opening kernel 3

val mean IoU       0.9176
val mean precision 0.9532
val mean recall    0.9610
```

Best test postprocess setting:

```text
threshold 0.40
opening kernel 3

test mean IoU       0.9098
test mean precision 0.9477
test mean recall    0.9582
```

Interpretation:

```text
Postprocessing gives only tiny changes. The false positives observed in visual
inspection are not mostly removable tiny components; they are entangled with
the main predicted region or represent a model/data generalization issue.
```

#### Empty-Frame Fine-Tune

Tried fine-tuning from the retained checkpoint with empty/background train
frames included and a more precision-oriented Tversky loss:

```text
init checkpoint: outputs/experiments/hand_segmentor/deeplab_r50_512/best.pt
include empty train frames: yes
tversky beta: 0.42
lr: 6e-5
```

Validation trajectory before early stop:

```text
epoch 1 val IoU 0.901 P 0.955 R 0.937
epoch 2 val IoU 0.882 P 0.952 R 0.925
epoch 3 val IoU 0.902 P 0.948 R 0.951
epoch 4 val IoU 0.899 P 0.947 R 0.944
```

Interpretation:

```text
The empty-frame fine-tune increased precision pressure but harmed overall IoU.
It was stopped early and its output/cache were removed.
```

#### SegFormer-B0 Hand Segmentor

Tried a separate SegFormer-B0 binary hand segmentor, initialized from
`nvidia/segformer-b0-finetuned-ade-512-512`.

Validation trajectory:

```text
epoch 1  val IoU 0.832 P 0.896 R 0.919
epoch 2  val IoU 0.855 P 0.909 R 0.936
epoch 3  val IoU 0.874 P 0.919 R 0.945
epoch 5  val IoU 0.883 P 0.926 R 0.948
epoch 9  val IoU 0.883 P 0.913 R 0.961
```

Interpretation:

```text
SegFormer-B0 is clearly below the retained DeepLab model and did not approach
the 0.95 IoU target. The temporary trainer/checkpoint were removed after
recording the result.
```

Current status:

```text
0.95 IoU has not been reached. The best retained model remains DeepLabV3-R50.
The next credible hand direction is targeted data expansion or hard-negative
HITL correction for the weak video/domain, not more generic postprocessing.
```

### 1. Original Scene Simplification Baselines

The repository started from a scene simplification pipeline inspired by the Han
baseline. It combined:

- saliency
- depth
- segmentation
- edge/simplification fusion

During cleanup, `han_baseline.py` and `combination1.py` were refactored to use
model adapters under `src/models/`, and shared fusion logic was consolidated in
`src/simplification/fusion.py`.

This stage was not a hand-mask segmentor. It is included because it established
the later need for a dedicated hand/active-object component.

### 2. Manual DEVA / Combination1 Direction

`combination1` used manual DEVA-style segmentation rather than automatic SAM
grid prompting. The intent was:

- never run DEVA in automatic mode
- use manually supplied classes/prompts
- keep categories separable for future logic:
  - hands/arms
  - foreground objects
  - scene/background
- combine the masks for display initially, matching the baseline output format

VISOR mask extraction logic was moved toward evaluation rather than model
inference, because VISOR labels are ground truth and should not drive inference.

Quantitative retained artifact for `combination1` is from the P06 10-second
pipeline evaluation, not from hand-only segmentation:

Source:

```text
outputs/evaluation/combination1_p06_10s/summary.json
```

| Metric | Value |
|---|---:|
| clips | 1 |
| output_frames | 200 |
| evaluated_frames | 200 |
| foreground_recall | 1.0000 |
| foreground_overlap | 0.8222 |
| background_recall | 1.0000 |
| background_overlap | 0.6502 |
| target_pixel_precision | 0.00495 |
| target_pixel_recall | 0.6376 |
| target_pixel_jaccard | 0.00494 |
| pseudo_target_pixel_precision | 0.2743 |
| pseudo_target_pixel_recall | 0.5610 |
| pseudo_target_pixel_jaccard | 0.2258 |
| output_load | 0.1802 |
| output_active_area | 0.4560 |
| flow_compensated_flicker | 0.0618 |
| track_dropout_rate | 0.0000 |
| object_frame_miss_rate | 0.0000 |
| track_fragmentation_rate | 0.0000 |

Interpretation:

- It could represent all foreground/background categories in the small clip.
- It was not sufficiently precise for hand-focused segmentation.
- It retained too much broad mask area when measured against exact target pixels.
- This helped motivate a simpler, dedicated hand/glove/arm segmentor.

### 3. Han Baseline Scene Simplification Reference

Retained P06 10-second evaluation for the Han baseline:

Source:

```text
outputs/evaluation/han_baseline_p06_10s/summary.json
```

| Metric | Value |
|---|---:|
| clips | 1 |
| output_frames | 200 |
| evaluated_frames | 200 |
| foreground_recall | 1.0000 |
| foreground_overlap | 0.9648 |
| background_recall | 1.0000 |
| background_overlap | 0.9185 |
| target_pixel_precision | 0.00530 |
| target_pixel_recall | 0.9125 |
| target_pixel_jaccard | 0.00530 |
| pseudo_target_pixel_precision | 0.3466 |
| pseudo_target_pixel_recall | 0.9474 |
| pseudo_target_pixel_jaccard | 0.3400 |
| output_load | 0.4648 |
| output_active_area | 0.6166 |
| flow_compensated_flicker | 0.0956 |
| track_dropout_rate | 0.0000 |
| object_frame_miss_rate | 0.0000 |
| track_fragmentation_rate | 0.0000 |

Interpretation:

- The baseline had higher target recall than `combination1` on this clip, but
  also much higher output active area.
- For exact target pixels, precision was extremely low because the task here was
  broad scene simplification rather than tight hand segmentation.
- This reinforced the need for a dedicated hand segmentor with a tighter
  foreground definition.

### 4. MediaPipe Hands Consideration

MediaPipe Hands was considered and rejected as the primary method because it is
a keypoint/landmark detector rather than a segmentation model. It may be useful
as a later ROI prior or temporal sanity check, but it cannot directly produce
the dense hand/glove/arm masks required here.

No quantitative MediaPipe segmentation metrics were produced.

### 5. Grounded-SAM2 / GroundingDINO Hand Experiments

The next major direction was open-vocabulary hand segmentation with
GroundingDINO + SAM2. The experiment originally lived under:

```text
experiments/grounded_sam2_hands/
```

It was later renamed/generalized to:

```text
experiments/hand_segmentor/
```

The open-vocabulary segmentor was attractive because it required no training
and could potentially segment:

- hand
- hands
- left hand / right hand
- glove / right glove
- arm
- hand + arm

Prompts and settings tried or planned in the Grounded-SAM2 phase:

| Prompt / prompt set | Notes |
|---|---|
| `hand.` | Reduced false positives from broader body-part prompts, but could miss arms/wrists. |
| `hands.` | Similar to `hand.`; used as a prompt variant in candidate generation. |
| `left hand. right hand.` | Considered before deciding left/right distinction was unnecessary. |
| `glove. right glove.` | Added because VISOR frames include gloves/attachments and bare-hand-only prompts could miss them. |
| `hand. arm.` / `hand.arm` | Improved coverage of visible forearms, but increased false positives such as feet/body parts in some frames. |

Thresholds and generation settings explored:

| Parameter | Values / notes |
|---|---|
| GroundingDINO box threshold | Tried values including `0.15`, `0.18`, `0.22`, `0.25`, and later raised to `0.40`. |
| GroundingDINO text threshold | Raised/lowered during prompt tuning; exact retained row metrics are no longer available. |
| max detections | Initially `4`, later high-recall candidate generation used `8`, then postprocessing reduced masks. |
| NMS threshold | Discussed as part of GroundingDINO candidate filtering; exact retained value not preserved in artifacts. |
| SAM variant | SAM2 / SAM2.1 tiny was used initially; larger SAM2 variants were considered. |

Observed failure modes:

- Open-vocabulary prompts could over-segment body-like objects, especially when
  `arm` was included.
- Feet could be segmented because GroundingDINO/SAM matched exposed limbs or
  skin-like regions under broad prompts.
- Gloves on tables could be detected even when they were not the wearer’s hand.
- Tight `hand` prompts improved precision but missed forearms and sometimes
  missed glove-like attachments.
- Recall-first settings produced large candidate masks, and simple thresholding
  was not enough to recover precision.

Postprocessing ideas tried or prototyped:

- bottom-origin egocentric heuristic: hands/arms usually enter from the lower
  image boundary in egocentric video
- area filtering
- connected-component filtering
- candidate score filtering
- temporal adjacency checks
- short-gap fill
- temporal median/majority smoothing
- optical-flow propagation for low-recall gaps
- class-aware logic:
  - keep `hand`/`glove` candidates by default
  - keep `arm` candidates only if connected downward or temporally adjacent to a
    hand/glove mask

The strongest chat-recorded retained number from this family was:

| Source | Prompt | Box threshold | Text threshold | Max detections | Frames | IoU | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| chat-recorded / artifact removed | `hand. arm.` | 0.25 | 0.18 | 4 | 1,838 | 0.453 | 0.644 | 0.607 |

Interpretation:

- The model was far below the desired IoU range.
- It was useful as a diagnostic and candidate generator, but not good enough as
  the final segmentor.
- The main pain point was inconsistent generalization across videos, especially
  glove-heavy or visually cluttered regions.

Invalid/overfit Grounded-SAM2 note:

- At one point, iterative tuning reached very high IoU on a development/training
  set, approximately `0.95`, but this did not transfer cleanly to held-out data.
- The user correctly rejected this as over-tuning / hold-out leakage.
- Those numbers are intentionally not treated as valid evidence.

### 6. Candidate Cache And Dev/Holdout Discipline

To avoid rerunning Grounded-SAM2 repeatedly, a candidate-cache design was
implemented during the open-vocabulary phase:

- generate raw candidates once per frame
- cache masks, boxes, labels, scores, prompt source, area, and origin stats
- evaluate/refine cached masks without rerunning Grounded-SAM2
- keep VISOR ground truth out of inference-time candidate files

Planned deterministic sets:

```text
dev100   = 100 evenly spaced frames from long hand-mask spans
long1838 = all frames from P06_110:430-1126, P07_103:409-728, P07_103:765-1583
```

Long hand-mask spans identified:

| Video | Frame span | Consecutive frames |
|---|---:|---:|
| P06_110 | 430-1126 | 697 |
| P07_103 | 409-728 | 320 |
| P07_103 | 765-1583 | 819 |

This direction was later cleaned up because the final supervised VISOR segmentor
was simpler, stronger, and more directly aligned with the target class.

### 7. Larger SAM2 / Other Model Exploration

Larger SAM2 models and other segmentation approaches were considered once the
open-vocabulary results plateaued. The conclusion was that the dominant failure
was not just SAM mask quality; it was open-vocabulary detection and prompt
semantics. In other words:

- if GroundingDINO produced the wrong box, SAM could faithfully segment the
  wrong object
- broader prompts improved coverage but harmed precision
- tighter prompts improved precision but missed gloves/arms

No retained artifact shows a larger-SAM2 run exceeding the final supervised
model. The project therefore pivoted to supervised binary segmentation using
VISOR hand/glove masks.

### 8. Supervised Binary Segmentor

Final model:

```text
DeepLabV3-ResNet50
input size: 512x912
output: one binary hand/glove/visible-arm mask channel
pretraining: torchvision DeepLabV3 ResNet50 weights
loss: BCE + Tversky-style term, beta=0.65
test-time augmentation: horizontal flip averaging
selected threshold: 0.55
```

Training command:

```bash
PYTHONPATH=src:experiments/hand_segmentor .venv-models/bin/python \
  experiments/hand_segmentor/train_supervised_segmentor.py \
  --mode train \
  --model resnet50 \
  --image-size 512x912 \
  --cache-dir outputs/experiments/hand_segmentor/cache_512x912 \
  --output-dir outputs/experiments/hand_segmentor/deeplab_r50_512 \
  --train-split train \
  --val-split val \
  --epochs 10 \
  --batch-size 2 \
  --lr 1e-4 \
  --tversky-beta 0.65 \
  --thresholds 0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7 \
  --tta-flip
```

Evaluation command:

```bash
PYTHONPATH=src:experiments/hand_segmentor .venv-models/bin/python \
  experiments/hand_segmentor/train_supervised_segmentor.py \
  --mode eval \
  --model resnet50 \
  --image-size 512x912 \
  --cache-dir outputs/experiments/hand_segmentor/cache_512x912 \
  --output-dir outputs/experiments/hand_segmentor/deeplab_r50_512 \
  --checkpoint outputs/experiments/hand_segmentor/deeplab_r50_512/best.pt \
  --eval-split test \
  --tta-flip
```

MobileNetV3 note:

- `DeepLabV3-MobileNetV3-Large` is supported by the training script as a lighter
  backbone.
- The final retained artifact and final reported result are from
  `DeepLabV3-ResNet50`, not MobileNetV3.

## Supervised Segmentor Results

### Validation Threshold Sweep

Source:

```text
outputs/experiments/hand_segmentor/deeplab_r50_512/val_summary.json
```

| Threshold | Frames | Mean IoU | Precision | Recall | Empty frames |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 729 | 0.9175 | 0.9515 | 0.9612 | 1 |
| 0.50 | 729 | 0.9173 | 0.9487 | 0.9638 | 1 |
| 0.60 | 729 | 0.9173 | 0.9555 | 0.9583 | 2 |
| 0.45 | 729 | 0.9169 | 0.9459 | 0.9662 | 1 |
| 0.65 | 729 | 0.9167 | 0.9582 | 0.9549 | 2 |
| 0.40 | 729 | 0.9160 | 0.9429 | 0.9685 | 1 |
| 0.70 | 729 | 0.9155 | 0.9608 | 0.9511 | 2 |
| 0.35 | 729 | 0.9150 | 0.9398 | 0.9706 | 0 |
| 0.30 | 729 | 0.9136 | 0.9364 | 0.9727 | 0 |
| 0.25 | 729 | 0.9118 | 0.9326 | 0.9749 | 0 |

Best validation threshold:

```text
threshold=0.55
IoU=0.9175
precision=0.9515
recall=0.9612
frames=729
empty_frames=1
```

### Validation Per-Video Breakdown At Threshold 0.55

| Video | Frames | Mean IoU | Precision | Recall | Empty frames |
|---|---:|---:|---:|---:|---:|
| P04_06 | 132 | 0.9079 | 0.9272 | 0.9764 | 0 |
| P06_03 | 172 | 0.9306 | 0.9586 | 0.9695 | 0 |
| P06_10 | 80 | 0.9409 | 0.9636 | 0.9758 | 0 |
| P25_09 | 168 | 0.8892 | 0.9471 | 0.9308 | 1 |
| P37_102 | 177 | 0.9282 | 0.9613 | 0.9640 | 0 |

### Clean Test Result

Source:

```text
outputs/experiments/hand_segmentor/deeplab_r50_512/test_summary.json
```

Overall:

| Threshold | Frames | Mean IoU | Precision | Recall | Mean detections | Empty frames |
|---:|---:|---:|---:|---:|---:|---:|
| 0.55 | 1,171 | 0.9074 | 0.9553 | 0.9480 | 1.0000 | 0 |

Per-video:

| Video | Frames | Mean IoU | Precision | Recall | Empty frames |
|---|---:|---:|---:|---:|---:|
| P03_120 | 454 | 0.9094 | 0.9538 | 0.9523 | 0 |
| P06_108 | 218 | 0.9466 | 0.9823 | 0.9631 | 0 |
| P08_17 | 175 | 0.9293 | 0.9610 | 0.9663 | 0 |
| P22_107 | 324 | 0.8664 | 0.9362 | 0.9221 | 0 |

Interpretation:

- The final supervised model reaches the target regime (`~0.9 IoU`) on a
  video-disjoint clean test set.
- Precision and recall are both high, with precision slightly higher than
  recall overall.
- `P22_107` is the hardest retained test video: IoU `0.8664`, precision
  `0.9362`, recall `0.9221`.
- `P06_108` is the strongest retained test video: IoU `0.9466`.
- The validation threshold sweep is flat near the optimum; thresholds from
  `0.45` to `0.65` all produce mean IoU above `0.916`, suggesting the model is
  not extremely brittle to the exact cutoff.

## Visual Artifacts

Static test overlays:

```text
outputs/experiments/hand_segmentor/deeplab_r50_512/test_overlays/
```

Continuous snippet overlays:

```text
outputs/experiments/hand_segmentor/continuous_segment_runs/
```

Overlay color convention for static ground-truth comparisons:

```text
green = true positive
red   = false positive
blue  = false negative
```

Continuous-video overlays use green for predicted hand/glove/visible-arm mask.

## What Worked

- A supervised binary DeepLabV3-ResNet50 trained on VISOR hand/glove frames was
  much stronger than open-vocabulary hand prompts.
- Whole-video disjoint splitting was essential. Earlier tuning on narrow frame
  subsets created misleadingly high estimates.
- Test-time horizontal flip averaging helped enough to keep as part of the final
  command.
- Selecting threshold on validation only produced a stable threshold (`0.55`)
  with clean test IoU above `0.90`.

## What Did Not Work Well

- Grounded-SAM2 / GroundingDINO prompt tuning did not reach high enough IoU on
  held-out data.
- Broad prompts such as `hand. arm.` introduced false positives.
- Tight prompts such as `hand.` improved some false positives but could miss
  arms, wrists, and gloves.
- Egocentric bottom-origin filtering was conceptually useful but insufficient
  to close the gap by itself.
- Iterating on a small or already-inspected hold-out set created invalidly high
  results; future claims should use fresh, untouched splits.

## Paper-Ready Claims That Are Currently Supported

The following statements are supported by retained artifacts:

1. A DeepLabV3-ResNet50 binary hand/glove/visible-arm segmentor trained on
   VISOR dense masks achieved mean IoU `0.9074`, precision `0.9553`, and recall
   `0.9480` on a video-disjoint clean test set of `1,171` EPIC-KITCHENS VISOR
   frames.

2. The selected threshold was chosen on a separate validation set of `729`
   frames, where threshold `0.55` achieved mean IoU `0.9175`, precision
   `0.9515`, and recall `0.9612`.

3. The test split was whole-video disjoint from training and validation and
   contained four videos: `P03_120`, `P06_108`, `P08_17`, and `P22_107`.

4. Open-vocabulary Grounded-SAM2 was tested as an earlier direction. The best
   chat-recorded long-span result was IoU `0.453`, precision `0.644`, recall
   `0.607` for `hand. arm.` with box threshold `0.25`, text threshold `0.18`,
   and max detections `4` over `1,838` frames. This artifact was later removed,
   so this number should be treated as historical context unless regenerated.

## Open Issues / Next Steps

- Reserve a new unseen test split before changing the model again.
- Add active-object segmentation next, ideally with split discipline from the
  start.
- Consider temporal smoothing only after evaluating the framewise model on a new
  validation set; otherwise temporal rules can overfit to known videos.
- If MediaPipe is revisited, use it as an auxiliary ROI/keypoint prior, not as a
  replacement for dense segmentation.
- If Grounded-SAM2 is revisited, regenerate all candidate-cache metrics and keep
  the exact CSV/JSON summaries rather than relying on chat-recorded values.
