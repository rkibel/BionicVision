# Active Object Experiment Log

This file is intentionally uncapped. It records the active-object work in enough
detail to preserve both successes and dead ends.

## Ground Truth Proxy

VISOR does not consistently expose a direct held/interacted-object label in the
local annotations. The active-object target used here is derived from VISOR
masks:

```text
active object =
  non-hand VISOR object mask
  with area <= 35% of the frame
  that touches a VISOR hand mask dilated by 12 px
```

This is a contact-derived proxy. It captures hand-object contact well, but it is
not an official VISOR label for intention, grasp, or persistence after release.

## Frozen Prior

The hand segmentor from `experiments/hand_segmentor` is treated as frozen. The
active-object experiments may read its checkpoint and cached predictions, but
must not modify any code in `experiments/hand_segmentor`.

Frozen hand checkpoint:

```text
outputs/experiments/hand_segmentor/deeplab_r50_512/best.pt
```

## Oracle Proposal Ceiling

Script:

```text
experiments/active_object/evaluate_contact_baselines.py
```

This experiment uses VISOR object masks as inference-time proposals, then uses
the frozen hand segmentor to choose which VISOR object masks are active.

Best validation-selected parameters:

```text
hand_threshold      0.45
contact_radius      12
max_area_frac       0.35
min_contact_pixels  1
persist_steps       0
```

Validation positive-frame metrics:

```text
IoU       0.9833
P         0.9864
R         0.9920
obj IoU   0.9768
obj P     0.9841
obj R     0.9878
```

Held-out test positive-frame metrics:

```text
IoU       0.9853
P         0.9907
R         0.9905
obj IoU   0.9811
obj P     0.9899
obj R     0.9871
```

Interpretation:

```text
If high-quality object proposals are available, hand-contact selection is more
than strong enough to exceed the 0.90 IoU target.
```

This is not yet a raw-video segmentor because VISOR object masks are used as
proposals.

## Failed Non-Oracle Attempts

These were removed from the active experiment code/output after documentation to
keep the repo lean.

### Full-Frame DeepLab With Hand Overlay

Input:

```text
RGB with predicted hand probability rendered as a green overlay
```

Target:

```text
full-frame derived active-object mask
```

Observed validation peak:

```text
IoU ~0.39
```

Reason for stopping:

```text
The target is too sparse at full-frame scale. The model mostly learns a weak
foreground prior and does not recover active object boundaries.
```

### Full-Frame Four-Channel DeepLab

Input:

```text
RGB + frozen hand probability as a fourth channel
```

Variants tried:

```text
overlay RGB + hand channel       peak IoU ~0.41
raw RGB + hand channel           peak IoU ~0.44
positive active frames only      peak IoU ~0.44
```

Reason for stopping:

```text
Adding the hand channel helped only modestly. Full-frame active-object
segmentation remains proposal-limited and far below 0.90 IoU.
```

### SAM ViT-H Point Proposals Around Hand

Input:

```text
SAM ViT-H
positive points sampled in a ring around the predicted hand mask
negative point sampled at the hand centroid
```

Observed smoke metrics:

```text
IoU ~0.17-0.23
```

Reason for stopping:

```text
The prompts often selected hand fragments or background surfaces instead of the
contact object. Runtime was high.
```

### SAM ViT-H Automatic Masks In Hand Crop

Input:

```text
SAM automatic masks on hand-centered crops with hand-contact filtering
```

Observed smoke metrics:

```text
best IoU ~0.27
```

Reason for stopping:

```text
Larger crops increased recall but precision collapsed. Runtime was still high.
```

### Hand Dilation Only

Input:

```text
dilated predicted hand mask, excluding the hand itself
```

Observed validation-cache sample:

```text
best IoU ~0.30
```

Reason for stopping:

```text
Large dilation increases recall but includes too much background.
```

### Mask2Former Panoptic Proposals

Input:

```text
facebook/mask2former-swin-base-coco-panoptic proposals
frozen hand-contact selector
```

Observed smoke metrics:

```text
val smoke best IoU ~0.23
test smoke IoU ~0.07
```

Reason for stopping:

```text
COCO panoptic proposals do not align well with VISOR kitchen object masks in
the active-object contact region. Recall can be high, but precision and mask
shape are poor.
```

## Current Serious Attempt

The next attempt is a hand-centered crop segmentor:

```text
input:  crop RGB + frozen hand probability channel
target: crop of derived active-object mask
output: pasted back into full VISOR annotation frame for IoU scoring
```

Rationale:

```text
Full-frame training dilutes small active-object masks. Cropping around the
predicted hand makes the active object much larger in the training view while
still using only the frozen hand prior at inference.
```

Initial result:

```text
crop active-object target, margin 128, DeepLabV3-R50
peak val IoU ~0.41

crop objectness target + connected-component hand-contact selection
peak val IoU ~0.44
```

Interpretation:

```text
Cropping helps target scale but still does not solve proposal quality. The
learner finds rough contact regions, not full VISOR-quality object masks.
```

## SegFormer Crop Attempts

SegFormer was tried because DeepLab appeared to be underfitting object
boundaries. The input was a hand-centered crop with the frozen hand probability
rendered into the RGB image as a green prior. The target remained the derived
active-object crop.

### SegFormer-B2

Command shape:

```text
model: nvidia/segformer-b2-finetuned-ade-512-512
crop cache: outputs/experiments/active_object/crop_cache_384
batch size: 4
lr: 6e-5
epochs: 8
```

Validation results:

```text
epoch 1 IoU 0.507 P 0.637 R 0.721
epoch 2 IoU 0.501 P 0.649 R 0.700
epoch 3 IoU 0.512 P 0.654 R 0.719
epoch 4 IoU 0.479 P 0.578 R 0.754
epoch 5 IoU 0.516 P 0.646 R 0.752
epoch 6 IoU 0.497 P 0.615 R 0.758
epoch 7 IoU 0.514 P 0.659 R 0.736
epoch 8 IoU 0.529 P 0.642 R 0.768
```

Continuation from the best B2 checkpoint with a lower learning rate did not
improve the result:

```text
best continuation IoU ~0.522
```

### SegFormer-B5

Command shape:

```text
model: nvidia/segformer-b5-finetuned-ade-640-640
batch size: 2
lr: 4e-5
epochs: 6
```

Validation results:

```text
epoch 1 IoU 0.505 P 0.640 R 0.719
epoch 2 IoU 0.498 P 0.646 R 0.719
epoch 3 IoU 0.505 P 0.660 R 0.703
epoch 4 IoU 0.501 P 0.629 R 0.736
epoch 5 IoU 0.481 P 0.687 R 0.633
epoch 6 IoU 0.510 P 0.699 R 0.695
```

Interpretation:

```text
SegFormer improves over DeepLab, but the non-oracle learned crop setup still
tops out at about 0.53 validation IoU. Scaling model size alone does not solve
the proposal-quality gap.
```

Current non-oracle best:

```text
SegFormer-B2 crop active-object segmentor
validation IoU 0.529
```

Held-out test score at the validation-best threshold:

```text
threshold 0.35
frames 1290
positive frames 963

test IoU 0.4719
test P   0.5488
test R   0.8026
```

Per-video test IoU:

```text
P03_120  0.5425
P06_108  0.4028
P08_17   0.4951
P22_107  0.3823
```

This is not close to the 0.90 target. The gap is mostly precision: the model
often finds the general contact region, but its masks bleed into nearby surfaces
or do not align to VISOR object-instance boundaries.

## HITL Active-Object Ground Truth Comparison

After the HITL annotation pass, all prompted frames were submitted:

```text
total frames: 475
positive active-object frames: 437
empty active-object frames: 38

train: 275 positive, 25 empty
eval:   68 positive, 7 empty
test:   94 positive, 6 empty
```

The annotations were intentionally conservative: if the active object was
unclear, absent, or not worth highlighting, the frame could be submitted empty.
This makes recall important, but it also means broad VISOR-contact masks are
penalized heavily for including nearby surfaces or incidental objects.

Output summary:

```text
outputs/experiments/active_object_hitl_oracle/summary.json
```

### Compared Methods

`all_visible_area035`

Predict every non-hand VISOR object whose area is at most 35% of the frame.

`old_visor_hand_contact_oracle_r12_area035`

The previous oracle/proxy rule: use VISOR hand masks, dilate by 12 px, and
select every non-hand VISOR object touching that hand-contact zone, with the
same 35% area cap.

`greedy_visor_object_ceiling_area035`

Diagnostic ceiling only. It uses the HITL target to greedily choose VISOR
object masks that maximize IoU. This is not an inference method; it answers
whether VISOR object proposals can express the HITL target.

### Results On Positive HITL Frames

Frame-macro metrics:

```text
method                                  IoU     precision  recall
all_visible_area035                     0.338   0.358      0.803
old_visor_hand_contact_oracle_r12       0.415   0.447      0.779
VISOR object proposal ceiling, area035  0.658   0.725      0.791
VISOR object proposal ceiling, no cap   0.658   0.725      0.791
```

Pixel-micro metrics for the old VISOR-contact oracle on positive frames:

```text
IoU       0.237
precision 0.249
recall    0.827
```

Split-level positive-frame macro metrics for the old contact oracle:

```text
split   IoU     precision  recall
train   0.444   0.478      0.789
eval    0.298   0.326      0.664
test    0.413   0.444      0.833
```

Contact-radius check with the old 35% VISOR object area cap:

```text
radius  IoU     precision  recall
4       0.413   0.451      0.771
8       0.413   0.447      0.776
12      0.415   0.447      0.779
24      0.410   0.440      0.785
48      0.400   0.426      0.796
```

### Interpretation

The old oracle is not a good match for the HITL target. It has useful recall,
especially by pixel-micro recall, but precision is low because it selects too
many nearby VISOR objects or broad surfaces. Increasing the contact radius
raises recall only slightly and lowers precision, so this is not just a radius
tuning issue.

The VISOR proposal ceiling is also below the desired range:

```text
best VISOR-object ceiling IoU on positive HITL frames: ~0.658
```

That means a substantial part of the gap comes from target/proposal mismatch:
some HITL masks are more conservative, more specific, or simply different from
the available VISOR object instances. A future active-object segmentor should
therefore be evaluated directly against HITL masks rather than against the
VISOR-contact proxy alone.

## HITL Supervised Segmentor Iteration

The first direct HITL model used RGB plus a frozen hand-segmentor prior painted
into the input. Architecture was DeepLabV3-ResNet50, initialized from ImageNet
segmentation weights, trained on the 300 HITL train frames and selected on the
75 HITL eval frames. The objective used a recall-heavy Tversky term
(`beta=0.85`).

Best eval result:

```text
threshold 0.15
positive frames 68
empty frames 7

positive-frame macro IoU       0.294
positive-frame macro precision 0.326
positive-frame macro recall    0.832
all-frame macro IoU            0.267
```

Interpretation:

```text
The model learns a broad contact-region prior and gives decent recall, but it
does not localize the active-object boundary. This is substantially below the
0.80 IoU target and worse than the VISOR object-mask ceiling, so the next
direction should be proposal-based rather than another full-frame semantic
segmentor.
```

### Hand-Centered Crop DeepLab

The next supervised attempt cropped around the frozen predicted hand mask before
training DeepLabV3-ResNet50. This made the active object larger in the input but
did not materially change the result.

Best eval behavior:

```text
positive-frame macro IoU       ~0.36
positive-frame macro recall    ~0.72
```

Interpretation:

```text
The active object is not simply lost because it is small in the full frame.
The model still learns contact-region texture rather than a stable object
boundary.
```

## SAM Proposal Experiments On HITL

Because the HITL masks were produced using SAM clicks, the next question was
whether SAM can provide a usable proposal set without manual clicks.

### SAM Hand-Ring Prompts

Method: sample positive SAM points in a ring around the frozen hand mask and
union filtered masks that touch the hand-contact zone.

Smoke result on 10 eval frames:

```text
IoU       0.157
precision 0.159
recall    0.658
```

Even the proposal oracle over these ring-prompt masks was uneven:

```text
10-frame proposal-oracle mean IoU ~0.61
```

Interpretation:

```text
Hand-ring point prompts sometimes produce the right mask exactly, but they miss
too many cases. The prompt strategy is not reliable enough to pursue.
```

### SAM Automatic Proposals

SAM automatic masks were much more promising. On the same 10-frame smoke subset,
a greedy oracle over automatic masks reached:

```text
IoU       ~0.905
precision ~0.922
recall    ~0.982
```

This shows that SAM automatic proposals often contain the HITL object mask. The
problem becomes selecting the right masks.

### SAM Automatic Mask Selector

Attempted selector:

```text
proposal source: SAM ViT-H automatic masks
hand prior: frozen hand segmentor
active prior: full-frame HITL DeepLab heatmap
selector: small MLP over geometry, hand-contact, SAM score, and active-heatmap features
training labels: HITL train masks
```

Full train/eval selector result:

```text
best recall-oriented eval setting:
threshold 0.15, topk 8

positive-frame macro IoU       0.384
positive-frame macro precision 0.398
positive-frame macro recall    0.850
```

Best more balanced eval settings were around:

```text
positive-frame macro IoU       ~0.49
positive-frame macro recall    ~0.69-0.80
```

Changing the training target from candidate IoU to candidate precision/purity
on a 50-train/20-eval smoke run improved some high-recall tradeoffs but did not
break through:

```text
best high-recall smoke setting:
positive-frame macro IoU       ~0.45-0.51
positive-frame macro recall    ~0.81
```

Interpretation:

```text
SAM automatic proposals have enough mask quality in principle, but the current
feature-only selector does not reliably identify the active object. The best
recall so far is useful, but still far from the requested 0.80 IoU target.
The next serious direction should use stronger visual features for each SAM
proposal, such as masked DINO/CLIP embeddings or a learned crop classifier,
rather than only geometry/contact/heatmap statistics.
```

## HITL Iteration Toward 0.70 IoU

Goal for this round:

```text
target: positive-frame macro IoU >= 0.70 on HITL eval
constraints: keep the frozen hand segmentor unmodified; use HITL masks only for
training/scoring, not as inference inputs
```

The high-level finding did not change: SAM automatic proposals contain the
right masks, but selecting the right proposals remains the bottleneck.

### SAM Automatic Proposal Ceiling

Full HITL eval greedy proposal oracle over SAM ViT-H automatic masks:

```text
frames          75
positive frames 68

positive-frame macro IoU       0.923
positive-frame macro precision 0.946
positive-frame macro recall    0.955
```

Interpretation:

```text
The 0.70 target is feasible with this proposal pool. Failure is selection, not
mask generation.
```

### TCMonoDepth Proposal Features

Added TCMonoDepth statistics to the SAM proposal selector:

```text
features: proposal depth mean/median/std, proposal-vs-hand depth difference,
          fraction of proposal near hand depth bands
smoke:    50 HITL train frames, 20 HITL eval frames
```

Best high-recall smoke behavior:

```text
positive-frame macro IoU       0.454-0.467
positive-frame macro precision 0.531-0.544
positive-frame macro recall    0.758
```

Interpretation:

```text
Depth did not improve over the non-depth selector. The monocular depth prior is
not discriminative enough for active object selection in these sparse frames.
```

### Oracle-Target SAM Proposal Selector

Changed the training target from per-proposal purity to whether a proposal was
selected by a greedy frame-level oracle on HITL train.

Full HITL eval result:

```text
best balanced setting:
threshold 0.50, topk 2

positive-frame macro IoU       0.569
positive-frame macro precision 0.659
positive-frame macro recall    0.692

highest-recall setting:
threshold 0.15, topk 8

positive-frame macro IoU       0.403
positive-frame macro precision 0.428
positive-frame macro recall    0.885
```

Interpretation:

```text
Oracle-target labels help recall, but top-k union still overselects nearby
objects and surfaces.
```

### Classical Tabular Selectors

Tried stronger tabular models over geometry, SAM scores, hand-contact, and
active-heatmap features:

```text
models: HistGradientBoostingClassifier, RandomForestClassifier,
        ExtraTreesClassifier
targets: greedy-oracle proposal label, proposal precision label
```

Best full HITL eval behavior:

```text
ExtraTrees oracle target, topk 2:
positive-frame macro IoU       0.554
positive-frame macro precision 0.644
positive-frame macro recall    0.650

RandomForest oracle target, topk 3:
positive-frame macro IoU       0.552
positive-frame macro precision 0.642
positive-frame macro recall    0.735
```

Interpretation:

```text
The weak point is not just the tiny MLP optimizer. The hand/contact/heatmap
feature set alone does not identify active-object proposals reliably enough.
```

### DINOv2 Proposal Embeddings

Added masked proposal-crop DINOv2 embeddings to the SAM proposal selector.

Best 50-train/20-eval smoke:

```text
encoder: DINOv2-small
classifier: HistGradientBoosting
threshold 0.10, topk 8

positive-frame macro IoU       0.614
positive-frame macro precision 0.651
positive-frame macro recall    0.829
```

Full HITL train/eval result:

```text
encoder: DINOv2-small
classifier: HistGradientBoosting
threshold 0.05, topk 2

positive-frame macro IoU       0.605
positive-frame macro precision 0.715
positive-frame macro recall    0.705
```

Other DINO variants:

```text
DINOv2-base smoke, best IoU                 0.608
context-preserving DINOv2-small smoke       0.568
DINOv2-small regression target, full eval   0.594
DINOv2-small PyTorch MLP, full eval         0.609
```

Interpretation:

```text
DINO proposal embeddings are the best non-oracle direction so far, but they
still plateau around 0.60-0.61 eval IoU. Bigger DINO features and a deeper MLP
did not close the gap.
```

### SegFormer Dense HITL Segmentor

Tried a direct SegFormer-B0 dense segmentor on HITL masks using the frozen hand
prior rendered into the RGB image.

```text
model: nvidia/segformer-b0-finetuned-ade-512-512
input: 512x912 RGB with hand-prior overlay
train: 300 HITL train frames
eval:  75 HITL eval frames
```

Best eval result:

```text
threshold 0.25
positive-frame macro IoU       0.354
positive-frame macro precision 0.492
positive-frame macro recall    0.642
```

Interpretation:

```text
Dense semantic segmentation still learns rough contact regions rather than
object boundaries. This branch was removed from active code after recording the
result.
```

### Heatmap-Guided SAM Prompts

Tried using the dense HITL DeepLab heatmap to drive SAM point and box prompts.

Point-prompt smoke on 20 eval frames:

```text
best focused setting:
radius 96, points 8, topk 5

positive-frame macro IoU       0.373
positive-frame macro precision 0.662
positive-frame macro recall    0.435
```

Box-prompt smoke on 20 eval frames:

```text
best focused setting:
heat threshold 0.40, near radius 96, topk 3

positive-frame macro IoU       0.246
positive-frame macro precision 0.284
positive-frame macro recall    0.425
```

Interpretation:

```text
The dense heatmap is not accurate enough to place reliable SAM prompts. It can
produce excellent masks on some frames and complete misses on others.
```

### Heatmap-Explaining SAM Proposal Rule

Tried selecting SAM automatic proposals that best explain thresholded dense
heatmaps, with hand/contact/area penalties.

Train-selected best settings reached:

```text
HITL train positive-frame macro IoU 0.677
HITL train precision                0.767
HITL train recall                   0.818
```

But transferred poorly:

```text
HITL eval positive-frame macro IoU  0.503
HITL eval precision                 0.611
HITL eval recall                    0.685
```

Interpretation:

```text
The rule overfits train videos and does not generalize to eval. The heatmap is
still useful as a weak prior but not as a standalone selector.
```

## Current Status

Best non-oracle HITL eval result:

```text
DINOv2-small masked proposal embeddings + SAM automatic proposals
positive-frame macro IoU       0.605
positive-frame macro precision 0.715
positive-frame macro recall    0.705
```

Best high-recall non-oracle setting:

```text
SAM automatic selector, oracle-target training, topk 8
positive-frame macro IoU       0.403
positive-frame macro precision 0.428
positive-frame macro recall    0.885
```

Conclusion:

```text
The current frame-wise proposal-selection formulation has not reached 0.70 IoU.
The most promising next change is not another per-frame selector; it is temporal
object tubes on continuous clips. The HITL benchmark is sparse, so optical flow
cannot be fairly measured there yet. For continuous videos, we should propagate
high-recall SAM proposals through time, select objects that persist near the
hand, and decay released objects rather than making independent frame decisions.
```

### 2026-05-31 Train+Eval Selector Check

To test whether the DINO/SAM selector was primarily data-limited, the best
proposal-selector family was trained on HITL `train+eval` and evaluated on the
held-out HITL `test` split.

Method:

```text
proposal source: SAM ViT-H automatic masks
features: geometry + SAM scores + frozen hand prior + active heatmap + masked DINOv2-small proposal embedding
train split: HITL train + HITL eval
test split: HITL test
```

Best held-out HITL test result:

```text
ExtraTrees selector
threshold 0.15, topk 3

positive-frame macro IoU       0.599
positive-frame macro precision 0.712
positive-frame macro recall    0.776
```

Best high-recall held-out setting:

```text
ExtraTrees selector
threshold 0.15, topk 5

positive-frame macro IoU       0.592
positive-frame macro precision 0.651
positive-frame macro recall    0.836
```

Interpretation:

```text
Adding the eval labels to training did not reach 0.70 on held-out test. The
frame-wise DINO/SAM proposal selector appears to plateau around 0.60 IoU.
The remaining gap is likely temporal/interaction reasoning or additional HITL
labels for hard negatives, not simply a small train-set issue.
```
