# VISOR Hand Segmentor

This experiment trains a binary hand/glove/visible-arm segmentor for
EPIC-KITCHENS VISOR frames. The final model is a `DeepLabV3-ResNet50` with one
output mask channel.

For the full iteration history, including earlier DEVA/Grounded-SAM2 attempts
and retained quantitative results, see:

```text
experiments/hand_segmentor/EXPERIMENT_LOG.md
```

Generated outputs live under:

```text
outputs/experiments/hand_segmentor/
```

## Splits

The split names are intentionally boring:

```text
train = model weight updates
val   = checkpoint and threshold selection
test  = clean audit only
```

All splits are whole-video disjoint.

Current split sizes:

```text
train: 1,845 hand/glove frames
val:     729 hand/glove frames
test:  1,171 hand/glove frames
```

The clean test videos are:

```text
P03_120, P06_108, P08_17, P22_107
```

Older experimental split labels have been removed from the code. The final
audited split is now simply `test`.

## Download Data

```bash
PYTHONPATH=src:experiments/hand_segmentor .venv-models/bin/python \
  experiments/hand_segmentor/download_visor_subset.py \
  --splits train,val,test
```

The downloader skips extracted videos that already exist and removes source zips
after extraction.

## Train

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

Best validation checkpoint:

```text
val IoU=0.9175  precision=0.9515  recall=0.9612  threshold=0.55
```

## Evaluate

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

Clean test result:

```text
IoU=0.9074  precision=0.9553  recall=0.9480  threshold=0.55  frames=1171
```

Per-video:

```text
P03_120  IoU=0.9094  P=0.9538  R=0.9523  frames=454
P06_108  IoU=0.9466  P=0.9823  R=0.9631  frames=218
P08_17   IoU=0.9293  P=0.9610  R=0.9663  frames=175
P22_107  IoU=0.8664  P=0.9362  R=0.9221  frames=324
```

Now that `test` has been audited, treat it as spent for future research claims.
If the model changes, reserve a new untouched test split.

## Visualize Test Overlays

```bash
PYTHONPATH=src:experiments/hand_segmentor .venv-models/bin/python \
  experiments/hand_segmentor/make_overlays.py \
  --split test \
  --samples 24 \
  --tta-flip
```

Overlay colors:

```text
green = true positive
red   = false positive
blue  = false negative
```

## Run On A Video

The generated `test_set/inputs` clips are sparse VISOR annotated frames. For
temporal inspection, cut three 30-second continuous snippets per test video
from the public EPIC videos:

```bash
PYTHONPATH=src:experiments/hand_segmentor .venv-models/bin/python \
  experiments/hand_segmentor/make_continuous_segments.py
```

The snippet manifest records target source-frame ranges and all VISOR annotated
frames that fall inside each snippet:

```text
data/epic_kitchens/video_snippets/test_set/continuous_segments/manifest.json
```

Run the segmentor on one continuous segment:

```bash
PYTHONPATH=src:experiments/hand_segmentor .venv-models/bin/python \
  experiments/hand_segmentor/run_video.py \
  --input data/epic_kitchens/video_snippets/test_set/continuous_segments/P03_120_continuous_01_frames_0007447_0008946.mp4 \
  --output-dir outputs/experiments/hand_segmentor/video_runs \
  --batch-size 6 \
  --tta-flip
```

The video overlay uses green for the predicted hand/glove/visible-arm mask.
`P08_17` has no public MP4 endpoint, so the segment cutter streams the public
RGB-frame tar once, extracts only the selected snippet frames, encodes the three
clips, and deletes the temporary JPEG frames.

## Kept Artifacts

The kept model artifacts are:

```text
outputs/experiments/hand_segmentor/deeplab_r50_512/
```

Intermediate caches are generated artifacts and can be deleted after summaries,
overlays, and videos are produced.
