# BionicVision

BionicVision is a research codebase for scene simplification in egocentric
kitchen video. The current `src` codebase contains the reusable pipelines,
datasets, models, and evaluations.

The main progression is:

- `han_fusion_baseline_models`: reproduction-oriented Han et al. fusion baseline.
- `han_fusion_temporal_models`: Han-style fusion with temporal model inputs.
- `scheme1`: temporally propagated DEVA object segmentation.
- `scheme2`: Scheme 1 plus always-retained hands from the custom hand segmentor.
- `scheme3`: learned dense interacting-object segmentation plus retained hands.
- `scheme4`: Scheme 3 with a TCMonoDepth prior.
- `scheme5`: Scheme 3 with a GLC saliency prior.

## Repository Layout

```text
src/
  datasets/          Dataset adapters used by current src evaluation/training.
  evaluation/        Hand-segmentor and dense-scheme evaluation entrypoints.
  models/            Reusable model adapters, training code, and wrappers.
  pipelines/         Runnable video simplification pipelines.

external/
  baselines/         External baseline source checkouts.
  model_sources/     External model source checkouts.
  model_weights/     Retained model weights used by src.
data/                Local datasets and video clips.
outputs/             Generated runs, masks, videos, metrics, and new checkpoints.
```

`src` should not depend on pre-existing files in `outputs`. Retained model
weights live in `external/model_weights`; `outputs` is for newly generated run
artifacts and same-run intermediates.

## Environment

The repo expects Python commands to run with `PYTHONPATH=src`. The local working
environment used here is `.venv-models`.

```bash
python -m venv .venv-models
source .venv-models/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124
pip install --no-build-isolation \
  "detectron2 @ git+https://github.com/facebookresearch/detectron2.git@e0ec4e189d438848521aee7926f9900e114229f5"
pip install --no-build-isolation \
  "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git@856dde20aee659246248e20734ef9ba5214f5e44"
pip install -r requirements.txt
```

Detectron2 and GroundingDINO are installed separately because their build
metadata imports `torch`.

Some external model adapters also require their upstream source checkouts under
`external/model_sources`. The large retained checkpoints are not produced by
`src` at runtime; they should already be placed under `external/model_weights`.

## Data

The current code expects these local dataset roots when running the full
training/evaluation stack:

```text
data/epic_kitchens/
  continuous_segments/
  visor/

data/egohos/
  data/train/
  data/val/
  data/test_indomain/
  data/test_outdomain/

data/egoexo4d/
  annotations/
  takes/
```

The continuous EPIC-KITCHENS clips currently include P22 examples such as:

```text
data/epic_kitchens/continuous_segments/P22_107_continuous_01_frames_0003893_0005392.mp4
```

## Retained Weights

The canonical src defaults point at:

```text
external/model_weights/hand_segmentor.pt
external/model_weights/scheme3.pt
external/model_weights/scheme4.pt
external/model_weights/scheme5.pt
external/model_weights/glc_ego4d.pyth
external/model_weights/ade20k-resnet50dilated-ppm_deepsup/
```

Training commands still write new checkpoints under `outputs/models` unless an
explicit output path is provided.

## Run Pipelines

All pipeline commands below write into `outputs/`.

```bash
source .venv-models/bin/activate
```

Run the Han fusion baseline:

```bash
PYTHONPATH=src python -m pipelines.han_fusion_baseline_models \
  --clip-dir data/epic_kitchens/continuous_segments \
  --output-root outputs/han_fusion_baseline_models \
  --target-fps 10 \
  --device cuda
```

Run the Han fusion temporal-model variant:

```bash
PYTHONPATH=src python -m pipelines.han_fusion_temporal_models \
  --clip-dir data/epic_kitchens/continuous_segments \
  --output-root outputs/han_fusion_temporal_models \
  --target-fps 10 \
  --device cuda
```

Run Scheme 1, DEVA-only temporal object segmentation:

```bash
PYTHONPATH=src python -m pipelines.scheme1 \
  --clip-dir data/epic_kitchens/continuous_segments \
  --output-root outputs/scheme1 \
  --target-fps 10 \
  --device cuda
```

Run Scheme 2, temporal DEVA objects plus always-retained hands:

```bash
PYTHONPATH=src python -m pipelines.scheme2 \
  --clip-dir data/epic_kitchens/continuous_segments \
  --output-root outputs/scheme2 \
  --target-fps 10 \
  --device cuda
```

Run Scheme 3, dense interacting-object segmentation plus retained hands:

```bash
PYTHONPATH=src python -m pipelines.scheme3 \
  --clip-dir data/epic_kitchens/continuous_segments \
  --output-root outputs/scheme3 \
  --target-fps 10 \
  --device cuda
```

Run Scheme 4, dense segmentation with TCMonoDepth prior:

```bash
PYTHONPATH=src python -m pipelines.scheme4 \
  --clip-dir data/epic_kitchens/continuous_segments \
  --output-root outputs/scheme4 \
  --target-fps 10 \
  --device cuda
```

Run Scheme 5, dense segmentation with GLC saliency prior:

```bash
PYTHONPATH=src python -m pipelines.scheme5 \
  --clip-dir data/epic_kitchens/continuous_segments \
  --output-root outputs/scheme5 \
  --target-fps 10 \
  --device cuda
```

For quick local smoke runs, add `--max-frames N` to any scheme command.

## Evaluation

Evaluate the reusable EgoHOS-trained hand segmentor:

```bash
PYTHONPATH=src python -m evaluation.hand_segmentor.egohos \
  --output outputs/evaluation/hand_segmentor/egohos.json \
  --device cuda

PYTHONPATH=src python -m evaluation.hand_segmentor.visor \
  --output outputs/evaluation/hand_segmentor/visor.json \
  --device cuda
```

Evaluate dense schemes on supervised EgoHOS/Ego-Exo masks plus unsupervised
flow-corrected temporal stability over multiple horizons:

```bash
PYTHONPATH=src python -m evaluation.scheme3_dense.run_evaluation \
  --output outputs/evaluation/scheme3_dense/results.json \
  --device cuda

PYTHONPATH=src python -m evaluation.scheme4_dense.run_evaluation \
  --output outputs/evaluation/scheme4_dense/results.json \
  --device cuda

PYTHONPATH=src python -m evaluation.scheme5_dense.run_evaluation \
  --output outputs/evaluation/scheme5_dense/results.json \
  --device cuda
```

Useful dense-evaluation flags:

```bash
--egoexo-samples 180
--egohos-samples 240
--egohos-splits val,test_indomain,test_outdomain
--flow-horizons 1,2,5,10,15,30
--max-video-frames 900
--flow-size 256
```

## Training

Train the hand segmentor on EgoHOS:

```bash
PYTHONPATH=src python -m models.segmentation.hand_segmentor.train \
  --output-dir outputs/models/hand_segmentor \
  --device cuda
```

Train Scheme 3 from the reusable hand segmentor:

```bash
PYTHONPATH=src python -m models.segmentation.scheme3_dense.train \
  --output outputs/models/scheme3_dense/best.pt \
  --device cuda
```

Train Scheme 4 or Scheme 5 starting from the retained Scheme 3 checkpoint:

```bash
PYTHONPATH=src python -m models.segmentation.scheme4_dense.train --device cuda
PYTHONPATH=src python -m models.segmentation.scheme5_dense.train --device cuda
```

For integration checks, the hand segmentor and Scheme 3 trainers support
`--dev-run`.

## Development Notes

- Keep retained model weights in `external/model_weights`.
- Treat `outputs` as disposable generated state.
- Prefer adding scheme-independent metrics to `src/evaluation/dense_schemes`.
- Prefer adding reusable segmentation components under `src/models/segmentation`.

## Verification

Basic import and path verification:

```bash
PYTHONPATH=src python - <<'PY'
from models.segmentation.hand_segmentor.adapter import DEFAULT_CHECKPOINT as hand
from models.segmentation.scheme3_dense.adapter import DEFAULT_CHECKPOINT as scheme3
from models.segmentation.scheme4_dense.adapter import DEFAULT_CHECKPOINT as scheme4
from models.segmentation.scheme5_dense.adapter import DEFAULT_CHECKPOINT as scheme5
from models.saliency.glc.adapter import DEFAULT_CHECKPOINT as glc, assert_glc_available

for path in [hand, scheme3, scheme4, scheme5, glc]:
    print(path, path.exists())

assert all(path.exists() for path in [hand, scheme3, scheme4, scheme5, glc])
assert_glc_available()
PY
```

Compile the current src modules:

```bash
PYTHONPATH=src python -m compileall -q src
```
