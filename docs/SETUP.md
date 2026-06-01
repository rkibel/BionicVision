# Setup And Reproduction

This document is the reproducibility contract for the current EPIC-KITCHENS
experiments. A different environment is ready only when it can reproduce the Han
baseline, run `combination1`, and evaluate both with the same code.

## Hardware And System Packages

Use a Linux machine with an NVIDIA GPU. The validated environment used CUDA
12.4 wheels, PyTorch `2.6.0+cu124`, and torchvision `0.21.0+cu124`.

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ffmpeg \
  git \
  libgl1 \
  libglib2.0-0 \
  python3.10 \
  python3.10-dev \
  python3.10-venv \
  unzip \
  wget
```

## Python Environment

Create the project venv at the repo root. The commands expect `.venv-models`.

```bash
python3.10 -m venv .venv-models
source .venv-models/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0+cu124 \
  torchvision==0.21.0+cu124
pip install -r requirements.txt
```

If Detectron2 fails to build, confirm that the active Python is from
`.venv-models`, PyTorch imports successfully, and `python3.10-dev` plus
`build-essential` are installed.

## External Source Trees

The runtime imports and subprocess calls expect these external repositories under
`external/`. Clone the exact revisions below:

```bash
mkdir -p external/model_sources/depth
git clone https://github.com/nianticlabs/monodepth2 \
  external/model_sources/depth/monodepth2
git -C external/model_sources/depth/monodepth2 checkout b676244

git clone https://github.com/yu-li/TCMonoDepth \
  external/model_sources/depth/TCMonoDepth
git -C external/model_sources/depth/TCMonoDepth checkout 128a98c

mkdir -p external/model_sources/saliency
git clone https://github.com/matthias-k/DeepGaze \
  external/model_sources/saliency/DeepGaze
git -C external/model_sources/saliency/DeepGaze checkout c87b106

mkdir -p external/model_sources/segmentation
git clone https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git \
  external/model_sources/segmentation/Tracking-Anything-with-DEVA
git -C external/model_sources/segmentation/Tracking-Anything-with-DEVA checkout 404a112

git clone https://github.com/CSAILVision/semantic-segmentation-pytorch \
  external/model_sources/segmentation/semantic-segmentation-pytorch
git -C external/model_sources/segmentation/semantic-segmentation-pytorch checkout 8f27c9b

mkdir -p external/baselines
git clone https://github.com/bionicvisionlab/2021-han-scene-simplification.git \
  external/baselines/2021-han-scene-simplification
git -C external/baselines/2021-han-scene-simplification checkout 59264e2
```

The Han baseline reproduction in this repo does not shell out to
`external/baselines/2021-han-scene-simplification`; it is kept as the reference
source for the reproduced algorithm.

## Model Weights

The following weights are required.

### Han Baseline

Monodepth2, default `mono+stereo_640x192`:

```bash
mkdir -p external/model_sources/depth/monodepth2/models
wget -O /tmp/mono_stereo_640x192.zip \
  "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip"
unzip -q /tmp/mono_stereo_640x192.zip \
  -d external/model_sources/depth/monodepth2/models
```

MIT scene parsing weights:

```bash
mkdir -p external/model_weights/mit_scene_parsing/ade20k-resnet50dilated-ppm_deepsup
wget -O external/model_weights/mit_scene_parsing/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth \
  http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
wget -O external/model_weights/mit_scene_parsing/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth \
  http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth
```

Detectron2 downloads its Mask R-CNN checkpoint through `model_zoo` on first use.

### Combination1

TCMonoDepth small checkpoint:

```bash
mkdir -p external/model_sources/depth/TCMonoDepth/weights
gdown --fuzzy \
  "https://drive.google.com/file/d/1MGefuek7_fW_9vu5bP6l0IIF72tg-n-M/view" \
  -O external/model_sources/depth/TCMonoDepth/weights/_ckpt_small.pt.tar
```

DEVA propagation checkpoint:

```bash
mkdir -p external/model_sources/segmentation/Tracking-Anything-with-DEVA/saves
wget -O external/model_sources/segmentation/Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth \
  https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
```

`combination1` uses fixed-prompt manual DEVA with GroundingDINO + SAM. It does
not use automatic SAM grid prompting and does not use VISOR masks at inference
time.

```bash
CUDA_VISIBLE_DEVICES="" python -m pip install --no-build-isolation \
  "groundingdino @ git+https://github.com/IDEA-Research/GroundingDINO.git"

wget -O external/model_sources/segmentation/Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py \
  https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/GroundingDINO_SwinT_OGC.py
wget -O external/model_sources/segmentation/Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -O external/model_sources/segmentation/Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Required Data

`data/` is gitignored. For the current benchmark subset, keep VISOR dense
annotations and sparse RGB frames for the held-out test videos, plus generated
test-set snippets and ground-truth videos:

```text
data/epic_kitchens/video_snippets/test_set/manifest.json
data/epic_kitchens/video_snippets/test_set/inputs/P03_120_test_frames.mp4
data/epic_kitchens/video_snippets/test_set/inputs/P06_108_test_frames.mp4
data/epic_kitchens/video_snippets/test_set/inputs/P08_17_test_frames.mp4
data/epic_kitchens/video_snippets/test_set/inputs/P22_107_test_frames.mp4
data/epic_kitchens/video_snippets/test_set/ground_truth/*_test_hand_gt_masks.mp4
data/epic_kitchens/video_snippets/test_set/ground_truth/*_test_hand_gt_overlay.mp4
data/epic_kitchens/video_snippets/test_set/ground_truth/*_test_all_gt_masks.mp4
data/epic_kitchens/video_snippets/test_set/ground_truth/*_test_all_gt_overlay.mp4
data/epic_kitchens/video_snippets/test_set/labels/*_test_labels.json
data/epic_kitchens/visor/dense_annotations/{P03_120,P06_108,P08_17,P22_107}/
data/epic_kitchens/visor/sparse_rgb_frames/{P03_120,P06_108,P08_17,P22_107}/
```

The fastest exact setup is to copy the known-good asset bundle from an existing
machine:

```bash
# On the known-good machine, from the repo root:
tar -czf bionicvision-epic-subset-assets.tgz \
  data/epic_kitchens/video_snippets/test_set \
  data/epic_kitchens/visor/dense_annotations \
  data/epic_kitchens/visor/sparse_rgb_frames \
  external/model_weights \
  external/model_sources/depth/monodepth2/models/mono+stereo_640x192 \
  external/model_sources/depth/TCMonoDepth/weights/_ckpt_small.pt.tar \
  external/model_sources/segmentation/Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth

# On the new machine, from the repo root:
tar -xzf bionicvision-epic-subset-assets.tgz
```

To rebuild the VISOR data and generated test videos directly:

```bash
PYTHONPATH=src:experiments/hand_segmentor python \
  experiments/hand_segmentor/download_visor_subset.py \
  --splits test

PYTHONPATH=src:experiments/hand_segmentor python \
  experiments/hand_segmentor/make_test_videos.py \
  --split test
```

## Sanity Checks

Run unit tests first:

```bash
PYTHONPATH=src python -m unittest discover tests
```

Then run a small smoke test for each pipeline.

Mandatory baseline smoke:

```bash
PYTHONPATH=src python -m pipelines.han_baseline \
  --clip-dir data/epic_kitchens/video_snippets/test_set/inputs \
  --output-root outputs/han_baseline_smoke \
  --target-fps 10 \
  --max-frames 3 \
  --device cuda
```

Combination1 smoke:

```bash
PYTHONPATH=src python -m pipelines.combination1 \
  --clip-dir data/epic_kitchens/video_snippets/test_set/inputs \
  --output-root outputs/combination1_smoke \
  --target-fps 10 \
  --max-frames 3 \
  --device cuda \
  --deva-memory-reset-interval 1
```

Inspect:

```text
outputs/combination1_smoke/<clip>/deva_raw/Annotations/frame00000.png
outputs/combination1_smoke/<clip>/deva_raw/Visualizations/frame00000.png
```

Manual DEVA writes only three semantic colors in those images:

```text
arms:    red   RGB(255, 0, 0)
objects: green RGB(0, 255, 0)
scenes:  blue  RGB(0, 0, 255)
```

There should be no `deva_masks/` output and no `deva_raw/Grouped/` output.

## Full Reproduction Commands

Run the baseline first. This is mandatory.

```bash
PYTHONPATH=src python -m pipelines.han_baseline \
  --clip-dir data/epic_kitchens/video_snippets/test_set/inputs \
  --output-root outputs/han_baseline_test_set \
  --target-fps 10 \
  --device cuda

PYTHONPATH=src python -m evaluation.pipeline_outputs \
  --data-root data/epic_kitchens \
  --output-root outputs/han_baseline_test_set \
  --results-dir outputs/evaluation/han_baseline_test_set
```

Then run `combination1`:

```bash
PYTHONPATH=src python -m pipelines.combination1 \
  --clip-dir data/epic_kitchens/video_snippets/test_set/inputs \
  --output-root outputs/combination1_test_set \
  --target-fps 10 \
  --device cuda \
  --deva-detection-every 1 \
  --deva-memory-reset-interval 4 \
  --deva-size 360

PYTHONPATH=src python -m evaluation.pipeline_outputs \
  --data-root data/epic_kitchens \
  --output-root outputs/combination1_test_set \
  --results-dir outputs/evaluation/combination1_test_set
```

`evaluation.pipeline_outputs` is the shared evaluator for any pipeline that
produces the same output directory contract: `frames/`,
`combination_frames/`, and videos.

## Expected Output Layout

Baseline:

```text
outputs/han_baseline_test_set/<clip>/
  frames/
  saliency_frames/
  segmentation_frames/
  depth/frames/
  combination_frames/
  videos/
```

Combination1:

```text
outputs/combination1_test_set/<clip>/
  frames/
  saliency_frames/
  segmentation_frames/
  depth/frames/
  deva_raw/Annotations/
  deva_raw/Visualizations/
  combination_frames/
  videos/
```

Evaluations:

```text
outputs/evaluation/<run_name>/summary.json
outputs/evaluation/<run_name>/frames.csv
```

## Metrics Notes

The evaluator reports gold sparse VISOR metrics and pseudo dense VISOR metrics.
Gold recall is the primary exact-annotation check, but it can saturate on this
small subset. Pseudo dense metrics are useful for coverage and dropout trends,
but they inherit interpolation noise. Brightness quality is intentionally not
scored yet. Pipeline outputs are evaluated as unpercepted video frames; they are
not passed through a phosphene/perceptual model.

For the current validated run, both baseline and `combination1` preserve gold
foreground/background recall. The useful comparison is therefore pseudo dense
coverage plus output load, active area, and flow-compensated flicker.
