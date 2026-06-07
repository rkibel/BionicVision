# BionicVision

This repository reproduces the Han et al. 2021 scene-simplification baseline on
EPIC-KITCHENS clips, evaluates those outputs against VISOR annotations, and adds
`han_fusion_temporal_models`: the same Han fusion scheme with temporal model inputs
(DeepGaze III, TCMonoDepth, and fixed-prompt manual DEVA).
It also includes `scheme1`, a DEVA-only indoor kitchen segmentation pipeline
with temporal propagation and no hand prompts.

Baseline reproduction is mandatory for this project. Run and evaluate the Han
baseline before comparing `han_fusion_temporal_models`; the comparison is only meaningful when
both outputs are produced from the same clips, frame rate, environment, and
evaluation code.

The full setup, data manifest, model weight locations, and run commands are in
[docs/SETUP.md](docs/SETUP.md).

Quick command map:

```bash
source .venv-models/bin/activate

# Mandatory baseline reproduction.
PYTHONPATH=src python -m pipelines.han_baseline \
  --clip-dir data/epic_kitchens/video_snippets/test_set/inputs \
  --output-root outputs/han_baseline_test_set \
  --target-fps 10 \
  --device cuda

PYTHONPATH=src python -m evaluation.pipeline_outputs \
  --data-root data/epic_kitchens \
  --output-root outputs/han_baseline_test_set \
  --results-dir outputs/evaluation/han_baseline_test_set

# Han fusion with temporal models after the baseline exists.
PYTHONPATH=src python -m pipelines.han_fusion_temporal_models \
  --clip-dir data/epic_kitchens/video_snippets/test_set/inputs \
  --output-root outputs/han_fusion_temporal_models_test_set \
  --target-fps 10 \
  --device cuda

PYTHONPATH=src python -m evaluation.pipeline_outputs \
  --data-root data/epic_kitchens \
  --output-root outputs/han_fusion_temporal_models_test_set \
  --results-dir outputs/evaluation/han_fusion_temporal_models_test_set

# Scheme 1: DEVA-only indoor kitchen segmentation.
PYTHONPATH=src python -m pipelines.scheme1 \
  --clip-dir data/epic_kitchens/video_snippets/test_set/inputs \
  --output-root outputs/scheme1_test_set \
  --target-fps 10 \
  --device cuda
```
