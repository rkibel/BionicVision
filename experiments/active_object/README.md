# Active Object Experiment

This folder contains the human-in-the-loop active-object annotation tooling and
the current benchmark experiments against the HITL masks. Older dead-end
scripts were removed after their results were documented in `EXPERIMENT_LOG.md`.

## Annotation Goal

For each prompted VISOR frame, annotate the active object: the object being held, manipulated, or otherwise directly interacted with by the visible hand. The derived VISOR-contact proxy was useful for bootstrapping, but these HITL masks are intended to become the cleaner ground truth for later active-object models.

## Files

```text
experiments/active_object/common.py
experiments/active_object/hitl_ground_truth_editor.py
experiments/active_object/train_hitl_segmentor.py
experiments/active_object/train_sam_auto_selector.py
experiments/active_object/README.md
experiments/active_object/EXPERIMENT_LOG.md
```

Prompt frames live in:

```text
data/epic_kitchens/HITL/active_object_prompt_frames.json
```

Submitted annotations live in split folders:

```text
data/epic_kitchens/HITL/active_objects/train/<frame_id>/
data/epic_kitchens/HITL/active_objects/eval/<frame_id>/
data/epic_kitchens/HITL/active_objects/test/<frame_id>/
```

Each submitted frame folder contains:

```text
active_object_mask.png
overlay.png
metadata.json
```

There are intentionally no annotation manifests. The editor decides whether a frame is already submitted by checking whether its `<frame_id>/` folder exists. If a mask is not good enough, delete that frame folder and it will reappear in the editor.

The editor re-checks these folders every time the page loads, so browser refresh
is enough to pick up manual folder deletions or newly added annotations.

## Run The Editor

```bash
.venv-models/bin/python experiments/active_object/hitl_ground_truth_editor.py --server-port 7861
```

Then open:

```text
http://127.0.0.1:7861
```

Workflow:

```text
click positive/negative SAM points -> accept proposal if adding multiple regions -> submit HITL ground truth
```

The visible pending SAM proposal is saved on submit, even if it was not explicitly accepted. Accepted regions are useful when the active object has multiple disconnected pieces.

Keyboard shortcuts:

```text
a  accept the current proposal
s  submit the current ground truth
```

## Prompt Set

The current prompt file contains:

```text
300 train frames
75 eval frames
100 test frames
```

The prompt set is a fixed sampling list, not an annotation manifest.

## Historical Results

The active-object modeling attempts, including VISOR-proposal ceilings,
supervised HITL segmentors, SAM proposal selectors, and quantitative metrics,
are kept in:

```text
experiments/active_object/EXPERIMENT_LOG.md
```

Current retained experiment outputs:

```text
outputs/experiments/active_object/hitl_deeplab_r50_hand_prior/
outputs/experiments/active_object/hitl_cache_384/
outputs/experiments/active_object/sam_auto_selector_full/
```
