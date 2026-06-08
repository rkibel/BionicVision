# Scheme 3 Outputs

This output tree keeps only the current best dense checkpoint, its statistics,
and the hand-prior checkpoint required to run it.

## Retained Files

```
outputs/experiments/scheme3/
  checkpoints/
    best.pt
    best_summary.json
    best_eval.json
  hand_segmentor/
    best.pt
```

## Why These Stay

`best.pt` is the retained broad-object reference checkpoint and the default in
the training, evaluation, and rendering scripts. `best_summary.json` records
its training metrics, and `best_eval.json` records postprocessed supervised and
flow-aligned temporal metrics.

`hand_segmentor/best.pt` is the dense-model hand prior. It is required for
training, evaluation, and rendering.

## Reference Metrics

Best training summary:

```
Ego-Exo val IoU: 0.9369
EgoHOS combined val IoU: 0.6325
EgoHOS source-min val IoU: 0.5511
held-out target IoU: 0.8663
threshold: 0.18
```

Retained postprocessed metric report:

```
Ego-Exo supervised IoU: 0.9395
EgoHOS val IoU: 0.6236
full-FPS flow IoU h=1: 0.9376
full-FPS flow IoU h=5: 0.8396
full-FPS flow IoU h=30: 0.6450
```

## Regenerable Output

Rendering videos, flow caches, temporary diagnostics, train logs, and probe
JSONs are intentionally not retained here. They can be regenerated from the
checkpoint and scripts when needed.
