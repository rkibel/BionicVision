# Scheme 3 Outputs

This output tree is intentionally small. It keeps only the original dense
baseline, the current best v7 dense checkpoint, and the hand-prior checkpoint
needed to run either dense model.

## Retained Files

```
outputs/experiments/scheme3/
  checkpoints/
    dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt
    dense_union_unetpp_b4_raw_ring_outer_distance_finetune_summary.json
    dense_union_unetpp_b4_raw_ring_outer_distance_egohos_ego4dweight_v7.pt
    dense_union_unetpp_b4_raw_ring_outer_distance_egohos_ego4dweight_v7_summary.json
    dense_union_motion_metric_egohos_ego4dweight_v7.json
  hand_segmentor/
    best.pt
```

Current size after cleanup:

```
outputs/experiments/scheme3: 241M
```

## Why These Stay

`dense_union_unetpp_b4_raw_ring_outer_distance_finetune.pt` is the original
Ego-Exo dense baseline. Its summary records the pre-EgoHOS training behavior.

`dense_union_unetpp_b4_raw_ring_outer_distance_egohos_ego4dweight_v7.pt` is the
retained best broad-object checkpoint. It is the default checkpoint in the
training, evaluation, and rendering scripts.

`dense_union_motion_metric_egohos_ego4dweight_v7.json` is the retained
postprocessed evaluation report for v7, including supervised IoU and
flow-aligned temporal IoU.

`hand_segmentor/best.pt` is the dense-model hand prior. It is required for
training, evaluation, and rendering.

## Reference Metrics

Original baseline summary:

```
Ego-Exo val IoU: 0.9500
held-out target IoU: 0.8688
held-out target temporal IoU: 0.4731
threshold: 0.54
```

v7 training summary:

```
Ego-Exo val IoU: 0.9369
EgoHOS combined val IoU: 0.6325
EgoHOS source-min val IoU: 0.5511
held-out target IoU: 0.8663
threshold: 0.18
```

v7 retained postprocessed metric report:

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
