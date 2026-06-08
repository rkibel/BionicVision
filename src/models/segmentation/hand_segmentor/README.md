# Hand Segmentor

Train the reusable EgoHOS hand segmentor:

```bash
PYTHONPATH=src .venv-models/bin/python -m models.segmentation.hand_segmentor.train
```

New checkpoints are written to `outputs/models/hand_segmentor` by default and
remain compatible with `HandSegmentor`. Use `--dev-run` for a small integration
test, or `--init-checkpoint` to continue from an existing compatible checkpoint.
