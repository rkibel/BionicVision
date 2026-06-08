# Hand Segmentor Evaluation

Evaluate the reusable EgoHOS-trained hand segmentor at its checkpoint threshold:

```bash
PYTHONPATH=src .venv-models/bin/python -m evaluation.hand_segmentor.egohos
PYTHONPATH=src .venv-models/bin/python -m evaluation.hand_segmentor.visor
```

The EgoHOS evaluator defaults to `val,test_indomain,test_outdomain`. The VISOR
evaluator defaults to the held-out `test` videos. Both commands support
`--thresholds`, `--max-frames`, `--batch-size`, and `--write-masks`.
