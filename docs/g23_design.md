# G2.3 measurement, causal design and development feasibility

The restricted Cloud study preserves G1, G2.1 and G2.2. It compares three trajectory estimators at common interior samples, assesses synthetic interval overlap and angle error on independent calibration and evaluation runs, and presents blinded path examples for independent review. A marker in a point trajectory is a path annotation. Biological movement and joint loading require separate measurements.

The reliability exercise uses a fixed ADEMP grid with repeat attempts, order effects, heteroscedastic error and informative rejection. The causal exercise enumerates known binary generators and separates interventions, information and associations. The NFL diagram is a proposed set of assumptions for review.

The real pilot is restricted to the 2023 development cohort. The frozen private protocol defines the rookie opportunity threshold and the conditional mean of observed separation, plus one global Combine speed summary. The pilot uses fixed ridge penalties and player-level splits. Data work for later cohorts is limited to structural availability. Approximate route classes remain candidates pending independent football review.

## Reproduction

Use the authorized Cloud workspace with the G1, G2.1 and G2.2 checkpoints. Install the optional scientific dependencies documented for G2.1 with SciPy 1.18.1. Use a private run directory with the frozen protocol and pilot protocol. Set a writable `MPLCONFIGDIR` in the task workspace.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/g23_run.py "$run" motion reliability causal football
.venv/bin/python scripts/g23_supplement.py "$run"
```

The runner verifies source digests at initialization and verifies recorded artifacts when resuming. The run stores detailed evidence, timings, software and hashes in the ignored data directory. Keep the blinded packet separate from the private identifier mapping and model outputs.
