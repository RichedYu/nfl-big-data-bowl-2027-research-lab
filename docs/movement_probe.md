# G2.1 movement measurement probe

The private Cloud run uses the verified G1 DuckDB and fingerprints all nine raw CSVs against G1. Exploratory movement measurements use the 2023 draft cohort only. The two WR/TE drills are speed out left and curl right. The two DB drills are backpedal then 45° transition and 90° break to ball. The DB tasks probe geometry across different instructions. Their timing and movement demands differ from receiver routes. For other cohorts, the code returns availability counts only.

The default position filter is a centered three-frame average. Compute velocity and acceleration from timestamps using centered differences with one-sided edge formulas. The signed tangent and normal components are dot(v,a)/|v| and cross(v,a)/|v|. Signed curvature is cross(v,a)/|v|³. Mask these ratios below 1 yd/s. Reject an event with an interval differing from 100 ms by more than 1 ms. Detect the largest wrapped heading change across a roughly ±0.3 s window, keeping at least 0.6 s of support on either side and requiring 30° of change. At 5 Hz, use the nearest integer window and report both phases. Turn center is an algorithmic landmark.

Entry and exit are mean speeds in the adjoining 0.6 s windows. Braking amplitude is entry speed minus the minimum speed in the turn search window. Recovery is the time from that minimum to the first sample at ≥90% of entry speed. Recovery is missing when the event ends before the threshold is observed. A speed dip around a sharp corner can arise from smoothing and numerical differentiation. Record it as a trajectory statistic.

Sensitivity configurations change smoothing to one or five samples, sample both phases at 5 Hz, and add independent zero-mean Gaussian coordinate noise at 0.03 and 0.10 yd SD with five draws per trial. These are simulation parameters. Compare each resulting measurement to the same trial's baseline after re-detecting its cut. Report missing pairs and cut shifts along with median absolute change and the 90th percentile. Normalize median change by the task median absolute feature value. Report recovery errors in seconds alongside relative differences. The exercise does not estimate instrument error.

Within/between variance uses the first two usable trials ordered by official attempt index and event ID for athletes with repeated trials. Within variance is half the mean squared difference and the between estimate subtracts half the within component from the variance of pair means. Negative estimates are retained. Collapse trials by athlete before comparing empirical entry-speed strata and computing Spearman correlations with Combine test times. Strata are descriptive and may reflect how features share the same segmentation. No outcome model or significance screening enters the probe.

For game transfer, use regular-season games in draft year + 1 and deduplicate tracking by player-play before linking. Report availability for trajectory, 40-yard time, tracking joined to play metadata, and at least one role-specific eligible row. WR/TE uses route and separation at throw on the same row, DB uses coverage assignment at snap, OL uses pressure allowed availability and DL uses get-off availability. These are minimum field-availability rules. A final endpoint still needs opportunity thresholds and a missingness rule.

## Reproduce in the authorized Cloud task

Install NumPy and Matplotlib for this probe. The private evidence records exact versions. Run the synthetic tests before the measurement script.

```bash
.venv/bin/python -m pip install numpy==2.5.5 matplotlib==3.11.2
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/movement_probe.py --g1 data/audit_runs/g1-cloud-final-20261126T0942Z --output data/g2/your-run-id
MPLCONFIGDIR="$PWD/data/g2/mpl" .venv/bin/python scripts/summarize_movement_probe.py data/g2/your-run-id
```

Use a fresh run directory for changed code or inputs. Trials and figures remain under the ignored data directory. The official field definitions are available on the [Kaggle competition Data page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data).
