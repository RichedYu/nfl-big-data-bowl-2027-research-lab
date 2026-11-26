# Research Roadmap

## G1 — Cloud data audit (complete; review pending)

- Prepare the selected private Codex Cloud environment and install the repository dependencies and official Kaggle CLI.
- Retrieve and strictly parse all nine official CSV files. Keep raw files, per-file fingerprints, full audit output, and detailed logs in the private ignored workspace.
- Profile frame-level, play-level, player-level, and draft-cohort data; verify candidate keys and cross-table coverage.
- Assess A (WR/TE), B (DB), and C (OL/DL) from observed sample sizes, repeat attempts, Combine measurements, future-season coverage, and outcome definitions.
- Save resumable download and audit checkpoints. Do not fit a model or select a direction from held-out outcomes.
- Current evidence and limitations are in the private task workspace. The public dashboard has not been updated; its status remains pending review.

## G2 — Scientific design review and preregistration

- Select or revise the candidate position group based on data coverage and football-science review, without looking at a held-out test result.
- Define the prediction timestamp, eligible draft cohorts, minimum future-outcome horizon, censoring rules, and the unit of analysis (player or player-season).
- Freeze outcome definitions and the pre-observation covariates available at that timestamp. Decide how incomplete traditional Combine measurements will be handled.
- Specify the train/validation/test split by draft cohort and the uncertainty procedure before fitting any model.

## G3 — Trajectory feature specification

- Define raw trajectory T and a pre-registered feature map X using only Combine tracking (for example braking, turning, and re-acceleration summaries for routes).
- Define baseline B from traditional Combine tests and anthropometrics; define Z from covariates available at the declared prediction time.
- Document frame filtering, missingness, event segmentation, attempt aggregation, and sensitivity checks. Keep the player as the independent sampling unit.

## G4 — Estimation plan for additional information

- Evaluate the target I(X;Y | B,Z) only after G2 and G3 are approved.
- Compare a prespecified baseline using (B,Z) with an augmented model using (X,B,Z), using grouped validation and uncertainty intervals.
- Treat finite-sample estimates as model- and estimator-dependent. Check sensitivity to feature representation and estimator misspecification.
- Use the data processing inequality as a representation constraint: a deterministic X=f(T) cannot contain more information than T under the same conditioning variables.

## G5 — Cross-cohort validation and robustness

- Test temporal transport across draft cohorts using fixed outcome horizons and player-grouped splits.
- Report calibration, uncertainty, missing-data sensitivity, cohort drift, and position-specific sample limitations.
- Do not choose features, outcomes, or candidate direction based on the final held-out cohort.

## G6 — Competition submission and publication

- Consider a competition submission only after the research design and validation review are complete.
- Publish only a reviewed, disclosure-safe aggregate audit summary. Track dashboard review, approval, deployment, and verification as separate states.

**Current stop:** G1. No model has been fitted and no competition submission has been prepared.
