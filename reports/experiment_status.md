# Experiment Status — G1 Cloud Data Audit

**Phase:** G1 data preparation and audit complete in the selected private Codex Cloud task. Scientific review and publication remain pending. No model has been fitted.

| Item | Status |
|---|---|
| Goal / phase | NFL Big Data Bowl 2027 G1 — resumable cloud data audit |
| Last updated | 2026-11-26T09:40:24.066127+00:00 |
| Execution environment | Selected private Codex Cloud task; raw data and full reports remain in its ignored `data/` workspace |
| Branch / base commit / run | `g1-resumable-audit` / `fcf40905d50539db334900ac641d9f974e51b1b9` / `g1-cloud-final-20261126T0942Z` |
| Main commit observed | `528c0ef28943015c0a7086417c5caedd9fa6d3e1` |
| LOCAL verified files | Not checked in this task |
| CLOUD verified files | 9/9 official CSVs; per-file byte counts, strict-parse result, and SHA-256 are in the private checkpoint |
| Completed audit steps | Nine-table profiles, keys, relationships, cohort coverage, and sampled 10 Hz intervals |
| Tests | `setup.sh`: 2 passed; resumable-audit suite: 3 passed |
| Evidence | Private `data/checkpoints/download_verification.json`, `data/audit_runs/g1-cloud-final-20261126T0942Z/checkpoint.json`, and `data/checkpoints/audit/g1-cloud-final-20261126T0942Z/report/` |
| Current blocker | None for G1 data acquisition or audit. Candidate outcome horizons and position grouping require scientific review. |
| Next executable action | Review the private A/B/C feasibility matrix and settle a pre-registered outcome horizon before any model work. |
| User action required | Scientific review before selecting a direction; separate approval before dashboard publication. |
| PR / Actions / Dashboard | PR not yet created; Actions not run for this branch; dashboard unchanged and awaiting review. |

The official Kaggle API file listing succeeded. The first object-storage download was rejected because the restricted environment allowed the Kaggle API host but not its verified object-storage host. The cloud egress policy was updated to add `storage.googleapis.com` while retaining the Kaggle API and package-manager access. All nine files then passed size checks and strict DuckDB parsing.

Full raw data, SHA-256 values, field-level missingness, and detailed join results are excluded from this public status page and remain in the private task workspace.
