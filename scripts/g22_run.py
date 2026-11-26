"""Run private G2.2 stages and record artifact hashes."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

from movement_probe import atomic_json, digest
import g22_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("stage", choices=("synthetic", "real", "report", "information", "outcomes"))
    args = parser.parse_args()
    run = args.run
    with (run / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(run, args.stage)


def execute(run, stage):
    cp = run / "checkpoint.json"
    ck = json.loads(cp.read_text())
    if ck["protocol_sha256"] != digest(run / "protocol.md"):
        raise ValueError("Protocol no longer matches the frozen copy")
    modules = {"synthetic": "g22_motion", "real": "g22_motion", "report": "g22_report",
               "information": "g22_information", "outcomes": "g22_outcomes"}
    module = __import__(modules[stage])
    outputs = {"synthetic": ["synthetic_truth_results.json"], "real": ["development_trials.json"],
               "report": ["segmentation_comparison.md", "repeatability_report.md", "motion_summary.json"],
               "information": ["information_theory_experiments.md", "information_results.json"],
               "outcomes": ["outcome_feasibility.md", "outcome_coverage.json"]}
    identity = {Path(x).name: digest(x) for x in (module.__file__, __file__,
                                               g22_motion.__file__, "scripts/movement_probe.py")}
    dependencies = {"report": ("synthetic", "real"), "outcomes": ("real",)}
    for dependency in dependencies.get(stage, ()):
        prior = ck["stages"][dependency]
        for name, sha in prior["artifacts"].items():
            if digest(run/name) != sha:
                raise ValueError(f"Changed {dependency} artifact {name}")
    if stage in ("real", "outcomes"):
        if digest(g22_motion.G1) != ck["database_sha256"]:
            raise ValueError("Changed G1 database")
    previous = ck["stages"].get(stage)
    if previous and previous.get("complete"):
        if previous["code"] == identity and all(
            (run/p).exists() and digest(run/p)==sha for p,sha in previous["artifacts"].items()
        ):
            print(stage, "checkpoint reused")
            return
        raise ValueError(f"{stage} completed under different code or artifacts. Use a new run.")
    ck["stages"][stage] = dict(complete=False, code=identity)
    atomic_json(cp, ck)
    getattr(module, stage)(run)
    names = outputs[stage]
    if stage in ("report", "information"):
        prefix = "motion_" if stage=="report" else "learning_"
        names += [str(p.relative_to(run)) for p in sorted((run/"figures").glob(prefix+"*.png"))]
    ck["stages"][stage].update(complete=True, artifacts={n:digest(run/n) for n in names},
                              at_utc=datetime.now(timezone.utc).isoformat(),
                              git_base=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    atomic_json(cp, ck)
    print(stage, "complete")


if __name__=="__main__":
    main()
