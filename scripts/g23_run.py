"""Private G2.3 stage runner with fingerprint and artifact verification."""
import argparse
from datetime import datetime, timezone
import fcntl
from importlib import import_module
import json
from pathlib import Path
import platform
import subprocess
import time

import duckdb
import numpy
import scipy

from movement_probe import atomic_json, digest

ROOT=Path("data/g2")
G1=Path("data/audit_runs/g1-cloud-final-20261126T0942Z")
G21=ROOT/"g2-1-20261126-final"
G22=ROOT/"g2-2-20261127"
STAGES=dict(motion="g23_motion",reliability="g23_reliability",causal="g23_causal",
            football="g23_football")
FILES={
    "motion":("measurement_review.md","trajectory_estimator_comparison.md","risk_coverage_report.md",
              "development_measurements.json","motion_results.json","annotation_key_private.json",
              "blind_annotation_packet/*","figures/risk_coverage.png"),
    "reliability":("reliability_ademp_report.md","reliability_results.json","figures/reliability_coverage.png"),
    "causal":("causal_experiments.md","causal_estimand_registry.md","causal_results.json"),
    "football":("route_correspondence_matrix.md","outcome_design.md","development_prediction_pilot.md",
                "football_results.json"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_artifacts(root,artifacts):
    for name,expected in artifacts.items():
        if digest(root/name)!=expected:
            raise ValueError(f"Fingerprint changed: {root/name}")


def initialize(run):
    start=time.monotonic()
    checkpoints=[json.loads((p/"checkpoint.json").read_text()) for p in (G1,G21,G22)]
    one,two,three=checkpoints
    if one["phase"]!="reports_complete" or two["status"]!="evidence_complete" or three["status"]!="awaiting_scientific_review":
        raise ValueError("Expected completed source stages")
    inputs=one["fingerprint"]["inputs"]
    if inputs!=two["fingerprint"]["inputs"] or inputs!=three["inputs"]:
        raise ValueError("Historical file manifests disagree")
    verify_artifacts(Path("data/raw/nfl-big-data-bowl-2027"),{f["file"]:f["sha256"] for f in inputs})
    db=digest(G1/"audit.duckdb")
    if db!=two["fingerprint"]["database_sha256"] or db!=three["database_sha256"]:
        raise ValueError("Database digest does not match historical runs")
    verify_artifacts(G21,two["review_files"])
    verify_artifacts(G22,three["review_artifacts"])
    for s in three["stages"].values():
        verify_artifacts(G22,s["artifacts"])
    if digest(ROOT/"official_pages.json")!=two["official_dictionary_sha256"]:
        raise ValueError("Historical competition documentation changed")
    return dict(run_id=run.name,initialized=now(),verify_wall_s=time.monotonic()-start,
                inputs=inputs,db_sha256=db,source_artifacts={
                    str(p/"checkpoint.json"):digest(p/"checkpoint.json") for p in (G1,G21,G22)}
                | {str(ROOT/"official_pages.json"):digest(ROOT/"official_pages.json")},
                protocol_sha256={n:digest(run/n) for n in ("protocol.md","pilot_protocol.md")},
                environment=dict(python=platform.python_version(),duckdb=duckdb.__version__,
                                 numpy=numpy.__version__,scipy=scipy.__version__),stages={})


def execute(run,stage):
    cp=run/"checkpoint.json"
    state=json.loads(cp.read_text()) if cp.exists() else initialize(run)
    verify_artifacts(run,state["protocol_sha256"])
    verify_artifacts(Path("."),state["source_artifacts"])
    prior=state["stages"].get(stage,{})
    mod=import_module(STAGES[stage])
    code={str(Path(p).relative_to(Path.cwd())):digest(p) for p in
          (Path(mod.__file__).resolve(),Path(__file__).resolve(),
           *[Path("scripts",n).resolve() for n in ("movement_probe.py","g22_motion.py")])}
    if prior.get("complete"):
        if code!=prior["code"]:
            raise ValueError("Use a new run for altered stage code")
        verify_artifacts(run,prior["artifacts"])
        prior["cache_hits"]=prior.get("cache_hits",0)+1
        atomic_json(cp,state)
        print(stage,"checkpoint reused")
        return
    if stage=="football":
        verify_artifacts(run,state["stages"]["motion"]["artifacts"])
        if digest(G1/"audit.duckdb")!=state["db_sha256"]:
            raise ValueError("Audit database changed")
    state["stages"][stage]=record=dict(start=now(),complete=False,code=code)
    atomic_json(cp,state)
    start,cpu=time.monotonic(),time.process_time()
    counts=getattr(mod,stage)(run)
    files=[f for glob in FILES[stage] for f in run.glob(glob) if f.is_file()]
    record.update(end=now(),wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,
                  counts=counts,complete=True,git_at_run=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  artifacts={str(f.relative_to(run)):digest(f) for f in files})
    atomic_json(cp,state)
    print(stage,"complete",counts,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("stages",choices=tuple(STAGES),nargs="+")
    a=p.parse_args()
    with (a.run/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in a.stages:
            execute(a.run,stage)


if __name__=="__main__":
    main()
