"""Checkpointed and protocol-gated G3C development execution."""
import argparse
from datetime import datetime,timezone
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

from g23_run import G1, G22, verify_artifacts
from g3c_construct import R1,G3A,G23
from movement_probe import atomic_json,digest

PROTOCOL="observed_representation_protocol.md"
STAGES={
    "construct":("geometry_construct_review.md","baseline_registry.md","construct.json","frozen_predictors_private.json"),
    "policy":("observed_information_results.md","paired_loss_analysis.md","policy_results.json","paired_player_losses_private.json"),
    "context":("contextual_variability_report.md","contextual_variance.json"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize(run):
    r=json.loads((R1/"checkpoint.json").read_text())
    assert r["status"]=="measurement_review_pending" and not r["model_authorized_features"]
    verify_artifacts(R1,r["files"])
    verify_artifacts(Path("."),r["source_files"])
    verify_artifacts(Path("data/raw/nfl-big-data-bowl-2027"),{x["file"]:x["sha256"] for x in r["input_manifest"]})
    assert digest(G1/"audit.duckdb")==r["database_sha256"]
    paths=[p/"checkpoint.json" for p in (G1,G22,G23,G3A,R1)]
    return dict(run=run.name,created_utc=now(),protocol=digest(run/PROTOCOL),
                sources={str(p):digest(p) for p in paths},inputs=r["input_manifest"],
                db_sha256=r["database_sha256"],sealed_cohorts=[2024,2025],
                versions=dict(python=platform.python_version(),numpy=numpy.__version__,
                             scipy=scipy.__version__,duckdb=duckdb.__version__),stages={})


def execute(run,kind):
    checkpoint=run/"checkpoint.json"
    data=json.loads(checkpoint.read_text()) if checkpoint.exists() else initialize(run)
    verify_artifacts(Path("."),data["sources"])
    verify_artifacts(run,{PROTOCOL:data["protocol"]})
    code={str(p):digest(p) for p in map(Path,(f"scripts/g3c_{kind}.py","scripts/g3c_run.py",
          "scripts/g3c_construct.py","scripts/g3a_policy.py","scripts/g3ar1_measure.py","scripts/g23_run.py",
          "scripts/movement_probe.py"))}
    previous=data["stages"].get(kind)
    if previous and previous["complete"]:
        if previous["code"]!=code:
            raise ValueError("Use a new run with changed code")
        verify_artifacts(run,previous["files"])
        previous["cache_hits"]=previous.get("cache_hits",0)+1
        atomic_json(checkpoint,data)
        print(kind,"reused",flush=True)
        return
    if kind!="construct":
        assert data["stages"]["construct"]["complete"]
        verify_artifacts(run,data["stages"]["construct"]["files"])
    verify_artifacts(G1,{"audit.duckdb":data["db_sha256"]})
    if previous:
        data.setdefault("attempts",[]).append(dict(stage=kind,record=previous))
    t,cpu=time.monotonic(),time.process_time()
    rec=data["stages"][kind]=dict(start=now(),complete=False,code=code,
                                  git_at_run=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    atomic_json(checkpoint,data)
    try:
        rec["counts"]=getattr(import_module(f"g3c_{kind}"),kind)(run)
        rec["files"]={name:digest(run/name) for name in STAGES[kind]}
        rec["complete"]=True
    except Exception as e:
        rec["failure"]=type(e).__name__
        raise
    finally:
        rec.update(end=now(),wall_s=time.monotonic()-t,cpu_s=time.process_time()-cpu)
        atomic_json(checkpoint,data)
    print(kind,"complete",rec["counts"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("stages",choices=tuple(STAGES),nargs="+")
    a=p.parse_args()
    with (a.run/".lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for k in a.stages:
            execute(a.run,k)
