"""Checkpointed measurement-only runner for G3A-R1."""
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

from g23_run import G1, verify_artifacts
from g3ar1_measure import G3A,G22
from movement_probe import digest,atomic_json

DICTIONARY=Path("data/g2/official_pages.json")
G21=Path("data/g2/g2-1-20261126-final")
STAGES={
    "analytic":("g3ar1_measure",("analytical_bias_verification.md","analytical_results.json")),
    "measurement":("g3ar1_measure",("multiscale_measurement_report.md","candidate_feature_freeze.md","multiscale_results.json")),
    "reference":("g3ar1_reference",("reference_measurement_audit.md","reference_audit.json")),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def initialise(run):
    g3a=json.loads((G3A/"checkpoint.json").read_text())
    if g3a["status"]!="G3A_complete_awaiting_scientific_review":
        raise ValueError("G3A is not complete")
    verify_artifacts(G3A,g3a["deliverables"])
    verify_artifacts(Path("."),g3a["source_checkpoints"])
    verify_artifacts(Path("data/raw/nfl-big-data-bowl-2027"),{x["file"]:x["sha256"] for x in g3a["input_manifest"]})
    g21=json.loads((G21/"checkpoint.json").read_text())
    assert digest(DICTIONARY)==g21["official_dictionary_sha256"]
    assert digest(G1/"audit.duckdb")==g3a["database_sha256"]
    return dict(run_id=run.name,created=now(),protocol_sha256=digest(run/"protocol.md"),
                database_sha256=g3a["database_sha256"],
                source_files={str(x):digest(x) for x in (G3A/"checkpoint.json",G22/"checkpoint.json",
                      G21/"checkpoint.json",G1/"checkpoint.json",DICTIONARY)},
                input_manifest=g3a["input_manifest"],sealed_draft_cohorts=[2024,2025],
                versions=dict(python=platform.python_version(),numpy=numpy.__version__,
                              duckdb=duckdb.__version__,scipy=scipy.__version__),stages={})


def execute(run,stage):
    cp=run/"checkpoint.json"
    state=json.loads(cp.read_text()) if cp.exists() else initialise(run)
    verify_artifacts(run,{"protocol.md":state["protocol_sha256"]})
    verify_artifacts(Path("."),state["source_files"])
    name,files=STAGES[stage]
    mod=import_module(name)
    code={str(x):digest(x) for x in [Path("scripts")/p for p in
            sorted(set((name+".py","g3ar1_run.py","g3ar1_measure.py","g23_motion.py","movement_probe.py")))]}
    old=state["stages"].get(stage)
    if old and old["complete"]:
        if old["code"]!=code:
            raise ValueError("Completed stage code changed. Use a new run.")
        verify_artifacts(run,old["files"])
        old["cache_hits"]=old.get("cache_hits",0)+1
        atomic_json(cp,state)
        print(stage,"reused",flush=True)
        return
    if stage=="reference":
        verify_artifacts(G1,{"audit.duckdb":state["database_sha256"]})
    if old:
        state.setdefault("failed_attempts",[]).append(dict(stage=stage,**old))
    record=state["stages"][stage]=dict(start=now(),complete=False,code=code,
               commit_at_run=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    atomic_json(cp,state)
    wall,cpu=time.monotonic(),time.process_time()
    try:
        counts=getattr(mod,stage)(run)
        result={f:digest(run/f) for f in files}
    except Exception as exc:
        record.update(failure=type(exc).__name__,end=now(),wall_s=time.monotonic()-wall,
                      cpu_s=time.process_time()-cpu)
        atomic_json(cp,state)
        raise
    record.update(complete=True,end=now(),wall_s=time.monotonic()-wall,cpu_s=time.process_time()-cpu,
                  counts=counts,files=result)
    atomic_json(cp,state)
    print(stage,"completed",counts,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("stage",choices=tuple(STAGES),nargs="+")
    a=p.parse_args()
    with (a.run/".lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for s in a.stage:
            execute(a.run,s)
