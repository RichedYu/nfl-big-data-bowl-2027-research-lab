"""Run G3A stages and bind private artifacts to sources, code and protocol."""
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
from g3a_measurement import G24,G23
from movement_probe import atomic_json,digest

STAGES={
    "measurement":("g3a_measurement",("measurement_results.json","feature_freeze.json",
                                     "representation_validation.md","feature_freeze.md")),
    "models":("g3a_policy",("full_policy_evaluation.md","paired_loss_analysis.md","statistical_precision.md",
                            "policy_summary.json","player_oof.json","refit_distributions.json")),
    "mechanism":("g3a_annotation",("mechanism_transfer_feasibility.md","annotation_materials/*")),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def execute(run,stage):
    cp=run/"checkpoint.json"
    if cp.exists():
        state=json.loads(cp.read_text())
    else:
        g24=json.loads((G24/"checkpoint.json").read_text())
        if digest(G1/"audit.duckdb")!=g24["datasource_sha256"]:
            raise ValueError("G1 database changed")
        state=dict(run_id=run.name,protocol_sha256=digest(run/"protocol.md"),created_utc=now(),
                   source_checkpoints={str(p/"checkpoint.json"):digest(p/"checkpoint.json") for p in (G24,G23,G1)},
                   database_sha256=g24["datasource_sha256"],environment=dict(python=platform.python_version(),
                   numpy=numpy.__version__,scipy=scipy.__version__,duckdb=duckdb.__version__),stages={})
    verify_artifacts(Path("."),state["source_checkpoints"])
    verify_artifacts(run,{"protocol.md":state["protocol_sha256"]})
    name,patterns=STAGES[stage]
    module=import_module(name)
    code={str(p):digest(p) for p in sorted(Path("scripts").glob("g3a_*.py"))
          if p.name in (name+".py", "g3a_run.py", "g3a_measurement.py")}
    inherited=("movement_probe.py","g23_motion.py","g23_run.py","g24_annotation.py")
    code.update({f"scripts/{p}":digest(Path("scripts")/p) for p in inherited})
    old=state["stages"].get(stage)
    if old and old.get("complete"):
        if old["code"]!=code:
            raise ValueError("Stage code changed. Use a new run.")
        verify_artifacts(run,old["files"])
        old["cache_hits"]=old.get("cache_hits",0)+1
        atomic_json(cp,state)
        print(stage,"checkpoint reused",flush=True)
        return
    if stage=="models":
        prior=state["stages"].get("measurement",{})
        if not prior.get("complete"):
            raise ValueError("Freeze features before outcome access")
        verify_artifacts(run,prior["files"])
        if digest(G1/"audit.duckdb")!=state["database_sha256"]:
            raise ValueError("G1 database changed")
    count=len(state.get("attempt_history",[]))
    if old:
        state.setdefault("attempt_history",[]).append(dict(stage=stage,record=old,sequence=count))
    record=state["stages"][stage]=dict(start_utc=now(),code=code,complete=False,
                                      git_at_run=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    atomic_json(cp,state)
    start,cpu=time.monotonic(),time.process_time()
    try:
        counts=getattr(module,stage)(run)
        files=[p for pattern in patterns for p in run.glob(pattern) if p.is_file()]
        if any(not list(run.glob(pattern)) for pattern in patterns):
            raise ValueError("Missing expected artifact")
    except Exception as e:
        record.update(failure=type(e).__name__,end_utc=now(),wall_s=time.monotonic()-start,
                      cpu_s=time.process_time()-cpu)
        atomic_json(cp,state)
        raise
    record.update(complete=True,end_utc=now(),wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,
                  counts=counts,files={str(p.relative_to(run)):digest(p) for p in files})
    atomic_json(cp,state)
    print(stage,"complete",counts,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run",type=Path)
    ap.add_argument("stage",choices=tuple(STAGES),nargs="+")
    args=ap.parse_args()
    with (args.run/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in args.stage:
            execute(args.run,stage)
