"""Post-pilot descriptive checks of the frozen feature and candidate game window."""
from collections import defaultdict
import json
from pathlib import Path
import sys
from datetime import datetime,timezone
import time

import duckdb
import numpy as np

from g23_football import development, precision, THRESHOLDS
from g23_motion import G22, global_repr
from g22_motion import noise
from g23_run import G1, verify_artifacts
from movement_probe import atomic_json, digest


def main(run):
    start=time.monotonic()
    cpu=time.process_time()
    started_at=datetime.now(timezone.utc).isoformat()
    path=run/"checkpoint.json"
    cp=json.loads(path.read_text())
    verify_artifacts(run,cp["protocol_sha256"])
    verify_artifacts(Path("."),cp["source_artifacts"])
    verify_artifacts(run,cp["stages"]["football"]["artifacts"])
    verify_artifacts(run,cp["stages"]["motion"]["artifacts"])
    code={str(f):digest(f) for f in (Path(__file__).resolve().relative_to(Path.cwd()),
                                    Path("scripts/g23_football.py"),Path("scripts/g23_motion.py"))}
    if "supplement" in cp["stages"]:
        old=cp["stages"]["supplement"]
        if old["code"]!=code:
            raise ValueError("Supplement code changed")
        verify_artifacts(run,old["artifacts"])
        old["cache_hits"]=old.get("cache_hits",0)+1
        atomic_json(path,cp)
        return
    if digest(G1/"audit.duckdb")!=cp["db_sha256"]:
        raise ValueError("Changed database")
    rng=np.random.default_rng(23038)
    changes=defaultdict(list)
    trials=json.loads((G22/"development_trials.json").read_text())
    usable=0
    for r in trials:
        if r["drill"]!="CURL_ROUTE_RIGHT" or "xy" not in r:
            continue
        t,xy=np.asarray(r["t"]),np.asarray(r["xy"])
        base=global_repr(t,xy)
        if not base:
            continue
        usable+=1
        for event,sd,related in (("iid03",.03,False),("iid10",.1,False),("ar10",.1,True)):
            vs=[global_repr(t,xy+noise(rng,len(t),.1,sd,related)) for _ in range(5)]
            changes[event].append(float(np.median([abs(v["speed"]-base["speed"]) for v in vs if v])))
        trim=global_repr(t[2:-2],xy[2:-2])
        if trim:
            changes["trim_0.2_each_end"].append(abs(trim["speed"]-base["speed"]))
    with duckdb.connect(str(G1/"audit.duckdb"),read_only=True,config={"threads":2,"memory_limit":"4GB"}) as db:
        roster,records,stages,_=development(db)
    classes={(p,g,k,y):r for p,g,k,y,r,s in records if r is not None and s is not None}
    roles=defaultdict(list)
    for p,g,k,year,speed in stages:
        roles[year,"all",p].append((g,speed))
        roles[year,classes[p,g,k,year],p].append((g,speed))
    out={}
    for year in (2023,2024):
        for label in ("all","OUT","HITCH"):
            measures={}
            for n in THRESHOLDS:
                rows=[roles[year,label,p] for p,*_ in roster if len(roles[year,label,p])>=n]
                values=[np.mean([v for _,v in r]) for r in rows]
                se=[s for r in rows if (s:=precision(r,rng)) is not None]
                measures[n]=dict(players=len(rows),median_speed=float(np.median(values)) if values else None,
                                 median_se=float(np.median(se)) if se else None,se_players=len(se))
            out[f"{year}/{label}"]=measures
    summary={k:dict(n=len(a),median_abs=float(np.median(a)),p90=float(np.quantile(a,.9))) for k,a in changes.items()}
    atomic_json(run/"supplement_results.json",dict(frozen_global_speed=summary,candidate_transfer=out))
    lines=["# Descriptive supplement to E32/E36",
           "These post-pilot checks use the frozen predictor and the unchanged game-window definition. They are sensitivity descriptions and did not select the endpoint, predictor or model. Seed 23038.",
           "", "| Curl perturbation | Trials | Median / P90 absolute speed change (yd/s) |",
           "|---|---:|---:|"]
    for k,s in summary.items():
        lines.append(f"| {k} | {s['n']} | {s['median_abs']:.3f} / {s['p90']:.3f} |")
    lines+=["","The global statistic is repeatable to the extent tested by artificial noise or trimming, with no independent validation of event boundaries.",
            "", "| Game window / threshold 25 | Players | Median of player means (yd/s) | Median game-block SE (yd/s) |",
            "|---|---:|---:|---:|"]
    for k,d in out.items():
        row=d[25]
        lines.append(f"| {k} | {row['players']} | {row['median_speed']:.2f} | {row['median_se']:.2f} |")
    lines+=["","Game summaries use 2023-drafted players only. A game mean of snap-to-throw median movement speed is a transfer candidate. It includes tactical pace and pass timing, and does not identify a cut, an isolated movement mechanism or a Combine effect. All thresholds and game-bootstrap denominators are in supplement_results.json."]
    (run/"measurement_transfer_sensitivity.md").write_text("\n".join(lines)+"\n")
    cp["stages"]["supplement"]=dict(start=started_at,end=datetime.now(timezone.utc).isoformat(),
        wall_s=time.monotonic()-start,cpu_s=time.process_time()-cpu,complete=True,code=code,
        counts=dict(curl_trials=usable,players=len(roster),game_windows=len(stages),
                    simulated_coordinate_jitters=15*usable),
        artifacts={n:digest(run/n) for n in ("measurement_transfer_sensitivity.md","supplement_results.json")})
    atomic_json(path,cp)


if __name__=="__main__":
    main(Path(sys.argv[1]))
