"""G2.4 read-only availability, representation and precision diagnostics."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path

import duckdb
import numpy as np

from g23_run import G1, G22, verify_artifacts
from g24_annotation import FIELDS
from movement_probe import atomic_json, digest

SOURCE=Path("data/g2/g2-3-20261127")
CONTEXT=("down","yards_to_go","yardline_number","quarter","game_clock","offense_formation",
         "receiver_alignment","team_coverage_man_zone","team_coverage_type","target","pass_length",
         "separation_at_pass_forward")


def inputs():
    cp=json.loads((SOURCE/"checkpoint.json").read_text())
    assert cp["status"]=="awaiting_scientific_review"
    verify_artifacts(SOURCE,cp["final_artifacts"])
    for stage in cp["stages"].values():
        verify_artifacts(SOURCE,stage["artifacts"])
    verify_artifacts(Path("."),cp["source_artifacts"])
    if digest(G1/"audit.duckdb")!=cp["db_sha256"]:
        raise ValueError("G1 database fingerprint changed")
    return cp


def annotation(run, old):
    trials={tuple(r[k] for k in ("drill","player","event","attempt")):r for r in old}
    key=json.loads((SOURCE/"annotation_key_private.json").read_text())
    folder=run/"annotation_materials"
    folder.mkdir(exist_ok=True)
    with (folder/"cases.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("case","last_frame_s"))
        w.writerows((case,round(trials[tuple(row)]["t"][-1],6)) for case,row in sorted(key.items()))
    for reviewer in ("A","B"):
        with (folder/f"reviewer_{reviewer}_blank.csv").open("w",newline="") as f:
            w=csv.writer(f)
            w.writerow(FIELDS)
            w.writerows([name,reviewer]+[""]*(len(FIELDS)-2) for name in sorted(key))
    with (folder/"adjudication_blank.csv").open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(("case","event_A","event_B","reason_for_review","status","adjudicator","onset_s",
                    "bend_start_s","bend_center_s","bend_end_s","exit_s","direction","confidence","rationale"))
    return len(key)


def stratified_rejection(rows):
    by=defaultdict(list)
    for r in rows:
        by[r["drill"]].append(r)
    result={}
    for drill,group in sorted(by.items()):
        def rates(rs):
            local=[r for r in rs if r["accepted"]]
            return dict(trials=len(rs),accepted=len(local),
                        distinct_players=len({r["player"] for r in rs}),
                        players_with_accepted=len({r["player"] for r in local}))
        strata={}
        for field in ("speed","complexity"):
            values=[r["global_rep"][field] for r in group]
            lo,hi=np.quantile(values,[1/3,2/3])
            bins=[[],[],[]]
            for r,v in zip(group,values):
                bins[int(v>lo)+int(v>hi)].append(r)
            strata[field]=dict(cuts=[lo,hi],bins=[rates(x) for x in bins])
        result[drill]={"overall":rates(group)}|strata
    return result


def path_summary(t,xy):
    steps=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    total=steps.sum()
    if not total or len(steps)<10:
        return None
    s=np.r_[0,np.cumsum(steps)]
    unique=np.r_[True,np.diff(s)>0]
    q=np.linspace(0,total,21)
    u=np.column_stack([np.interp(q,s[unique],p[unique]) for p in xy.T])
    middle=(s[1:]+s[:-1])/2/total
    speed=steps/np.diff(t)
    bands=[speed[((middle>=i/3)&(middle<(i+1)/3)) & (speed>=1)] for i in range(3)]
    return dict(chord_ratio=float(np.linalg.norm(u[-1]-u[0])/total),
                distance=float(total), duration=float(t[-1]-t[0]),
                speed_bands=[float(np.median(x)) if len(x)>=3 else None for x in bands])


def representation(old):
    rng=np.random.default_rng(24032)
    by=defaultdict(list)
    for r in old:
        if "t" not in r or "xy" not in r:
            continue
        t,xy=np.asarray(r["t"]),np.asarray(r["xy"])
        if len(t)<11 or np.any(abs(np.diff(t)-.1)>.001):
            continue
        base=path_summary(t,xy)
        if base is None:
            continue
        record=dict(player=r["player"],complete=all(x is not None for x in base["speed_bands"]),variants={})
        for label,noise in (("iid03",.03),("iid10",.10),("trim_0.2",None)):
            sample=[path_summary(t,xy+rng.normal(0,noise,xy.shape)) for _ in range(5)] if noise else [path_summary(t[2:-2],xy[2:-2])]
            errors=defaultdict(list)
            for s in sample:
                if s is None:
                    continue
                for metric in ("chord_ratio","distance","duration"):
                    errors[metric].append(abs(s[metric]-base[metric]))
                for j,(p,q) in enumerate(zip(s["speed_bands"],base["speed_bands"]),1):
                    if p is not None and q is not None:
                        errors[f"band_{j}_speed"].append(abs(p-q))
            record["variants"][label]={k:float(np.median(v)) for k,v in errors.items()}
        by[r["drill"]].append(record)
    out={}
    for drill,group in sorted(by.items()):
        variants={}
        for label in ("iid03","iid10","trim_0.2"):
            pool=defaultdict(list)
            for r in group:
                for k,v in r["variants"][label].items():
                    pool[k].append(v)
            variants[label]={k:dict(pairs=len(v),median_abs=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in pool.items()}
        out[drill]=dict(trials=len(group),players=len({r["player"] for r in group}),
                        complete_bands=sum(r["complete"] for r in group),sensitivity=variants)
    return out


def availability(curl, pilot):
    with duckdb.connect(str(G1/"audit.duckdb"),read_only=True) as db:
        # No held-out roster or outcome values are queried. Separation is used solely in IS NOT NULL.
        roster=db.execute("""
          WITH r AS (SELECT nfl_id FROM players WHERE draft_year=2023 AND nfl_position IN ('WR','TE')),
          eligible AS (SELECT p.nfl_id,count(*) AS opportunities,count(DISTINCT p.game_id) AS games
            FROM r JOIN player_play p USING(nfl_id) JOIN games g USING(game_id)
            WHERE g.season=2023 AND g.season_type='REG' AND p.route_ran IS NOT NULL
            AND p.separation_at_pass_forward IS NOT NULL GROUP BY p.nfl_id)
          SELECT r.nfl_id,b.forty IS NOT NULL,coalesce(opportunities,0),coalesce(games,0)
          FROM r LEFT JOIN combine_results b USING(nfl_id) LEFT JOIN eligible e USING(nfl_id)
        """).fetchall()
        row=db.execute("""
          SELECT count(*), """+", ".join(f'count(p."{field}")' for field in CONTEXT)+"""
          FROM players r JOIN player_play p USING(nfl_id) JOIN games g USING(game_id)
          WHERE r.draft_year=2023 AND r.nfl_position IN ('WR','TE') AND g.season=2023
          AND g.season_type='REG' AND p.route_ran IS NOT NULL
          AND p.separation_at_pass_forward IS NOT NULL
        """).fetchone()
        vocab=db.execute("""
          SELECT p.route_ran,count(*),count(DISTINCT p.nfl_id)
          FROM players r JOIN player_play p USING(nfl_id) JOIN games g USING(game_id)
          WHERE r.draft_year=2023 AND r.nfl_position IN ('WR','TE') AND g.season=2023
          AND g.season_type='REG' AND p.route_ran IS NOT NULL
          AND p.separation_at_pass_forward IS NOT NULL GROUP BY 1 ORDER BY 2 DESC
        """).fetchall()
    result={}
    for label,subset in (("R_roster",roster),("Y_eligible",[r for r in roster if r[2]>=25])):
        g=sum(f and p in curl for p,f,_,_ in subset)
        b=sum(f for _,f,_,_ in subset)
        result[label]=dict(total=len(subset),global_plus_forty=g,forty_fallback=b-g,
                           roster_only=len(subset)-b,traditional_coverage=b,
                           common_pilot=g,local_authorized=0)
    thresholds={}
    for k,v in pilot["coverage"]["2023"]["opportunities"]["overall"].items():
        selected=[r for r in roster if r[2]>=int(k)]
        assert len(selected)==v["players"]
        thresholds[k]=dict(players=len(selected),median_opportunities=float(np.median([r[2] for r in selected])),
                           median_distinct_games=float(np.median([r[3] for r in selected])),
                           min_distinct_games=min(r[3] for r in selected),
                           median_game_block_se=v["median_se"],bootstrap_players=v["bootstrap_players"])
    return dict(policy=result,opportunities=thresholds,field_denominator=row[0],
                context_fields=dict(zip(CONTEXT,row[1:])),vocabulary=vocab)


def precision_plan(pilot):
    result={}
    for endpoint,measure,deltas in (("opportunity","log_loss",(.01,.02)),
                                     ("performance","mae",(.02,.05))):
        record=pilot[endpoint]
        r=record["metrics"][measure]
        lo,hi=r["interval"][1]
        sd=(hi-lo)/(2*1.959964)*math.sqrt(record["n"])
        result[measure]=dict(n=record["n"],difference=r["differences"][1],interval=[lo,hi],
                             proxy_sd=sd,scenarios=[dict(delta=d,half_width=d/2,
                             n_fixed=math.ceil((1.959964*sd/(d/2))**2),
                             n_sd_1_5=math.ceil((1.959964*1.5*sd/(d/2))**2))
                             for d in deltas])
    return result


def design(run):
    cp=inputs()
    g22=json.loads((G22/"development_trials.json").read_text())
    r=json.loads((SOURCE/"development_measurements.json").read_text())
    pilot=json.loads((SOURCE/"football_results.json").read_text())
    syn=json.loads((SOURCE/"motion_results.json").read_text())
    local=[v for v in r if v["method"]=="local_quadratic"]
    # Evaluate conditional error using true, fixed bookends. Compare available denominators.
    clean=[]
    for gate in (20,35,50):
        rows=[v for v in syn["evaluation"] if v["method"]=="local_quadratic" and v["gate"]==gate
              and len(v["oracle_errors"])==1]
        usable=[v for v in rows if v["oracle_errors"][0] is not None]
        # Select only trials with one matched feature. Non-matched events and feature failures remain in coverage.
        selected=[e for v in rows if v["detections"]==1 and len(v["matched"])==1
                  for e in v["matched"] if e["angle"] is not None]
        clean.append(dict(gate=gate,target_trials=len(rows),oracle_available=len(usable),
                          accepted_correct_matches=len(selected),acceptance=sum(v["detections"]==1 and v["features"]==1 for v in rows)/len(rows),
                          conditional_angle_mae=float(np.mean([abs(e["angle"]) for e in selected])) if selected else None,
                          event_f1=syn["grid"][f"local_quadratic/{gate}"]["evaluation"]["f1"]))
    data=dict(source_id=cp["run_id"],annotation_cases=annotation(run,g22),
              rejection=stratified_rejection(local),representation=representation(g22),
              risk_coverage=clean,policy_and_outcome=availability(
                  {v["player"] for v in r if v["drill"]=="CURL_ROUTE_RIGHT" and v["method"]=="moving_mean"},pilot["pilot"] | {"coverage":pilot["coverage"]}),
              precision=precision_plan(pilot["pilot"]))
    atomic_json(run/"design_diagnostics.json",data)
    atomic_json(run/"checkpoint.json",dict(run_id=run.name,state="diagnostics_complete",
                completed_utc=datetime.now(timezone.utc).isoformat(),
                source_checkpoint_sha256=digest(SOURCE/"checkpoint.json"),datasource_sha256=cp["db_sha256"],
                code_sha256=digest(__file__),rng_seed=24032,files={"design_diagnostics.json":digest(run/"design_diagnostics.json")}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    a=p.parse_args()
    if (a.run/"checkpoint.json").exists():
        p.error("run already exists")
    design(a.run)
