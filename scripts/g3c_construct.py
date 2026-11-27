"""Outcome-free observed geometry and baseline registration."""
from collections import defaultdict
import json
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import spearmanr

from g23_run import G1, G22, verify_artifacts
from g3a_measurement import TASK
from g3ar1_measure import FEATURES, features
from movement_probe import atomic_json, digest, wrap

R1=Path("data/g3/g3a-r1-20261127")
G3A=Path("data/g3/g3a-20261127")
G23=Path("data/g2/g2-3-20261127")
COBS="C_obs"


def geometry(nodes):
    nodes=np.asarray(nodes)
    length=float(np.linalg.norm(np.diff(nodes,axis=0),axis=1).sum())
    return float(np.linalg.norm(nodes[-1]-nodes[0])/length),length


def perturbation_bound(c,length,epsilon):
    if epsilon<0 or length<=6*epsilon:
        raise ValueError("Bound requires epsilon>=0 and L>6 epsilon")
    return (2*epsilon+6*epsilon*c)/(length-6*epsilon)


def verify_bound():
    rng=np.random.default_rng(33001)
    max_fraction=0.
    n=0
    for _ in range(2000):
        nodes=rng.normal(size=(4,2))
        c,l=geometry(nodes)
        epsilon=l*rng.uniform(.002,.15)
        direction=rng.uniform(0,2*np.pi,4)
        drag=np.c_[np.cos(direction),np.sin(direction)]*(epsilon*rng.uniform(0,1,4)[:,None])
        perturbed,_=geometry(nodes+drag)
        bound=perturbation_bound(c,l,epsilon)
        assert abs(perturbed-c)<=bound+1e-12
        max_fraction=max(max_fraction,abs(perturbed-c)/bound)
        n+=1
    return dict(cases=n,max_bound_fraction=max_fraction)


def quantiles(x):
    a=np.asarray(x)
    return np.quantile(a,[.1,.25,.5,.75,.9]).tolist() if len(a) else None


def rank(x,y):
    if len(x)<3 or not np.ptp(x) or not np.ptp(y):
        return None
    return float(spearmanr(x,y).statistic)


def baseline_data(db):
    return db.execute("""
      SELECT p.nfl_id,p.nfl_position,c.forty,c.ten_yd_split,c.vertical,c.broad_jump,c.short_shuttle,c.three_cone
      FROM players p LEFT JOIN combine_results c ON p.nfl_id=c.nfl_id AND c.draft_year=2023
      WHERE p.draft_year=2023 AND p.nfl_position IN ('WR','TE')
      ORDER BY p.nfl_id
    """).fetchall()


def construct(run):
    r1=json.loads((R1/"checkpoint.json").read_text())
    verify_artifacts(R1,r1["files"])
    old=json.loads((G22/"checkpoint.json").read_text())
    verify_artifacts(G22,old["stages"]["real"]["artifacts"])
    original=json.loads((G3A/"checkpoint.json").read_text())
    verify_artifacts(G3A,original["stages"]["measurement"]["files"])
    verify_artifacts(G23,json.loads((G23/"checkpoint.json").read_text())["stages"]["motion"]["artifacts"])
    records=[r for r in json.loads((G22/"development_trials.json").read_text()) if r["drill"]==TASK]
    assert len(records)==43
    prior=json.loads((G3A/"measurement_results.json").read_text())
    legacy_tech={r["players"] for r in prior["trial_records"] if r["drill"]==TASK and r["pred"]["quadratic"]["accepted"]}
    g23=json.loads((G23/"development_measurements.json").read_text())
    common={r["player"] for r in g23 if r["drill"]==TASK and r["method"]=="moving_mean"}
    with duckdb.connect(str(G1/"audit.duckdb"),read_only=True) as db:
        table=baseline_data(db)
        # The vendor's same-source speed on observed first/last frames of this drill.
        ends=db.execute("""
          SELECT nfl_id,event_id,first(s ORDER BY time),last(s ORDER BY time)
          FROM combine_tracking WHERE draft_year=2023 AND entity_type='PLAYER' AND drill_name=?
          GROUP BY 1,2
        """,[TASK]).fetchall()
    official={(p,e):(s,z) for p,e,s,z in ends}
    group=defaultdict(list)
    diagnostics=[]
    recorded={r["player"] for r in records}
    for r in records:
        t,xy=np.asarray(r.get("t",[])),np.asarray(r.get("xy",[]))
        c=features(t,xy)[FEATURES[0]]
        if c is None:
            continue
        q=t[0]+np.arange(4)
        nodes=np.column_stack([np.interp(q,t,x) for x in xy.T])
        steps=np.diff(nodes,axis=0)
        delta=np.linalg.norm(steps,axis=1)
        fine=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        heading=np.arctan2(steps[:,1],steps[:,0])
        angles=np.abs(wrap(np.diff(heading)))
        first,last=official[(r["player"],r["event"])]
        v=dict(cobs=c,record_duration=float(t[-1]-t[0]),first_s=first,last_s=last,
               observed_fraction=float(fine[((t[1:]+t[:-1])/2)<t[0]+3].sum()/fine.sum()),
               first_second_rate=float(delta[0]),first_frame_rate=float(fine[0]/(t[1]-t[0])),
               length=float(delta.sum()),bend_total=float(np.degrees(angles.sum())),
               largest_bend_node=int(np.argmax(angles))+1)
        g=group[r["player"]]
        g.append((r["attempt"],r["event"],v))
        diagnostics.append(v)
    ordered=[sorted(values) for values in group.values() if len(values)>=2]
    repeat=[(v[0][2]["cobs"],v[1][2]["cobs"]) for v in ordered]
    a,b=zip(*repeat) if repeat else ([],[])
    values={str(p):float(np.mean([v[2]["cobs"] for v in vv])) for p,vv in group.items()}
    c=np.asarray([r["cobs"] for r in diagnostics])
    varnames=("record_duration","first_s","last_s","observed_fraction","first_second_rate","first_frame_rate",
              "length","bend_total","largest_bend_node")
    ds={k:dict(quantiles=quantiles([v[k] for v in diagnostics]),spearman_with_cobs=rank(c,[v[k] for v in diagnostics]))
        for k in varnames}
    ds["first_s_le_030"]=sum(r["first_s"]<=.30 for r in diagnostics)
    availability=json.loads((R1/"multiscale_results.json").read_text())["real"][TASK][FEATURES[0]]
    assert availability["valid_trials"]==len(diagnostics) and availability["valid_players"]==len(values)
    checks=dict(trial_coverage=len(diagnostics)/len(records)>=.8,player_coverage=len(values)/len(recorded)>=.8,
                defined_observed_origin=all("t" in r and len(r["t"])>0 for r in records),
                same_source_event_join=len(official)==len(records))
    for noise in ("iid10","ar10","trim"):
        s=availability["changes"][noise]
        checks[noise+"_retention"]=s["potential"]>0 and s["retained"]/s["potential"]>=.80
    checks["trim_p90"]=availability["changes"]["trim"]["p90"]<=.10
    use=all(checks.values())
    if len({r[0] for r in table})!=len(table):
        raise ValueError("Combine table is not one row per roster player")
    baselinecounts={name:sum(row[i] is not None for row in table)
                    for i,name in enumerate(("forty","split","vertical","broad_jump","shuttle","three_cone"),2)}
    joint=sum(r[2] is not None and r[3] is not None and 0<r[3]<r[2] for r in table)
    base_enabled=baselinecounts["forty"]>0 and joint/baselinecounts["forty"]>=.8
    q=quantiles(list(values.values()))
    bounds=verify_bound()
    atomic_json(run/"construct.json",dict(cobs_enabled=use,physical_validity="PENDING",checks=checks,
                players_count=len(values),recorded_players=len(recorded),trials_valid=len(diagnostics),trials_total=len(records),
                cobs_iqr=q[3]-q[1] if q else None,quantiles=q,repeat_players=len(repeat),
                repeat_abs_diff=float(np.median(np.abs(np.asarray(a)-b))) if repeat else None,repeat_rank=rank(a,b),
                diagnostics=ds,retention=availability["changes"],bounds=bounds,bplus_enabled=base_enabled))
    atomic_json(run/"frozen_predictors_private.json",dict(room={
        str(p):dict(te=int(pos=="TE"),forty=f,split=s if f is not None and s is not None and 0<s<f else None,
                    tech=int(p in legacy_tech),cobs=values.get(str(p)) if use else None,
                    recorded=p in recorded,legacy_common=p in common and f is not None)
        for p,pos,f,s,*_ in table},cobs_enabled=use,bplus_enabled=base_enabled,
        protocol_hash=digest(run/"observed_representation_protocol.md")))
    lines=["# Geometry construct review","",
           "C_obs is net displacement divided by sum of three one-second node chords, over exactly the first three visible seconds. Regular 10 Hz timestamps, full support and L≥2 yd are required. The task and estimator are the registered Curl candidate from R1. First visible timestamp is a recording boundary; metadata identify a continuous event, not a synchronized movement start. The expression combines pace, path and where the bends fall in the capture window.",
           f"Trials {len(diagnostics)}/{len(records)}, players {len(values)}/{len(recorded)}. First reported speed ≤.30 yd/s on {ds['first_s_le_030']} of {len(diagnostics)} valid traces. Between-player C_obs IQR {q[3]-q[1]:.4f}. Usable repeated players {len(repeat)}, median absolute first-two difference {float(np.median(np.abs(np.asarray(a)-b))) if repeat else None}, Spearman {rank(a,b)}. Small repeated subset is descriptive only.",
           "", "| Recording diagnostic | P10 / median / P90 | Trial Spearman with C_obs |", "|---|---|---:|"]
    for k in varnames:
        zz=ds[k]
        rho=zz["spearman_with_cobs"]
        lines.append(f"| {k} | "+" / ".join(f"{zz['quantiles'][i]:.3f}" for i in (0,2,4))+" | "+
                     (f"{rho:.3f}" if rho is not None else "—")+" |")
    lines+=["","Observed progress is fraction of the *visible* path over the first 3 seconds, so it is not known fraction of the whole physical drill. Bend node is an unvalidated coarse diagnostic. Correlations include mathematical coupling and player repeats and are not inference.",
            "", "| R1 sensitivity reused | Retained / potential | Absolute change P90 |", "|---|---:|---:|"]
    for k,s in availability["changes"].items():
        lines.append(f"| {k} | {s['retained']}/{s['potential']} | {s['p90']:.4f} |")
    lines+=["",f"Operational continuation conditions: {', '.join(k+'='+str(v) for k,v in checks.items())}. "+
            ("Exploratory observed-input fit permitted by this protocol." if use else "Observed-input fit stopped."),
            "Independent calibration, physical equivalence and human phase labels remain PENDING.",
            "", "## Deterministic perturbation bound",
            "Write C=D/L with endpoint norm D and three-chord length L. If all node-coordinate perturbations have norm ≤ε, the reverse triangle inequality gives |ΔD|≤2ε and |ΔL|≤6ε. Then |Ĉ−C|=|ΔD−CΔL|/|L+ΔL|≤(2ε+6εC)/(L−6ε) for L>6ε. This is a deterministic certificate conditional on bounded node error. A Gaussian standard deviation is not a uniform bound and no NFL sensor epsilon is assumed.",
            f"Verified {bounds['cases']} seeded perturbed node configurations, maximum observed error/bound ratio {bounds['max_bound_fraction']:.3f}."]
    (run/"geometry_construct_review.md").write_text("\n".join(lines)+"\n")
    lines=["# Baseline registry","",
           "M0 is position as known pre-draft; M1 includes traditional forty with roster fallback; M1G adds technical Curl availability. B+ adds the official electronic 10-yard split in seconds to forty; B+G adds technical availability. Same-roster scope for all comparisons.",
           f"Coverage among {len(table)} development WR/TE: "+", ".join(f"{k} {v}" for k,v in baselinecounts.items())+".",
           f"Plausible paired split/forty coverage is {joint}/{baselinecounts['forty']}. Registered B+ enabled: {base_enabled}. Split requires 0 < split < forty. Official player-level columns are pre-draft measurements, without synchronization to captured tracking or proof both timings correspond to one filmed attempt. B+ uses the supplied values independently, not a derived 10-to-40 interval. If split unavailable, fall back to the corresponding forty/roster branch.",
           "Other traditional fields remain coverage diagnostics. No outcome-guided search. Technical availability is measured before draft, but may reflect participation, release and recording selection. Missing/rejected can have different reasons and imply no ability ordering.",
           "[Official competition data dictionary](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data)"]
    (run/"baseline_registry.md").write_text("\n".join(lines)+"\n")
    return dict(roster_players=len(table),valid_trials=len(diagnostics),feature_players=len(values),
                repeated_players=len(repeat),bound_checks=bounds["cases"],cobs_enabled=use,bplus_enabled=base_enabled)
