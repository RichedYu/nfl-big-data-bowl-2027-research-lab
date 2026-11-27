"""2023-only Combine reference and recording-coverage audit."""
from collections import defaultdict
import json

import duckdb
import numpy as np

from g23_run import G1
from g3ar1_measure import METHODS,native,ready
from movement_probe import atomic_json

BASE=("forty","ten_yd_split","vertical","broad_jump","short_shuttle","three_cone")


def crossing(t,value,mark):
    idx=np.flatnonzero(value>=mark)
    if not len(idx) or idx[0]==0:
        return None
    i=idx[0]
    return float(t[i-1]+(t[i]-t[i-1])*(mark-value[i-1])/(value[i]-value[i-1]))


def read(db):
    records=db.execute("""
      SELECT nfl_id,event_id,drill_name,attempt,time,x,y,s,a,dir,dis
      FROM combine_tracking WHERE draft_year=2023 AND entity_type='PLAYER'
      ORDER BY nfl_id,event_id,time
    """).fetchall()
    fields=db.execute("""
      SELECT p.nfl_id,p.nfl_position,"""+", ".join('c."'+c+'"' for c in BASE)+"""
      FROM players p LEFT JOIN combine_results c ON c.nfl_id=p.nfl_id AND c.draft_year=2023
      WHERE p.draft_year=2023
    """).fetchall()
    return records,fields


def stat(x):
    a=np.asarray(x)
    if not len(a):
        return dict(n=0,median=None,p90=None)
    return dict(n=len(a),median=float(np.median(a)),p90=float(np.quantile(a,.9)))


def reference(run):
    with duckdb.connect(str(G1/"audit.duckdb"),read_only=True) as db:
        records,fields=read(db)
    grouped=defaultdict(list)
    for line in records:
        grouped[line[:4]].append(line[4:])
    roster={f[0]:dict(zip(("position",*BASE),f[1:])) for f in fields}
    off={m:defaultdict(list) for m in METHODS}
    recorded_gap=0
    low=defaultdict(list)
    dash=[]
    tried=set()
    dash_players={p for p,_,name,_ in grouped if name=="FORTY_YARD_DASH"}
    for (p,event,drill,attempt),rows in grouped.items():
        t=np.asarray([(r[0]-rows[0][0]).total_seconds() for r in rows])
        xy=np.asarray([r[1:3] for r in rows],dtype=float)
        s=np.asarray([r[3] for r in rows],dtype=float)
        if not ready(t,xy) or not np.isfinite(s).all():
            recorded_gap+=1
            continue
        for method in METHODS:
            q,v=native(t,xy,method)
            if not len(q):
                continue
            official=np.interp(q,t,s)
            d=v-official
            off[method][p].append([len(d),d.mean(),np.mean(d*d),np.mean(abs(d))])
        for j in range(len(t)-10):
            if np.mean(s[j:j+11])<=.30:
                endpoint=float(np.linalg.norm(xy[j+10]-xy[j])/(t[j+10]-t[j]))
                low[drill].append(dict(player=p,event=event,window_origin=float(t[j]),
                    official_mean_s=float(np.mean(s[j:j+11])),d10=endpoint,passes_displacement_screen=endpoint<=.30))
                break  # first eligible window only; any low-speed window identifies a trial
        if drill!="FORTY_YARD_DASH":
            continue
        tried.add(p)
        v=xy[-1]-xy[0]
        d=np.linalg.norm(v)
        span=(xy-xy[0])@(v/d) if d>0 else np.zeros(len(t))
        y10,y40=(crossing(t,span,x) for x in (10,40))
        ref=roster.get(p,{})
        r10,r40=ref.get("ten_yd_split"),ref.get("forty")
        dash.append(dict(player=p,event=event,attempt=attempt,first_s=float(s[0]),last_s=float(s[-1]),
            max_progress=float(span.max()),chord=float(d),elapsed=float(t[-1]),proxy_crossing10=y10,
            proxy_crossing40=y40,official_split10=r10,official_forty=r40,
            proxy_minus_10=float(y10-r10) if y10 is not None and r10 is not None else None,
            proxy_minus_40=float(y40-r40) if y40 is not None and r40 is not None else None))
    timing={}
    for key,rk,tag in (("proxy_minus_10","official_split10","10"),("proxy_minus_40","official_forty","40")):
        counts=defaultdict(list)
        for d in dash:
            if d[key] is not None:
                counts[d["player"]].append(d[key])
        timing[tag]=dict(official_trials=sum(d[rk] is not None for d in dash),
            crossings=sum(d["proxy_crossing"+tag] is not None for d in dash),
            players=len(counts),paired_trials=sum(len(x) for x in counts.values()),
            player_mean_proxy_difference=stat([np.mean(v) for v in counts.values()]))
    rng=np.random.default_rng(31103)
    comparisons={}
    for m,group in off.items():
        x=np.asarray([np.mean([v[1:] for v in rows],axis=0) for rows in group.values()])
        b=x[rng.integers(0,len(x),(500,len(x)))].mean(1)
        comparisons[m]=dict(players=len(x),trials=sum(len(v) for v in group.values()),
            method_frames=sum(int(z[0]) for v in group.values() for z in v),bias=float(x[:,0].mean()),
            bias_ci=np.quantile(b[:,0],[.025,.975]).tolist(),rmse=float(np.sqrt(x[:,1].mean())),
            mae=float(x[:,2].mean()))
    baselines={}
    for label,entries in (("all_2023",list(roster.values())),
                          ("WR_TE_2023",[z for z in roster.values() if z["position"] in ("WR","TE")])):
        baselines[label]=dict(n=len(entries),available={k:sum(z[k] is not None for z in entries) for k in BASE},
            both_forty_and_split=sum(z["forty"] is not None and z["ten_yd_split"] is not None for z in entries))
    diag=dict(recorded_events=len(grouped),recorded_players=len({k[0] for k in grouped}),
        ineligible_events=recorded_gap,comparisons=comparisons,baseline=baselines,
        low_candidate_counts={k:dict(trials=len(v),players=len({z["player"] for z in v}),
                          corroborating_same_source=sum(r["passes_displacement_screen"] for r in v)) for k,v in low.items()},
        low_candidate_windows=low,dash=dict(recorded_trials=sum(k[2]=="FORTY_YARD_DASH" for k in grouped),
            recorded_players=len(dash_players),usable_trials=len(dash),usable_players=len(tried),
            recorded_first_speed_low=sum(r["first_s"]<=.30 for r in dash),
            recorded_last_speed_low=sum(r["last_s"]<=.30 for r in dash),
            max_progress=stat([r["max_progress"] for r in dash]),timing=timing,private_trials=dash))
    atomic_json(run/"reference_audit.json",diag)
    z=diag["dash"]
    content=["# Reference measurement audit · 2023 Combine",
             "The authenticated Kaggle data-description archived earlier documents optical 10 Hz PLAYER paths, instantaneous s in yd/s, a in yd/s², dir as direction of travel in degrees and dis as distance since preceding frame. It does not specify hardware-error variance, independent calibration, vendor smoothing, reference timing triggers or separate speed sensing. The dis discrepancy remains unresolved and dis is excluded. Neither travel direction nor acceleration identifies body rotation or joint loading.",
             "", "## Same-source speed consistency",
             "Reported s interpolated at native estimator centers. Metrics weight trials equally within player and players equally; uncertainty is a 500-draw player bootstrap for mean bias. Chord span and instantaneous s have different meaning on turns.",
             "", "| Method | Trials / players | Mean difference | RMSE | MAE | Bootstrap bias 95% |",
             "|---|---:|---:|---:|---:|---|"]
    for k,v in comparisons.items():
        lo,hi=v["bias_ci"]
        content.append(f"| {k} | {v['trials']}/{v['players']} | {v['bias']:+.3f} | {v['rmse']:.3f} | {v['mae']:.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    content+=["","Candidate low-motion windows use first 1 s of samples per trial meeting mean s ≤0.30 yd/s anywhere in the record. A second gate uses 1 s endpoint displacement. Both use the same recorded trajectories. No candidate is a verified stationary or controlled low-speed reference.",
              "", "| Drill | Candidate trials | Players | Same-source chord rate ≤.30 |","|---|---:|---:|---:|"]
    for k,v in sorted(diag["low_candidate_counts"].items()):
        content.append(f"| {k} | {v['trials']} | {v['players']} | {v['corroborating_same_source']} |")
    content+=["","## Electronic 10-yard and 40-yard references",
              f"2023 FORTY_YARD_DASH has {z['recorded_trials']} attempts from {z['recorded_players']} players. All technical analyses use {z['usable_trials']} attempts from {z['usable_players']}. The code uses PLAYER only. There is an attempt index and timestamp but no start/finish or timing beam marker. First s ≤0.30: {z['recorded_first_speed_low']}/{z['usable_trials']}; last s ≤0.30: {z['recorded_last_speed_low']}/{z['usable_trials']}. Median maximum relative start-to-end projection is {z['max_progress']['median']:.2f} yd.",
              "Reference times are electronic player-level results with no trial-key match. The NFL drill defines 10 and 40 yard distances, whereas our crossings subtract the first visible coordinate and project onto the endpoint direction. This can neither locate the surveyed start line nor establish that the clock starts at the first frame. Crossing coverage checks reveal edge clipping and uncertain origin. Retain all attempts and never select the one closest to an official time.",
              "", "| Proxy | Trials crossing / usable | Trials with official time | Paired trials / players | Median player-average proxy minus official (s) |",
              "|---|---:|---:|---:|---:|"]
    for label,x in timing.items():
        q=x["player_mean_proxy_difference"]["median"]
        content.append(f"| {label} yd | {x['crossings']}/{z['usable_trials']} | {x['official_trials']} | {x['paired_trials']}/{x['players']} | "+(f"{q:+.3f}" if q is not None else "—")+" |")
    content+=["","The official timing may supply an independently meaningful distance-time endpoint once the timing trigger, exact landmarks and paired attempt are established. This release does not document synchronization for those comparisons.",
              "", "## Traditional baseline candidates",
              "", "| 2023 scope | Roster | Forty | 10 yd split | Both | Vertical | Broad jump | Shuttle | Three cone |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k,b in baselines.items():
        a=b["available"]
        content.append(f"| {k} | {b['n']} | {a['forty']} | {a['ten_yd_split']} | {b['both_forty_and_split']} | {a['vertical']} | {a['broad_jump']} | {a['short_shuttle']} | {a['three_cone']} |")
    content+=["","For a later frozen baseline, consider choosing either 10-yard time or one nonredundant early/late split representation with fixed shrinkage and fallback, then compare with forty-only on the same roster. Verify attempts/split semantics before taking timing differences. Jumps are separate explosiveness constructs; shuttle and three cone have selection/coverage implications. This audit did not fit a model or choose a baseline from game performance.",
              "", "[Kaggle Data](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data) · [NFL 40-yard Combine drill](https://www.nfl.com/news/combine-events-40-yard-dash-09000d5d806ce6bb)"]
    (run/"reference_measurement_audit.md").write_text("\n".join(content)+"\n")
    return dict(real_trials=len(grouped),real_players=diag["recorded_players"],
                dash_trials=z["usable_trials"],candidate_low_windows=sum(len(v) for v in low.values()))
