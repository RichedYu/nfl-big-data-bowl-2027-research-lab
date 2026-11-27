"""Outcome-free geometry and velocity validation for G3A."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from g23_motion import fit, truth
from g23_run import G22, verify_artifacts
from movement_probe import atomic_json, digest

G24=Path("data/g2/g2-4-20261127")
G23=Path("data/g2/g2-3-20261127")
TASK="CURL_ROUTE_RIGHT"
FEATURES=("chord_ratio","velocity_contrast")
METHODS=("raw_full","raw_interior","quadratic")


def describe(t,xy,method):
    if len(t)<15 or not np.isfinite(xy).all() or np.any(abs(np.diff(t)-.1)>.001):
        return {"accepted":False,"reason":"insufficient_or_irregular_frames"}
    if method=="quadratic":
        t,xy,*_=fit(t,xy,"local_quadratic")
    elif method=="raw_interior":
        t,xy=t[2:-2],xy[2:-2]
    step=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    arc=step.sum()
    speeds=step/np.diff(t)
    moving=speeds>=1.
    result=dict(duration=float(t[-1]-t[0]),length=float(arc),moving_fraction=float(moving.mean()),
                moving_intervals=int(moving.sum()),accepted=bool(moving.sum()>=10 and arc>0))
    if not result["accepted"]:
        return result|dict(reason="too_few_moving_intervals")
    p=(np.cumsum(step)-step/2)/arc
    bands=[speeds[moving & (p>=i/3) & (p<(i+1)/3)] for i in range(3)]
    values=[float(np.median(x)) if len(x)>=3 else None for x in bands]
    return result|dict(chord_ratio=float(np.linalg.norm(xy[-1]-xy[0])/arc),band_speed=values,
                       velocity_contrast=values[2]-values[0] if values[2] is not None and values[0] is not None else None)


def summaries(records):
    out=dict(trials=len(records),accepted=sum(x["accepted"] for x in records))
    for k in ("length","moving_fraction","chord_ratio","velocity_contrast"):
        v=[x[k] for x in records if x.get(k) is not None]
        out[k]={"n":len(v),"median":float(np.median(v)) if v else None}
    out["reason"]={k:sum(x.get("reason")==k for x in records)
                   for k in sorted({x["reason"] for x in records if "reason" in x})}
    if records and "players" in records[0]:
        out["players"]=len({x["players"] for x in records})
        out["valid_players"]=len({x["players"] for x in records if x["accepted"]})
    return out


def synthetic():
    rng=np.random.default_rng(31001)
    grid=np.arange(41)*.1
    ref_t=np.arange(4001)*.001
    results=[]
    for name,speed,turns in (("stationary",0,[]),("slow",.6,[]),("straight",4,[]),
                             ("smooth_turn",4,[(1.5,1.,90,0)])):
        ref,*_=truth(ref_t,speed,turns)
        p=ref[::100]
        for sd in (0,.03,.10):
            jitter=[p] if sd==0 else [p+rng.normal(0,sd,p.shape) for _ in range(100)]
            for method in METHODS:
                rows=[describe(grid,xy,method) for xy in jitter]
                length=speed*(4 if method=="raw_full" else 3.6)
                error=np.asarray([r["length"]-length for r in rows])
                results.append(dict(case=name,noise=sd,method=method,truth_length=length,
                                    length_bias=float(error.mean()),length_rmse=float(np.sqrt(np.mean(error**2))),
                                    **summaries(rows)))
    return results


def level_stats(samples):
    a=np.asarray(samples)
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a,.9)) if len(a) else None)


def measurement(run):
    g24=json.loads((G24/"checkpoint.json").read_text())
    verify_artifacts(G24,g24["files"])
    g23=json.loads((G23/"checkpoint.json").read_text())
    if digest(G23/"checkpoint.json")!=g24["source_checkpoint_sha256"]:
        raise ValueError("G2.4 source binding changed")
    verify_artifacts(Path("."),g23["source_artifacts"])
    g22=json.loads((G22/"checkpoint.json").read_text())
    verify_artifacts(G22,g22["stages"]["real"]["artifacts"])
    old=json.loads((G22/"development_trials.json").read_text())
    assert len(old)==193
    syn=synthetic()
    def stationary_gate(r):
        if r["case"] in ("stationary","slow"):
            limit=0 if r["noise"]==0 else .05
            return r["accepted"]/r["trials"]<=limit
        return r["accepted"]==r["trials"]
    controls=all(stationary_gate(r) for r in syn
                 if r["method"]=="quadratic" and
                 ((r["case"] in ("stationary","slow") and r["noise"] in (0,.10)) or
                  (r["case"]=="straight" and r["noise"]==0)))
    rng=np.random.default_rng(31002)
    trial_records=[]
    for r in old:
        t,xy=np.asarray(r.get("t",[])),np.asarray(r.get("xy",[]))
        record=dict(drill=r["drill"],players=r["player"],event=r["event"],pred={},
                    perturbations={})
        for m in METHODS:
            record["pred"][m]=describe(t,xy,m)
        record["perturbations"]={mode:{m:[describe(tt,points,m) for points in variants] for m in METHODS}
                                  for mode,tt,variants in (
                                      ("iid03",t,[xy+rng.normal(0,.03,xy.shape) for _ in range(5)]),
                                      ("iid10",t,[xy+rng.normal(0,.10,xy.shape) for _ in range(5)]),
                                      ("trim",t[2:-2],[xy[2:-2]]))}
        trial_records.append(record)
    summary={}
    freeze=[]
    checks={}
    for task in sorted({r["drill"] for r in trial_records}):
        rows=[r for r in trial_records if r["drill"]==task]
        summary[task]={}
        for m in METHODS:
            base=[r["pred"][m]|{"players":r["players"]} for r in rows]
            variant_summary={}
            for label in ("iid03","iid10","trim"):
                metrics={}
                for k in ("length","moving_fraction",*FEATURES):
                    err=[]
                    signed=[]
                    pairs=eligible=0
                    for r in rows:
                        original=r["pred"][m].get(k)
                        if original is None:
                            continue
                        batch=r["perturbations"][label][m]
                        delta=[x[k]-original for x in batch if x.get(k) is not None]
                        pairs+=len(delta)
                        eligible+=len(batch)
                        if delta:
                            err.append(float(np.median(abs(np.asarray(delta)))))
                            signed.append(float(np.median(delta)))
                    metrics[k]=level_stats(err)|dict(pairs=pairs,possible_pairs=eligible,
                                                      median_signed=float(np.median(signed)) if signed else None)
                perturbed=[x for r in rows for x in r["perturbations"][label][m]]
                variant_summary[label]=dict(metrics=metrics,acceptance=summaries(perturbed))
            summary[task][m]=summaries(base)|dict(variants=variant_summary)
            if task!=TASK or m!="quadratic":
                continue
            for k,limit in zip(FEATURES,(.10,1.25)):
                grouped=defaultdict(list)
                for r in base:
                    if r.get(k) is not None:
                        grouped[r["players"]].append(r[k])
                iqr=float(np.ptp(np.quantile([np.mean(v) for v in grouped.values()],[.25,.75]))) if grouped else 0
                clean=[r for r in base if r.get(k) is not None]
                details={"synthetic_controls":controls,"trial_coverage":len(clean)/len(rows)>=.8,
                         "player_coverage":len(grouped)/len({r["players"] for r in rows})>=.8,
                         "iqr_positive":iqr>0}
                for v in ("iid10","trim"):
                    q=variant_summary[v]["metrics"][k]
                    details[v+"_retention"]=q["possible_pairs"]>0 and q["pairs"]/q["possible_pairs"]>=.8
                    details[v+"_p90"]=q["p90"] is not None and q["p90"]<=limit
                    if v=="iid10":
                        details["noise_over_iqr"]=q["median"] is not None and q["median"]<=.5*iqr
                passed=all(details.values())
                checks[k]=dict(passed=passed,details=details,iqr=iqr,trials=len(clean),players=len(grouped),
                               p90_limit=limit)
                if passed:
                    freeze.append(k)
    paired=[]
    for task in summary:
        rows=[r for r in trial_records if r["drill"]==task]
        for k in ("length","moving_fraction",*FEATURES):
            a=[r["pred"]["quadratic"][k]-r["pred"]["raw_interior"][k] for r in rows
               if r["pred"]["quadratic"].get(k) is not None and r["pred"]["raw_interior"].get(k) is not None]
            paired.append(dict(drill=task,metric=k,quadratic_minus_raw=level_stats(a)))
    atomic_json(run/"measurement_results.json",dict(synthetic=syn,controls_pass=controls,summary=summary,
                                                      paired=paired,trial_records=trial_records))
    atomic_json(run/"feature_freeze.json",dict(features=freeze,checks=checks,task=TASK,estimator="quadratic",
                                                protocol_sha256=digest(run/"protocol.md"),
                                                measurement_sha256=digest(run/"measurement_results.json")))
    out=["# Feature freeze · before outcome access","",
         "Fixed task: Curl Route Right. Local features: PENDING. The 2023 trajectory patches and acceptance limits were applied before reading outcomes.",
         "Frozen overall features: "+(", ".join(freeze) or "none. M2 stops by protocol.")+".",
         "", "| Feature | Valid trials / players | Between-player IQR | Screen | Failed checks |",
         "|---|---:|---:|---|---|"]
    for k,v in checks.items():
        out.append(f"| {k} | {v['trials']} / {v['players']} | {v['iqr']:.4f} | {'pass' if v['passed'] else 'fail'} | "+
                   ", ".join(p for p,f in v["details"].items() if not f)+" |")
    out+=["","A pass permits an exploratory comparison and does not establish repeatability, physical stage validity or generalization. No substitute feature enters if a candidate fails."]
    (run/"feature_freeze.md").write_text("\n".join(out)+"\n")
    out=["# Representation validation","",
         "Matched-interior raw coordinate segments and reconstructed local-quadratic coordinate segments share the same frame support. Raw full-recording rows account for support effects. Arc length is approximated by summing chords on the reconstructed 10 Hz coordinates, and segment speeds use their actual elapsed time. Simulated jitter affects both distance and the count exceeding the moving threshold.",
         "", "| Synthetic truth, 0.10 yd jitter | Method | Length bias / RMSE (yd) | Moving fraction median | Accepted |",
         "|---|---|---:|---:|---:|"]
    for r in syn:
        if r["noise"]==.10 and r["method"]!="raw_full":
            out.append(f"| {r['case']} | {r['method']} | {r['length_bias']:.3f} / {r['length_rmse']:.3f} | {r['moving_fraction']['median']:.2f} | {r['accepted']}/{r['trials']} |")
    out+=["","| Task / method | Raw trials / players | Valid trials / players | Baseline length median (yd) | Moving fraction | .10 yd jitter accepted | Trim accepted |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for task,ms in summary.items():
        for m,v in ms.items():
            a=v["variants"]["iid10"]["acceptance"]
            b=v["variants"]["trim"]["acceptance"]
            out.append(f"| {task} / {m} | {v['trials']} / {v['players']} | {v['accepted']} / {v['valid_players']} | {v['length']['median']:.2f} | {v['moving_fraction']['median']:.2f} | {a['accepted']}/{a['trials']} | {b['accepted']}/{b['trials']} |")
    out += ["","Jitter and trim denominators are trial×perturbation and trial, respectively. The JSON contains size-matched paired changes, per-band speeds, path geometry, reasons and both perturbation levels. Adding coordinate noise changes measured chord sums even when the underlying trajectory is fixed. Edge trimming changes which motion is visible. Arc-length thirds are geometric bins; physiological or route-phase equivalence is pending."]
    (run/"representation_validation.md").write_text("\n".join(out)+"\n")
    return dict(trials=len(old),players=len({r["player"] for r in old}),
                synthetic_paths=4*201,features=len(freeze))
