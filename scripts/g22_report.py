"""Private measurement, repeatability and trajectory review summaries."""
from collections import Counter, defaultdict
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from g22_motion import G21, METRICS, METHODS, match
from movement_probe import atomic_json


def stats(x):
    return dict(n=len(x), bias=float(np.mean(x)), mae=float(np.mean(np.abs(x))),
                rmse=float(np.sqrt(np.mean(np.square(x))))) if len(x) else {"n":0}


def compare(pool, left, right):
    if not left or not right:
        return
    for key in METRICS:
        if left[key] is not None and right[key] is not None:
            pool[key].append(left[key]-right[key])


def synthetic_summary(data):
    detection, errors = defaultdict(Counter), defaultdict(lambda:defaultdict(list))
    clean_oracle = {(r["case"],r["hz"],r["phase"]):r["filtered_oracle"]
                    for r in data["observations"] if r["noise"]=="clean"}
    for r in data["observations"]:
        design = data["design"][r["case"]]
        kind, truth = design["kind"], design["truth"]
        hz, noise = r["hz"], r["noise"]
        scope = f"{kind}/{hz}/{noise}"
        if noise=="clean":
            for raw, filtered, q in zip(r["sampled_oracle"],r["filtered_oracle"],truth):
                compare(errors[f"sampling/{scope}"],raw,q["features"])
                compare(errors[f"filter/{scope}"],filtered,raw)
        else:
            for raw, clean, q in zip(r["noisy_oracle"],clean_oracle[(r["case"],hz,r["phase"])],truth):
                compare(errors[f"noisy_oracle_total/{scope}"],raw,q["features"])
                compare(errors[f"noise_only_fixed_oracle/{scope}"],raw,clean)
        predictions = r["redetected"] | ({"A_legacy":r["legacy"]} if noise=="clean" else {})
        for m,pred in predictions.items():
            pairs = match(pred,truth)
            totals = detection[f"{m}/{scope}"]
            totals.update(dict(trials=1,accepted=int(bool(pred)), truth=len(truth),
                               tp=len(pairs), fp=len(pred)-len(pairs), fn=len(truth)-len(pairs)))
            for i,j in pairs:
                f,q = pred[i]["features"], truth[j]
                compare(errors[f"{m}/total/{scope}"],f,q["features"])
                compare(errors[f"{m}/segmentation/{scope}"],f,r["noisy_oracle"][j])
                errors[f"{m}/timing/{scope}"]["center_s"].append(pred[i]["center"]-q["center"])
            if m in METHODS:
                frozen = r["fixed"][m]
                for i,j in match(pred,frozen):
                    compare(errors[f"{m}/redetect_vs_fixed/{scope}"],pred[i]["features"],frozen[j]["features"])
    return dict(detection=dict(detection),errors={k:{f:stats(v) for f,v in fields.items()}
                                                  for k,fields in errors.items()})


def rank(x):
    _, inverse, counts = np.unique(x,return_inverse=True,return_counts=True)
    return (np.cumsum(counts)-(counts-1)/2)[inverse]


def components(pair):
    within = np.mean(np.diff(pair,axis=1)**2)/2
    between = np.var(pair.mean(axis=1),ddof=1)-within/2
    a,b = map(rank,pair.T)
    rho = np.corrcoef(a,b)[0,1] if a.std() and b.std() else None
    return [within,between,float(np.mean(abs(np.diff(pair,axis=1)))),rho,
            between/(within+between) if within+between>0 else None]


def repeats(trials):
    old = json.loads((G21/"trials.json").read_text())
    groups = defaultdict(list)
    for r in old:
        if r["drill"]=="BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION" and r["base"]:
            groups[r["player"]].append((r["player"],r["event"],r["attempt"]))
    keys = [v[:2] for v in groups.values() if len(v)>=2]
    assert len(keys)==31
    index = {(r["player"],r["event"],r["attempt"]):r for r in trials}
    rng = np.random.default_rng(22023)
    results = {}
    for m in METHODS:
        results[m] = {}
        for metric in METRICS:
            pair = []
            for p in keys:
                values = []
                for k in p:
                    segments = index[k]["methods"][m]
                    f = segments[0]["features"] if len(segments)==1 else None
                    values.append(f[metric] if f else None)
                if all(v is not None for v in values):
                    pair.append(values)
            a = np.asarray(pair)
            fields = ["within","between","mean_abs_difference","spearman","icc"]
            if len(a)<5:
                results[m][metric] = {"pairs":len(a)}
                continue
            if np.ptp(a)<1e-9:
                results[m][metric] = {"pairs":len(a),"constant":True}
                continue
            draws = np.array([[np.nan if x is None else x for x in components(a[ix])]
                              for ix in rng.integers(0,len(a),(2000,len(a)))])
            results[m][metric] = {"pairs":len(a)} | {
                name: {"estimate":est,"interval_95":np.quantile(col[np.isfinite(col)],[.025,.975]).tolist()
                       if np.isfinite(col).any() else None, "defined_bootstrap_draws":int(np.isfinite(col).sum())}
                for name,est,col in zip(fields,components(a),draws.T)}
    return results


def real_summary(trials):
    lookup = {}
    for drill in sorted({r["drill"] for r in trials}):
        rows = [r for r in trials if r["drill"]==drill]
        lookup[drill] = group = {"trials":len(rows)}
        for m in (*METHODS,"legacy"):
            def parts(r):
                return r.get("legacy",[]) if m=="legacy" else r["methods"][m]
            segments = [p for r in rows for p in parts(r) if p["features"]]
            failures = Counter(r["failures"].get(m) or "no_estimate" for r in rows if not parts(r))
            group[m] = dict(accepted=sum(bool(parts(r)) for r in rows),
                            multiple=sum(len(parts(r))>1 for r in rows), failures=dict(failures), features={})
            for metric in METRICS:
                levels=[p["features"][metric] for p in segments if p["features"][metric] is not None]
                result=dict(n=len(levels),median=float(np.median(levels)) if levels else None)
                if m in METHODS:
                    for mode in ("fixed","redetected"):
                        error=[]
                        eligible=0
                        for r in rows:
                            baseline=r.get("fixed_clean",{}).get(m,[]) if mode=="fixed" else parts(r)
                            expected=[p for p in baseline if p["features"] and p["features"][metric] is not None]
                            for v in r.get("variants",[]):
                                eligible+=len(expected)
                                pred=[p for p in v[mode][m] if p["features"] and p["features"][metric] is not None]
                                for i,j in match(pred,expected):
                                    error.append(pred[i]["features"][metric]-expected[j]["features"][metric])
                        med=float(np.median(abs(np.asarray(error)))) if error else None
                        scale=float(np.median(abs(np.asarray(levels)))) if levels else None
                        result[mode]=stats(error)|dict(eligible=eligible,median_abs=med,
                                                       relative=med/scale if med is not None and scale else None)
                group[m]["features"][metric]=result
            group[m]["recovery"]=dict(Counter(s["features"]["recovery_status"] for s in segments))
            group[m]["stage_failure_trials"]=sum(not parts(r) and r["failures"].get(m)!="missing_or_irregular_frames" for r in rows)
            group[m]["data_missing_trials"]=sum(r["failures"].get(m)=="missing_or_irregular_frames" for r in rows)
        group["method_differences"]={}
        for left,right in (("A","B"),("A","C"),("B","C")):
            center, angle=[],[]
            for r in rows:
                x,y = r["methods"][left],r["methods"][right]
                for i,j in match(x,y):
                    center.append(x[i]["center"]-y[j]["center"])
                    if x[i]["features"] and y[j]["features"]:
                        angle.append(x[i]["features"]["angle"]-y[j]["features"]["angle"])
            group["method_differences"][left+right]={"center":stats(center),"angle":stats(angle)}
    return lookup


def examples(run,trials):
    def agreement(r):
        parts=[r["methods"][m] for m in METHODS]
        return all(parts) and max(x[0]["center"] for x in parts)-min(x[0]["center"] for x in parts)<=.2
    categories = {
        "agreement":agreement,
        "disagreement":lambda r:any(r["methods"].values()) and not agreement(r),
        "rejection":lambda r:not any(r["methods"].values()),
        "low_speed":lambda r:r.get("below_floor",False),
        "multiple":lambda r:len(r["methods"]["B"])>1,
    }
    for i,drill in enumerate(sorted({r["drill"] for r in trials}),1):
        categories[f"task_{i}"]=lambda r,task=drill:r["drill"]==task and bool(r["methods"]["A"])
    selected={}
    for name,predicate in categories.items():
        record=next((r for r in trials if "xy" in r and predicate(r)),None)
        selected[name]=record["drill"] if record else None
        if record is None:
            continue
        t,xy=np.asarray(record["t"]),np.asarray(record["xy"])
        xy=xy-xy[0]
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained",sharex="row",sharey="row")
        fig.suptitle(f'{name.title()} · {record["drill"]}')
        for j,m in enumerate(METHODS):
            ax, speed = axes[:,j]
            ax.plot(*xy.T,color=".6")
            speed.plot(record["speed_t"],record["speed"],color=".5")
            ax.set(title=f'Method {m} · {record["failures"][m] or "accepted"}',aspect="equal",adjustable="box",xlabel="Δx (yd)",ylabel="Δy (yd)")
            speed.set(xlabel="Seconds",ylabel="Speed (yd/s)")
            for p in record["methods"][m]:
                a,b,k=(p[key] for key in ("start","end","center"))
                for start,end,color in ((a-.4,a,"C2"),(a,b,"C1"),(b,b+.4,"C0")):
                    path=(t>=start)&(t<=end)
                    ax.plot(*xy[path].T,color=color,linewidth=2)
                    speed.axvspan(start,end,color=color,alpha=.19)
                ax.scatter(*xy[np.argmin(abs(t-k))],color="black",s=20)
                speed.axvline(k,color="black",linestyle=":")
        fig.savefig(run/"figures"/f"motion_{name}.png",dpi=145)
        plt.close(fig)
    return selected


def report(run):
    trials=json.loads((run/"development_trials.json").read_text())
    syn=json.loads((run/"synthetic_truth_results.json").read_text())
    synthetic=synthetic_summary(syn)
    real=real_summary(trials)
    repeat=repeats(trials)
    review=examples(run,trials)
    distances={k:float(np.median([r["dis"][k] for r in trials if r.get("dis",{}).get(k) is not None]))
               for k in next(r["dis"] for r in trials if "dis" in r)}
    atomic_json(run/"motion_summary.json",dict(synthetic=synthetic,real=real,repeat=repeat,
                                               review=review,dis=distances))
    lines=["# E21–E23 Segmentation and validation", "",
           "A legacy is the original G2.1 detector. A uses a roughly ±0.3 s window and the same physical smoothing support as B/C; 5 Hz rounds to 0.4 s. B uses a sustained same-direction change in heading. C requires stable entry/exit headings around B candidates and therefore cannot detect a turn B missed.",
           "The clean oracle comparison separates raw sampled-minus-reference and filtered-minus-raw differences. Detected-minus-filtered oracle includes segmentation and, for A/C, a different angle definition. Fixed noisy intervals use net path angle on the frozen interval, while redetection also allows endpoint and representation changes. Variants are paired within trial; missing estimates are counted rather than imputed.",
           "", "| Method / 10 Hz smooth turns | Noise | Matched / true | False | Angle bias / MAE / RMSE (deg) |",
           "|---|---|---:|---:|---:|"]
    for m in (*METHODS,"A_legacy"):
        for condition in ("clean","iid03","iid10","ar10") if m!="A_legacy" else ("clean",):
            scope=f"smooth/10/{condition}"
            d=synthetic["detection"][f"{m}/{scope}"]
            f=synthetic["errors"].get(f"{m}/total/{scope}",{}).get("angle")
            numbers=" / ".join(f"{f[k]:.1f}" for k in ("bias","mae","rmse")) if f else "—"
            lines.append(f"| {m} | {condition} | {d['tp']} / {d['truth']} | {d['fp']} | {numbers} |")
    lines+=["","All noise, sampling rates, edge cases, timing errors, raw estimates and integration checks are preserved in JSON. The continuous arc has no discrete event and is reported separately when judging false detections.",
            "", "| Real drill | Total | A | B | C | Legacy |", "|---|---:|---:|---:|---:|---:|"]
    for drill,v in real.items():
        lines.append(f"| {drill} | {v['trials']} | {v['A']['accepted']} | {v['B']['accepted']} | {v['C']['accepted']} | {v['legacy']['accepted']} |")
    lines+=["","Per-method refusal counts, multi-turn counts, paired differences, phase features and perturbation acceptance denominators are in motion_summary.json.",
            "Figures mark entry (green), detected turn (orange), exit (blue) and center (black), for human review.",
            "Review categories: "+", ".join(f"{k}={v or 'absent'}" for k,v in review.items())+".",
            "", "## Official dis diagnosis", "",
            "Trial-median summaries, in yards except the ratio: "+", ".join(f"{k}={v:.3f}" for k,v in distances.items())+".",
            "The discrepancy is unresolved. A fitted ratio is descriptive and does not validate a unit conversion. The vendor processing, axis conventions across trials, and precise displacement definition require independent documentation."]
    (run/"segmentation_comparison.md").write_text("\n".join(lines)+"\n")
    lines=["# DB 45° paired repeatability","","The 31 original G2.1 player pairs are the denominator. Retain pairs with exactly one usable segment in both attempts. The moment decomposition assumes exchangeable repeats, equal residual variance and no systematic order or side effect. A negative between-player estimate indicates unresolved between-player variance. Bootstrap intervals resample matched players, with 2,000 draws.",
           "", "| Method | Metric | Pairs | Within variance | Between estimate | Mean absolute difference | Spearman [95%] | ICC [95%] |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for method, fields in repeat.items():
        for name,v in fields.items():
            def cell(key,interval=False):
                value=v.get(key,{})
                est=value.get("estimate")
                ci=value.get("interval_95")
                return ("—" if est is None else f"{est:.3g}")+(f" [{ci[0]:.2g}, {ci[1]:.2g}]" if interval and ci else "")
            lines.append(f"| {method} | {name} | {v['pairs']} | {cell('within')} | {cell('between')} | {cell('mean_abs_difference')} | {cell('spearman',True)} | {cell('icc',True)} |")
    lines+=["","Bootstrap intervals for variance components and absolute differences are also in motion_summary.json. Constant durations, including A’s imposed duration, have undefined reliability. Small or selected complete-case subsets cannot support a WR/TE ability ranking."]
    (run/"repeatability_report.md").write_text("\n".join(lines)+"\n")
