"""Analytical and outcome-free multiscale measurement checks."""
from collections import defaultdict
from math import pi,sqrt
from pathlib import Path
import json

import numpy as np
from scipy.integrate import cumulative_trapezoid

from g23_motion import fit
from movement_probe import digest,atomic_json

G3A=Path("data/g3/g3a-20261127")
G22=Path("data/g2/g2-2-20261127")
METHODS=("d01","d05","d10","quadratic_derivative")
FEATURES=("coarse_1s_chord_ratio","half_second_late_minus_early")
NOISES={"clean":(0,False),"iid03":(.03,False),"iid10":(.10,False),"ar10":(.10,True)}
CASES=("stationary","slow","v080","v095","v105","v120","straight","turn","decel","reverse")


def history():
    cp=json.loads((G3A/"checkpoint.json").read_text())
    for name in ("measurement_results.json","feature_freeze.json","protocol.md"):
        if digest(G3A/name)!=cp["deliverables"][name]:
            raise ValueError("Historical measurement changed")
    return json.loads((G3A/"measurement_results.json").read_text()),json.loads((G3A/"feature_freeze.json").read_text())


def analytic(run):
    old,freeze=history()
    h=np.arange(-2,3)
    design=np.c_[np.ones(5),h*.1,(h*.1)**2/2]
    meanweight,slopeweight=np.linalg.pinv(design)[:2]
    weights=np.array([-3,12,17,12,-3])/35
    difference=np.r_[0,weights]-np.r_[weights,0]
    factor=float(difference@difference)
    np.testing.assert_allclose(meanweight,weights,atol=1e-14)
    np.testing.assert_allclose(factor,518/1225)
    # Exercise the actual code instead of relying only on the Vandermonde derivation.
    t=np.arange(9)*.1
    actual=[fit(t,np.c_[np.eye(9)[:,k],np.zeros(9)],"local_quadratic")[1][0,0] for k in range(5)]
    np.testing.assert_allclose(actual,weights,atol=1e-14)
    rayleigh=sqrt(pi/2)*.10
    expected={"raw_interior":36*sqrt(2)*rayleigh,"quadratic":36*sqrt(factor)*rayleigh}
    comparison={}
    for m,e in expected.items():
        o=next(x for x in old["synthetic"] if x["case"]=="stationary" and x["noise"]==.10 and x["method"]==m)
        comparison[m]=dict(theoretical=e,historical_mean=o["length_bias"],historical_acceptance=o["accepted"],
                           historical_trials=o["trials"])
    result=dict(weights=weights.tolist(),difference_weights=difference.tolist(),diff_variance_factor=factor,
                derivative_weights=slopeweight.tolist(),comparison=comparison,
                historical_failed={k:[n for n,v in a["details"].items() if not v] for k,a in freeze["checks"].items()})
    atomic_json(run/"analytical_results.json",result)
    (run/"analytical_bias_verification.md").write_text(f"""# Analytical bias verification

With location errors iid N₂(0, σ²I), each adjacent difference has independent axes with variance 2σ². Its norm is Rayleigh with scale √2 σ. Linearity of expectation over 36 intervals gives 36 σ√π = **{expected['raw_interior']:.6f} yd** at σ=.10 yd. Overlapping differences are correlated, so this formula makes no independence claim for the total's variance.

The fitted center intercept has weights [-3,12,17,12,-3]/35. An adjacent difference has coefficients [3,-15,-5,5,15,-3]/35, giving single-axis variance 518σ²/1225. The Rayleigh mean gives 36 σ√(518/1225) √(π/2) = **{expected['quadratic']:.6f} yd**. The sign of the difference coefficients reverses if the order of subtraction reverses.

Sealed G3A means were {comparison['raw_interior']['historical_mean']:.3f} and {comparison['quadratic']['historical_mean']:.3f} yd. This check reused those simulations. G3A summed distances of adjacent reconstructed positions and divided each by Δt. The G2.3 fitted derivative is a different linear operator with weights [-2,-1,0,1,2]/(10Δt) for the same evenly spaced five points. Its magnitude has noise bias too.

The original 1 yd/s, ten-interval and ≤5% gates remain historical. Both candidates failed only synthetic_controls. Trial and player coverage, positive IQR, all perturbation/trim retention and stability checks passed. The 0.6 yd/s control moved by construction and must not be described as stationary. No historical flag changes.
""")
    return dict(historical_experiments_recomputed=0,analytical_checks=3)


def ready(t,xy):
    return len(t)>=6 and np.isfinite(xy).all() and np.all(abs(np.diff(t)-.1)<=.001)


def native(t,xy,method):
    if not ready(t,xy):
        return np.array([]),np.array([])
    if method=="quadratic_derivative":
        q,_,vel,_=fit(t,xy,"local_quadratic")
        return q,np.linalg.norm(vel,axis=1)
    stride={"d01":1,"d05":5,"d10":10}[method]
    dt=t[stride:]-t[:-stride]
    return (t[stride:]+t[:-stride])/2,np.linalg.norm(xy[stride:]-xy[:-stride],axis=1)/dt


def features(t,xy):
    """Construct-defined endpoint chord sums on the first 3.0 observed seconds."""
    output=dict.fromkeys(FEATURES)
    if not ready(t,xy) or t[-1]-t[0]<3.-1e-8:
        return output
    q=t[0]+np.arange(7)*.5
    node=np.column_stack([np.interp(q,t,x) for x in xy.T])
    d=np.linalg.norm(np.diff(node,axis=0),axis=1)
    coarse=np.linalg.norm(np.diff(node[::2],axis=0),axis=1).sum()
    if coarse>=2:
        output[FEATURES[0]]=float(np.linalg.norm(node[-1]-node[0])/coarse)
    if d.sum()>=2:
        output[FEATURES[1]]=float((d[3:].mean()-d[:3].mean())/.5)
    return output


def dense(case):
    t=np.linspace(0,4.1,4101)
    levels={"stationary":0.,"slow":.6,"v080":.8,"v095":.95,"v105":1.05,"v120":1.2,"straight":4.}
    h=np.zeros_like(t)
    if case in levels:
        signed=np.full_like(t,levels[case])
    elif case=="turn":
        u=np.clip(t-1.5,0,1)
        h=(pi/2)*(3*u*u-2*u**3)
        signed=np.full_like(t,4.)
    elif case=="decel":
        signed=4-3.4*np.exp(-((t-2.)/.36)**2)
    elif case=="reverse":
        signed=-3*np.tanh((t-2.)/.18)
    else:
        raise ValueError(case)
    velocity=signed[:,None]*np.c_[np.cos(h),np.sin(h)]
    xy=cumulative_trapezoid(velocity,t,initial=0,axis=0)
    arc=cumulative_trapezoid(abs(signed),t,initial=0)
    return t,xy,abs(signed),arc


def jitter(rng,shape,sd,correlated):
    e=rng.normal(0,sd,shape)
    if correlated:
        rho=np.exp(-.1/.25)
        for i in range(1,len(e)):
            e[i]=rho*e[i-1]+sqrt(1-rho*rho)*e[i]
    return e


def wilson(k,n):
    if not n:
        return None
    z=1.959964
    p=k/n
    mid=(p+z*z/(2*n))/(1+z*z/n)
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0,mid-half),min(1,mid+half)]


def average(values):
    v=np.asarray(values,dtype=float)
    if not len(v):
        return dict(n=0,mean=None,mcse=None,interval=None)
    m=float(v.mean())
    se=float(v.std(ddof=1)/sqrt(len(v))) if len(v)>1 else None
    return dict(n=len(v),mean=m,mcse=se,interval=[m-1.96*se,m+1.96*se] if se is not None else None)


def common_eval(t,xy,method,grid,physical,clean_xy,ref_t,ref_xy):
    q,v=native(t,xy,method)
    _,v0=native(t,clean_xy,method)
    keep=(grid>=q[0]-1e-8)&(grid<=q[-1]+1e-8)
    filtered=np.interp(grid[keep],q,v)
    p=physical[keep]
    delta=filtered-p
    label=p>=1
    prediction=filtered>=1
    def errors(mask,err):
        return float(err[mask].mean()) if mask.any() else None
    def shift(x):
        return np.column_stack([np.interp(x,ref_t,d) for d in ref_xy.T])
    if method!="quadratic_derivative":
        width={"d01":.1,"d05":.5,"d10":1.}[method]
        chord_ref=np.linalg.norm(shift(q+width/2)-shift(q-width/2),axis=1)/width
        chord_mse=float(np.mean((v-chord_ref)**2))
        chord_bias=float(np.mean(v-chord_ref))
    else:
        chord_mse=chord_bias=None
    return dict(bias=float(np.mean(delta)),mse=float(np.mean(delta*delta)),
                clean_target_bias=float(np.mean(np.interp(grid[keep],q,v0)-p)),
                chord_mse=chord_mse,chord_bias=chord_bias,
                stationary_decision=bool(np.median(v)<=.30),
                false_positive=errors(~label,prediction),false_negative=errors(label,~prediction),
                positives=int(label.sum()),negatives=int((~label).sum()),
                native=len(q),native_coverage=len(q)/len(t),common_coverage=float(keep.mean()))


def sensitivity(rng, t, xy, sd, ar, count):
    return [xy+jitter(rng,xy.shape,sd,ar) for _ in range(count)]


def simulation():
    rng=np.random.default_rng(31101)
    speed_out=[]
    construct_out=[]
    for case in CASES:
        rt,rxy,rs,arc=dense(case)
        sample=lambda q:np.column_stack([np.interp(q,rt,v) for v in rxy.T])
        for phase in (0,.05):
            t=np.arange(41)*.1+phase
            x=sample(t)
            grid=t[5:-5]
            physical=np.interp(grid,rt,rs)
            clean=features(t,x)
            ends=t[0]+np.array([0,1.5,3.])
            distance=np.diff(np.interp(ends,rt,arc))
            target_geo=np.linalg.norm(sample(ends)[-1]-sample(ends)[0])/distance.sum() if distance.sum() else None
            target_velocity=(distance[1]-distance[0])/1.5
            trimmed=features(t[2:-2],x[2:-2])
            for label,(sd,ar) in NOISES.items():
                paths=sensitivity(rng,t,x,sd,ar,120) if sd else [x]
                candidate=[features(t,p) for p in paths]
                for k in FEATURES:
                    values=[z[k] for z in candidate if z[k] is not None]
                    ref=clean[k]
                    dif=np.asarray(values)-ref if ref is not None else np.array([])
                    construct_out.append(dict(case=case,phase=phase,noise=label,feature=k,
                        n=len(paths),available=len(values),coverage_ci=wilson(len(values),len(paths)),
                        clean_coarse=ref,physical_comparator=target_geo if k==FEATURES[0] else target_velocity,
                        trimmed_clean=trimmed[k],bias=average(dif.tolist()),p90_abs=float(np.quantile(abs(dif),.9)) if len(dif) else None,
                        rmse=float(np.sqrt(np.mean(dif*dif))) if len(dif) else None))
                for m in METHODS:
                    a=[common_eval(t,p,m,grid,physical,x,rt,rxy) for p in paths]
                    stat=sum(p["stationary_decision"] for p in a)
                    result=dict(case=case,phase=phase,noise=label,method=m,n=len(a),
                                stationary_calls=stat,stationary_call_ci=wilson(stat,len(a)),
                                positives=a[0]["positives"],negatives=a[0]["negatives"])
                    for key in ("bias","mse","clean_target_bias","chord_mse","chord_bias","false_positive",
                                "false_negative","native_coverage","common_coverage"):
                        result[key]=average([p[key] for p in a if p[key] is not None])
                    result["rmse"]=sqrt(result["mse"]["mean"])
                    result["chord_rmse"]=sqrt(result["chord_mse"]["mean"]) if result["chord_mse"]["n"] else None
                    speed_out.append(result)
    return speed_out,construct_out


def real(run):
    cp=json.loads((G22/"checkpoint.json").read_text())
    old=G22/"development_trials.json"
    if digest(old)!=cp["stages"]["real"]["artifacts"][old.name]:
        raise ValueError("Development coordinates changed")
    trials=json.loads(old.read_text())
    rng=np.random.default_rng(31102)
    groups=defaultdict(list)
    for r in trials:
        t,xy=np.asarray(r.get("t",[])),np.asarray(r.get("xy",[]))
        base=features(t,xy)
        draw={k:[features(t,z) for z in sensitivity(rng,t,xy,sd,ar,5)]
              for k,(sd,ar) in NOISES.items() if sd}
        draw["trim"]=[features(t[2:-2],xy[2:-2])]
        groups[r["drill"]].append((r["player"],base,draw))
    output={}
    for group,records in sorted(groups.items()):
        entries={}
        for k in FEATURES:
            valid=[(p,f,d) for p,f,d in records if f[k] is not None]
            changes={}
            for mode in ("iid03","iid10","ar10","trim"):
                differences=[]
                retained=0
                potential=0
                for _,f,d in valid:
                    ds=[abs(q[k]-f[k]) for q in d[mode] if q[k] is not None]
                    retained+=len(ds)
                    potential+=len(d[mode])
                    if ds:
                        differences.append(float(np.median(ds)))
                changes[mode]=dict(retained=retained,potential=potential,paired_trials=len(differences),
                    median=float(np.median(differences)) if differences else None,
                    p90=float(np.quantile(differences,.9)) if differences else None)
            entries[k]=dict(trials=len(records),players=len({p for p,_,_ in records}),valid_trials=len(valid),
                            valid_players=len({p for p,_,_ in valid}),changes=changes)
        output[group]=entries
    return output,len(trials),len({r["player"] for r in trials})


def measurement(run):
    speed,geometry=simulation()
    natural,n,n_player=real(run)
    checks={}
    curl=natural["CURL_ROUTE_RIGHT"]
    for k,bound,tolerance in zip(FEATURES,(.07,.50),(.10,.70)):
        cells=[x for x in geometry if x["feature"]==k and x["case"] in ("straight","turn","decel","reverse")]
        r=curl[k]
        assertions=dict(
            synthetic_coverage=all(x["available"]/x["n"]>=.95 for x in cells),
            synthetic_p90=all(x["p90_abs"] is not None and x["p90_abs"]<=bound for x in cells),
            stationary_spurious=all(x["available"]/x["n"]<=.05 for x in geometry if x["feature"]==k and x["case"]=="stationary"),
            real_trial_coverage=r["valid_trials"]/r["trials"]>=.8,
            real_player_coverage=r["valid_players"]/r["players"]>=.8)
        for mode in ("iid10","ar10","trim"):
            a=r["changes"][mode]
            assertions[mode+"_retention"]=a["potential"]>0 and a["retained"]/a["potential"]>=.8
            assertions[mode+"_p90"]=a["p90"] is not None and a["p90"]<=tolerance
        checks[k]=dict(conditions=assertions,conditional_pass=all(assertions.values()),
            max_simulation_p90=max(x["p90_abs"] for x in cells if x["p90_abs"] is not None),
            simulation_p90_limit=bound,real_p90_limit=tolerance,
            failed_cells=[{q:x[q] for q in ("case","phase","noise","p90_abs","available","n")} for x in cells
                          if x["available"]/x["n"]<.95 or x["p90_abs"] is None or x["p90_abs"]>bound])
    speeds={}
    for m in METHODS:
        s=[r for r in speed if r["method"]==m]
        stationary=[]; attainment=[]
        for r in s:
            if r["case"] in ("slow","stationary"):
                rate=r["stationary_calls"]/r["n"]
                stationary.append((1-rate if r["case"]=="stationary" else rate)<=.05)
            if r["case"] in ("stationary","slow","v080","v120","straight"):
                field="false_negative" if r["case"] in ("v120","straight") else "false_positive"
                attainment.append(r[field]["mean"]<=.10)
        speeds[m]=dict(static_pass=all(stationary),attainment_pass=all(attainment),dynamic_phase_status="PENDING")
    report=dict(speed=speed,geometry=geometry,real=natural,checks=checks,speed_checks=speeds,
                trial_count=n,player_count=n_player)
    atomic_json(run/"multiscale_results.json",report)
    render(run,report)
    return dict(synthetic_paths_per_case_phase=361,synthetic_paths_total=361*len(CASES)*2,
                real_trials=n,real_players=n_player,
                conditional_features=sum(x["conditional_pass"] for x in checks.values()))


def render(run,r):
    lines=["# Multiscale measurement report","",
           "10 Hz acquisition, two sampling phases and independent/AR(1) jitter are separate cells. Endpoint displacement rate uses a chord divided by physical span. Physical instantaneous speed is a different reference, especially on turns, deceleration and reversals. Analysis time support is a common .5 s guard; native coverage is count of estimates divided by recorded frames. Both support comparisons are retained.",
           "", "| Scenario, iid .10 and phase 0 | Method | Physical bias / RMSE (yd/s) | Chord target RMSE | FPR / FNR | Native coverage |",
           "|---|---|---:|---:|---|---:|"]
    for x in r["speed"]:
        if x["noise"]!="iid10" or x["phase"] or x["case"] not in ("stationary","slow","v095","v105","turn","decel","reverse"):
            continue
        def fmt(v):
            return f"{v:.3f}" if v is not None else "—"
        lines.append(f"| {x['case']} | {x['method']} | {x['bias']['mean']:+.3f} / {x['rmse']:.3f} | {fmt(x['chord_rmse'])} | {fmt(x['false_positive']['mean'])} / {fmt(x['false_negative']['mean'])} | {x['native_coverage']['mean']:.1%} |")
    lines += ["","The full JSON reports absolute frame-category denominators, individual estimated rates across independent trajectories, path-level Monte Carlo SE and intervals for bias, MSE and error rates. Static decisions and feature availability also have Wilson intervals at independent path level. One noiseless realization has no sampling interval. Frames are not independent experimental replicates.",
              "", "| Method | Stationary discrimination screen | Sustained speed attainment screen |", "|---|---|---|"]
    for m,x in r["speed_checks"].items():
        lines.append(f"| {m} | {x['static_pass']} | {x['attainment_pass']} |")
    lines+=["","A slow case is real motion and belongs in the below-1 yd/s category. A static whole-recording rule is not a pointwise motion detector. Long spans reduce stationary noise but may suppress reversals or hide a dip in speed; clearest comparisons are the clean-target-bias and dynamic error fields.",
            "", "| Real 2023 task | Candidate | Valid trials / total | Valid players / total | P90 iid / AR / trim change |",
            "|---|---|---:|---:|---|"]
    for task,d in r["real"].items():
        for k,v in d.items():
            context=" / ".join(f"{v['changes'][c]['p90']:.3f}" if v['changes'][c]['p90'] is not None else "—"
                               for c in ("iid10","ar10","trim"))
            lines.append(f"| {task} | {k} | {v['valid_trials']}/{v['trials']} | {v['valid_players']}/{v['players']} | {context} |")
    lines+=["","First 3 s are defined from the observed origin. Edge trim shifts the construct and measures support sensitivity as well as estimator sensitivity. Fine-versus-coarse comparators are retained per clean scenario and phase in JSON; short-distance denominators are rejected rather than stabilized silently."]
    (run/"multiscale_measurement_report.md").write_text("\n".join(lines)+"\n")
    a=["# R1 candidate feature freeze","",
       "G3A freeze stays unchanged. Candidate status here is conditional to the prespecified simulations and 2023 sensitivity; any candidate that passes still waits for independent calibration before model eligibility.",
       "", "| Candidate | R1 conditional certificate | Worst supported simulated P90 | Failed checks |",
       "|---|---|---:|---|"]
    for k,v in r["checks"].items():
        a.append(f"| {k} | {'PASS conditional' if v['conditional_pass'] else 'FAIL'} | {v['max_simulation_p90']:.4f} | "+
                 (", ".join(c for c,b in v["conditions"].items() if not b) or "—")+" |")
    a+=["","Each boolean and failing cell is in multiscale_results.json. The two pre-registered candidates are the entire search. The R1 conditional set is distinct from the empty set authorized for modeling. Artificial noise lacks device calibration, actual recording origins are undocumented, and independent path review is pending. No threshold revisions based on results."]
    (run/"candidate_feature_freeze.md").write_text("\n".join(a)+"\n")
