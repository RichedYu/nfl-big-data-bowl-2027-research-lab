"""Measurement audit, blind packet and independent synthetic estimator evaluation."""
from collections import Counter, defaultdict
import csv
from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import linear_sum_assignment

from g22_motion import detect, noise
from movement_probe import atomic_json, kinematics, wrap

G22 = Path("data/g2/g2-2-20261127")
METHODS = ("moving_mean", "local_quadratic", "spline")
GATES = (20, 35, 50)


def corner_truth(t, start, duration, degrees):
    """A finite constant-rate surrogate. A true instantaneous corner has no ordinary acceleration there."""
    u = np.clip((t-start)/duration, 0, 1)
    theta = np.radians(degrees)
    rate = np.where((t>start)&(t<start+duration),theta/duration,0.)
    return theta*u, rate


def truth(t, speed, turns):
    h = np.zeros_like(t)
    hdot = np.zeros_like(t)
    s = np.full_like(t,float(speed))
    sd = np.zeros_like(t)
    for a,d,angle,drop in turns:
        u = np.clip((t-a)/d,0,1)
        h += np.radians(angle)*(3*u*u-2*u**3)
        hdot += np.radians(angle)*6*u*(1-u)/d
        s -= drop*np.sin(np.pi*u)**2
        sd -= drop*np.pi*np.sin(2*np.pi*u)/d
    e = np.c_[np.cos(h),np.sin(h)]
    v = s[:,None]*e
    a = sd[:,None]*e + (s*hdot)[:,None]*np.c_[-e[:,1],e[:,0]]
    xy = np.vstack([np.zeros(2),np.cumsum(np.diff(t)[:,None]*(v[:-1]+v[1:])/2,axis=0)])
    return xy,v,a,h


def fit(t,xy,method):
    dt = np.median(np.diff(t))
    half = max(1,round(.2/dt))
    # Exclude the same edge region for scoring. Baseline derivatives use centered
    # derivatives throughout that region wherever the stencil fits.
    if method=="moving_mean":
        pos=np.column_stack([np.convolve(x,np.ones(2*half+1)/(2*half+1),mode="valid") for x in xy.T])
        grid=t[half:-half]
        v=np.gradient(pos,grid,axis=0,edge_order=2)
        a=np.gradient(v,grid,axis=0,edge_order=2)
        return grid,pos,v,a
    grid=t[half:-half]
    if method=="local_quadratic":
        z=(np.arange(2*half+1)-half)*dt
        weights=np.linalg.pinv(np.c_[np.ones(len(z)),z,z*z/2])
        p,v,a=(np.asarray([weights[i]@xy[j-half:j+half+1]
                           for j in range(half,len(t)-half)]) for i in range(3))
        return grid,p,v,a
    spl=[UnivariateSpline(t,x,k=3,s=len(t)*.05**2) for x in xy.T]
    estimates=[np.column_stack([f.derivative(d)(grid) if d else f(grid) for f in spl]) for d in range(3)]
    return grid,*estimates


def curves(fit_result):
    t,p,v,a=fit_result
    s=np.linalg.norm(v,axis=1)
    safe=np.where(s>=1,s,np.nan)
    cross=v[:,0]*a[:,1]-v[:,1]*a[:,0]
    return t,p,s,np.sum(v*a,axis=1)/safe,cross/safe,cross/safe**3,np.arctan2(v[:,1],v[:,0])


def bookend_angle(t,v,a,b):
    left=(t>=a-.35-1e-8)&(t<a-.04)
    right=(t>b+.04)&(t<=b+.35+1e-8)
    if not left.any() or not right.any():
        return None
    x,y=v[left].mean(0),v[right].mean(0)
    if min(np.linalg.norm(x),np.linalg.norm(y))<1:
        return None
    return abs(float(np.degrees(wrap(np.arctan2(y[1],y[0])-np.arctan2(x[1],x[0])))))


def match_intervals(pred, ref):
    iou=np.zeros((len(pred),len(ref)))
    for i,(a,b,*_) in enumerate(pred):
        for j,(c,d,*_) in enumerate(ref):
            overlap=max(0,min(b,d)-max(a,c))
            s=overlap/(max(b,d)-min(a,c))
            if s>=.20 and abs((a+b-c-d)/2)<=.35:
                iou[i,j]=s
    a,b=linear_sum_assignment(iou,maximize=True)
    return [(int(i),int(j),float(iou[i,j])) for i,j in zip(a,b) if iou[i,j]>0]


def design(seed):
    rng=np.random.default_rng(seed)
    grid=list(product((45,90,135,180),(3,6),(.4,.8),(0,1.2)))
    for i in range(240):
        angle,speed,duration,loss=grid[i%len(grid)]
        hz=int(rng.choice([5,10]))
        phase=float(rng.choice([0,.5]))
        label=str(rng.choice(("clean","iid03","iid10","ar10")))
        sd={"clean":0,"iid03":.03,"iid10":.10,"ar10":.10}[label]
        start=float(rng.uniform(1.15,1.55))
        if i%10==0:
            kind,turns="straight",[]
        elif i%10==1:
            kind,turns="multi",[(start,duration,angle,loss),(2.7,.6,-60,0)]
        else:
            kind,turns="smooth",[(start,duration,angle,loss)]
        t=np.arange(phase/hz,4.50001,1/hz)
        ref_t=np.arange(4501)*.001
        xp,vp,ap,h=truth(ref_t,speed,turns)
        resample=lambda q,f:np.column_stack([np.interp(q,ref_t,x) for x in f.T])
        xy=resample(t,xp)+noise(rng,len(t),1/hz,sd,label=="ar10")
        yield i,kind,hz,phase,label,t,xy,turns,resample,(xp,vp,ap)


def evaluate(seed):
    rows=[]
    for i,kind,hz,phase,label,t,xy,turns,resample,refs in design(seed):
        for method in METHODS:
            est=fit(t,xy,method)
            q,p,v,a=est
            keep=(q>=.4-1e-8)&(q<=t[-1]-.4+1e-8)
            errors={name:float(np.mean(np.sum((estimates[keep]-resample(q[keep],ref))**2,axis=1)))
                    for name,estimates,ref in zip(("position","velocity","acceleration"),(p,v,a),refs)}
            oracle=[bookend_angle(q,v,o,o+d) for o,d,*_ in turns]
            reference=[(o,o+d,abs(theta)) for o,d,theta,_ in turns]
            segments,reason=detect(curves(est),"B")
            for gate in GATES:
                candidates=[r for r in segments if r[3]>=gate]
                scored=[(*r,bookend_angle(q,v,r[0],r[1])) for r in candidates]
                pairs=match_intervals(scored,reference)
                errors_event=[dict(truth=j,iou=s,center=scored[k][2]-(reference[j][0]+reference[j][1])/2,
                                   start=scored[k][0]-reference[j][0],end=scored[k][1]-reference[j][1],
                                   angle=scored[k][4]-reference[j][2] if scored[k][4] is not None else None)
                              for k,j,s in pairs]
                rows.append(dict(case=i,kind=kind,hz=hz,phase=phase,noise=label,method=method,gate=gate,
                                 oracle_errors=[None if x is None else x-ref[2] for x,ref in zip(oracle,reference)],
                                 squared_errors=errors,matched=errors_event,truth=len(reference),
                                 detections=len(candidates),features=sum(r[4] is not None for r in scored),
                                 reject=None if candidates else reason or "angle_gate"))
    return rows


def report_metrics(rows):
    def moments(values):
        if not values:
            return None
        a=np.asarray(values)
        return dict(n=len(a),bias=float(a.mean()),mae=float(abs(a).mean()),median=float(np.median(a)))
    event=[e for r in rows for e in r["matched"]]
    tp=len(event)
    detections=sum(r["detections"] for r in rows)
    truth_count=sum(r["truth"] for r in rows)
    return dict(trials=len(rows),tp=tp,detected=detections,truth=truth_count,
                precision=tp/detections if detections else 0,recall=tp/truth_count if truth_count else 0,
                f1=2*tp/(detections+truth_count) if detections+truth_count else 0,
                features=sum(r["features"] for r in rows),accepted=sum(r["features"]>0 for r in rows),
                rejections=dict(Counter(r["reject"] or "no_valid_angle" for r in rows if not r["features"])),
                oracle_angle=moments([x for r in rows for x in r["oracle_errors"] if x is not None]),
                rmse={k:float(np.sqrt(np.mean([r["squared_errors"][k] for r in rows])))
                      for k in ("position","velocity","acceleration")} if rows else {},
                events={k:moments([e[k] for e in event if e[k] is not None])
                        for k in ("center","start","end","iou","angle")})


def global_repr(t,xy):
    dt=np.diff(t)
    if len(dt)<10 or np.any(abs(dt-.1)>.001):
        return None
    d=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    speeds=d/dt
    moving=speeds>=1
    if np.sum(moving)<10:
        return None
    length=float(d.sum())
    chord=float(np.linalg.norm(xy[-1]-xy[0]))
    s=np.r_[0,np.cumsum(d)]
    keep=np.r_[True,np.diff(s)>0]
    path=np.column_stack([np.interp(np.linspace(0,length,21),s[keep],q[keep]) for q in (xy-xy[0]).T])
    displ=np.diff(path,axis=0)
    headings=np.arctan2(displ[:,1],displ[:,0])
    return dict(speed=float(np.median(speeds[moving])),duration=float(t[-1]-t[0]),length=length,
                chord=chord,complexity=length/max(chord,.1),path=path.tolist(),
                geometry_turn=float(np.degrees(abs(wrap(np.diff(headings))).sum())),
                speed_quantiles=np.quantile(speeds[moving],[.25,.5,.75]).tolist(),moving=int(moving.sum()),
                edge_speeds=[float(speeds[0]),float(speeds[-1])])


def blinded(run,trials):
    packet=run/"blind_annotation_packet"
    chosen={}
    for drill in sorted({r["drill"] for r in trials}):
        group=[r for r in trials if r["drill"]==drill and "xy" in r]
        def label(r):
            segments=[r["methods"][m] for m in "ABC"]
            if len(segments[1])>1:
                return "multiple"
            if not any(segments):
                return "rejected"
            if all(segments) and np.ptp([p[0]["center"] for p in segments])<=.2:
                return "agreement"
            return "ambiguous"
        for kind in ("agreement","ambiguous","rejected","multiple"):
            for r in [r for r in group if label(r)==kind][:2]:
                chosen[(r["player"],r["event"])]=r
    rng=np.random.default_rng(23031)
    key={}
    seq=list(chosen.values())
    with (packet/"review_template.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["case","reviewer","event_index","start_s","center_s","end_s","direction","confidence",
                         "ambiguity","notes","review_pass"])
        for j,i in enumerate(rng.permutation(len(seq)),1):
            case=f"case_{j:03d}"
            r=seq[i]
            t=np.asarray(r["t"])
            xy=np.asarray(r["xy"])-np.asarray(r["xy"])[0]
            colors=np.arange(len(t))
            fig,ax=plt.subplots(1,2,figsize=(9,4),layout="constrained")
            ax[0].plot(*xy.T,color=".6")
            points=ax[0].scatter(*xy.T,c=t,s=10)
            fig.colorbar(points,ax=ax[0],label="time (s)")
            ax[0].set(aspect="equal",xlabel="relative x (yd)",ylabel="relative y (yd)")
            ax[1].plot(t[1:],np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(t))
            ax[1].set(xlabel="time (s)",ylabel="frame-difference speed (yd/s)")
            fig.suptitle(case)
            fig.savefig(packet/f"{case}.png",dpi=125)
            plt.close(fig)
            key[case]=[r["drill"],r["player"],r["event"],r["attempt"]]
            writer.writerow([case]+[""]*10)
    atomic_json(run/"annotation_key_private.json",key)
    (packet/"annotation_guide.md").write_text("""# Blinded path annotation

Reviewers work independently on randomized cases. Use only these point trajectories, time colors and coordinate-derived speed. Do not infer assigned task, athlete identity, body orientation, foot contact or kinetics. Use seconds from the plotted first frame. Mark onset of a coherent change from a preceding direction, center of the principal bend, and end when a distinct outgoing direction is established. Use one row per distinguishable turn. Mark left/right or indeterminate in the plotted coordinate frame. A pause without coherent outgoing direction, a clipped bend or overlapping bends should be marked ambiguous with a reason. If no event can be assessed, leave event times blank and say why. Confidence: high = both leg directions and interval clear; medium = one uncertain boundary; low = clips, drift, low speed or indistinct legs.

Keep the original reviewer sheets. Only after both are frozen, compare matched intervals (center within 0.35 s and IoU ≥0.20). Report event-count disagreement and interval and timing differences. A third reviewer adjudicates unmatched or low-confidence events and records reasons in a separate sheet. Preserve unresolved cases. Reviewers see neither the private key nor the algorithm predictions before adjudication. This task labels visible paths, not independent physical or joint-motion truth.
""")
    return len(seq)


def motion(run):
    trials=json.loads((G22/"development_trials.json").read_text())
    n=blinded(run,trials)
    # Exercise both generators explicitly. The numerical slope inside G2.2's
    # linear ramp is constant, while its analytic normal term used a cubic slope.
    time=np.array([1.6005,1.6010,1.6015])
    heading,rate=corner_truth(time,1.6,.002,90)
    assert np.allclose(np.gradient(heading,time),rate)
    audit=json.loads((G22/"motion_summary.json").read_text())["real"]
    real=[]
    for r in trials:
        if "xy" not in r:
            continue
        t,xy=np.asarray(r["t"]),np.asarray(r["xy"])
        g=global_repr(t,xy)
        if g is not None:
            for m in METHODS:
                est=fit(t,xy,m)
                segments,reason=detect(curves(est),"B")
                local=[dict(start=a,end=b,center=k,angle=bookend_angle(est[0],est[2],a,b))
                       for a,b,k,d in segments if d>=35]
                real.append(dict(drill=r["drill"],player=r["player"],event=r["event"],method=m,
                                 global_rep=g,local=local,accepted=any(x["angle"] is not None for x in local),
                                 rejection=reason or "angle_gate_or_bookend"))
    atomic_json(run/"development_measurements.json",real)
    cal,test=evaluate(23032),evaluate(23033)
    grid={}
    for method in METHODS:
        for gate in GATES:
            name=f"{method}/{gate}"
            grid[name]={k:report_metrics([r for r in data if r["method"]==method and r["gate"]==gate])
                        for k,data in (("calibration",cal),("evaluation",test))}
    selected=max(grid,key=lambda k:grid[k]["calibration"]["f1"])
    common=set.intersection(*[{r["case"] for r in test if r["method"]==m and r["gate"]==35 and r["features"]}
                              for m in METHODS])
    grouped={}
    for m in METHODS:
        subset=[r for r in test if r["method"]==m and r["gate"]==35]
        grouped[m]={f"{hz}/{noise_type}":report_metrics([r for r in subset if r["hz"]==hz and r["noise"]==noise_type])
                    for hz,noise_type in product((5,10),("clean","iid03","iid10","ar10"))}
        grouped[m]["common"]=report_metrics([r for r in subset if r["case"] in common])
    rejection=[]
    for m,drill in product(METHODS,sorted({r["drill"] for r in real})):
        subset=[r for r in real if r["method"]==m and r["drill"]==drill]
        def summary(rows):
            players={r["player"] for r in rows}
            return dict(n=len(rows),players=len(players),medians={
                key:float(np.median([r["global_rep"][key] for r in rows])) if rows else None
                for key in ("speed","duration","complexity","geometry_turn")})
        rejection.append(dict(method=m,drill=drill,accepted=summary([r for r in subset if r["accepted"]]),
                              rejected=summary([r for r in subset if not r["accepted"]]),
                              reasons=dict(Counter(r["rejection"] for r in subset if not r["accepted"]))))
    by_group={}
    for m in METHODS:
        valid=[r for r in real if r["method"]==m]
        athletes=defaultdict(list)
        for r in valid:
            athletes[r["player"]].append(r["accepted"])
        by_group[m]={"segments":sum(sum(s["angle"] is not None for s in r["local"]) for r in valid),
                     "trials":sum(r["accepted"] for r in valid),"trial_denominator":len(valid),
                     "players":sum(any(v) for v in athletes.values()),"player_denominator":len(athletes),
                     "equal_player_acceptance":float(np.mean([np.mean(v) for v in athletes.values()]))}
    arc_rows=[r["global_rep"] for r in real if r["method"]==METHODS[0]]
    integration=[sum(g["edge_speeds"][i]>=1 for g in arc_rows) for i in (0,1)]
    atomic_json(run/"motion_results.json",dict(calibration=cal,evaluation=test,grid=grid,chosen=selected,
                strata=grouped,common_cases=len(common),rejection=rejection,weights=by_group,blind_cases=n,
                moving_boundaries=integration,boundary_denominator=len(arc_rows)))
    lines=["# E32 Estimator comparison",
           "The common target angle is the smallest angle between average velocity headings in true pre and post bookends. This is distinct from prescribed angle and accumulated rotation. Scoring uses a common interior time mask at every sampling phase. Position, velocity and acceleration are vector RMSE, in yd, yd/s and yd/s².",
           "", "| Estimator | Position | Velocity | Acceleration | Oracle angle MAE / n |", "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        r=grid[f"{m}/35"]["evaluation"]
        e=r["oracle_angle"]
        lines.append(f"| {m} | "+" | ".join(f"{v:.3f}" for v in r["rmse"].values())+f" | {e['mae']:.2f}° / {e['n']} |")
    lines+=["",f"{integration[0]} of {len(arc_rows)} eligible trials begin moving and {integration[1]} end moving at the 1 yd/s floor. File limits are therefore recording boundaries. Full-trial path length, duration and speed are exposure-dependent, and cannot be interpreted as start-to-stop athletic tests.",
            "Global representation: moving-interval median speed and speed quantiles, geometry length/chord. Local representation: interval bookend angle and timestamps with success flags. Functional representation: 21 equally spaced cumulative path-length coordinates, translated to the start, alongside temporal speed summaries. Geometry preserves left/right orientation in file axes, needs no cut, and is sensitive to spatial noise and recording boundaries. No body or peak-braking inference. Quality fields record timing, moving intervals, edge speeds, candidate count and missing angles."]
    (run/"trajectory_estimator_comparison.md").write_text("\n".join(lines)+"\n")
    lines=["# E33 Risk and coverage",
           f"One-to-one maximum-total-IoU assignment, with IoU ≥0.20 and center error ≤0.35 s. Frozen calibration winner: {selected}. Evaluation uses seed 23033. The confidence gates are descriptive and do not establish calibrated coverage guarantees.",
           "", "| Estimator/gate | Calibration F1 | Evaluation precision | Recall | Valid feature trials | Angle MAE ° | Mean IoU |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for k,v in grid.items():
        e=v["evaluation"]
        lines.append(f"| {k} | {v['calibration']['f1']:.3f} | {e['precision']:.3f} | {e['recall']:.3f} | {e['accepted']}/240 | {e['events']['angle']['mae']:.1f} | {e['events']['iou']['bias']:.2f} |")
    lines += ["",f"Common accepted subset at the middle gate includes {len(common)} evaluation tracks. Its conditional estimates and all-track estimates are separate in JSON. Matched-event start/end/center signed and absolute error, angle bias, detection versus feature counts and rejection counts are also in JSON.",
              "", "| Real drill / estimator | Accepted / total | Speed accepted / rejected | Duration accepted / rejected | Path/chord accepted / rejected |",
              "|---|---:|---:|---:|---:|"]
    for r in rejection:
        a,b=r["accepted"],r["rejected"]
        pair=lambda k:" / ".join(f"{q['medians'][k]:.2f}" if q["medians"][k] is not None else "—" for q in (a,b))
        lines.append(f"| {r['drill']} / {r['method']} | {a['n']}/{a['n']+b['n']} | {pair('speed')} | {pair('duration')} | {pair('complexity')} |")
    lines+=["","These are descriptive comparisons on measured 2023 trajectories. Differential rejection changes the target population, even when errors are small among accepted cases. No real angle target was used."]
    (run/"risk_coverage_report.md").write_text("\n".join(lines)+"\n")
    lines=["# E31 Independent measurement audit",
           "G2.2 remains intact. Nominal task angle is an instruction label. The A window describes local motion only. B integrates in a detected rate interval; C compares stable bookends conditional on B finding an interval. Those estimands and accepted populations differ.",
           "G2.2's idealized corner used a linear heading ramp within a 2 ms interval but reported the derivative of a smooth cubic ramp for the reference normal acceleration. Inside the corrected linear ramp the angular derivative is constant; the derivative has a discontinuity at each end. An instantaneous mathematical corner provides no ordinary finite acceleration at the vertex. The new audit checks that consistency and excludes the corner from derivative error scoring.",
           "The previous match used center distance only, so short fragments could match a whole turn. The new matching checks IoU and one-to-one assignment. A detected candidate can fail stable bookend and low-speed requirements. Real figures for annotation intentionally hide all predictions and task names.",
           f"Created {n} cases for two independent reviewers. No human labels yet. A/B/C accepted trial counts by task from G2.2 follow. These are trial counts, even when B/C find several segments.",
           "", "| Drill | A | B | C | Total |","|---|---:|---:|---:|---:|"]
    for drill,g in audit.items():
        lines.append(f"| {drill} | {g['A']['accepted']} | {g['B']['accepted']} | {g['C']['accepted']} | {g['trials']} |")
    lines+=["","G2.3 reports segment counts, any valid estimate per trial, any accepted trial per player, and equally weighted player acceptance in motion_results.json. Segment pooling overweights multi-turn and high-attempt players; inferential samples should group on athletes."]
    (run/"measurement_review.md").write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(figsize=(7,5),layout="constrained")
    for m in METHODS:
        e=[grid[f"{m}/{g}"]["evaluation"] for g in GATES]
        ax.plot([r["recall"] for r in e],[r["precision"] for r in e],marker="o",label=m)
        for g,r in zip(GATES,e):
            ax.annotate(str(g),(r["recall"],r["precision"]),xytext=(4,2),textcoords="offset points")
    ax.set(xlabel="Event recall",ylabel="Event precision",title="Independent synthetic risk and coverage",xlim=(0,1),ylim=(0,1))
    ax.legend()
    fig.savefig(run/"figures"/"risk_coverage.png",dpi=145)
    plt.close(fig)
    return dict(trials=len(trials),players=len({r["player"] for r in trials}),simulations=480,blind_cases=n)
