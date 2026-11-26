"""G2.2 synthetic truth and private development-cohort measurement experiments."""
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import json

import duckdb
import numpy as np

from movement_probe import TASKS, atomic_json, describe, digest, kinematics, wrap

G1 = Path("data/audit_runs/g1-cloud-final-20261126T0942Z/audit.duckdb")
G21 = Path("data/g2/g2-1-20261126-final")
METHODS = ("A", "B", "C")
METRICS = ("angle", "duration", "entry", "exit", "retention", "normal75", "curvature75",
           "accumulated", "speed_loss", "recovery")


def estimates(t, xy, smooth):
    return kinematics(t, xy, smooth)


def detect(curves, method):
    t, _, speed, _, _, _, heading = curves
    dt = np.median(np.diff(t))
    # Compute physically comparable windows. A_legacy invokes the original threshold and smoothing.
    if method == "A_legacy":
        f = describe(curves)
        if f is None:
            return [], "legacy_rejection"
        c = f["cut_s"]
        return [(c-.3, c+.3, c, f["turn_deg"])], None
    if method == "A":
        width = max(1, round(.3 / dt))
        i = np.arange(width, len(t)-width)
        delta = np.degrees(wrap(heading[i+width] - heading[i-width]))
        valid = ((speed[i-width] >= 1) & (speed[i+width] >= 1)
                 & (t[i-width]>=t[0]+.4-1e-8) & (t[i+width]<=t[-1]-.4+1e-8))
        if not valid.any():
            return [], "low_speed_or_edge"
        score = np.where(valid, abs(delta), -1)
        ties = np.flatnonzero(np.isclose(score, score.max(), atol=1e-8, rtol=0))
        plateau = np.split(ties, np.flatnonzero(np.diff(ties)>1)+1)[0]
        j = plateau[len(plateau)//2]
        if score[j] < 30:
            return [], "no_window_turn"
        k = i[j]
        return [(t[k-width], t[k+width], t[k], abs(delta[j]))], None
    delta = np.degrees(wrap(np.diff(heading)))
    rate = delta / np.diff(t)
    moving = (speed[:-1] >= 1) & (speed[1:] >= 1)
    active = np.flatnonzero(moving & (abs(rate) >= 35))
    runs = np.split(active, np.flatnonzero((np.diff(active) != 1)
                    | (np.diff(np.sign(rate[active])) != 0))+1)
    segments = []
    reasons = Counter()
    for ix in runs:
        if not len(ix):
            continue
        a, b = ix[0], ix[-1]+1
        if t[b]-t[a] < .19:
            reasons["short_excursion"] += 1
            continue
        if t[a] < t[0]+.4-1e-8 or t[b] > t[-1]-.4+1e-8:
            reasons["edge"] += 1
            continue
        angle = abs(delta[ix].sum())
        if method == "C":
            windows = [np.flatnonzero((t[:-1]>=t[a]-.4-1e-8) & (t[1:]<=t[a]+1e-8)),
                       np.flatnonzero((t[:-1]>=t[b]-1e-8) & (t[1:]<=t[b]+.4+1e-8))]
            if any(not len(w) or np.sum(np.diff(t)[w][moving[w] & (abs(rate[w])<=20)]) < .19
                   for w in windows):
                reasons["unstable_bookends"] += 1
                continue
            stable = [w[moving[w] & (abs(rate[w]) <= 20)] for w in windows]
            before, after = [np.angle(np.mean(np.exp(1j*heading[w]))) for w in stable]
            angle = abs(np.degrees(wrap(after-before)))
        segments.append((t[a], t[b], (t[a]+t[b])/2, angle))
    return segments, (reasons.most_common(1)[0][0] if reasons else "no_sustained_turn") if not segments else None


def metrics(curves, a, b, angle=None):
    t, _, speed, _, normal, curve, heading = curves
    masks = ((t>=a-.4-1e-8) & (t<a-1e-8), (t>b+1e-8) & (t<=b+.4+1e-8),
             (t>=a-1e-8) & (t<=b+1e-8))
    pre, post, turn = masks
    if not all(mask.any() for mask in masks):
        return None
    en, ex = np.mean(speed[pre]), np.mean(speed[post])
    angular_steps = np.degrees(wrap(np.diff(heading[turn])))
    trough_ix = np.flatnonzero(turn)[np.argmin(speed[turn])]
    attained = np.flatnonzero((np.arange(len(t))>trough_ix) & (speed>=.9*en))
    status = "observed" if len(attained) else "right_censored"
    def p75(x):
        x = np.abs(x[turn])
        return np.quantile(x[np.isfinite(x)], .75) if np.isfinite(x).any() else None
    good_heading = np.all(speed[turn]>=1)
    return dict(angle=angle if angle is not None else (abs(angular_steps.sum()) if good_heading else None),
                duration=b-a, entry=en, exit=ex, retention=ex/en if en>=1 else None,
                normal75=p75(normal), curvature75=p75(curve),
                accumulated=np.abs(angular_steps).sum() if good_heading else None,
                speed_loss=en-speed[trough_ix], recovery=t[attained[0]]-t[trough_ix] if len(attained) else None,
                followup=t[-1]-t[trough_ix], recovery_status=status)


def trajectory(t, speed, turns, corner=False):
    angle = np.zeros_like(t)
    angle_rate = np.zeros_like(t)
    s = np.full_like(t, float(speed))
    for start, dur, turn, drop in turns:
        u = np.clip((t-start)/dur, 0, 1)
        angle += np.radians(turn)*(u if corner else u*u*(3-2*u))
        angle_rate += np.radians(turn)*6*u*(1-u)/dur
        s -= drop*np.sin(np.pi*u)**2
    v = np.c_[s*np.cos(angle), s*np.sin(angle)]
    xy = np.vstack([np.zeros(2), np.cumsum((v[:-1]+v[1:])*np.diff(t)[:,None]/2, axis=0)])
    normal = s * angle_rate
    safe = np.where(s>=1, s, np.nan)
    return t, xy, s, np.gradient(s,t), np.where(s>=1,normal,np.nan), normal/safe**2, angle


def noise(rng, size, dt, sd, related):
    z = rng.normal(0, sd, (size,2))
    if related:
        rho = np.exp(-dt/.25)
        for i in range(1,size):
            z[i] = rho*z[i-1] + np.sqrt(1-rho*rho)*z[i]
    return z


def measured(curves, segments):
    return [dict(start=a,end=b,center=c,features=metrics(curves,a,b,angle))
            for a,b,c,angle in segments]


def match(pred, truth, tol=.35):
    options = sorted((abs(p["center"]-q["center"]),i,j)
                     for i,p in enumerate(pred) for j,q in enumerate(truth))
    used_i, used_j, paired = set(), set(), []
    for distance,i,j in options:
        if distance <= tol and i not in used_i and j not in used_j:
            used_i.add(i)
            used_j.add(j)
            paired.append((i,j))
    return paired


def cases():
    for angle, speed, dur, drop in product((45,90,135,180),(3,6),(.4,.8),(0,1.2)):
        yield "smooth", speed, [(1.4,dur,angle,drop)]
    yield "straight", 4, []
    yield "arc", 4, [(0,4,120,0)]
    yield "corner", 4, [(1.6,.002,90,0)]
    yield "same_direction", 4, [(1,.5,60,0),(2.5,.6,80,1)]
    yield "opposite_direction", 4, [(1,.5,60,0),(2.5,.6,-80,1)]
    yield "edge", 4, [(.05,.4,90,0)]
    yield "low_speed", .7, [(1.4,.8,90,0)]


def synthetic(run):
    rng = np.random.default_rng(22021)
    records, design = [], []
    for n, (kind,speed,turns) in enumerate(cases()):
        ref = trajectory(np.arange(4001)*.001,speed,turns,kind=="corner")
        fine = trajectory(np.arange(8001)*.0005,speed,turns,kind=="corner")
        if kind == "arc":
            def circle(t):
                omega = np.pi/6
                return (t, np.c_[speed*np.sin(omega*t)/omega, speed*(1-np.cos(omega*t))/omega],
                        np.full(len(t),speed), t*0, t*0+speed*omega, t*0+omega/speed, omega*t)
            ref, fine = circle(ref[0]), circle(fine[0])
        truth = [dict(start=a,end=a+d,center=a+d/2,features=metrics(ref,a,a+d,abs(theta)),
                      nominal_angle=abs(theta), initial_speed=speed, speed_loss=drop)
                 for a,d,theta,drop in turns] if kind!="arc" else []
        design.append(dict(kind=kind,speed=speed,turns=turns,truth=truth,
                           max_integration_discrepancy=float(np.linalg.norm(ref[1]-fine[1][::2],axis=1).max())))
        for hz,phase in product((5,10),(0,.5)):
            t = np.arange(phase/hz,4.00001,1/hz)
            xy = np.c_[*[np.interp(t,ref[0],ref[1][:,j]) for j in range(2)]]
            clean_raw = estimates(t,xy,0)
            clean = estimates(t,xy,round(.2*hz))
            oracle = lambda c: [metrics(c,q["start"],q["end"]) for q in truth]
            fixed = {m:detect(clean,m)[0] for m in METHODS}
            legacy = measured(estimates(t,xy,1),detect(estimates(t,xy,1),"A_legacy")[0])
            for label, sd, related in (("clean",0,False),("iid03",.03,False),
                                       ("iid10",.10,False),("ar10",.10,True)):
                for draw in range(1 if not sd else 5):
                    cur = clean if not sd else estimates(t,xy+noise(rng,len(t),1/hz,sd,related),round(.2*hz))
                    predicted = {m:measured(cur,detect(cur,m)[0]) for m in METHODS}
                    records.append(dict(case=n,hz=hz,phase=phase,noise=label,draw=draw,
                         sampled_oracle=oracle(clean_raw) if not sd else None,
                         filtered_oracle=oracle(clean) if not sd else None,
                         noisy_oracle=oracle(cur),
                         redetected=predicted, fixed={m:measured(cur,[(a,b,k,None) for a,b,k,_ in fixed[m]]) for m in METHODS},
                         legacy=legacy if not sd else None))
    atomic_json(run/"synthetic_truth_results.json",dict(design=design,observations=records))


def real(run):
    rng = np.random.default_rng(22023)
    with duckdb.connect(str(G1), read_only=True) as db:
        rows = db.execute("""
          SELECT c.drill_name,c.nfl_id,c.attempt,c.event_id,p.nfl_position,time,x,y,s,dis
          FROM combine_tracking c JOIN players p USING(nfl_id)
          WHERE c.draft_year=2023 AND c.entity_type='PLAYER' AND drill_name IN (SELECT unnest(?))
          ORDER BY 1,2,3,4,6
        """,[list(TASKS)]).fetchall()
    trials = defaultdict(list)
    for row in rows:
        if TASKS[row[0]] == ("WR/TE" if row[4] in ("WR","TE") else "DB") and row[4] in ("WR","TE","CB","FS","SS","DB"):
            trials[row[:5]].append(row[5:])
    out = []
    for (drill,player,attempt,event,position), frames in trials.items():
        t = np.array([(r[0]-frames[0][0]).total_seconds() for r in frames])
        xy = np.array([r[1:3] for r in frames])
        record = dict(drill=drill,player=player,attempt=attempt,event=event,position=position)
        if len(t)<7 or np.any(abs(np.diff(t)-.1)>.001) or not np.isfinite(xy).all():
            out.append(record | dict(methods={m:[] for m in METHODS},
                        failures={m:"missing_or_irregular_frames" for m in METHODS}))
            continue
        c = estimates(t,xy,2)
        seg = {m:detect(c,m) for m in METHODS}
        legacy = estimates(t,xy,1)
        step = np.linalg.norm(np.diff(xy,axis=0),axis=1)
        s, dis = np.array([r[3:] for r in frames]).T
        cumulative = np.r_[0,np.cumsum(step)]
        rel = step>.15
        ratio = np.median(dis[1:][rel]/step[rel]) if rel.any() else None
        distances = dict(lag_0=np.median(abs(dis[1:]-step)), lag_minus1=np.median(abs(dis[:-1]-step)),
                         lag_plus1=np.median(abs(dis[2:]-step[:-1])),
                         first_difference=np.median(abs(np.diff(dis)-step)),
                         cumulative=np.median(abs(dis-dis[0]-cumulative)),
                         ratio=ratio, yards_to_metres=np.median(abs(dis[1:]-.9144*step)),
                         metres_to_yards=np.median(abs(dis[1:]-step/.9144)),
                         fitted_ratio_residual=np.median(abs(dis[1:]-ratio*step)) if ratio is not None else None,
                         speed_times_dt=np.median(abs(dis[1:]-.1*s[1:])))
        variants = []
        for _ in range(5):
            v = estimates(t,xy+noise(rng,len(t),.1,.1,False),2)
            variants.append(dict(fixed={m:measured(v,[(a,b,k,None) for a,b,k,_ in seg[m][0]]) for m in METHODS},
                                 redetected={m:measured(v,detect(v,m)[0]) for m in METHODS}))
        out.append(record | dict(methods={m:measured(c,seg[m][0]) for m in METHODS},
                                 fixed_clean={m:measured(c,[(a,b,k,None) for a,b,k,_ in seg[m][0]]) for m in METHODS},
                                 failures={m:seg[m][1] for m in METHODS}, variants=variants,
                                 legacy=measured(legacy,detect(legacy,"A_legacy")[0]),
                                 t=t.tolist(),xy=xy.tolist(), speed_t=c[0].tolist(), speed=c[2].tolist(),
                                 below_floor=bool(np.any(c[2]<1)), dis=distances))
    atomic_json(run/"development_trials.json",out)
