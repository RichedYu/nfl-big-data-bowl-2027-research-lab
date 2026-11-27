"""Paired 2023-only strategy comparison with weighted player refits."""
from collections import defaultdict
import json
import math

import duckdb
import numpy as np
from scipy.special import expit, logit

from g23_run import G1, verify_artifacts
from g3a_measurement import G23,TASK
from movement_probe import atomic_json,digest

SEEDS=(31011,31012,31013)
PENALTY=4
MODELS=("M0","M1","M1G","M2")


def read_outcomes(db):
    # Materialize only the 2023 draft roster before accessing any target values.
    db.execute("""CREATE TEMP TABLE g3a_roster AS
        SELECT nfl_id,nfl_position FROM players
        WHERE draft_year=2023 AND nfl_position IN ('WR','TE')""")
    rows=db.execute("""
        WITH eligible AS (
          SELECT p.nfl_id,count(*) AS opportunities,
                 CASE WHEN count(*)>=25 THEN avg(p.separation_at_pass_forward) END AS y
          FROM g3a_roster r JOIN player_play p USING(nfl_id) JOIN games g USING(game_id)
          WHERE g.season=2023 AND g.season_type='REG'
            AND p.route_ran IS NOT NULL AND p.separation_at_pass_forward IS NOT NULL
          GROUP BY p.nfl_id
        )
        SELECT r.nfl_id,r.nfl_position,c.forty,coalesce(e.opportunities,0),e.y
        FROM g3a_roster r LEFT JOIN combine_results c USING(nfl_id)
        LEFT JOIN eligible e USING(nfl_id) ORDER BY 1
    """).fetchall()
    return rows


def ridge(x,y,train,predict,weights,binary,offset=None):
    """Training-fold scaling and fixed penalty. Return predictions or no viable fit."""
    w=weights[train]
    if len(train)<4 or w.sum()<4:
        return None,"parent_insufficient_rows"
    target=y[train]
    if binary and len(np.unique(target))<2:
        if offset is not None:
            return None,"parent_single_class"
        prior=(w@target+.5)/(w.sum()+1)
        return np.repeat(prior,len(predict)),"half_count_prior"
    a,b=x[train],x[predict]
    if x.shape[1]:
        mean=np.average(a,axis=0,weights=w)
        sd=np.sqrt(np.average((a-mean)**2,axis=0,weights=w))
        sd[sd<1e-12]=1
        a,b=(a-mean)/sd,(b-mean)/sd
    if offset is None:
        a,b=np.c_[np.ones(len(a)),a],np.c_[np.ones(len(b)),b]
        penalty=np.diag([0]+[PENALTY]*(a.shape[1]-1))
        off_a,off_b=np.zeros(len(a)),np.zeros(len(b))
    else:
        penalty=np.eye(a.shape[1])*PENALTY
        off_a,off_b=offset[train],offset[predict]
    if not binary:
        beta=np.linalg.solve(a.T@(w[:,None]*a)+penalty,a.T@(w*(target-off_a)))
        return off_b+b@beta,"fit"
    beta=np.zeros(a.shape[1])
    if offset is None:
        beta[0]=logit((w@target+.5)/(w.sum()+1))
    def objective(coef):
        z=off_a+a@coef
        return w@(np.logaddexp(0,z)-target*z)+.5*coef@penalty@coef
    for _ in range(60):
        p=expit(off_a+a@beta)
        grad=a.T@(w*(p-target))+penalty@beta
        hessian=a.T@((w*p*(1-p))[:,None]*a)+penalty
        step=np.linalg.solve(hessian,grad)
        if np.max(abs(grad)) <= 1e-9*(1+w.sum()) or np.max(abs(step)) < 1e-8:
            break
        before=objective(beta)
        for shrink in (1.,.5,.25,.125,.0625,.03125,.015625):
            update=beta-shrink*step
            if objective(update)<=before+8*np.finfo(float).eps*max(1.,abs(before)):
                beta=update
                break
        else:
            raise RuntimeError("No descent in weighted logistic fit")
    else:
        raise RuntimeError("Weighted logistic fit did not converge")
    return np.clip(expit(off_b+b@beta),1e-9,1-1e-9),"fit"


def make_splits(y,binary):
    n=len(y)
    splits=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        if binary:
            parts=[np.array_split(rng.permutation(np.flatnonzero(y==i)),5) for i in (0,1)]
            folds=[np.sort(np.r_[parts[0][k],parts[1][k]]) for k in range(5)]
        else:
            folds=[np.sort(fold) for fold in np.array_split(rng.permutation(n),5)]
        assert np.array_equal(np.sort(np.concatenate(folds)),np.arange(n))
        splits.append(folds)
    return splits


def fold_policy(data,train,test,w,binary):
    y,pos,forty,available,features=data
    forty_ok=np.isfinite(forty)
    all_rows=np.arange(len(y))
    train=train[w[train]>0]
    decisions={}
    def branch(name,x,indices,rows,offset=None):
        pred,status=ridge(x,y,indices,rows,w,binary,offset)
        decisions[name]=dict(status=status,training_players=len(indices),training_weight=int(w[indices].sum()))
        return pred
    # Fallback intercept is technical protection against an empty/tiny resample.
    if len(train):
        v=np.average(y[train],weights=w[train]) if not binary else (w[train]@y[train]+.5)/(w[train].sum()+1)
    else:
        raise ValueError("A bootstrap resample has no training labels")
    base=branch("M0",pos,train,all_rows)
    if base is None:
        base=np.repeat(v,len(y))
    trad=base.copy()
    forty_train=train[forty_ok[train]]
    forty_rows=all_rows[forty_ok]
    gdata=np.c_[pos,available]
    gbase=branch("M1G_roster",gdata,train,all_rows)
    if gbase is None:
        gbase=base.copy()
    g=gbase.copy()
    tfit=branch("M1_forty",np.c_[pos,forty],forty_train,forty_rows)
    if tfit is not None:
        trad[forty_rows]=tfit
    gfit=branch("M1G_forty",np.c_[gdata,forty],forty_train,forty_rows)
    if gfit is not None:
        g[forty_rows]=gfit
    pred=[base,trad,g]
    if features.shape[1]:
        m2=g.copy()
        good=forty_ok & np.isfinite(features).all(1)
        correct=branch("M2_offset",features,train[good[train]],all_rows[good],
                       offset=logit(g) if binary else g)
        if correct is not None:
            m2[good]=correct
        pred.append(m2)
    return np.asarray(pred).T[test],decisions


def predict(data,splits,weights,binary,active_only=False):
    scores=np.zeros((len(SEEDS),len(data[0]),3+bool(data[-1].shape[1])))
    status=[]
    ids=np.arange(len(data[0]))
    for i,folds in enumerate(splits):
        for j,fold in enumerate(folds):
            test=fold[weights[fold]>0] if active_only else fold
            if not len(test):
                continue
            y,details=fold_policy(data,np.setdiff1d(ids,fold),test,weights,binary)
            scores[i,test]=y
            status.append(dict(repeat=i,fold=j,branches=details))
    return scores,status


def losses(y,p,binary):
    error=p-y[None,:,None]
    if binary:
        prob=np.clip(p,1e-9,1-1e-9)
        ll=-(y[None,:,None]*np.log(prob)+(1-y[None,:,None])*np.log1p(-prob))
        return dict(log_loss=ll,brier=error**2)
    return dict(mse=error**2,mae=abs(error))


def intervals(values,observed):
    return dict(interval=np.quantile(values,[.025,.975]).tolist(),sd=float(np.std(values,ddof=1)),
                observed=float(observed))


def run_scope(rows,features,binary,seed):
    out_ids=[r["player"] for r in rows]
    y=np.asarray([r["r"] if binary else r["y"] for r in rows],dtype=float)
    pos=np.asarray([[r["te"]] for r in rows],dtype=float)
    forty=np.asarray([r["forty"] for r in rows],dtype=float)
    availability=np.asarray([r["available"] for r in rows],dtype=float)
    x=np.asarray([[r["features"].get(f) for f in features] for r in rows],dtype=float)
    data=(y,pos,forty,availability,x)
    n=len(y)
    splits=make_splits(y,binary)
    p,decisions=predict(data,splits,np.ones(n,dtype=int),binary)
    per_repeat=losses(y,p,binary)
    per_player={k:v.mean(0) for k,v in per_repeat.items()}
    rng=np.random.default_rng(seed)
    draws=rng.multinomial(n,np.full(n,1/n),size=400)
    fixed=defaultdict(list)
    refit=defaultdict(list)
    failed=[]
    for i,w in enumerate(draws):
        q=None
        try:
            q,_=predict(data,splits,w,binary,active_only=True)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:
            failed.append(dict(draw=i,reason=str(e)))
        for metric,a in per_player.items():
            fixed[metric].append(w@a/n)
        if q is not None:
            for metric,value in losses(y,q,binary).items():
                refit[metric].append(w@value.mean(0)/n)
    labels=MODELS[:p.shape[-1]]
    pairs=[(0,1),(1,2)]+([(2,3),(1,3)] if "M2" in labels else [])
    report={}
    distributions={}
    for metric in (*per_player,*(["rmse"] if not binary else [])):
        source="mse" if metric=="rmse" else metric
        scale=np.sqrt if metric=="rmse" else lambda x:x
        v=per_player[source]
        point=scale(v.mean(0))
        fixed_array=scale(np.asarray(fixed[source]))
        refit_array=scale(np.asarray(refit[source]))
        report[metric]={"scores":dict(zip(labels,point.tolist())),"comparisons":{}}
        distributions[metric]={"fixed":fixed_array.tolist(),"refit":refit_array.tolist()}
        for a,b in pairs:
            observed=point[b]-point[a]
            f=fixed_array[:,b]-fixed_array[:,a]
            r=refit_array[:,b]-refit_array[:,a] if len(refit_array) else []
            within_repeat=scale(per_repeat[source].mean(1))
            comp=dict(difference=float(observed),fixed=intervals(f,observed),
                      refit=intervals(r,observed) if len(r)>1 else None,
                      repeat_differences=(within_repeat[:,b]-within_repeat[:,a]).tolist())
            if metric!="rmse":
                d=v[:,b]-v[:,a]
                comp.update(player_sd=float(np.std(d,ddof=1)),
                            influential_order=[out_ids[j] for j in np.argsort(-abs(d-d.mean()))[:3]])
            report[metric]["comparisons"][labels[b]+"−"+labels[a]]=comp
    prediction=p.mean(0)
    calibration={}
    for j,label in enumerate(labels):
        pred=prediction[:,j]
        slope=float(np.cov(y,pred,ddof=1)[0,1]/np.var(pred,ddof=1)) if np.ptp(pred)>1e-10 else None
        calibration[label]=dict(mean_prediction=float(pred.mean()),mean_observation=float(y.mean()),
                                mean_error=float((pred-y).mean()),linear_slope=slope)
    grades={}
    for status in sorted({r["tracking_status"] for r in rows}):
        flag=np.asarray([r["tracking_status"]==status for r in rows])
        grades[status]=dict(players=int(flag.sum()),outcomes_mean=float(y[flag].mean()),
                           losses={k:dict(zip(labels,a[flag].mean(0).tolist())) for k,a in per_player.items()})
    private=dict(players=out_ids,rows=rows,model_labels=labels,split_seeds=SEEDS,
                 test_folds=[[[out_ids[i] for i in x] for x in fs] for fs in splits],
                 predictions=p.tolist(),losses={k:a.tolist() for k,a in per_repeat.items()},
                 averaged_player_losses={k:a.tolist() for k,a in per_player.items()},fit_branches=decisions)
    return dict(n=n,metrics=report,calibration=calibration,by_availability=grades,
                completed_refits=400-len(failed),failed_refits=failed),private,distributions


def grouped_rows(run,features):
    f=json.loads((run/"feature_freeze.json").read_text())
    if (f["measurement_sha256"]!=digest(run/"measurement_results.json")
            or f["protocol_sha256"]!=digest(run/"protocol.md")):
        raise ValueError("Feature freeze does not match private inputs")
    stage=json.loads((G23/"checkpoint.json").read_text())["stages"]["motion"]
    verify_artifacts(G23,stage["artifacts"])
    legacy=json.loads((G23/"development_measurements.json").read_text())
    legacy_players={r["player"] for r in legacy if r["drill"]==TASK and r["method"]=="moving_mean"}
    source=json.loads((run/"measurement_results.json").read_text())
    candidate=defaultdict(list)
    recorded=set()
    for row in source["trial_records"]:
        if row["drill"]!=TASK:
            continue
        key=row["players"]
        recorded.add(key)
        value=row["pred"]["quadratic"]
        if value["accepted"] and all(value.get(f) is not None for f in features):
            candidate[key].append(value)
    with duckdb.connect(str(G1/"audit.duckdb"),read_only=True) as db:
        roster=read_outcomes(db)
    result=[]
    for p,pos,forty,opportunity,y in roster:
        vals=candidate[p]
        status="technically_available" if vals else ("recorded_rejected" if p in recorded else "no_task_record")
        if (opportunity>=25)!=(y is not None):
            raise ValueError("Inconsistent R/Y definition")
        result.append(dict(player=p,te=int(pos=="TE"),forty=forty,available=int(bool(vals)),
                           features={k:float(np.mean([v[k] for v in vals])) for k in features} if vals else {},
                           common_legacy=p in legacy_players and forty is not None,
                           tracking_status=status,opportunities=opportunity,r=int(opportunity>=25),y=y))
    assert len(result)==51 and sum(r["common_legacy"] for r in result)==34
    return result


def render(run,summary,features):
    policy=["# Full-policy and common-player evaluation","",
            "All values are repeated player-fold development scores for the 2023 draft cohort. R covers the full roster. Y is restricted to R=1. R=0 describes publication eligibility. The 34-player historical common R cohort and its Y subset are refit separately.",
            "M0 uses position. M1 adds forty where available and otherwise uses M0. M1G adds technical Curl availability to the applicable arms. Its no-forty model is trained with position and availability across training players. Availability is pre-draft metadata; no record and rejected measurement are distinct statuses with a shared zero indicator. The indicator is not a movement score.",
            ("M2 corrects M1G with the frozen feature(s) and otherwise uses M1G." if features else
             "M2 stopped before outcome access because no global feature passed measurement. The technical indicator can describe the available record but does not authorize its coordinate-derived features."),
            "", "| Scope | N | Metric | "+" | ".join(MODELS[:3+bool(features)])+" |",
            "|---|---:|---|"+"---:|"*(3+bool(features))]
    analysis=["# Paired loss analysis","",
              "Negative changes favor the new strategy. Each player's losses average three predetermined repeats before inference. The fixed bootstrap resamples actual paired losses. The refit bootstrap reweights players and refits all training folds from their positive-weight training players, evaluating the same weighted held-out identities. Intervals reflect player and refitting sensitivity conditional on the fixed splits. Each run uses the same number of real players.",
              "", "| Scope | Metric | Difference | Estimate | Fixed 95% | Refit 95% |",
              "|---|---|---|---:|---|---|"]
    precision=["# Precision scenarios","",
               "Planning only. Primary metrics are R log loss and conditional-mean Y MSE. Approximate effective paired n = (1.96 SD / h)², with h=δ/2. Fixed SD comes from actual paired player losses, and SD-equivalent for refit uses bootstrap SD×sqrt(current n). No MC bootstrap draw is an independent athlete. The sensitivity is conditional on these development folds/learners. Uncertain effect and transport are outside this calculation.",
               "", "| Scope | Comparison | Metric | δ | Half-width | Paired SD | Fixed n | Refit SD equivalent | Refit n |",
               "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for scope,r in summary.items():
        for metric,m in r["metrics"].items():
            policy.append(f"| {scope} | {r['n']} | {metric} | "+" | ".join(f"{v:.5f}" for v in m["scores"].values())+" |")
            for name,c in m["comparisons"].items():
                a,b=c["fixed"]["interval"]
                lo,hi=c["refit"]["interval"] if c["refit"] else (float("nan"),)*2
                analysis.append(f"| {scope} | {metric} | {name} | {c['difference']:+.5f} | [{a:+.5f}, {b:+.5f}] | [{lo:+.5f}, {hi:+.5f}] |")
                if metric not in ("log_loss","mse") or name not in ("M1G−M1","M2−M1G"):
                    continue
                for d in ((.01,.02) if metric=="log_loss" else (.02,.05)):
                    sd=c["player_sd"]
                    rd=c["refit"]["sd"]*math.sqrt(r["n"]) if c["refit"] else None
                    required=lambda s:"—" if s is None or s<1e-12 else str(math.ceil((1.96*s/(d/2))**2))
                    precision.append(f"| {scope} | {name} | {metric} | {d:.2f} | {d/2:.3f} | {sd:.5f} | {required(sd)} | "+
                                     (f"{rd:.5f}" if rd is not None else "—")+f" | {required(rd)} |")
    policy+=["","## Calibration and availability","",
              "Calibration is descriptive on mean per-player fold predictions. Mean error is prediction minus observation. Linear slope regresses observation on prediction; for R it is a probability-scale diagnostic, not a logistic calibration slope.",
              "", "| Scope | Model | Mean error | Linear slope |", "|---|---|---:|---:|"]
    for scope,r in summary.items():
        for model,c in r["calibration"].items():
            policy.append(f"| {scope} | {model} | {c['mean_error']:+.4f} | "+(f"{c['linear_slope']:.3f}" if c["linear_slope"] is not None else "—")+" |")
    policy+=["","Availability-specific counts and losses are in policy_summary.json. Same-player comparisons on the full target population do not discard missing forty or unavailable tracking. M2 coverage is zero if no representation was authorized."]
    analysis+=["","Refit success counts, failed draws, fold assignments, each repeat prediction, player losses and influence ranks are persisted privately. Models can be misspecified, especially for heterogeneous route/context mixtures, roster selection and the small tied availability strata. Observable opportunity may reflect downstream publication and playing opportunities. A loss difference does not identify true conditional mutual information."]
    precision+=["","δ is a planning sensitivity, not a validated minimum. A constant comparison has zero dispersion because the two procedures coincide on that scope; a dash leaves its sample requirement undefined. Plan needs effective, common paired players and enough cases in each availability tier, corrected for effective clustered sample size and attrition. New-team/cohort calibration, misspecification and measurement error can dominate nominal precision."]
    for name,lines in (("full_policy_evaluation",policy),("paired_loss_analysis",analysis),("statistical_precision",precision)):
        (run/(name+".md")).write_text("\n".join(lines)+"\n")


def models(run):
    features=json.loads((run/"feature_freeze.json").read_text())["features"]
    rows=grouped_rows(run,features)
    summary,private,distribution={},{},{}
    for i,(endpoint,binary,common) in enumerate(
            (("R",True,False),("Y",False,False),("R",True,True),("Y",False,True))):
        label=endpoint+("_common" if common else "_full")
        r=[v for v in rows if (binary or v["r"]) and (not common or v["common_legacy"])]
        summary[label],private[label],distribution[label]=run_scope(r,features,binary,31021+i)
    atomic_json(run/"policy_summary.json",dict(features=features,scopes=summary))
    atomic_json(run/"player_oof.json",private)
    atomic_json(run/"refit_distributions.json",distribution)
    render(run,summary,features)
    return dict(scopes=len(summary),fits_per_scope=15,refit_attempts_per_scope=400,
                successful_refits={s:r["completed_refits"] for s,r in summary.items()},
                feature_count=len(features))
