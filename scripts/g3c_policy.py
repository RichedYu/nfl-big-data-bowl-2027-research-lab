"""Frozen observed-input comparisons with identical held-out player identities."""
from collections import defaultdict
import json

import duckdb
import numpy as np
from scipy.special import logit

from g23_run import G1
from g3a_policy import SEEDS, fold_policy, losses, make_splits, read_outcomes, ridge
from g3c_construct import COBS
from movement_probe import atomic_json,digest

LABELS=("M0","M1","M1G","B+","B+G","M2-observed")
CONTRASTS=((0,1),(1,2),(1,3),(3,4),(4,5),(3,5))


def predict(data,split,w,binary,enabled,active=False):
    y,pos,forty,tech,short,cobs=data
    n=len(y)
    result=np.zeros((len(SEEDS),n,5+int(enabled[1])))
    ids=np.arange(n)
    empty=np.empty((n,0))
    for j,folds in enumerate(split):
        for k,fold in enumerate(folds):
            train=np.setdiff1d(ids,fold)
            chosen=train[w[train]>0]
            test=fold[w[fold]>0] if active else fold
            if not len(test):
                continue
            # Compute in-fold fitted parent for the offset; only write held-out results.
            parent,_=fold_policy((y,pos,forty,tech,empty),train,ids,w,binary)
            b=parent[:,1:3].copy()
            use=np.flatnonzero(np.isfinite(forty)&np.isfinite(short))
            if enabled[0]:
                rows=chosen[np.isin(chosen,use)]
                for i,design in enumerate((np.c_[pos,forty,short],np.c_[pos,forty,short,tech])):
                    q,_=ridge(design,y,rows,use,w,binary)
                    if q is not None:
                        b[use,i]=q
            values=np.c_[parent,b]
            if enabled[1]:
                aug=b[:,1].copy()
                good=np.flatnonzero(np.isfinite(cobs)&np.isfinite(forty))
                q,_=ridge(cobs[:,None],y,chosen[np.isin(chosen,good)],good,w,binary,
                          offset=logit(aug) if binary else aug)
                if q is not None:
                    aug[good]=q
                values=np.c_[values,aug]
            result[j,test]=values[test]
    return result


def summary(rows,binary,enabled,seed):
    n=len(rows)
    ids=[r["id"] for r in rows]
    y=np.asarray([r["r"] if binary else r["y"] for r in rows],dtype=float)
    te=np.asarray([r["te"] for r in rows],dtype=float)[:,None]
    columns=[np.asarray([r[k] for r in rows],dtype=float) for k in ("forty","tech","split","cobs")]
    data=(y,te,*columns)
    folds=make_splits(y,binary)
    result=predict(data,folds,np.ones(n,dtype=int),binary,enabled)
    per_rep=losses(y,result,binary)
    per_player={k:v.mean(0) for k,v in per_rep.items()}
    rng=np.random.default_rng(seed)
    fixed=defaultdict(list)
    refit=defaultdict(list)
    failures=[]
    for j,w in enumerate(rng.multinomial(n,np.full(n,1/n),size=400)):
        for m,v in per_player.items():
            fixed[m].append(w@v/n)
        try:
            p=predict(data,folds,w,binary,enabled,active=True)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            failures.append(dict(draw=j,error=type(exc).__name__,reason=str(exc)))
            continue
        for m,v in losses(y,p,binary).items():
            refit[m].append(w@v.mean(0)/n)
    metrics={}
    names=LABELS[:result.shape[-1]]
    for m in (*per_player,*(("rmse",) if not binary else ())):
        k="mse" if m=="rmse" else m
        scale=np.sqrt if m=="rmse" else lambda x:x
        point=scale(per_player[k].mean(0))
        f=scale(np.asarray(fixed[k]))
        r=scale(np.asarray(refit[k]))
        split=scale(per_rep[k].mean(1))
        out={}
        for a,b in CONTRASTS:
            if b>=len(names):
                continue
            diff=per_player[k][:,b]-per_player[k][:,a]
            out[names[b]+" − "+names[a]]=dict(delta=float(point[b]-point[a]),
                 fixed=np.quantile(f[:,b]-f[:,a],[.025,.975]).tolist(),
                 refit=np.quantile(r[:,b]-r[:,a],[.025,.975]).tolist() if len(r)>=380 else None,
                 split_deltas=(split[:,b]-split[:,a]).tolist(),player_sd=float(diff.std(ddof=1)) if m!="rmse" else None)
        metrics[m]=dict(scores=dict(zip(names,point.tolist())),comparisons=out)
    results=dict(n=n,eligible_R=int(sum(r["r"] for r in rows)),technical=sum(r["tech"] for r in rows),
                 forty=sum(r["forty"] is not None for r in rows),both_timing=sum(r["split"] is not None for r in rows),
                 no_task_record=sum(not r["recorded"] for r in rows),cobs_computable=sum(r["cobs"] is not None for r in rows),
                 recorded_but_no_cobs=sum(r["recorded"] and r["cobs"] is None for r in rows),
                 cobs_model_arm=sum(r["cobs"] is not None and r["forty"] is not None for r in rows),
                 calibration_mean_error=dict(zip(names,(result.mean(axis=(0,1))-y.mean()).tolist())),
                 metrics=metrics,refits=400-len(failures),failures=failures)
    raw=dict(players=ids,outcome=y.tolist(),predictors=rows,model_labels=names,predictions=result.tolist(),
             tests=[[[ids[i] for i in f] for f in s] for s in folds],seeds=SEEDS,
             player_losses={k:v.tolist() for k,v in per_player.items()},
             repeat_losses={k:v.tolist() for k,v in per_rep.items()})
    return results,raw


def policy(run):
    freeze=json.loads((run/"frozen_predictors_private.json").read_text())
    if freeze["protocol_hash"]!=digest(run/"observed_representation_protocol.md"):
        raise ValueError("Protocol changed")
    enabled=(freeze["bplus_enabled"],freeze["cobs_enabled"])
    with duckdb.connect(str(G1/"audit.duckdb"),read_only=True) as db:
        roster=read_outcomes(db)
    result=[]
    for i,pos,forty,opportunity,y in roster:
        row=freeze["room"][str(i)]
        assert row["te"]==int(pos=="TE") and row["forty"]==forty
        assert (opportunity>=25)==(y is not None)
        result.append(row|dict(id=i,r=int(opportunity>=25),y=y))
    assert len(result)==51 and sum(r["legacy_common"] for r in result)==34
    stats,private={},{}
    for i,(y,full) in enumerate(((False,True),(True,True),(False,False),(True,False))):
        key=("Y" if y else "R")+("_full" if full else "_common")
        rows=[r for r in result if (r["r"] or not y) and (full or r["legacy_common"])]
        stats[key],private[key]=summary(rows,not y,enabled,33101+i)
    atomic_json(run/"policy_results.json",dict(enabled=enabled,scopes=stats))
    atomic_json(run/"paired_player_losses_private.json",private)
    a=["# Observed information results · 2023 development",
       "B+ uses official 10-yard split plus forty, and B+G adds the G3A technical-availability flag. The observed model adds one C_obs slope to frozen B+G parent predictions when forty and C_obs are present. Fallback predictions are retained for all others. R_eligible is published rookie eligibility, and Y is the player mean among eligible players.",
       f"Approved by this *exploratory* protocol: B+ {enabled[0]}, C_obs {enabled[1]}. Physical measurement status PENDING.",
       "", "| Scope | Players | R eligible | Forty / both times | Technical flag | C_obs arm | No Curl record / recorded no C_obs | Refit draws |",
       "|---|---:|---:|---:|---:|---:|---|---:|"]
    for state,z in stats.items():
        a.append(f"| {state} | {z['n']} | {z['eligible_R']} | {z['forty']}/{z['both_timing']} | {z['technical']} | {z['cobs_model_arm']} | {z['no_task_record']}/{z['recorded_but_no_cobs']} | {z['refits']}/400 |")
    a+=["","| Scope | Primary score | "+" | ".join(LABELS[:5+int(enabled[1])])+" |",
        "|---|---|"+"---:|"*(5+int(enabled[1]))]
    for state,z in stats.items():
        metric="log_loss" if state.startswith("R") else "mse"
        a.append(f"| {state} | {metric} | "+" | ".join(f"{x:.5f}" for x in z["metrics"][metric]["scores"].values())+" |")
    a+=["","Calibration mean error, Brier, RMSE and MAE retained in paired_loss_analysis.md. Fixed-score and weighted refit intervals remain conditional on these development splits. Point gain alone is inconclusive where plausible differences include zero or precision is insufficient. Neither rule provides a real NFL conditional mutual information estimator."]
    (run/"observed_information_results.md").write_text("\n".join(a)+"\n")
    b=["# Paired losses and uncertainty","",
       "Differences are new minus comparator, so negative is favorable. Three predetermined repeats share the same players. Average each player's own losses across repeats. The 400 paired fixed-loss draws and 400 player-weighted refits retain held-out identities, and capture separate aspects of uncertainty conditional on frozen split plans. Failed draws are logged; a refit interval needs ≥380 successful draws.",
       "", "| Scope | Metric | Comparison | Difference | Fixed 95% | Refit 95% | Split differences min / max |",
       "|---|---|---|---:|---|---|---|"]
    def ci(x):
        return f"[{x[0]:+.5f}, {x[1]:+.5f}]" if x else "PENDING"
    for state,z in stats.items():
        for metric,a in z["metrics"].items():
            for k,v in a["comparisons"].items():
                s=v["split_deltas"]
                b.append(f"| {state} | {metric} | {k} | {v['delta']:+.5f} | {ci(v['fixed'])} | {ci(v['refit'])} | {min(s):+.5f} / {max(s):+.5f} |")
    b+=["","## Ancillary scores and calibration","",
        "| Scope | Metric | "+" | ".join(LABELS[:5+int(enabled[1])])+" |",
        "|---|---|"+"---:|"*(5+int(enabled[1]))]
    for state,z in stats.items():
        for k,value in z["metrics"].items():
            if k in ("log_loss","mse"):
                continue
            b.append(f"| {state} | {k} | "+" | ".join(f"{v:.5f}" for v in value["scores"].values())+" |")
        b.append(f"| {state} | mean prediction minus observation | "+" | ".join(f"{v:+.5f}" for v in z["calibration_mean_error"].values())+" |")
    b+=["","All player identities, predictions and loss vectors remain in the private workspace. This report exports only aggregate comparisons."]
    (run/"paired_loss_analysis.md").write_text("\n".join(b)+"\n")
    return dict(scopes=len(stats),attempted_refits=1600,successes=sum(x["refits"] for x in stats.values()),cobs=enabled[1])
