"""Known-generator information experiments, with log loss in bits."""
from itertools import product
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from movement_probe import atomic_json


def entropy(p):
    p=np.asarray(p)
    with np.errstate(divide="ignore",invalid="ignore"):
        return np.sum(np.where(p>0,-p*np.log2(p),0))


def mean_loss(p,q,w):
    return float(np.sum(w*(-p*np.log2(q)-(1-p)*np.log2(1-q))))


def conditional(p,w,keys):
    _, ix = np.unique(keys,axis=0,return_inverse=True)
    return (np.bincount(ix,w*p)/np.bincount(ix,w))[ix]


def xor_lab(rng):
    output=[]
    for flip,n in product((0,.1,.25),(24,100,500,10000)):
        observations=[]
        for _ in range(100):
            a,b=rng.integers(0,2,(2,n))
            y=a^b^(rng.random(n)<flip)
            h=lambda v:entropy(np.bincount(v)/n)
            observations.append([h(a)+h(y)-h(2*a+y),h(b)+h(y)-h(2*b+y),
                                 h(2*a+b)+h(y)-h(4*a+2*b+y)])
        observations=np.array(observations)
        expected=np.array([0,0,1-entropy([flip,1-flip])])
        errors=observations-expected
        output.append(dict(n=n,flip=flip,truth=expected.tolist(),mean=observations.mean(0).tolist(),
                           bias=errors.mean(0).tolist(),rmse=np.sqrt(np.mean(errors**2,0)).tolist(),
                           interval=np.quantile(observations,[.1,.9],axis=0).tolist()))
    return output


def shift(p):
    return .08+.84*p


def compression_lab(rng):
    rows=np.array(list(product((0,1),repeat=3)))
    useful,dynamic,junk=rows.T
    t=np.linspace(0,1,101)
    xy=np.asarray([
        np.c_[(2+a)*t+(2*b-1)*.07*np.sin(2*np.pi*t),(2*j-1)*.10*np.sin(4*np.pi*t)]
        for a,b,j in rows])
    traditional=np.rint(xy[:,-1,0]-2).astype(int)[:,None]
    early=(xy[:,25,0]>(2+traditional[:,0])*.25).astype(int)[:,None]
    mechanism=np.c_[traditional,early]
    rich=xy[:,[25,35,100],:].reshape(len(rows),-1)
    w=np.full(8,1/8)
    models={
        "sufficient":shift(useful),
        "omitted_dynamic":shift(useful^dynamic),
        "remove_nuisance":shift((useful+dynamic)/2),
    }
    result={}
    for name,p in models.items():
        rep={}
        for label,f in (("traditional",traditional),("mechanism",mechanism),("rich",rich)):
            true=mean_loss(p,np.full(len(p),np.sum(w*p)),w)-mean_loss(p,conditional(p,w,f),w)
            _, codes=np.unique(f,axis=0,return_inverse=True)
            draws=[]
            for _ in range(100):
                sample=rng.integers(0,len(p),500)
                y=(rng.random(len(sample))<p[sample]).astype(int)
                x=codes[sample]
                h=lambda v:entropy(np.bincount(v)/len(sample))
                draws.append(h(x)+h(y)-h(2*x+y))
            rep[label]=dict(truth=true,estimate=float(np.mean(draws)),
                            rmse=float(np.sqrt(np.mean((np.asarray(draws)-true)**2))))
        result[name]=rep
    return result


def learner(keys,outcomes,query,model,weights=None,alpha=.5):
    if weights is None:
        weights=np.ones(len(outcomes))
    if model=="cell":
        def encode(x):
            return x @ (2**np.arange(x.shape[1]))
        k=encode(keys)
        width=2**keys.shape[1]
        n=np.bincount(k,weights=weights,minlength=width)
        successes=np.bincount(k,weights=weights*outcomes,minlength=width)
        return (successes+alpha)[encode(query)]/(n+2*alpha)[encode(query)]
    answer=np.empty(len(query))
    w=keys[:,0]
    for group in (0,1):
        mask=(w==group)
        qry=(query[:,0]==group)
        loglikes=[]
        for y in (0,1):
            ww=weights*mask*((outcomes if y else 1-outcomes))
            total=ww.sum()
            prior=(total+alpha)/(weights[mask].sum()+2*alpha)
            q=(ww @ keys[:,1:]+alpha)/(total+2*alpha)
            loglikes.append(np.log(prior)+(query[qry,1:]*np.log(q)+(1-query[qry,1:])*np.log1p(-q)).sum(axis=1))
        answer[qry]=1/(1+np.exp(np.clip(loglikes[0]-loglikes[1],-700,700)))
    return np.clip(answer,1e-12,1-1e-12)


def prediction_lab(rng):
    states=np.array(list(product((0,1),repeat=5)))
    w,a,b,j,flip=states.T
    weight=np.where(flip,.2,.8)/16
    p=.07+.10*w+.18*a+.16*b+.27*(a^b)
    strategies={
        "independent_junk":(np.c_[w,a,b],np.c_[w,a,b,j]),
        "noisy_signal":(np.c_[w,a^flip,b],np.c_[w,a,b,a^flip]),
        "compressed":(np.c_[w,a],np.c_[w,a,b]),
        "added_measurement":(np.c_[w,a],np.c_[w,a,b,j]),
    }
    out={}
    for label,(x,t) in strategies.items():
        px,pt=[conditional(p,weight,k) for k in (x,t)]
        hx,ht=[mean_loss(p,r,weight) for r in (px,pt)]
        def kl(true, estimated):
            return float(sum(weight*(true*np.log2(true/estimated)
                         +(1-true)*np.log2((1-true)/(1-estimated)))))
        info=hx-ht
        # Generate independent large held-out samples, then reduce to exact state/label counts.
        test=rng.choice(len(p),100000,p=weight)
        observed=rng.random(len(test))<p[test]
        test_counts=np.bincount(test,minlength=len(p))/len(test)
        test_success=np.bincount(test,weights=observed,minlength=len(p))
        test_probs=np.divide(test_success,test_counts*len(test),out=np.zeros(len(p)),where=test_counts>0)
        experiments={}
        for model in ("naive","cell"):
            population=[mean_loss(p,learner(k,p,k,model,weight,alpha=0),weight) for k in (x,t)]
            results=[]
            for n in (24,32,50,100,500):
                draws=[]
                for _ in range(40):
                    take=rng.choice(len(p),n,p=weight)
                    y=(rng.random(n)<p[take]).astype(float)
                    qx,qt=[learner(k[take],y,k,model) for k in (x,t)]
                    lx,lt=[mean_loss(p,q,weight) for q in (qx,qt)]
                    ex,et=[mean_loss(test_probs,q,test_counts) for q in (qx,qt)]
                    dx,dt=kl(px,qx),kl(pt,qt)
                    draws.append((lx,lt,ex,et,dx,dt,lx-lt-info-dx+dt,ex-et-info-dx+dt,
                                  lx-population[0],lt-population[1]))
                draws=np.asarray(draws)
                results.append(dict(n=n,mean=draws.mean(0).tolist(),
                                    interval_10_90=np.quantile(draws,[.1,.9],axis=0).tolist(),
                                    max_exact_residual=float(abs(draws[:,6]).max())))
            experiments[model]=dict(population_excess=[population[0]-hx,population[1]-ht],learning=results)
        out[label]=dict(bayes_x=hx,bayes_t=ht,cmi=info,models=experiments)
    return out


def information(run):
    rng=np.random.default_rng(22024)
    xor=xor_lab(rng)
    compression=compression_lab(np.random.default_rng(2202402))
    predictions=prediction_lab(rng)
    keys=["expected_x","expected_t","heldout_x","heldout_t","d_x","d_t","exact_identity_residual",
          "empirical_identity_residual","training_excess_x","training_excess_t"]
    assert max(r["max_exact_residual"] for v in predictions.values() for m in v["models"].values()
               for r in m["learning"])<1e-12
    atomic_json(run/"information_results.json",dict(xor=xor,compression=compression,
                                                   predictions=predictions,columns=keys))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for ax,(label,v) in zip(axes.ravel(),predictions.items()):
        for model,m in v["models"].items():
            for j,rep in ((2,"X"),(3,"T")):
                p=m["learning"]
                n=[r["n"] for r in p]
                line,=ax.plot(n,[r["mean"][j] for r in p],label=f"{model} {rep}")
                ax.fill_between(n,*np.asarray([r["interval_10_90"] for r in p])[:,:,j].T,color=line.get_color(),alpha=.12)
        for key in ("bayes_x","bayes_t"):
            ax.axhline(v[key],color=".55",linestyle=":")
        ax.set(xscale="log",title=label.replace("_"," "),xlabel="Independent training samples",ylabel="Test log loss (bits)")
    axes[0,0].legend(fontsize=8)
    fig.savefig(run/"figures"/"learning_log_loss.png",dpi=150)
    plt.close(fig)
    lines=["# E24 Synthetic information laboratory","","## A. XOR",
           "Independent fair A,B and an independent Y flip of probability e give I(A;Y)=I(B;Y)=0 and I(A,B;Y)=1−h₂(e). The plug-in estimator uses empirical joint frequencies. Each row averages 100 independently sampled datasets.",
           "", "| Flip | n | Joint truth | Joint estimate | Joint RMSE |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['flip']} | {r['n']} | {r['truth'][2]:.4f} | {r['mean'][2]:.4f} | {r['rmse'][2]:.4f} |" for r in xor]
    lines+=["","## B. Sufficiency of trajectory representations",
            "Eight equiprobable tracks cross two mean speeds, two early/late velocity redistributions and two lateral oscillation signs. Decode mean speed from end-to-end displacement and early/late redistribution from the quarter point. Rich snapshots preserve all three bits. The mechanism representation drops lateral oscillation. The target uses exactly specified Bernoulli probabilities.",
            "", "| Generator | Traditional | Mechanism | Rich |", "|---|---:|---:|---:|"]
    for name,values in compression.items():
        lines.append(f"| {name} | "+" | ".join(f"{v['truth']:.4f} / {v['estimate']:.4f}" for v in values.values())+" |")
    lines.append("Each cell is exact information / mean empirical information from 100 independent samples of size 500. Monte Carlo RMSE is preserved in JSON.")
    lines+=["","In sufficient, Y depends only on mean speed. In omitted_dynamic, it depends on XOR of speed and redistribution. In remove_nuisance, both are useful and lateral sign only adds nuisance state count. These are exact discrete information values for the constructed supports.",
            "", "## C. Noise, mismatch and sample size",
            "W is observed by both representations; X is determined by T and W. Noisy X is derived from a corrupted bit explicitly included in T. Conditional naive Bayes is compared with a richer smoothed cell table. Both use half-count smoothing at finite n. Population naive Bayes parameters are calculated from exact conditional marginals; their excess risk is approximation mismatch, not an optimized projection over all possible predictors. Finite training excess is measured relative to that population fit.",
            "", "| Generator | H(Y|X,W) | H(Y|T,W) | CMI | Mean cell test X/T at n=24 | Mean cell test X/T at n=500 |",
            "|---|---:|---:|---:|---:|---:|"]
    for label,v in predictions.items():
        f=v["models"]["cell"]["learning"]
        pattern=lambda r:f"{r['mean'][2]:.3f} / {r['mean'][3]:.3f}"
        lines.append(f"| {label} | {v['bayes_x']:.4f} | {v['bayes_t']:.4f} | {v['cmi']:.4f} | {pattern(f[0])} | {pattern(f[-1])} |")
    lines+=["","Expected loss satisfies L(q_X)−L(q_T)=I(T;Y|X,W)+D_X−D_T for every fitted replicate by finite-state enumeration. Empirical held-out loss includes Monte Carlo sampling error; the residual and learning intervals are preserved in JSON. A more informative signal can still worsen small-n held-out loss. The independent junk condition has exactly zero CMI. None of these synthetic information quantities is an estimate for real NFL trajectories."]
    (run/"information_theory_experiments.md").write_text("\n".join(lines)+"\n")
