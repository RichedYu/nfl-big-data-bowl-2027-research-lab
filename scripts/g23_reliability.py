"""ADEMP simulations of two-attempt measurement reliability."""
from itertools import product
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, expit
from scipy.stats import t as student

from movement_probe import atomic_json

FIELDS=("within_sd","icc","absolute_difference","loa_low","loa_high","mdc")


def estimators(a):
    diff=a[:,1]-a[:,0]
    mean=a.mean(1)
    residual=np.mean(diff**2)/2
    alt=np.var(diff,ddof=1)/2
    def build(v):
        between=np.var(mean,ddof=1)-v/2
        sd=np.sqrt(v)
        return [sd,between/(between+v),np.mean(np.abs(diff)),
                diff.mean()-1.96*np.std(diff,ddof=1),diff.mean()+1.96*np.std(diff,ddof=1),
                1.96*sqrt(2)*sd]
    return np.asarray([build(residual),build(alt)])


def truths(tau,sig,regime):
    shift=.6 if regime=="order" else 0.
    multipliers=np.array([.55,1.35]) if regime=="hetero" else np.ones(2)
    s=sig*multipliers*sqrt(2)
    abs_diff=float(np.mean(s*np.sqrt(2/np.pi)*np.exp(-.5*(shift/s)**2)
                            +shift*(2*ndtr(shift/s)-1)))
    within=sig*sqrt(np.mean(multipliers**2))
    icc=tau*tau/(tau*tau+within*within)
    span=1.96*sqrt(2)*within
    return np.array([within,icc,abs_diff,shift-span,shift+span,span])


def simulate(seed=23034,reps=400):
    rng=np.random.default_rng(seed)
    cells=[]
    for n,k,tau,regime in product((14,30,50,100),(2,4),(.65,1.10),("balanced","order","hetero","selected")):
        sigma=1.
        target=truths(tau,sigma,regime)
        fitted, ci, sensitivity, denominators=[],[],[],[]
        for _ in range(reps):
            b=rng.normal(0,tau,n)
            sig=np.where(b[:,None]>=0,1.35,.55) if regime=="hetero" else 1.
            order=.6*(np.arange(k)%2) if regime=="order" else np.zeros(k)
            y=5+b[:,None]+order+sig*rng.normal(0,sigma,(n,k))
            observed=(rng.random((n,k))<expit(1.2+.7*b[:,None]+.65*(y-5))
                      if regime=="selected" else np.ones((n,k),dtype=bool))
            pair=y[observed[:,:2].all(1),:2]
            denominators.append(len(pair))
            n_pair=len(pair)
            if n_pair<5:
                continue
            fit=estimators(pair)
            # Vectorized delete-one pseudo-values for the two estimators.
            dif=pair[:,1]-pair[:,0]
            avg=pair.mean(1)
            mu=(dif.sum()-dif)/(n_pair-1)
            e2=(np.sum(dif**2)-dif**2)/(n_pair-1)
            vd=(e2-mu**2)*(n_pair-1)/(n_pair-2)
            av=(avg.sum()-avg)/(n_pair-1)
            va=((np.sum(avg**2)-avg**2)/(n_pair-1)-av**2)*(n_pair-1)/(n_pair-2)
            ab=(np.abs(dif).sum()-np.abs(dif))/(n_pair-1)
            se=[]
            for var in (e2/2,vd/2):
                bt=va-var/2
                jack=np.array([np.sqrt(var),bt/(bt+var),ab,mu-1.96*np.sqrt(vd),
                               mu+1.96*np.sqrt(vd),1.96*np.sqrt(2*var)])
                se.append(jack.std(1)*sqrt(n_pair-1))
            se=np.asarray(se)
            half=student.ppf(.975,n_pair-1)*se
            ci.append(np.abs(fit-target)<=half)
            fitted.append(fit)
            # Remove known simulated order for a labeled oracle sensitivity; aggregate residual DOF.
            trial=(y-order)*observed
            count=observed.sum(1)
            mean=np.divide(trial.sum(1),count,out=np.zeros(n),where=count>0)
            dof=np.maximum(count-1,0).sum()
            if dof:
                sensitivity.append(np.sqrt(np.sum(((y-order-mean[:,None])*observed)**2)/dof))
        errors=np.asarray(fitted)-target
        cover=np.asarray(ci).mean(0)
        diff=np.asarray(fitted)[:,1]-np.asarray(fitted)[:,0]
        cells.append(dict(n=n,attempts=k,tau=tau,sigma=sigma,regime=regime,reps=reps,success=len(fitted),
                          median_pairs=float(np.median(denominators)),truth=target.tolist(),
                          bias=errors.mean(0).tolist(),rmse=np.sqrt(np.mean(errors**2,0)).tolist(),
                          bias_mcse=(errors.std(0,ddof=1)/sqrt(len(fitted))).tolist(),
                          coverage=cover.tolist(),coverage_mcse=np.sqrt(cover*(1-cover)/len(fitted)).tolist(),
                          repaired_minus_naive=diff.mean(0).tolist(),
                          paired_difference_mcse=(diff.std(0,ddof=1)/sqrt(len(diff))).tolist(),
                          oracle_all_attempt_within_mean=float(np.mean(sensitivity))))
    return cells


def reliability(run):
    cells=simulate()
    atomic_json(run/"reliability_results.json",dict(fields=FIELDS,methods=["raw_pair","centered_pair"],cells=cells))
    lines=["# E34 ADEMP reliability",
           "Aims: estimate variance, repeatability and repeat agreement at small effective N. G2.2 DB entry repeat differences were around 1.1 yd/s, and between variance was weakly resolved. Simulation uses mean speed 5 yd/s, within SD 1 yd/s, and between SD .65 or 1.10 yd/s as sensitivity values.",
           "Data-generating mechanisms: Gaussian random intercept plus noise. Order adds 0.6 yd/s to even-numbered attempts. Heteroscedastic residual SD is 0.55 or 1.35 times baseline by latent intercept sign. Selection uses a Bernoulli logit of 1.2 + .7 player effect + .65 centered noisy value.",
           "Estimands: full-population within SD, ICC after accounting for order, mean absolute difference, moment-based Bland–Altman limits and minimum detectable change 1.96√2×within SD. For mixtures, limits are descriptive moment targets. Selection changes the observable population, so full-cohort comparisons expose selection bias.",
           "Methods: raw pair moment variance versus centering attempt differences to remove common order. Player jackknife t intervals; all observed attempts with oracle-known order removed are a sensitivity for within SD. The primary pair estimator always uses the first two scheduled attempts, so adding opportunities does not artificially inflate its pair count.",
           "Performance: 64 cells of 400 independent datasets. Report bias, RMSE, empirical interval coverage, corresponding Monte Carlo SE, paired method-difference MCSE and available replicates. No increase in replicate count was used.",
           "", "| n | Regime | Median pairs | ICC bias raw / centered | Centered ICC coverage (MCSE) |",
           "|---:|---|---:|---:|---:|"]
    for r in cells:
        if r["tau"]==.65 and r["attempts"]==2:
            lines.append(f"| {r['n']} | {r['regime']} | {r['median_pairs']:.0f} | {r['bias'][0][1]:.3f} / {r['bias'][1][1]:.3f} | {r['coverage'][1][1]:.3f} ({r['coverage_mcse'][1][1]:.3f}) |")
    lines+=["","Monte Carlo variation is distinct from uncertainty in one sample. Complete-pair estimates can be biased under informative rejection even when centering removes order. Negative between variance and ICC are retained. Jackknife intervals may have poor coverage in the smallest cells or for nonlinear quantities; full results include this failure mode."]
    (run/"reliability_ademp_report.md").write_text("\n".join(lines)+"\n")
    fig,ax=plt.subplots(layout="constrained")
    for regime in ("balanced","order","hetero","selected"):
        rs=[r for r in cells if r["regime"]==regime and r["attempts"]==2 and r["tau"]==.65]
        ax.plot([r["n"] for r in rs],[r["coverage"][1][1] for r in rs],marker="o",label=regime)
    ax.axhline(.95,color=".5",linestyle=":")
    ax.set(xlabel="Independent players",ylabel="Coverage of 95% ICC interval",ylim=(0,1))
    ax.legend()
    fig.savefig(run/"figures"/"reliability_coverage.png",dpi=145)
    plt.close(fig)
    return dict(simulations=sum(r["reps"] for r in cells),cells=len(cells))
