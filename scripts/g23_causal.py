"""Exact finite causal models and population information diagnostics."""
from itertools import product
import numpy as np

from movement_probe import atomic_json

NAMES=("U","M","W","D","Y","R")


def probabilities(world,u,m,w,d,y):
    p_m={"A_latent":.125+.75*u,"B_confounding":.2+.6*u}.get(world,.5)
    p_w=.15+.70*m
    p_d={"E_mediator":.2+.6*m,"E_collider":.1+.4*m+.4*u}.get(world,.5)
    p_y={"A_direct":.2+.6*m,"A_latent":.1+.8*u,"B_confounding":.1+.2*m+.5*u,
         "C_selection":.5,"D_error_selection":.2+.6*m,"E_mediator":.2+.6*d,
         "E_collider":.2+.6*u}[world]
    p_r={"C_selection":.10+.40*m+.40*y,"D_error_selection":.10+.40*w+.40*y}.get(world,.8)
    return .5,p_m,p_w,p_d,p_y,p_r


def distribution(world,force=None):
    rows=np.array(list(product((0,1),repeat=6)))
    weight=[]
    for u,m,w,d,y,r in rows:
        p=np.asarray(probabilities(world,u,m,w,d,y))
        state=np.array([u,m,w,d,y,r])
        factors=np.where(state,p,1-p)
        if force is not None:
            factors[1]=float(m==force)
        weight.append(factors.prod())
    return rows,np.array(weight)


def entropy(rows,w,cols):
    if not cols:
        return 0.
    _, ix=np.unique(rows[:,[NAMES.index(c) for c in cols]],axis=0,return_inverse=True)
    p=np.bincount(ix,w)
    p=p[p>0]
    return float(-p@np.log2(p))


def cmi(rows,w,x,y,z=()):
    return (entropy(rows,w,(x,*z))+entropy(rows,w,(y,*z))
            -entropy(rows,w,z)-entropy(rows,w,(x,y,*z)))


def coefficient(rows,w,x,z=()):
    design=np.column_stack([np.ones(len(rows)),rows[:,[NAMES.index(k) for k in (x,*z)]]])
    sq=np.sqrt(w)
    if np.linalg.matrix_rank(design*sq[:,None])<design.shape[1]:
        return None
    return float(np.linalg.lstsq(design*sq[:,None],rows[:,4]*sq,rcond=None)[0][1])


def experiment():
    worlds=("A_direct","A_latent","B_confounding","C_selection","D_error_selection","E_mediator","E_collider")
    out={}
    for world in worlds:
        rows,w=distribution(world)
        effect=float(distribution(world,1)[1]@rows[:,4]-distribution(world,0)[1]@rows[:,4])
        joint=[float(w[(rows[:,1]==m)&(rows[:,4]==y)].sum()) for m,y in product((0,1),repeat=2)]
        records={}
        for scope in ("full","selected"):
            p=w.copy() if scope=="full" else w*rows[:,5]
            mass=float(p.sum())
            p/=mass
            cells={}
            for x,z in product(("M","W"),((),("U",),("D",))):
                cells[f"{x}|{','.join(z) or '-'}"]=dict(cmi=cmi(rows,p,x,"Y",z),
                                                       lpm=coefficient(rows,p,x,z))
            records[scope]=dict(mass=mass,diagnostics=cells)
        out[world]=dict(effect=effect,observed_joint=joint,scopes=records)
    assert np.allclose(out["A_direct"]["observed_joint"],out["A_latent"]["observed_joint"])
    return out


def causal(run):
    data=experiment()
    atomic_json(run/"causal_results.json",data)
    lines=["# E35 Known causal worlds",
           "Exact Bernoulli structural equations in the associated code enumerate 64 possible states. U denotes background, M latent movement, W a noisy measurement, D draft position proxy, Y performance and R observed opportunity. The causal contrast replaces the assignment equation for M and marginalizes the rest of the system. CMI and weighted linear probability coefficients are calculated independently from the observational distribution. Positivity holds in these toy observational states.",
           "", "| World | True effect do(M=1)−do(M=0) | Full MI(M;Y) | Selected MI(M;Y) | Full slope | Adjust U | Adjust D | Selected slope |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,d in data.items():
        full,sel=[d["scopes"][k]["diagnostics"] for k in ("full","selected")]
        values=[d["effect"],full["M|-"]["cmi"],sel["M|-"]["cmi"],full["M|-"]["lpm"],
                full["M|U"]["lpm"],full["M|D"]["lpm"],sel["M|-"]["lpm"]]
        lines.append(f"| {name} | "+" | ".join(f"{x:.4f}" for x in values)+" |")
    lines+=["","A: observed M,Y joint tables are identical (0.4,0.1,0.1,0.4); direct intervention contrasts are different.",
            "B: U affects M and Y. Adjustment recovers the intentionally additive 0.2 effect when U is available.",
            "C: R depends on both M and Y. Full association and effect are zero, but selection creates an association.",
            "D: W flips M with probability .15. Measurement error changes association, and selection depends on W and Y. Inspect the W rows in causal_results.json; selected W association mixes both mechanisms.",
            "E: controlling draft D blocks a mediated path in one world, and opens a path through a collider in the other. Draft adjustment requires a stated estimand and graph.",
            "All conditional information values under U and D, samples selected, and W regressions are in causal_results.json. These counterexamples give no real NFL effect estimates."]
    (run/"causal_experiments.md").write_text("\n".join(lines)+"\n")
    (run/"causal_estimand_registry.md").write_text("""# Proposed NFL diagram and estimands

```mermaid
graph LR
 subgraph combine[Pre-draft]
  U[Prior skill, health, training and latent ability] --> T[Latent movement]
  U --> B[Traditional testing]
  Z[Position, cohort and role] --> T
  T --> X[Measured trajectory summary]
  P[Drill protocol and measurement process] --> X
  P --> Q[Measurement accepted]
  T --> Q
 end
 T --> D[Draft selection and capital]
 U --> D
 Z --> R[Observed eligible opportunities]
 D --> R
 D --> O[Team, coaching and playing opportunities]
 U --> O
 O --> R
 O --> Y[Football performance]
 T --> Y
 U --> Y
 Y --> R
 C[Release coverage and missing fields] --> R
```

Candidate graph only. D may mediate transitions or act as a collider. R may depend on underlying performance, team roles and publication coverage. Q can select on how the player moves. Y is recorded only for a stated observed population. This is not a causal discovery result.

| Estimand | Population and assumptions | Status |
|---|---|---|
| Descriptive measurement | Within drill and defined accepted path events; compare methods and same-player repeats | Development evidence, path annotation pending |
| Opportunity prediction | Probability of ≥25 eligible published rookie route/separation observations for comparable pre-draft profiles | Predictive only, not true participation |
| Conditional performance prediction | Mean separation conditional on observed R=1 and specified pre-draft predictors, evaluated on common eligible players | Development pilot only |
| Effect of training on later performance | Would require well-defined intervention, baseline exchangeability or randomized assignment, positivity, and handling of eligibility and measurement error | Unidentified in release |
| Direct effect excluding draft or opportunities | Requires an explicit mediation/intervention definition and additional assumptions on intermediate confounding and selection | Unidentified |
| Transfer of path mechanisms | Comparable route, phase and football context with independent marker validation | Deferred |
""")
    return dict(worlds=len(data),states_per_world=64)
