"""2023-only opportunity and exploratory player-level prediction."""
from collections import defaultdict
import json

import duckdb
import numpy as np
from scipy.special import expit

from g23_run import G1
from movement_probe import atomic_json

THRESHOLDS=(1,10,25,50)


def development(db):
    # Create the development roster before the only query accessing separation
    # values. Later cohorts are never in the outcome-query relation.
    db.execute("""CREATE TEMP TABLE development_roster AS
        SELECT nfl_id,nfl_position FROM players WHERE draft_year=2023 AND nfl_position IN ('WR','TE')""")
    roster=db.execute("""SELECT p.nfl_id,nfl_position,b.forty FROM development_roster p
        LEFT JOIN combine_results b USING(nfl_id) ORDER BY p.nfl_id""").fetchall()
    db.execute("""CREATE TEMP TABLE development_routes AS
        SELECT p.nfl_id,p.game_id,p.play_id,g.season,p.route_ran,p.separation_at_pass_forward AS separation
        FROM development_roster r JOIN player_play p USING(nfl_id) JOIN games g USING(game_id)
        WHERE g.season IN (2023,2024) AND g.season_type='REG'""")
    records=db.execute("""SELECT nfl_id,game_id,play_id,season,upper(route_ran),separation
                          FROM development_routes ORDER BY nfl_id,season,game_id,play_id""").fetchall()
    stages=[]
    for year in (2023,2024):
        # No tracing of athletes outside the temporary 2023 roster.
        stages.extend(db.execute(f"""
          WITH track AS (
            SELECT t.nfl_id,t.game_id,t.play_id,time,event,x,y,
              lag(time) OVER w prev_t,lag(x) OVER w AS px,lag(y) OVER w AS py
            FROM game_tracking_{year} t SEMI JOIN
              (SELECT nfl_id,game_id,play_id FROM development_routes
               WHERE season={year} AND route_ran IS NOT NULL AND separation IS NOT NULL) r
              USING(nfl_id,game_id,play_id)
            WINDOW w AS (PARTITION BY t.nfl_id,t.game_id,t.play_id ORDER BY time)
          ), marks AS (
            SELECT nfl_id,game_id,play_id,
                min(time) FILTER(WHERE event='ball_snap') snap_t,
                min(time) FILTER(WHERE event='pass_forward') throw_t
            FROM track GROUP BY 1,2,3
            HAVING count(*) FILTER(WHERE event='ball_snap')=1
               AND count(*) FILTER(WHERE event='pass_forward')=1
               AND count(*) FILTER(WHERE prev_t IS NOT NULL
                  AND date_diff('microsecond',prev_t,time) NOT BETWEEN 99000 AND 101000)=0
               AND date_diff('millisecond',snap_t,throw_t)>=600
               AND date_diff('millisecond',throw_t,max(time))>=600
          )
          SELECT t.nfl_id,t.game_id,t.play_id,{year},
                 median(sqrt((x-px)^2+(y-py)^2)*10)
          FROM track t JOIN marks m USING(nfl_id,game_id,play_id)
          WHERE time>snap_t AND time<=throw_t AND prev_t>=snap_t
          GROUP BY 1,2,3 HAVING count(*)>=6
        """).fetchall())
    aliases=db.execute("""SELECT c.draft_year,drill_name,count(DISTINCT c.nfl_id)
       FROM combine_tracking c JOIN players p USING(nfl_id)
       WHERE c.entity_type='PLAYER' AND p.nfl_position IN ('WR','TE') AND drill_name LIKE '%SPEED%OUT%'
       GROUP BY ALL ORDER BY ALL""").fetchall()
    return roster, records, stages, aliases


def precision(rows,rng):
    games=defaultdict(list)
    for game,value in rows:
        games[game].append(value)
    if len(games)<2:
        return None
    counts=np.array([len(v) for v in games.values()])
    totals=np.array([sum(v) for v in games.values()])
    ix=rng.integers(0,len(counts),(400,len(counts)))
    return float(np.std(totals[ix].sum(1)/counts[ix].sum(1),ddof=1))


def train(x,y,train_ix,test_ix,binary,penalty=4):
    x0,x1=x[train_ix],x[test_ix]
    if x.shape[1]:
        scale=x0.std(0)
        scale[scale==0]=1
        mean=x0.mean(0)
        x0,x1=(x0-mean)/scale,(x1-mean)/scale
    design=np.column_stack([np.ones(len(x0)),x0])
    test=np.column_stack([np.ones(len(x1)),x1])
    ridge=np.diag([0]+[penalty]*x.shape[1])
    target=y[train_ix]
    if not binary:
        coef=np.linalg.solve(design.T@design+ridge,design.T@target)
        return test@coef
    coef=np.zeros(design.shape[1])
    for _ in range(30):
        fitted=expit(design@coef)
        sd=fitted*(1-fitted)
        grad=design.T@(fitted-target)+ridge@coef
        step=np.linalg.solve(design.T@(sd[:,None]*design)+ridge,grad)
        coef-=step
        if np.max(abs(step))<1e-9:
            break
    return np.clip(expit(test@coef),1e-9,1-1e-9)


def pilot(rows,binary):
    n=len(rows)
    ids,y,x=zip(*rows) if rows else ([],[],[])
    a=np.asarray(y)
    counts={"positive":int(a.sum()),"negative":int(n-a.sum())} if binary else {}
    if n<24 or (binary and min(counts.values())<8):
        return dict(n=n,counts=counts,status="stopped_sample_gate",fits=0)
    rng=np.random.default_rng(23037)
    if binary:
        strata=[np.array_split(rng.permutation(np.flatnonzero(a==j)),5) for j in (0,1)]
        folds=[np.concatenate([z[i] for z in strata]) for i in range(5)]
    else:
        folds=np.array_split(rng.permutation(n),5)
        if min(map(len,folds))<4:
            return dict(n=n,counts=counts,status="stopped_fold_gate",fits=0)
    prediction=np.empty((n,3))
    b=np.asarray(x)
    for fold in folds:
        training=np.setdiff1d(np.arange(n),fold)
        for j,cols in enumerate((0,2,3)):
            prediction[fold,j]=train(b[:,:cols],a,training,fold,binary)
    dif=prediction-a[:,None]
    if binary:
        errors=dict(log_loss=-(a[:,None]*np.log(prediction)+(1-a[:,None])*np.log1p(-prediction)),
                    brier=dif**2)
    else:
        errors=dict(mae=abs(dif),rmse=dif**2)
    ix=rng.integers(0,n,(2000,n))
    results={}
    influence={}
    for name,losses in errors.items():
        scale=np.sqrt if name=="rmse" else lambda q:q
        model=scale(losses.mean(0))
        boot=scale(losses[ix].mean(1))
        drops=scale((losses.sum(0)-losses)/(n-1))
        delta=np.diff(model)
        gap=np.diff(drops,axis=1)-delta
        biggest=np.argmax(abs(gap),axis=0)
        results[name]=dict(scores=model.tolist(),differences=delta.tolist(),
                           interval=np.quantile(np.diff(boot,axis=1),[.025,.975],axis=0).T.tolist(),
                           leave_one_max=np.max(abs(gap),axis=0).tolist())
        influence[name]=[int(ids[j]) for j in biggest]
    return dict(n=n,counts=counts,status="completed",fits=15,metrics=results,
                private_influential_ids=influence,fold_sizes=list(map(len,folds)))


def football(run):
    with duckdb.connect(str(G1/"audit.duckdb"),read_only=True,config={"threads":2,"memory_limit":"4GB"}) as db:
        roster,records,stages,aliases=development(db)
    movement=json.loads((run/"development_measurements.json").read_text())
    attempts=defaultdict(list)
    for r in movement:
        if r["method"]=="moving_mean" and r["drill"]=="CURL_ROUTE_RIGHT":
            attempts[r["player"]].append(r["global_rep"]["speed"])
    tracks={(p,g,k,y):s for p,g,k,y,s in stages}
    eligible=defaultdict(list)
    available=defaultdict(list)
    vocab=defaultdict(lambda:defaultdict(lambda:[0,0,0]))
    for p,g,k,y,route,sep in records:
        available[p,y].append((route,sep is not None))
        if route is None:
            continue
        row=vocab[y][route]
        row[0]+=1
        if sep is not None:
            speed=tracks.get((p,g,k,y))
            row[1]+=1
            row[2]+=int(speed is not None)
            eligible[p,y].append((g,route,sep,speed))
    rng=np.random.default_rng(23036)
    coverage={}
    for year in (2023,2024):
        opportunities={}
        for label in ("overall","out","hitch","transfer","out_transfer","hitch_transfer"):
            keys=[]
            for p,*_ in roster:
                rows=[(g,sep) for g,route,sep,speed in eligible[p,year]
                      if (not label.startswith("out") or route=="OUT")
                      and (not label.startswith("hitch") or route=="HITCH")
                      and ("transfer" not in label or speed is not None)]
                keys.append(rows)
            thresholds={}
            for n in THRESHOLDS:
                matched=[r for r in keys if len(r)>=n]
                means=[x for r in matched if (x:=precision(r,rng)) is not None]
                thresholds[n]=dict(players=len(matched),median_se=float(np.median(means)) if means else None,
                                   bootstrap_players=len(means))
            opportunities[label]=thresholds
        coverage[year]=dict(opportunities=opportunities,
            no_published=sum(not available[p,year] for p,*_ in roster),
            no_route=sum(bool(available[p,year]) and not any(r for r,_ in available[p,year]) for p,*_ in roster),
            route_no_sep=sum(any(r for r,_ in available[p,year]) and not eligible[p,year] for p,*_ in roster),
            sep_no_transfer=sum(bool(eligible[p,year]) and not any(s is not None for _,_,_,s in eligible[p,year])
                                for p,*_ in roster))
    results={}
    for mode in ("opportunity","performance"):
        prepared=[]
        for p,pos,forty in roster:
            if forty is None or p not in attempts:
                continue
            available_y=eligible[p,2023]
            offered=len(available_y)>=25
            if mode=="performance" and not offered:
                continue
            target=int(offered) if mode=="opportunity" else float(np.mean([v for _,_,v,_ in available_y]))
            prepared.append((p,target,[forty,float(pos=="TE"),float(np.mean(attempts[p]))]))
        results[mode]=pilot(prepared,mode=="opportunity")
    counts=dict(roster=len(roster),forty=sum(f is not None for _,_,f in roster),
                curl_speed=len(attempts),both=sum(p in attempts and f is not None for p,_,f in roster),
                roster_r1=sum(len(eligible[p,2023])>=25 for p,*_ in roster),
                speed_available_r1=sum(p in attempts and len(eligible[p,2023])>=25 for p,*_ in roster),
                speed_unavailable_r1=sum(p not in attempts and len(eligible[p,2023])>=25 for p,*_ in roster))
    atomic_json(run/"football_results.json",dict(vocabulary=vocab,aliases=aliases,coverage=coverage,
                                                pilot=results,availability=counts))
    lines=["# E36 Route correspondence",
           "The Kaggle dictionary defines route_ran as a per-player route category and separation_at_pass_forward as distance to the nearest defender at throw. NFL Next Gen Stats documents broad route classes and many playbook variants. G2.2's eligibility only checked fields on the same row and optional snap/throw tracking support.",
           "", "| Combine drill | Candidate game class | Interpretation |",
           "|---|---|---|",
           "| SPEED_OUT_ROUTE_LEFT | OUT | Same coarse direction family. Game label does not encode break depth, speed, left/right or comparable timing. |",
           "| SPEED_OUT_LEFT | OUT only if protocol verified | 2024 naming could be an alias. No event instructions or external protocol evidence establish equivalence. |",
           "| CURL_ROUTE_RIGHT | HITCH | Approximate return-route family, with curl vs hitch depth and mechanics unresolved. Treat as sensitivity only. |",
           "| Any | All published route categories | Overall separation is descriptive, with no drill match. |",
           "| Either | OUT/HITCH and valid game tracking | Candidate timing support only; validate route phase in real examples before asserting movement transfer. |",
           "", "Structural exact-label coverage: "+", ".join(f"{y} {name}: {n}" for y,name,n in aliases)+".",
           "", "| Season | Published route class | Route rows | Separation present | Matching game speed window |",
           "|---|---|---:|---:|---:|"]
    for year,group in vocab.items():
        for label,ns in sorted(group.items()):
            lines.append(f"| {year} | {label} | "+" | ".join(map(str,ns))+" |")
    lines+=["","Sources: [Kaggle official dictionary](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data) and [NFL Next Gen Stats route classification](https://www.nfl.com/news/next-gen-stats-intro-to-new-route-recognition-model)."]
    (run/"route_correspondence_matrix.md").write_text("\n".join(lines)+"\n")
    lines=["# E36 Outcome design and precision",
           "Primary rookie regular season, secondary second regular season for 2023 development. Count only route-bearing player-play rows with observed separation. R is release eligibility at a defined threshold. Y is mean observed separation given R=1. Route-matched OUT/HITCH are provisional mappings.",
           "The game-transfer candidate is median coordinate-difference speed between snap and throw, requiring a clean 10 Hz event window and ≥0.6 s on either side of throw. Availability for the matched class is measured; timing around a specific cut is unverified. True movement-transfer modeling is deferred.",
           "", "| Horizon / endpoint | Players ≥1 / ≥10 / ≥25 / ≥50 | Median game-block SE at the four thresholds (yd) |",
           "|---|---|---|"]
    for year,v in coverage.items():
        for label,d in v["opportunities"].items():
            nn=" / ".join(str(r["players"]) for r in d.values())
            se=" / ".join("—" if r["median_se"] is None else f"{r['median_se']:.2f}" for r in d.values())
            lines.append(f"| {year} / {label} | {nn} | {se} |")
    lines+=["","The SE column uses separation values, including when the rows require tracking. It is conditional on publication and observed games, not precision of the game speed endpoint. More observations need not imply more independent games, or comparable players at a different threshold.",
            "Availability failures: "+str({k:{n:c for n,c in v.items() if n!="opportunities"} for k,v in coverage.items()})+".",
            "True participation cannot be inferred from an absent published row. Coverage, missing route, missing separation, unusable game stages and unusable Combine summaries require separate flags.",
            "nflverse participation is a candidate only. For 2023 onward its documentation attributes FTN and CC BY-SA 4.0, with completed-season release after postseason. A join would require a verified game-key crosswalk, play-key reconciliation, and GSIS ID crosswalk rather than equating those IDs with Kaggle nfl_id. Verify duplicate keys, franchise/season coverage, and identifier match rate on authorized data. Competition rules Section 6 permits qualifying accessible external data subject to rights and any host constraints; release-level compatibility and ID/coverage validation remain open. No external data joined.",
            "[Kaggle rules](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/rules) · [nflverse provenance and license](https://nflreadr.nflverse.com/reference/load_participation.html) · [nflverse participation schema](https://nflreadr.nflverse.com/articles/dictionary_participation.html)"]
    (run/"outcome_design.md").write_text("\n".join(lines)+"\n")
    lines=["# E37 Restricted development pilot",
           "Executed under the frozen pilot protocol. M0 intercept; M1 forty time plus TE indicator; M2 adds across-attempt Curl Right global median moving speed. All comparisons within identical eligible athletes with training-fold scaling and a fixed ridge penalty. M3 and functional modeling await independent review and a larger effective sample.",
           "Selection counts: "+", ".join(f"{k}={n}" for k,n in counts.items())+". Release eligibility differs by feature availability, reported as coverage rather than predictive improvement.",
           "", "| Endpoint | N | Metric | M0 / M1 / M2 | M1−M0 (95% interval) | M2−M1 (95% interval) | Max single-player score influence |",
           "|---|---:|---|---|---|---|---|"]
    for name,r in results.items():
        if r["status"]!="completed":
            lines.append(f"| {name} | {r['n']} | Stopped | {r['counts']} | — | — | — |")
            continue
        for metric,d in r["metrics"].items():
            models=" / ".join(f"{x:.4f}" for x in d["scores"])
            deltas=[f"{x:+.4f} [{lo:+.4f}, {hi:+.4f}]" for x,(lo,hi) in zip(d["differences"],d["interval"])]
            lines.append(f"| {name} | {r['n']} | {metric} | {models} | "+" | ".join(deltas)
                         +" | "+" / ".join(f"{x:.4f}" for x in d["leave_one_max"])+" |")
    lines+=["","Bootstrap intervals resample fixed out-of-fold player losses and omit model-refitting uncertainty. Influential-player IDs stay in the private JSON. Small differences with wide uncertainty are weak evidence. All 2024/25 draft-cohort outcome values remained sealed. Secondary season is the 2024 calendar season of the 2023 draft cohort."]
    (run/"development_prediction_pilot.md").write_text("\n".join(lines)+"\n")
    return dict(players=len(roster),route_rows=len(records),tracking_event_windows=len(stages),
                model_fits=sum(r["fits"] for r in results.values()))
