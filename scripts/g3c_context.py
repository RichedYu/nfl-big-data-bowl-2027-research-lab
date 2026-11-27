"""Retrospective decomposition of published 2023 rookie separation variation."""
from collections import Counter,defaultdict

import duckdb
import numpy as np

from g23_run import G1
from movement_probe import atomic_json

FIELDS=("player","game","play","route","sep","opponent","down","yards_to_go","yardline_number",
        "offense_formation","receiver_alignment","pass_length","coverage","target","dropback_duration",
        "snap_count","throw_count","seconds")


def read_context(db):
    db.execute("""
        CREATE TEMP TABLE cobs_eligible_context AS
        SELECT a.nfl_id,a.game_id,a.play_id,a.route_ran,a.separation_at_pass_forward,
               CASE WHEN a.team_abbr=g.home_team_abbr THEN g.visitor_team_abbr
                    WHEN a.team_abbr=g.visitor_team_abbr THEN g.home_team_abbr END AS opponent,
               a.down,a.yards_to_go,a.yardline_number,a.offense_formation,a.receiver_alignment,
               a.pass_length,a.team_coverage_type,a.target,a.dropback_duration
        FROM (SELECT nfl_id FROM players WHERE draft_year=2023 AND nfl_position IN ('WR','TE')) r
        JOIN player_play a USING(nfl_id) JOIN games g USING(game_id)
        WHERE g.season=2023 AND g.season_type='REG' AND a.route_ran IS NOT NULL
          AND a.separation_at_pass_forward IS NOT NULL
    """)
    rows=db.execute("""
        WITH markers AS (
          SELECT t.nfl_id,t.game_id,t.play_id,
            count(*) FILTER (WHERE t.event='ball_snap') AS snaps,
            count(*) FILTER (WHERE t.event='pass_forward') AS throws,
            min(t.time) FILTER (WHERE t.event='ball_snap') AS snap,
            min(t.time) FILTER (WHERE t.event='pass_forward') AS throw
          FROM game_tracking_2023 t SEMI JOIN cobs_eligible_context c USING(nfl_id,game_id,play_id)
          WHERE t.event IN ('ball_snap','pass_forward')
          GROUP BY 1,2,3
        )
        SELECT c.*,coalesce(m.snaps,0),coalesce(m.throws,0),
               CASE WHEN m.snaps=1 AND m.throws=1 AND m.throw>m.snap
                    THEN date_diff('microsecond',m.snap,m.throw)/1000000.0 END
        FROM cobs_eligible_context c LEFT JOIN markers m USING(nfl_id,game_id,play_id)
        ORDER BY c.nfl_id,c.game_id,c.play_id
    """).fetchall()
    if len({r[:3] for r in rows})!=len(rows):
        raise ValueError("Ambiguous player-play context join")
    return [dict(zip(FIELDS,r)) for r in rows]


def index(keys):
    m={}
    return np.asarray([m.setdefault(k,len(m)) for k in keys],dtype=int)


def components(y,w,groups):
    total=float(w.sum())
    if not total:
        raise ValueError("No positive-weight observation")
    center=float(w@y/total)
    var=float(w@(y-center)**2/total)
    outputs={}
    for name,g in groups.items():
        count=np.bincount(g,w)
        mean=np.divide(np.bincount(g,w*y),count,out=np.zeros_like(count),where=count>0)
        between=float(count@(mean-center)**2/total)
        within=float(w@(y-mean[g])**2/total)
        if not np.isclose(within+between,var,atol=1e-10):
            raise ValueError("Variance identity failed")
        outputs[name]={"between":between,"within":within}
    return var,outputs


def analysis(rows,boot=False):
    def timing(x):
        return "unknown" if x["seconds"] is None else "≤2s" if x["seconds"]<=2 else "2–3s" if x["seconds"]<=3 else ">3s"
    y=np.asarray([r["sep"] for r in rows],dtype=float)
    p=index([r["player"] for r in rows])
    g=index([r["game"] for r in rows])
    routes=[r["route"] for r in rows]
    context=[(r["route"],timing(r)) for r in rows]
    groups=dict(player=p,player_game=index(list(zip(p,g))),route=index(routes),route_time=index(context),
                player_route_time=index(list(zip(p,context))))
    # Base weights are equal athlete mass and uniform repetitions within athlete.
    w=1/np.bincount(p)[p]
    populations={}
    for flavor,ww in (("equal_player",w),("equal_play",np.ones(len(rows)))):
        total,parts=components(y,ww,groups)
        populations[flavor]=dict(total=total,components=parts,
                                player_game_increment=parts["player_game"]["between"]-parts["player"]["between"],
                                within_player_context_increment=parts["player_route_time"]["between"]-parts["player"]["between"])
    rng=np.random.default_rng(33002)
    sample=defaultdict(list)
    for j in range(400 if boot else 0):
        # Crossed multiplier resamples preserve shared player/game clusters.
        wt=w*rng.poisson(1,int(p.max()+1))[p]*rng.poisson(1,int(g.max()+1))[g]
        if not wt.sum():
            continue
        total,part=components(y,wt,groups)
        sample["total"].append(total)
        sample["between_player"].append(part["player"]["between"])
        sample["between_games_within_player"].append(part["player_game"]["between"]-part["player"]["between"])
        sample["within_player_game"].append(part["player_game"]["within"])
        sample["between_route_time"].append(part["route_time"]["between"])
    field_counts={k:sum(r[k] is not None for r in rows) for k in ("opponent","down","yards_to_go","yardline_number",
                    "offense_formation","receiver_alignment","pass_length","coverage","target","dropback_duration","seconds")}
    counts=np.bincount(groups["player_route_time"])
    return dict(rows=len(rows),players=int(p.max()+1),games=int(g.max()+1),player_games=len(set(zip(p,g))),
                observed_routes=len(set(routes)),observed_opponents=len({r["opponent"] for r in rows if r["opponent"]}),
                times=dict(Counter(timing(r) for r in rows)),
                timing_failures=dict(Counter("missing or multiple markers" if r["snap_count"]!=1 or r["throw_count"]!=1
                    else "nonpositive order" for r in rows if r["seconds"] is None)),
                player_game_median=float(np.median([len(set(g[p==i])) for i in np.unique(p)])),
                player_context_cells=len(counts),player_context_singletons=int((counts==1).sum()),
                player_context_lt5=int((counts<5).sum()),
                counts=field_counts,weights=populations,
                uncertainty={k:dict(draws=len(v),interval=np.quantile(v,[.025,.975]).tolist()) for k,v in sample.items()})


def context(run):
    with duckdb.connect(str(G1/"audit.duckdb"),read_only=True,config={"threads":2,"memory_limit":"4GB"}) as db:
        rows=read_context(db)
    tally=Counter(r["player"] for r in rows)
    x={"all_published":analysis(rows,True),
       "eligible_sensitivity":analysis([r for r in rows if tally[r["player"]]>=25])}
    atomic_json(run/"contextual_variance.json",x)
    first=x["all_published"]
    a=["# Contextual variability · 2023 retrospective",
       "Eligible same-row observed route and separation at throw only, REG and rookie calendar 2023 for the 2023 drafted WR/TE. These variables are excluded from pre-draft models. This report uses row-level separation S to distinguish it from the formal player-mean outcome Y. Variation in observed S is exploratory and does not replace R or Y.",
       f"Primary sample: {first['rows']} player-play observations, {first['players']} players, {first['games']} games and {first['player_games']} player-game groups. Observed routes: {first['observed_routes']}, coherent opponent teams: {first['observed_opponents']}. Median games per included player: {first['player_game_median']:.0f}.",
       "", "| Context field | Observed eligible rows |", "|---|---:|"]
    for k,num in first["counts"].items():
        a.append(f"| {k} | {num}/{first['rows']} |")
    a+=["","Opponent is the other team when team_abbr matches the verified home or visitor field, otherwise missing. Snap-to-pass timing comes from a unique player-game-play ball_snap and pass_forward in 2023 tracking with positive elapsed time. Missing, multiple, or incoherent markers retain an explicit unknown bin. Timing bins (seconds): "+
          ", ".join(f"{k} {v}" for k,v in first["times"].items())+". Official dropback_duration is separately audited and does not substitute for a missing matched interval.",
         "", "## Weighted law of total variance", "",
         "For any chosen C, Var(S) = E[Var(S|C)] + Var(E[S|C]). Primary gives equal total weight to each observed player, then weights their observed plays equally. Separate overall play-weighted sensitivity. Nested player→player-game decomposition separates between-player, between-game within player and within player-game. Route or route×timing partitions are separate and cannot be added to game or player components. The within-player context increment subtracts between-player variance from the joint player×route×time component.",
         "", "| Population / weight | Total | Between players | Between games within player | Within player-game | Between route×time | Within route×time | Within-player context increment |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key,z in x.items():
        for label,v in z["weights"].items():
            q=v["components"]
            values=(v["total"],q["player"]["between"],v["player_game_increment"],q["player_game"]["within"],
                    q["route_time"]["between"],q["route_time"]["within"],v["within_player_context_increment"])
            a.append(f"| {key} / {label} | "+" | ".join(f"{num:.4f}" for num in values)+" |")
    a+=["","Units are yd². Route-only decompositions, driver counts and marker failures are saved in contextual_variance.json.",
         "", "| Equal-player primary component | Crossed player/game bootstrap 95% diagnostic | Draws |",
         "|---|---|---:|"]
    for k,v in first["uncertainty"].items():
        lo,hi=v["interval"]
        a.append(f"| {k} | [{lo:.4f}, {hi:.4f}] | {v['draws']} |")
    a+=["",f"Player×route×timing cells: {first['player_context_cells']}, with {first['player_context_singletons']} singletons and {first['player_context_lt5']} fewer than five. Sampling diagnostics use product Poisson player and game weights, conditional on the observed release. No independence of repeated plays is assumed. Sparse context strata, play selection and measurement uncertainty prevent a complex random-effect model in this run.",
         "Future low-dimensional candidates are a pooled route/timing mean as a retrospective reference, then a random-intercept player model with game-level clustering if there is overlap across contexts and games. Within-game remaining variation mixes role, defender, quarterback, tracking, and residual error. A conditional residual cannot isolate a player's causal contribution or underlying movement ability."]
    (run/"contextual_variability_report.md").write_text("\n".join(a)+"\n")
    return dict(players=first["players"],rows=first["rows"],games=first["games"],
                player_game_groups=first["player_games"],audited_timing_rows=first["counts"]["seconds"])
