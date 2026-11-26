"""Compute only WR/TE field and stage availability for both outcome horizons."""
from collections import defaultdict
import json

import duckdb

from g22_motion import G1
from movement_probe import atomic_json

DRILLS=("SPEED_OUT_ROUTE_LEFT","CURL_ROUTE_RIGHT")
THRESHOLDS=(1,10,25,50)


def query(db):
    db.execute("""
      CREATE TEMP TABLE eligibility AS
      SELECT pp.nfl_id,pp.game_id,pp.play_id,g.season,pp.route_ran IS NOT NULL AS has_route,
        pp.separation_at_pass_forward IS NOT NULL AS has_sep
      FROM player_play pp JOIN games g USING(game_id) JOIN players p USING(nfl_id)
      WHERE p.nfl_position IN ('WR','TE') AND g.season_type='REG'
        AND g.season BETWEEN p.draft_year AND p.draft_year+1
    """)
    frames = []
    for year in (2023,2024,2025):
        frames.append(f"""
          SELECT t.nfl_id,t.game_id,t.play_id,time,event,{year} AS season
          FROM game_tracking_{year} t SEMI JOIN eligibility USING(nfl_id,game_id,play_id)
        """)
    rows = db.execute("""
      WITH frames AS ("""+" UNION ALL ".join(frames)+"""),
      time_bounds AS (
        SELECT *,lag(time) OVER w AS prev
        FROM frames WINDOW w AS(PARTITION BY nfl_id,game_id,play_id ORDER BY time)
      ), stages AS (
        SELECT nfl_id,game_id,play_id,season,
          min(time) FILTER (WHERE event='ball_snap') snap,
          min(time) FILTER (WHERE event='pass_forward') throw,
          count(*) FILTER (WHERE event='ball_snap') snap_count,
          count(*) FILTER (WHERE event='pass_forward') throw_count,
          min(time) AS first_frame,max(time) AS last_frame,
          count(*) FILTER (WHERE prev IS NOT NULL AND date_diff('microsecond',prev,time) NOT BETWEEN 99000 AND 101000) gaps
        FROM time_bounds GROUP BY ALL
      ), plays AS (
        SELECT e.nfl_id,e.season,count(*) player_plays,
          count(*) FILTER (WHERE has_route) route_rows,
          count(*) FILTER (WHERE has_route AND has_sep) label_rows,
          count(s.nfl_id) track_rows,
          count(*) FILTER (WHERE has_route AND has_sep AND s.nfl_id IS NOT NULL) label_matched,
          count(*) FILTER (WHERE has_route AND has_sep AND gaps=0 AND snap_count=1 AND throw_count=1
             AND date_diff('millisecond',snap,throw)>=600
             AND date_diff('millisecond',first_frame,throw)>=600
             AND date_diff('millisecond',throw,last_frame)>=600) stage_rows
        FROM eligibility e LEFT JOIN stages s USING(nfl_id,game_id,play_id,season) GROUP BY 1,2
      ), track AS (
        SELECT nfl_id,draft_year,
           count(*)>0 AS tracked,
           bool_or(drill_name='SPEED_OUT_ROUTE_LEFT') AS speed_out,
           bool_or(drill_name='CURL_ROUTE_RIGHT') AS curl
        FROM combine_tracking WHERE entity_type='PLAYER' GROUP BY 1,2
      ), years AS (SELECT unnest([0,1]) AS offset_year)
      SELECT p.nfl_id,p.draft_year,y.offset_year,
        p.draft_year+y.offset_year<2026 AS horizon_available,
        coalesce(t.tracked,false) tracked,coalesce(speed_out,false) speed_out,coalesce(curl,false) curl,
        b.forty IS NOT NULL AS forty,b.ten_yd_split IS NOT NULL AS split,
        b.short_shuttle IS NOT NULL AS shuttle,b.three_cone IS NOT NULL AS cone,
        coalesce(player_plays,0) player_plays,coalesce(route_rows,0) route_rows,
        coalesce(label_rows,0) label_rows,coalesce(track_rows,0) track_rows,
        coalesce(label_matched,0) label_matched,coalesce(stage_rows,0) stage_rows
      FROM players p CROSS JOIN years y LEFT JOIN track t USING(nfl_id,draft_year)
      LEFT JOIN combine_results b ON b.nfl_id=p.nfl_id
      LEFT JOIN plays pp ON pp.nfl_id=p.nfl_id AND pp.season=p.draft_year+y.offset_year
      WHERE p.nfl_position IN ('WR','TE')
      ORDER BY p.draft_year,y.offset_year,p.nfl_id
    """)
    names=[d[0] for d in rows.description]
    return [dict(zip(names,row)) for row in rows.fetchall()]


def outcomes(run):
    development=json.loads((run/"development_trials.json").read_text())
    eligible = {task:{r["player"] for r in development if r["drill"]==task and
                        any(len(r["methods"][m])==1 and r["methods"][m][0]["features"] is not None
                            for m in ("A","B","C"))} for task in DRILLS}
    with duckdb.connect(str(G1),read_only=True,config={"threads":2,"memory_limit":"4GB"}) as db:
        rows=query(db)
    groups=defaultdict(list)
    for row in rows:
        groups[(row["draft_year"],row["offset_year"])].append(row)
    results={}
    for (year,offset),group in groups.items():
        key=f"{year}/{offset}"
        def pair(predicate, fields):
            a,b=(set(r["nfl_id"] for r in group if predicate(r) and f(r)) for f in fields)
            return dict(speed_out=len(a),curl=len(b),intersection=len(a&b),union=len(a|b))
        flags=(lambda r:r["speed_out"],lambda r:r["curl"])
        valid=tuple(lambda r,task=task:r["nfl_id"] in eligible[task] for task in DRILLS)
        availability=dict(roster=len(group),tracked=sum(r["tracked"] for r in group),
            baselines={name:sum(r[name] for r in group) for name in ("forty","split","shuttle","cone")},
            complete_baselines=sum(all(r[name] for name in ("forty","split","shuttle","cone")) for r in group),
            drill_tracks=pair(lambda r:True,flags),
            measured=pair(lambda r:True,valid) if year==2023 else None)
        rowstats=None
        if group[0]["horizon_available"]:
            counts={n:sum(r[n] for r in group) for n in ("player_plays","route_rows","label_rows","track_rows",
                                                        "label_matched","stage_rows")}
            missing=dict(no_published_player_play=sum(r["player_plays"]==0 for r in group),
                         player_play_no_route=sum(r["player_plays"]>0 and not r["route_rows"] for r in group),
                         route_no_separation=sum(r["route_rows"]>0 and not r["label_rows"] for r in group),
                         labels_no_matching_tracking=sum(r["label_rows"]>0 and not r["label_matched"] for r in group),
                         matched_labels_no_stage=sum(r["label_matched"]>0 and not r["stage_rows"] for r in group))
            thresholds={}
            for n in THRESHOLDS:
                tiers={}
                for label,field in (("A","label_rows"),("B","stage_rows")):
                    pred=lambda r:r[field]>=n
                    cond=lambda r:pred(r) and r["forty"]
                    tiers[label]=dict(players=sum(pred(r) for r in group),
                        positive_below_threshold=sum(0<r[field]<n for r in group),
                        tracks_forty=sum(cond(r) and r["tracked"] for r in group),
                        measured_forty=pair(cond,valid) if year==2023 else None, drill_forty=pair(cond,flags))
                thresholds[n]=tiers
            rowstats=dict(opportunities=counts,availability_gaps=missing,thresholds=thresholds)
        results[key]=availability | {"outcomes":rowstats}
    atomic_json(run/"outcome_coverage.json",results)
    lines=["# E25 WR/TE outcome definition and availability",
           "G2.1 used the second season. Here rookie = draft year, second = draft year + 1. A requires only player_play with route and separation on the same regular-season row. B also requires a matching tracking key, unique snap and throw tags, no timestamp gaps, and ≥0.6 s between snap and throw and after throw. B is a candidate availability screen. Independent review is needed to establish that a route cut occurs during a comparable movement phase.",
           "A and B count players with the indicated minimum eligible rows. Counts in parentheses also require PLAYER Combine tracking and a nonmissing 40-yard time. Outcome values are excluded; null separation is never zero-filled.",
           "", "| Cohort / horizon | Roster | Tracking | Forty present | A thresholds 1 / 10 / 25 / 50 | B thresholds 1 / 10 / 25 / 50 |",
           "|---|---:|---:|---:|---|---|"]
    for key,v in results.items():
        stages=v["outcomes"]
        seq=lambda label:" / ".join(f"{d[label]['players']} ({d[label]['tracks_forty']})"
                                   for d in stages["thresholds"].values()) if stages else "unavailable"
        lines.append(f"| {key} | {v['roster']} | {v['tracked']} | {v['baselines']['forty']} | {seq('A')} | {seq('B')} |")
    lines+=["","## Within the 2023 development cohort",
            "A candidate Combine feature requires exactly one usable segment in an attempt under at least one prespecified method. This is algorithmic availability. Both tasks are tabulated separately, with intersections and unions. Held-out cohorts receive drill-record availability only.",
            "", "| Horizon / tier / minimum | Speed Out | Curl | Intersection | Union |",
            "|---|---:|---:|---:|---:|"]
    for key,v in results.items():
        if not key.startswith("2023"):
            continue
        for n,tiers in v["outcomes"]["thresholds"].items():
            for tier,d in tiers.items():
                lines.append(f"| {key} / {tier} / {n} | "+" | ".join(map(str,d["measured_forty"].values()))+" |")
    lines+=["","Per-horizon counts for other baseline tests, drill intersections/unions, raw opportunity totals, and each availability gap are in outcome_coverage.json. No published player-play row may mean no game participation or no coverage. The release alone cannot separate these possibilities. Player-play with no route, route with missing separation, missing tracking match, missing usable stages, and positive but below-threshold opportunities are counted separately.",
            "A two-stage design can represent probability of an observed eligible opportunity, then performance conditional on having it. True participation requires an independent roster/snap source. Published tracking selection, unequal opportunities, target selection and injury may alter the observed group. Do not condition the full-cohort endpoint on future opportunities without explicitly changing the estimand."]
    (run/"outcome_feasibility.md").write_text("\n".join(lines)+"\n")
