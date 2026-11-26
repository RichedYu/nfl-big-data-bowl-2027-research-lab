"""Measure a small, prespecified set of Combine movement features."""
import argparse
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import os

import duckdb
import numpy as np

TASKS = {
    "SPEED_OUT_ROUTE_LEFT": "WR/TE",
    "CURL_ROUTE_RIGHT": "WR/TE",
    "BACK_PEDAL_AND_TRANSITION_45_DEGREE_REACTION": "DB",
    "BACK_PEDAL_AND_90_DEGREE_BREAK_TO_BALL": "DB",
}
LOW_SPEED = 1.0
WINDOW = 0.6
MIN_TURN = 30
SEED = 2101


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def kinematics(time, xy, smooth=1):
    if len(time) < max(7, 2 * smooth + 3) or np.any(np.diff(time) <= 0):
        raise ValueError("Require a strictly increasing time series with enough samples")
    if smooth:
        weights = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        xy = np.column_stack([np.convolve(xy[:, axis], weights, mode="valid") for axis in range(2)])
        time = time[smooth:-smooth]
    velocity = np.gradient(xy, time, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
    speed = np.linalg.norm(velocity, axis=1)
    cross = velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0]
    safe = np.where(speed >= LOW_SPEED, speed, np.nan)
    tangential = (velocity * acceleration).sum(axis=1) / safe
    normal = cross / safe
    return time, xy, speed, tangential, normal, cross / safe**3, np.arctan2(velocity[:, 1], velocity[:, 0])


def describe(curves):
    time, _, speed, tan, normal, curvature, angle = curves
    # A centered heading change locates the dominant turn inside the event.
    # Require two-sided support and moving endpoints.
    width = max(1, round(0.3 / np.median(np.diff(time))))
    center = np.arange(width + 2, len(time) - width - 2)
    delta = wrap(angle[center + width] - angle[center - width])
    valid = ((time[center] - time[0] >= WINDOW) & (time[-1] - time[center] >= WINDOW)
             & (speed[center - width] >= LOW_SPEED) & (speed[center + width] >= LOW_SPEED))
    if not valid.any():
        return None
    center, delta = center[valid], delta[valid]
    tied = np.flatnonzero(np.isclose(np.abs(delta), np.max(np.abs(delta)), atol=1e-8, rtol=0))
    plateau = np.split(tied, np.flatnonzero(np.diff(center[tied]) != 1) + 1)[0]
    i = plateau[len(plateau) // 2]
    k = center[i]
    if abs(np.degrees(delta[i])) < MIN_TURN:
        return None
    pre = (time >= time[k] - WINDOW - 1e-8) & (time < time[k] - 1e-8)
    post = (time > time[k] + 1e-8) & (time <= time[k] + WINDOW + 1e-8)
    turn = (np.abs(time - time[k]) <= .3 + 1e-8)
    entry = float(np.mean(speed[pre]))
    trough = k - width + np.argmin(speed[k-width:k+width+1])
    recovered = np.flatnonzero((np.arange(len(time)) > trough) & (speed >= .9 * entry))
    valid_acc = pre & np.isfinite(tan)
    valid_curv = turn & np.isfinite(curvature)
    if not valid_acc.any() or not valid_curv.any():
        return None
    return dict(cut_s=float(time[k]), turn_deg=float(abs(np.degrees(delta[i]))),
                entry=float(entry), exit=float(np.mean(speed[post])),
                braking=float(entry - speed[trough]),
                recovery_s=float(time[recovered[0]] - time[trough]) if len(recovered) else None,
                recovery_followup_s=float(time[-1]-time[trough]),
                peak_brake=float(np.max(np.maximum(0, -tan[valid_acc]))),
                mean_brake=float(np.mean(np.maximum(0, -tan[valid_acc]))),
                peak_normal=float(np.max(np.abs(normal[valid_curv]))),
                p75_normal=float(np.quantile(np.abs(normal[valid_curv]), .75)),
                peak_curvature=float(np.max(np.abs(curvature[valid_curv]))),
                p75_curvature=float(np.quantile(np.abs(curvature[valid_curv]), .75)))


def measure(time, xy, step=1, phase=0, smooth=1):
    # No interpolation or differentiation across missing frames within an event.
    if np.any(np.abs(np.diff(time) - .1) > .001):
        return None, None
    time, xy = time[phase::step], xy[phase::step]
    if len(time) < max(7, 2 * smooth + 3):
        return None, None
    curves = kinematics(time, xy, smooth)
    return describe(curves), curves


def digest(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def atomic_json(path, data):
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(partial, path)


def coverage(db):
    """Availability only. No held-out outcome values leave the query."""
    return db.execute("""
      WITH frame_keys AS (
        SELECT 2024 AS future_year,game_id,play_id,nfl_id FROM game_tracking_2024 GROUP BY ALL
        UNION ALL
        SELECT 2025,game_id,play_id,nfl_id FROM game_tracking_2025 GROUP BY ALL
      ), opportunities AS (
        SELECT f.future_year,f.nfl_id,count(*) matched_plays,count(pp.route_ran) routes,
          count(pp.separation_at_pass_forward) separation,
          count(*) FILTER(WHERE pp.route_ran IS NOT NULL AND pp.separation_at_pass_forward IS NOT NULL) route_sep,
          count(pp.coverage_assignment_at_snap) assigned_plays,count(pp.pressure_allowed) pressure_allowed,
          count(pp.player_get_off) get_off,sum(CASE WHEN pp.target THEN 1 ELSE 0 END) targets
        FROM frame_keys f JOIN games g USING(game_id)
        JOIN player_play pp USING(game_id,play_id,nfl_id)
        WHERE g.season_type='REG' AND g.season=f.future_year GROUP BY ALL
      ), track AS (
        SELECT DISTINCT draft_year,nfl_id FROM combine_tracking WHERE entity_type='PLAYER'
      ), roster AS (
        SELECT p.*, CASE WHEN nfl_position IN ('WR','TE') THEN 'WR/TE'
           WHEN nfl_position IN ('CB','FS','SS','DB') THEN 'DB'
           WHEN nfl_position IN ('C','G','T') THEN 'OL'
           WHEN nfl_position IN ('DE','DT','NT') THEN 'DL' END AS pos_group
        FROM players p
      )
      SELECT pos_group,p.draft_year,count(*) roster,count(t.nfl_id) tracked,
        count(*) FILTER(WHERE t.nfl_id IS NOT NULL AND c.forty IS NOT NULL) track_forty,
        count(*) FILTER(WHERE t.nfl_id IS NOT NULL AND c.forty IS NOT NULL AND c.short_shuttle IS NOT NULL AND c.three_cone IS NOT NULL) track_forty_agility,
        CASE WHEN p.draft_year < 2025 THEN count(o.nfl_id) END next_reg_matched,
        CASE WHEN p.draft_year < 2025 THEN count(*) FILTER(WHERE t.nfl_id IS NOT NULL AND c.forty IS NOT NULL AND o.nfl_id IS NOT NULL) END common,
        CASE WHEN p.draft_year < 2025 THEN count(*) FILTER(WHERE t.nfl_id IS NOT NULL AND c.forty IS NOT NULL AND
          CASE pos_group WHEN 'WR/TE' THEN route_sep WHEN 'DB' THEN assigned_plays WHEN 'OL' THEN pressure_allowed WHEN 'DL' THEN get_off END > 0) END common_role_label,
        CASE WHEN p.draft_year=2023 THEN sum(matched_plays) END dev_matched_plays,
        CASE WHEN p.draft_year=2023 THEN sum(routes) END dev_routes,
        CASE WHEN p.draft_year=2023 THEN sum(separation) END dev_separation,
        CASE WHEN p.draft_year=2023 THEN sum(assigned_plays) END dev_assignment,
        CASE WHEN p.draft_year=2023 THEN sum(targets) END dev_targets
      FROM roster p LEFT JOIN track t USING(nfl_id,draft_year)
      LEFT JOIN combine_results c ON c.nfl_id=p.nfl_id
      LEFT JOIN opportunities o ON o.nfl_id=p.nfl_id AND o.future_year=p.draft_year+1
      WHERE pos_group IS NOT NULL GROUP BY 1,2 ORDER BY 1,2
    """).fetchall()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run = args.output.name
    g1 = json.loads((args.g1 / "checkpoint.json").read_text())
    raw = Path("data/raw/nfl-big-data-bowl-2027")
    for entry in g1["fingerprint"]["inputs"]:
        if digest(raw / entry["file"]) != entry["sha256"]:
            raise ValueError("CSV fingerprint differs from G1")
    if g1["phase"] != "reports_complete":
        raise ValueError("G1 checkpoint must be complete")
    fingerprint = dict(run_id=run, g1_run=g1["run_id"], inputs=g1["fingerprint"]["inputs"],
                       g1_checkpoint_sha256=digest(args.g1 / "checkpoint.json"),
                       database_sha256=digest(args.g1 / "audit.duckdb"), source_sha256=digest(__file__),
                       git=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       numpy=np.__version__, duckdb=duckdb.__version__, seed=SEED,
                       development_cohort=2023, tasks=TASKS, low_speed=LOW_SPEED, window_s=WINDOW,
                       minimum_turn_deg=MIN_TURN, smoothing_halfwidth_frames=[0, 1, 2],
                       noise_coordinate_sd=[.03, .10], downsample_step=2)
    cp = args.output / "checkpoint.json"
    state = dict(fingerprint=fingerprint, phases={})
    if cp.exists():
        state = json.loads(cp.read_text())
        if state["fingerprint"] != fingerprint:
            raise ValueError("Fingerprint changed. Supply a new output/run ID.")
    with duckdb.connect(str(args.g1 / "audit.duckdb"), read_only=True) as db:
        cover = args.output / "coverage.json"
        if not cover.exists() or state["phases"].get("coverage") != digest(cover):
            atomic_json(cover, coverage(db))
            state["phases"]["coverage"] = digest(cover)
            atomic_json(cp, state)
        results = args.output / "trials.json"
        if results.exists() and state["phases"].get("measurements") == digest(results):
            return
        rows = db.execute("""
          WITH opportunities AS (
              SELECT pp.nfl_id, count(*) FILTER(WHERE pp.route_ran IS NOT NULL AND pp.separation_at_pass_forward IS NOT NULL) AS route_sep,
                count(pp.coverage_assignment_at_snap) AS assigned_plays
              FROM player_play pp JOIN games g USING(game_id)
              JOIN players p USING(nfl_id)
              SEMI JOIN game_tracking_2024 USING(game_id,play_id,nfl_id)
              WHERE g.season=2024 AND g.season_type='REG' AND p.draft_year=2023
              GROUP BY pp.nfl_id
          )
          SELECT c.nfl_id,event_id,attempt,drill_name,time,x,y,s,a,dis,dir,p.nfl_position,
            b.forty,b.short_shuttle,b.three_cone, coalesce(o.route_sep,0),coalesce(o.assigned_plays,0)
          FROM combine_tracking c JOIN players p USING(nfl_id) JOIN combine_results b USING(nfl_id) LEFT JOIN opportunities o USING(nfl_id)
          WHERE c.draft_year=2023 AND entity_type='PLAYER' AND drill_name IN (SELECT unnest(?))
            AND nfl_position IN ('WR','TE','CB','FS','SS','DB')
          ORDER BY drill_name,c.nfl_id,attempt,event_id,time
        """, [list(TASKS)]).fetchall()
    groups = defaultdict(list)
    for row in rows:
        if TASKS[row[3]] == ("WR/TE" if row[11] in ("WR", "TE") else "DB"):
            groups[row[:4]].append(row)
    rng = np.random.default_rng(SEED)
    trials = []
    for (player, event, attempt, drill), frames in groups.items():
        time = np.array([(row[4] - frames[0][4]).total_seconds() for row in frames])
        xy = np.array([row[5:7] for row in frames])
        base, curves = measure(time, xy)
        variants = {}
        for smooth in (0, 2):
            variants[f"smooth_{smooth}"] = measure(time, xy, smooth=smooth)[0]
        for phase in (0, 1):
            variants[f"5Hz_phase_{phase}"] = measure(time, xy, step=2, phase=phase, smooth=0)[0]
        for sd in (.03, .10):
            for draw in range(5):
                variants[f"noise_{sd}_{draw}"] = measure(time, xy + rng.normal(0, sd, xy.shape))[0]
        displacement = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        supplied = np.array([row[7:11] for row in frames])
        heading = np.degrees(np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0])))
        moving = displacement > .15
        dir_error = lambda expected: np.median(np.abs(np.degrees(wrap(np.radians(supplied[1:, 3][moving] - expected[moving])))))
        diagnostics = dict(
            dis_median_abs_error=float(np.median(np.abs(supplied[1:, 2] - displacement))),
            speed_median_abs_error=float(np.median(np.abs(supplied[1:, 0] - displacement / .1))),
            dis_lag_median_abs_error=float(np.median(np.abs(supplied[:-1, 2] - displacement))),
            delta_dis_median_abs_error=float(np.median(np.abs(np.diff(supplied[:, 2]) - displacement))),
            median_supplied_dis=float(np.median(supplied[1:, 2])),
            median_xy_dis=float(np.median(displacement)),
            negative_a=int(np.sum(supplied[:, 1] < 0)),
            dir_error_xccw=float(dir_error(heading)) if moving.any() else None,
            dir_error_ycw=float(dir_error(90-heading)) if moving.any() else None,
            dir_error_xcw=float(dir_error(-heading)) if moving.any() else None,
            dir_error_yccw=float(dir_error(heading-90)) if moving.any() else None)
        record = dict(player=player,event=event,attempt=attempt,drill=drill,position=frames[0][11],
                      baseline=list(frames[0][12:15]), opportunity_counts=list(frames[0][15:17]), frames=len(frames),base=base,variants=variants,
                      diagnostics=diagnostics)
        if base:
            record["curves"] = [np.where(np.isfinite(array), array, None).tolist() for array in curves[:-1]]
        trials.append(record)
    atomic_json(results, trials)
    state["phases"]["measurements"] = digest(results)
    atomic_json(cp, state)


if __name__ == "__main__":
    main()
