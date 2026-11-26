"""Run the phase-one audit against nine original competition CSVs."""

import argparse
import hashlib
import json
import platform
import subprocess
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from ingest import COMPETITION, GAMES_TRACKING, NULL_MARKERS, discover, identity, load_csv, resolve
from ingest import identifier as q


PLAYER = ("nfl_id", "nflId")
GAME = ("game_id", "gameId")
PLAY = ("play_id", "playId")
TIME = ("time", "timestamp", "frame_id", "frameId")
EVENT = ("event_id", "eventId")
POSITION = ("nfl_position", "position", "position_abbr", "position_group", "official_position")
DRAFT = ("draft_year", "entry_year", "draft_class")
DRILL = ("drill_name", "drill_type")
KEYS = {
    "players": (PLAYER,),
    "combine_results": (PLAYER,),
    "player_career_successes": (PLAYER,),
    "games": (GAME,),
    "combine_tracking": (EVENT, PLAYER, TIME),
    "player_play": (GAME, PLAY, PLAYER),
    **{name: (GAME, PLAY, PLAYER, TIME) for name in GAMES_TRACKING},
}


def table_profile(db, name, schema):
    table = q(name)
    columns = [q(field[0]) for field in schema]
    nulls = [f"count(*) - count({column})" for column in columns]
    whitespace = [
        f"count(*) FILTER (WHERE {column} <> '' AND trim({column}) = '')" if field[1] == "VARCHAR" else "0"
        for column, field in zip(columns, schema)
    ]
    player = resolve(schema, *PLAYER)
    player_count = f"count(DISTINCT {q(player)})" if player else "NULL"
    rows, players, *counts = db.execute(
        f"SELECT {', '.join(['count(*)', player_count, *nulls, *whitespace])} FROM {table}"
    ).fetchone()
    return {
        "rows": rows,
        "unique_players": players,
        "fields": [
            {"name": field[0], "type": field[1], "null": counts[index],
             "whitespace_only": counts[len(schema) + index],
             "missing_pct": round(100 * (counts[index] + counts[len(schema) + index]) / rows, 4) if rows else None}
            for index, field in enumerate(schema)
        ],
    }


def key_profile(db, name, schema):
    fields = [resolve(schema, *aliases) for aliases in KEYS[name]]
    unresolved = [aliases[0] for aliases, field in zip(KEYS[name], fields) if field is None]
    if unresolved:
        return {"status": "unresolved", "missing_fields": unresolved}
    columns = ", ".join(map(q, fields))
    complete = " AND ".join(f"{q(field)} IS NOT NULL" for field in fields)
    where = "TRUE"
    entity = resolve(schema, "entity_type") if name == "combine_tracking" else None
    if entity:
        where = f"upper(trim(cast({q(entity)} AS VARCHAR))) = 'PLAYER'"
    checked, nulls = db.execute(f"""
        SELECT count(*), count(*) FILTER (WHERE NOT ({complete})) FROM {q(name)} WHERE {where}
    """).fetchone()
    groups, extra = db.execute(f"""
        SELECT count(*) FILTER (WHERE n > 1), coalesce(sum(n - 1), 0) FROM (
            SELECT count(*) AS n FROM {q(name)} WHERE {where} AND {complete} GROUP BY {columns}
        )
    """).fetchone()
    return {"status": "checked", "fields": fields, "scope": "player entity" if entity else "all rows",
            "checked_rows": checked, "null_key_rows": nulls, "duplicate_groups": groups, "extra_rows": extra}


def relationship(db, schemas, child, parent, keys):
    left = [resolve(schemas[child], *aliases) for aliases in keys]
    right = [resolve(schemas[parent], *aliases) for aliases in keys]
    result = {"child": f"{child}.csv", "parent": f"{parent}.csv", "candidate": [key[0] for key in keys]}
    if None in left or None in right:
        return result | {"status": "unresolved", "child_fields": left, "parent_fields": right}

    def grouped(table, columns):
        identities = ", ".join(f"cast({q(col)} AS VARCHAR) AS id_{i}" for i, col in enumerate(columns))
        return f"SELECT {identities}, count(*) AS n FROM {q(table)} GROUP BY ALL"

    join_keys = ", ".join(f"id_{i}" for i in range(len(keys)))
    complete = " AND ".join(f"c.id_{i} IS NOT NULL" for i in range(len(keys)))
    matched, missing, unmatched, orphan_keys, hits, parents = db.execute(f"""
        WITH c AS ({grouped(child, left)}), p AS ({grouped(parent, right)})
        SELECT coalesce(sum(c.n) FILTER (WHERE p.n IS NOT NULL), 0),
               coalesce(sum(c.n) FILTER (WHERE c.n IS NOT NULL AND NOT ({complete})), 0),
               coalesce(sum(c.n) FILTER (WHERE {complete} AND p.n IS NULL), 0),
               count(*) FILTER (WHERE {complete} AND p.n IS NULL),
               count(*) FILTER (WHERE c.n IS NOT NULL AND p.n IS NOT NULL),
               count(*) FILTER (WHERE p.n IS NOT NULL AND {" AND ".join(f"p.id_{i} IS NOT NULL" for i in range(len(keys)))})
        FROM c FULL OUTER JOIN p USING ({join_keys})
    """).fetchone()
    return result | {"status": "checked", "child_fields": left, "parent_fields": right,
                     "matched_rows": matched, "null_key_rows": missing, "unmatched_rows": unmatched,
                     "unmatched_keys": orphan_keys, "matched_parent_keys": hits, "parent_keys": parents}


def cohort_profile(db, schemas):
    players = schemas["players"]
    player, position, draft = (resolve(players, *field) for field in (PLAYER, POSITION, DRAFT))
    result = {"positions": None, "draft_years": None, "combine_coverage": None, "game_coverage": None}
    if not player:
        return result
    for key, field in (("positions", position), ("draft_years", draft)):
        if field:
            result[key] = [
                {"group": group, "players": count}
                for group, count in db.execute(f"""
                    SELECT cast({q(field)} AS VARCHAR), count(DISTINCT {q(player)})
                    FROM players WHERE {q(player)} IS NOT NULL GROUP BY 1 ORDER BY 1
                """).fetchall()
            ]

    # A player with multiple conflicting demographic rows is not assigned an arbitrary cohort.
    position_expr = f"min(cast({q(position)} AS VARCHAR))" if position else "NULL::VARCHAR"
    draft_expr = f"min(cast({q(draft)} AS VARCHAR))" if draft else "NULL::VARCHAR"
    checks = [f"count(DISTINCT {q(field)}) <= 1" for field in (position, draft) if field]
    reliable = " AND ".join(checks) or "TRUE"
    db.execute(f"""
        CREATE TEMP TABLE audit_roster AS
        SELECT cast({q(player)} AS VARCHAR) AS player_id, {position_expr} AS position, {draft_expr} AS draft_year
        FROM players WHERE {q(player)} IS NOT NULL GROUP BY 1 HAVING {reliable}
    """)
    result["conflicting_roster_ids"] = db.execute(f"""
        SELECT count(DISTINCT {q(player)}) - (SELECT count(*) FROM audit_roster) FROM players
    """).fetchone()[0]
    combine = schemas["combine_tracking"]
    track_id, event, drill, time = (resolve(combine, *field) for field in (PLAYER, EVENT, DRILL, TIME))
    if track_id and event and drill:
        tracked_year = resolve(combine, "combine_year", "draft_year", "year", "season")
        year_expr = f"min(cast(t.{q(tracked_year)} AS VARCHAR))" if tracked_year else "max(r.draft_year)"
        time_count = f"count(DISTINCT t.{q(time)})" if time else "NULL::BIGINT"
        entity = resolve(combine, "entity_type")
        player_only = f"AND upper(trim(cast(t.{q(entity)} AS VARCHAR))) = 'PLAYER'" if entity else ""
        columns = "position year drill players attempts players_with_repeats min_frames median_frames max_frames one_frame_attempts duplicate_or_null_time_rows".split()
        result["combine_coverage"] = [dict(zip(columns, row)) for row in db.execute(f"""
            WITH attempts AS (
                SELECT cast(t.{q(track_id)} AS VARCHAR) AS id, t.{q(event)} AS event_id,
                       cast(t.{q(drill)} AS VARCHAR) AS drill, max(r.position) AS position,
                       {year_expr} AS cohort_year, count(*) AS rows, {time_count} AS frames
                FROM combine_tracking t LEFT JOIN audit_roster r ON cast(t.{q(track_id)} AS VARCHAR) = r.player_id
                WHERE t.{q(track_id)} IS NOT NULL AND t.{q(event)} IS NOT NULL {player_only}
                GROUP BY 1, 2, 3
            ), repeated AS (
                SELECT *, count(*) OVER (PARTITION BY id, position, cohort_year, drill) AS reps FROM attempts
            )
            SELECT position, cohort_year, drill, count(DISTINCT id), count(*),
                   count(DISTINCT id) FILTER (WHERE reps > 1), min(frames), median(frames), max(frames),
                   count(*) FILTER (WHERE frames = 1), sum(rows - frames)
            FROM repeated GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
        """).fetchall()]
        result["combine_year_field"] = tracked_year or draft
        result["combine_time_field"] = time
        result["combine_drill_field"] = drill
    result["game_coverage"] = []
    for name in GAMES_TRACKING:
        player_id, game, play = (resolve(schemas[name], *field) for field in (PLAYER, GAME, PLAY))
        if not all((player_id, game, play)):
            result["game_coverage"].append({"file": f"{name}.csv", "status": "unresolved"})
            continue
        columns = "position draft_year players player_plays frames".split()
        groups = [dict(zip(columns, row)) for row in db.execute(f"""
            SELECT r.position, r.draft_year, count(DISTINCT t.{q(player_id)}),
                   count(DISTINCT (t.{q(game)}, t.{q(play)}, t.{q(player_id)}))
                       FILTER (WHERE t.{q(game)} IS NOT NULL AND t.{q(play)} IS NOT NULL), count(*)
            FROM {q(name)} t LEFT JOIN audit_roster r ON cast(t.{q(player_id)} AS VARCHAR) = r.player_id
            WHERE t.{q(player_id)} IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2
        """).fetchall()]
        result["game_coverage"].append({"file": f"{name}.csv", "status": "checked", "groups": groups})
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def code_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_audit(data_dir, scratch, memory_limit="2GB", threads=2, run_id=None):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    paths = discover(data_dir)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    source_dir = Path(__file__).resolve().parent
    inputs = [{"file": path.name, **identity(path)} for path in paths.values()]
    fingerprint = {
        "inputs": inputs,
        "commit": git.stdout.strip() if git.returncode == 0 else None,
        "audit_sha256": code_identity(__file__),
        "ingest_sha256": code_identity(source_dir / "ingest.py"),
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "memory_limit": memory_limit,
        "threads": threads,
    }
    checkpoint_path = scratch / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError("Checkpoint fingerprint mismatch; use a new run ID/directory instead of skipping prior work")
    else:
        checkpoint = {
            "run_id": run_id or datetime.now(timezone.utc).strftime("g1-%Y%m%dT%H%M%S%fZ"),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "phase": "table_profiles",
            "tables": {},
            "relationships": {},
            "cohorts": None,
        }
        atomic_json(checkpoint_path, checkpoint)

    with duckdb.connect(str(scratch / "audit.duckdb"), config={
        "memory_limit": memory_limit, "threads": threads, "preserve_insertion_order": False,
    }) as db:
        schemas, profiles = {}, []
        for name, path in paths.items():
            present = db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [name]).fetchone()[0]
            cached = checkpoint["tables"].get(name)
            if not (cached and present):
                if present:
                    db.execute(f"DROP TABLE {q(name)}")
                schema = load_csv(db, name, path)
                cached = {"schema": schema, **table_profile(db, name, schema), "key": key_profile(db, name, schema),
                          "source_sha256": identity(path)["sha256"], "status": "complete"}
                checkpoint["tables"][name] = cached
                checkpoint["phase"] = "table_profiles"
                atomic_json(checkpoint_path, checkpoint)
            schemas[name] = [tuple(field) for field in cached["schema"]]
            profiles.append({"file": path.name, **identity(path), "rows": cached["rows"],
                             "unique_players": cached["unique_players"], "fields": cached["fields"],
                             "key": cached["key"]})

        relation_specs = [(name, "players", (PLAYER,)) for name in paths if name not in {"players", "games"}]
        relation_specs += [(name, "games", (GAME,)) for name in ("player_play", *GAMES_TRACKING)]
        relation_specs += [(name, "player_play", (GAME, PLAY, PLAYER)) for name in GAMES_TRACKING]
        checkpoint["phase"] = "cross_table_relations"
        relations = []
        for child, parent, keys in relation_specs:
            task_id = f"{child}->{parent}"
            result = checkpoint["relationships"].get(task_id)
            if result is None:
                result = relationship(db, schemas, child, parent, keys)
                checkpoint["relationships"][task_id] = result
                atomic_json(checkpoint_path, checkpoint)
            relations.append(result)
        checkpoint["phase"] = "cohort_profiles"
        cohorts = checkpoint.get("cohorts")
        if cohorts is None:
            cohorts = cohort_profile(db, schemas)
            checkpoint["cohorts"] = cohorts
            atomic_json(checkpoint_path, checkpoint)
    checkpoint["phase"] = "audit_compute_complete"
    checkpoint["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(checkpoint_path, checkpoint)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True)
    return {"competition": COMPETITION, "run_id": checkpoint["run_id"],
            "audited_at_utc": checkpoint.get("completed_at_utc", datetime.now(timezone.utc).isoformat()),
            "commit": git.stdout.strip() if git.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "python": platform.python_version(), "duckdb": duckdb.__version__, "null_markers": list(NULL_MARKERS),
            "files": profiles, "relationships": relations, "cohorts": cohorts}


def write_reports(report, output):
    output.mkdir(parents=True, exist_ok=True)
    def write_text_atomic(path, content):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    write_text_atomic(output / "audit.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# NFL Big Data Bowl 2027 数据审计", "", f"- UTC 时间 `{report['audited_at_utc']}`",
             f"- Git `{report['commit'] or 'unavailable'}`，存在未提交修改 `{report['git_dirty']}`",
             f"- Python `{report['python']}`，DuckDB `{report['duckdb']}`", "", "## 文件与球员覆盖", "",
             "| 文件 | 字节 | 行数 | 字段 | 唯一球员 |", "|---|---:|---:|---:|---:|"]
    for record in report["files"]:
        count = record['unique_players'] if record['unique_players'] is not None else '—'
        lines.append(f"| {record['file']} | {record['bytes']:,} | {record['rows']:,} | {len(record['fields'])} | {count} |")
    lines += ["", "## 候选键和关联", ""]
    for record in report["files"]:
        key = record["key"]
        detail = f"空键 {key['null_key_rows']:,}，重复额外行 {key['extra_rows']:,}" if key['status'] == 'checked' else f"待确认字段 {key['missing_fields']}"
        lines.append(f"- `{record['file']}` 候选键。{detail}。")
    for rel in report["relationships"]:
        detail = (f"匹配 {rel['matched_rows']:,} 行，空键 {rel['null_key_rows']:,} 行，孤立 {rel['unmatched_rows']:,} 行"
                  if rel['status'] == 'checked' else "候选字段待确认")
        lines.append(f"- `{rel['child']}` → `{rel['parent']}`。{detail}。")
    lines += ["", "## 字段和缺失值", "", "详见 [字段表](data_dictionary.md)。`audit.json` 包含文件指纹、关联覆盖、位置与选秀年、Combine 重复测量和分赛季比赛追踪覆盖。"]
    write_text_atomic(output / "data_audit.md", "\n".join(lines) + "\n")
    dictionary = ["# 字段表", "", "类型为 DuckDB 对完整 CSV 的推断。缺失包含空值与纯空格。"]
    for record in report["files"]:
        dictionary += ["", f"## {record['file']}", "", f"SHA-256 `{record['sha256']}`", "",
                       "| 字段 | 类型 | 空值 | 纯空格 | 缺失比例 |", "|---|---|---:|---:|---:|"]
        for col in record["fields"]:
            name = col["name"].replace("|", "\\|")
            rate = f"{col['missing_pct']:.4f}%" if col['missing_pct'] is not None else "—"
            dictionary.append(f"| `{name}` | `{col['type']}` | {col['null']:,} | {col['whitespace_only']:,} | {rate} |")
    write_text_atomic(output / "data_dictionary.md", "\n".join(dictionary) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch-dir", type=Path, default=Path("data/audit_runs"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("g1-%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be a positive number")
    output = args.output or Path("reports/runs") / args.run_id
    run_dir = args.scratch_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = run_audit(args.data_dir, run_dir, args.memory_limit, args.threads, args.run_id)
    except ValueError as error:
        parser.error(str(error))
    write_reports(report, output)
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    checkpoint["phase"] = "reports_complete"
    checkpoint["report_directory"] = str(output)
    checkpoint["report_files"] = {name: identity(output / name) for name in
                                  ("audit.json", "data_audit.md", "data_dictionary.md")}
    atomic_json(run_dir / "checkpoint.json", checkpoint)
    print(output / "data_audit.md")


if __name__ == "__main__":
    main()
