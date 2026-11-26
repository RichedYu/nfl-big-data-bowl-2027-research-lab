"""Competition file inventory and strict CSV loading."""

import hashlib
import re
from pathlib import Path


COMPETITION = "nfl-big-data-bowl-2027"
SEASONS = (2023, 2024, 2025)
GAMES_TRACKING = tuple(f"game_tracking_{year}" for year in SEASONS)
TABLES = (
    "players", "combine_results", "player_career_successes", "games",
    "combine_tracking", "player_play", *GAMES_TRACKING,
)
NULL_MARKERS = ("", "NA", "N/A", "NULL", "NaN")


def identifier(name):
    return '"' + name.replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def discover(directory):
    paths = {}
    for name in TABLES:
        matches = sorted(directory.rglob(f"{name}.csv"))
        if len(matches) != 1:
            raise ValueError(f"{name}.csv: expected one file under {directory}, found {len(matches)}")
        paths[name] = matches[0]
    return paths


def identity(path: Path):
    with path.open("rb") as data:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(data, "sha256").hexdigest()}


def load_csv(db, name, path):
    nulls = ", ".join(map(literal, NULL_MARKERS))
    db.execute(f"""
        CREATE TABLE {identifier(name)} AS
        SELECT * FROM read_csv(?, header=true, delim=',', sample_size=-1,
                               strict_mode=true, nullstr=[{nulls}])
    """, [str(path)])
    return db.execute(f"DESCRIBE {identifier(name)}").fetchall()


def resolve(schema, *aliases):
    names = {normalize(field[0]): field[0] for field in schema}
    for alias in aliases:
        match = names.get(normalize(alias))
        if match is not None:
            return match
    return None
