"""Emit a reviewable aggregate dashboard summary from a complete audit."""

import argparse
import json
from pathlib import Path

from ingest import COMPETITION, TABLES


def summarize(audit):
    if audit["competition"] != COMPETITION:
        raise ValueError("Unexpected competition")
    expected = sorted(f"{name}.csv" for name in TABLES)
    if sorted(item["file"] for item in audit["files"]) != expected:
        raise ValueError("Expected one audit record for each of the nine files")
    return {"status": "audited", "audited_at_utc": audit["audited_at_utc"], "files": [
        {"file": item["file"], "rows": item["rows"], "unique_players": item["unique_players"], "fields": [
            {"name": field["name"], "missing_pct": field["missing_pct"]} for field in item["fields"]
        ]} for item in audit["files"]
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_json", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    print(json.dumps(summarize(audit), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
