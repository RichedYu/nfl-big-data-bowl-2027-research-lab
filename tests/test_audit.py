import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit import run_audit, write_reports
from dashboard_summary import summarize
from ingest import GAMES_TRACKING, discover


class DataAuditTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "competition"
        self.data.mkdir()
        files = {
            "players": "nfl_id,nfl_position,draft_year,name\n1,WR,2023,A\n2,DB,2024,B\n2,DB,2024,B\n3,TE,2025,C\n",
            "combine_results": "nflId,forty_yard_dash,note\n1,4.5,ok\n2,NA,   \n4,4.9,\n",
            "player_career_successes": "nfl_id,games_played\n1,0\n2,4\n3,1\n",
            "games": "game_id,season\n10,2023\n11,2024\n12,2025\n",
            "combine_tracking": (
                "event_id,nfl_id,entity_type,drill_name,draft_year,time,x\n"
                "100,1,PLAYER,turn,2023,2023-01-01T00:00:00,1\n"
                "100,1,PLAYER,turn,2023,2023-01-01T00:00:00.1,2\n"
                "100,,BALL,turn,2023,2023-01-01T00:00:00.1,4\n"
                "101,1,PLAYER,turn,2023,2023-01-01T00:00:01,2\n"
                "200,2,PLAYER,turn,2024,2023-01-01T00:00:02,2\n"
            ),
            "player_play": "game_id,play_id,nfl_id,route_ran\n10,101,1,GO\n10,101,1,GO\n10,102,2,\n",
        }
        tracking = "game_id,play_id,nfl_id,frame_id,x\n10,101,1,1,5\n10,101,1,2,6\n10,103,2,1,2\n11,201,,1,7\n"
        for name in GAMES_TRACKING:
            files[name] = tracking
        for name, content in files.items():
            (self.data / f"{name}.csv").write_text(content, encoding="utf-8")

    def test_nine_file_audit_and_public_summary(self):
        report = run_audit(self.data, self.root, "256MB", 1)
        files = {record["file"]: record for record in report["files"]}
        self.assertEqual(len(files), 9)
        self.assertEqual(files["players.csv"]["unique_players"], 3)
        self.assertEqual(files["players.csv"]["key"]["extra_rows"], 1)
        self.assertEqual(files["combine_tracking.csv"]["key"]["checked_rows"], 4)
        self.assertIsNone(files["games.csv"]["unique_players"])
        fields = {field["name"]: field for field in files["combine_results.csv"]["fields"]}
        self.assertEqual(fields["forty_yard_dash"]["null"], 1)
        self.assertEqual(fields["note"]["whitespace_only"], 1)
        self.assertAlmostEqual(fields["note"]["missing_pct"], 66.6667)
        relation = next(rel for rel in report["relationships"] if
                        (rel["child"], rel["parent"]) == ("game_tracking_2023.csv", "player_play.csv"))
        self.assertEqual((relation["matched_rows"], relation["null_key_rows"], relation["unmatched_rows"]), (2, 1, 1))
        self.assertEqual(relation["matched_parent_keys"], 1)
        self.assertEqual(relation["parent_keys"], 2)
        wr = next(group for group in report["cohorts"]["combine_coverage"] if group["position"] == "WR")
        self.assertEqual((wr["attempts"], wr["players_with_repeats"], wr["median_frames"]), (2, 1, 1.5))
        self.assertEqual(report["cohorts"]["combine_year_field"], "draft_year")
        resumed = run_audit(self.data, self.root, "256MB", 1)
        self.assertEqual(resumed["run_id"], report["run_id"])
        self.assertEqual(resumed["files"], report["files"])
        checkpoint = json.loads((self.root / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["phase"], "audit_compute_complete")
        self.assertEqual(len(checkpoint["tables"]), 9)
        self.assertEqual(len(checkpoint["relationships"]), 14)
        output = self.root / "reports"
        write_reports(report, output)
        self.assertEqual(json.loads((output / "audit.json").read_text()), report)
        self.assertIn("66.6667%", (output / "data_dictionary.md").read_text())
        self.assertIn("孤立 1 行", (output / "data_audit.md").read_text())
        exported = summarize(report)
        self.assertEqual(exported["status"], "audited")
        self.assertEqual(set(exported["files"][0]), {"file", "rows", "unique_players", "fields"})
        self.assertEqual(set(exported["files"][0]["fields"][0]), {"name", "missing_pct"})

    def test_ambiguous_input_fails(self):
        (self.data / "other").mkdir()
        (self.data / "other" / "players.csv").touch()
        with self.assertRaisesRegex(ValueError, "players.csv: expected one file.*found 2"):
            discover(self.data)

    def test_checkpoint_rejects_changed_input(self):
        run_audit(self.data, self.root, "256MB", 1)
        players = self.data / "players.csv"
        players.write_text(players.read_text() + "4,WR,2025,D\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            run_audit(self.data, self.root, "256MB", 1)
