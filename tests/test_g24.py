"""Checks for blind review and design calculations using invented inputs."""
import importlib.util
import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

AVAILABLE=all(importlib.util.find_spec(x) for x in ("numpy","scipy"))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from g24_annotation import FIELDS, compare, load, pair_events


@unittest.skipUnless(AVAILABLE, "G2.4 uses NumPy and SciPy")
class G24AnnotationTest(unittest.TestCase):
    @staticmethod
    def event(start,end,center,index=1):
        return dict(onset_s=start,exit_s=end,bend_center_s=center,bend_start_s=None,bend_end_s=None,
                    index=index,direction="left",confidence="medium")

    def test_pairing_checks_annotated_center_and_interval(self):
        a=self.event(0,2,.4)
        b=self.event(0,2,1.5)
        self.assertEqual(pair_events([a],[b]),[])
        b=self.event(0,2,.5)
        self.assertEqual(pair_events([a,self.event(0,2,.55,2)],[b]),[(0,0,1.)])
        self.assertEqual(pair_events([a],[self.event(.39,.42,.4)]),[])

    def test_status_denominators(self):
        left={"x":[("events",self.event(0,2,.5))],"y":[("unjudgeable",None)],"z":[("no_event",None)]}
        right={"x":[("events",self.event(.1,2,.7))],"y":[("events",self.event(0,2,.5))],"z":[("no_event",None)]}
        report=compare(left,right)
        self.assertEqual(report["comparable_cases"],2)
        self.assertEqual(report["matched_events"],1)
        self.assertEqual(report["estimates"]["matched_f1"],1)
        self.assertAlmostEqual(report["estimates"]["bend_center_s_mae"],.2)
        self.assertIn("unjudgeable × events",report["status_counts"])

    def test_missing_or_impossible_annotation_fails(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"sheet.csv"
            row=dict(case="x",reviewer="A",review_status="events",event_index=1,onset_s=-1,
                     direction="left",confidence="medium",reason="partial")
            with p.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=FIELDS)
                w.writeheader()
                w.writerow(row)
            with self.assertRaisesRegex(ValueError,"outside"):
                load(p,{"x":2})
