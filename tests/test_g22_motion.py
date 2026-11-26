"""Known-answer checks for the G2.2 estimators."""
import importlib.util
from pathlib import Path
import sys
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from g22_motion import detect, estimates, match, metrics, trajectory


@unittest.skipUnless(HAS_NUMPY, "G2.2 requires NumPy")
class G22MotionTest(unittest.TestCase):
    def test_analytic_turn_and_speed(self):
        ref = trajectory(np.arange(4001)*.001,6,[(1.4,.8,135,1.2)])
        f = metrics(ref,1.4,2.2)
        self.assertAlmostEqual(f["angle"],135)
        self.assertAlmostEqual(f["speed_loss"],1.2)
        self.assertAlmostEqual(f["retention"],1)
        self.assertAlmostEqual(f["recovery"],.2)

    def test_straight_and_opposite_turns(self):
        t = np.arange(41)*.1
        for m in ("A","B","C"):
            self.assertFalse(detect(estimates(t,np.c_[3*t,0*t],2),m)[0])
        hi = trajectory(np.arange(4001)*.001,4,[(1,.5,60,0),(2.5,.6,-80,1)])
        for m in ("B","C"):
            turn, _ = detect(estimates(t,hi[1][::100],2),m)
            self.assertEqual(len(turn),2)
            self.assertGreater(min(row[3] for row in turn),35)

    def test_one_to_one_match(self):
        f = lambda x: dict(center=x)
        self.assertEqual(match([f(1),f(1.1),f(3)],[f(1.04),f(3)]),[(2,1),(0,0)])
