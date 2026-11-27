"""Known-answer checks for G2.3 motion, causal and prediction contracts."""
import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE=all(importlib.util.find_spec(x) is not None for x in ("numpy","scipy","matplotlib"))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from g23_motion import METHODS,bookend_angle,corner_truth,fit,global_repr,match_intervals
    from g23_causal import experiment
    from g23_football import pilot,train
    from g23_reliability import truths


@unittest.skipUnless(AVAILABLE,"G2.3 requires NumPy, SciPy and Matplotlib")
class G23Test(unittest.TestCase):
    def test_straight_and_corner(self):
        t=np.arange(41)*.1
        xy=np.c_[4*t,0*t]
        for name in METHODS:
            g,p,v,a=fit(t,xy,name)
            np.testing.assert_allclose(p,np.c_[4*g,0*g],atol=1e-10)
            np.testing.assert_allclose(v[:,0],4,atol=1e-10)
            np.testing.assert_allclose(a,0,atol=1e-9)
            self.assertAlmostEqual(bookend_angle(g,v,1,2),0)
        time=np.array([.0005,.001,.0015])
        h,r=corner_truth(time,0,.002,90)
        np.testing.assert_allclose(np.gradient(h,time),r)
        self.assertAlmostEqual(global_repr(t,xy)["speed"],4)

    def test_interval_overlap_and_uniqueness(self):
        self.assertEqual(match_intervals([(1.29,1.31)],[(1,1.6)]),[])
        self.assertEqual(match_intervals([(1,1.6),(1.1,1.5)],[(1,1.6)]),[(0,0,1.0)])

    def test_causal_worlds(self):
        o=experiment()
        np.testing.assert_allclose(o["A_direct"]["observed_joint"],o["A_latent"]["observed_joint"])
        self.assertAlmostEqual(o["A_direct"]["effect"],.6)
        self.assertAlmostEqual(o["A_latent"]["effect"],0)
        self.assertGreater(o["C_selection"]["scopes"]["selected"]["diagnostics"]["M|-"]["cmi"],0)

    def test_reliability_targets(self):
        baseline=truths(1,1,"balanced")
        self.assertAlmostEqual(baseline[1],.5)
        self.assertAlmostEqual(truths(1,1,"order")[3]-baseline[3],.6)

    def test_pilot_fits_only_training_labels(self):
        x=np.array([[1.],[2.],[3.],[4.]])
        y=np.array([0,1,0,999])
        self.assertAlmostEqual(train(x[:,:0],y,[0,1,2],[3],False)[0],1/3)
        self.assertEqual(pilot([(1,1,[1,0,1])],True)["status"],"stopped_sample_gate")
