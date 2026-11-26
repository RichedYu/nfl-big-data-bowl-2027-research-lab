import importlib.util
import sys
from pathlib import Path
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
if HAS_NUMPY:
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from movement_probe import describe, kinematics, measure, wrap


@unittest.skipUnless(HAS_NUMPY, "Movement probe requires NumPy")
class MovementProbeTest(unittest.TestCase):
    def test_straight_constant_acceleration(self):
        t = np.linspace(0, 2, 21)
        _, _, speed, tan, normal, kappa, _ = kinematics(t, np.c_[2*t+t*t, 0*t], 0)
        np.testing.assert_allclose(speed, 2+2*t, atol=1e-10)
        np.testing.assert_allclose(tan, 2, atol=1e-10)
        np.testing.assert_allclose(normal, 0, atol=1e-10)
        np.testing.assert_allclose(kappa, 0, atol=1e-10)
        self.assertIsNone(describe(kinematics(t, np.c_[2*t, 0*t], 0)))

    def test_circle(self):
        t = np.linspace(0, 4, 401)
        _, _, speed, tan, normal, kappa, _ = kinematics(t, np.c_[4*np.cos(t), 4*np.sin(t)], 0)
        for actual, expected in ((speed, 4), (tan, 0), (normal, 4), (kappa, .25)):
            np.testing.assert_allclose(actual[2:-2], expected, atol=.001)

    def test_wrap_low_speed_and_gap(self):
        self.assertAlmostEqual(np.degrees(wrap(np.radians(-358))), 2)
        t = np.arange(21)*.1
        self.assertTrue(np.isnan(kinematics(t, np.c_[0*t, 0*t], 0)[3]).all())
        self.assertEqual(measure(np.delete(t, 7), np.c_[np.delete(t, 7), np.delete(t, 7)]), (None,None))

    def test_known_cut_and_recovery(self):
        t = np.arange(41)*.1
        xy = np.c_[4*np.minimum(t,2), 4*np.maximum(0,t-2)]
        feature, _ = measure(t, xy, smooth=0)
        self.assertAlmostEqual(feature["turn_deg"], 90)
        self.assertAlmostEqual(feature["entry"], 4)
        self.assertAlmostEqual(feature["exit"], 4)
        self.assertAlmostEqual(feature["braking"], 4-np.sqrt(8))
        self.assertAlmostEqual(feature["recovery_s"], .1)
        self.assertLessEqual(abs(feature["cut_s"]-2), .3+1e-8)
