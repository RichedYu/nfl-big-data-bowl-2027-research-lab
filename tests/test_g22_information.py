"""Exact finite-support checks for the information laboratory."""
import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE = all(importlib.util.find_spec(x) is not None for x in ("numpy","matplotlib"))
if AVAILABLE:
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from g22_information import compression_lab, entropy, prediction_lab


@unittest.skipUnless(AVAILABLE, "G2.2 information tests require NumPy and Matplotlib")
class G22InformationTest(unittest.TestCase):
    def test_sufficiency_and_loss(self):
        out=compression_lab(np.random.default_rng(2202402))
        self.assertAlmostEqual(out["omitted_dynamic"]["traditional"]["truth"],0)
        self.assertAlmostEqual(out["omitted_dynamic"]["mechanism"]["truth"],1-entropy([.08,.92]))
        for v in out.values():
            self.assertAlmostEqual(v["mechanism"]["truth"],v["rich"]["truth"])

    def test_expected_loss_identity(self):
        out=prediction_lab(np.random.default_rng(22024))
        self.assertAlmostEqual(out["independent_junk"]["cmi"],0)
        self.assertGreater(out["noisy_signal"]["cmi"],0)
        for scenario in out.values():
            for model in scenario["models"].values():
                self.assertTrue(all(x["max_exact_residual"]<1e-12 for x in model["learning"]))
