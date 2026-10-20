"""Tests for the Phase-1 measured-constants capacity model."""

import json
import unittest
from pathlib import Path

import phase1_model as p1

ROOT = Path(__file__).resolve().parent


class ExpertUnion(unittest.TestCase):
    def test_monotonic_and_bounded(self):
        u = [p1.expert_union_per_layer(t) for t in (4, 16, 32, 64, 128)]
        self.assertTrue(all(x < y for x, y in zip(u, u[1:])))
        self.assertTrue(all(x <= p1.EXPERTS for x in u))
        self.assertAlmostEqual(u[0], p1.EXPERTS * (1 - (1 - 10 / 512) ** 4), places=6)

    def test_tokens_per_pass_matches_target(self):
        self.assertEqual(p1.TARGET["requests"] * p1.TARGET["positions"], 128)


class FitAndValidation(unittest.TestCase):
    def setUp(self):
        self.a, self.b = p1.fit_expert_curve()

    def test_fit_reproduces_anchors(self):
        for n, ms in p1.MEASURED["tp1_gemm_ms"].items():
            union = p1.LAYERS * p1.expert_union_per_layer(n * 4)
            self.assertLess(abs((self.a + self.b * union) - ms) / ms, 0.25,
                            f"anchor N={n} off by too much")

    def test_validation_gate_passes(self):
        _, worst = p1.validate(self.a, self.b)
        self.assertLessEqual(worst, 1.3)


class Prediction(unittest.TestCase):
    def setUp(self):
        self.a, self.b = p1.fit_expert_curve()
        self.pred = p1.predict(self.a, self.b)

    def test_pipelining_never_hurts_and_components_sane(self):
        p = self.pred["per_gpu_pass_ms"]
        self.assertLess(p["pipelined_total"], p["serialized_total"])
        self.assertGreater(p["expert_gemm"], p["index_scan"])
        self.assertGreater(p["kv_miss_link_serialized"], 0)

    def test_memory_fits_reported(self):
        m = self.pred["memory"]
        self.assertTrue(m["gpu_fits"])
        self.assertTrue(m["host_fits"])
        self.assertLess(m["gpu_resident_gib_per_gpu"], 94.2)

    def test_sensitivity_widens_ep_cost(self):
        s = self.pred["sensitivity_p2p_15_gbs"]
        self.assertGreater(s["ep_dispatch_combine_ms"],
                           self.pred["per_gpu_pass_ms"]["ep_dispatch_combine"])


class ArtifactTest(unittest.TestCase):
    def test_report_written_and_valid(self):
        path = ROOT / "artifacts/phase1-20260930/capacity-model.json"
        self.assertTrue(path.exists(), "run phase1_model.py to write the artifact")
        report = json.loads(path.read_text())
        self.assertTrue(report["validation_gate"]["gate_1_3x"])
        self.assertIn("prediction_ep2_32x256k", report)


if __name__ == "__main__":
    unittest.main()
