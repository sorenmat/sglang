"""Tests for the next-window prep: overlap analyzer and batch sender selection."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "proof_kit"))

from cross_layer_overlap import overlap, pair_stats  # noqa: E402
from replay import write_fixture  # noqa: E402


class PairStats(unittest.TestCase):
    def test_identity_and_disjoint(self):
        ids = np.arange(10)
        stats = pair_stats(ids, ids)
        self.assertEqual(stats["jaccard"], 1.0)
        self.assertEqual(stats["coverage_of_b_by_a"], 1.0)
        self.assertEqual(stats["overfetch_fraction_of_a"], 0.0)
        disjoint = pair_stats(np.arange(10), np.arange(10, 20))
        self.assertEqual(disjoint["jaccard"], 0.0)
        self.assertEqual(disjoint["coverage_of_b_by_a"], 0.0)
        self.assertEqual(disjoint["overfetch_fraction_of_a"], 1.0)


class OverlapAnalyzer(unittest.TestCase):
    def _capture(self, path, layer, seed, request="synthetic-0"):
        write_fixture(path, seed=seed)
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in ("metadata", "query_positions",
                                            "native_selected_ids")}
        meta = json.loads(data["metadata"].item())
        meta["layer_id"] = layer
        meta["request_id"] = request
        np.savez_compressed(path, metadata=json.dumps(meta),
                            query_positions=data["query_positions"],
                            native_selected_ids=data["native_selected_ids"],
                            keys=np.zeros((2, 128), dtype=np.float32),
                            queries=np.zeros((1, 4, 128), dtype=np.float32),
                            valid_counts=np.array([2]))
        return path

    def test_same_request_different_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            early = self._capture(Path(tmp) / "a.npz", 3, seed=1)
            late = self._capture(Path(tmp) / "b.npz", 7, seed=2)
            report = overlap(early, late)
            self.assertEqual(report["early_layer"], 3)
            self.assertEqual(report["late_layer"], 7)
            self.assertGreaterEqual(report["per_position_mean"]["jaccard"], 0.0)
            self.assertLessEqual(report["per_position_mean"]["jaccard"], 1.0)
            # identical picks must give full coverage and zero overfetch
            same = overlap(self._capture(Path(tmp) / "c.npz", 3, seed=5),
                           self._capture(Path(tmp) / "d.npz", 9, seed=5))
            self.assertEqual(same["per_position_mean"]["coverage_of_b_by_a"], 1.0)
            self.assertEqual(same["per_position_mean"]["overfetch_fraction_of_a"], 0.0)

    def test_rejects_same_layer_and_cross_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self._capture(Path(tmp) / "a.npz", 3, seed=1)
            b = self._capture(Path(tmp) / "b.npz", 3, seed=2)
            with self.assertRaises(ValueError):
                overlap(a, b)
            c = self._capture(Path(tmp) / "c.npz", 7, seed=1, request="other")
            with self.assertRaises(ValueError):
                overlap(a, c)


class RouteCollect(unittest.TestCase):
    def test_selection_uses_band(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "route_collect", ROOT / "serverless" / "route_collect.py")
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)
        rows = rc.select_requests(ROOT / "artifacts" / "workload.jsonl",
                                  4, 25_000, 40_000)
        self.assertEqual(len(rows), 4)

    def test_union_curve_growth(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "route_collect", ROOT / "serverless" / "route_collect.py")
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)
        curve = rc.union_curve([{1, 2}, {2, 3}, {3, 4}])
        self.assertEqual(curve, [2, 3, 4])

    def test_extract_falls_back_cleanly(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "route_collect", ROOT / "serverless" / "route_collect.py")
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)
        self.assertIsNone(rc.extract_experts({"choices": [{}]}))
        self.assertEqual(rc.extract_experts(
            {"choices": [{"routed_experts": [[1, 2]]}]}), [[1, 2]])


class BatchSelection(unittest.TestCase):
    def test_selects_distinct_requests_in_band(self):
        from batch_profile_send import select_requests
        workload = ROOT / "artifacts" / "workload.jsonl"
        rows = select_requests(workload, 4, 25_000, 40_000)
        self.assertEqual(len(rows), 4)
        hashes = {r["request_sha256"] for r in rows}
        self.assertEqual(len(hashes), 4)
        for row in rows:
            self.assertGreaterEqual(row["observed_prompt_tokens"], 25_000)
            self.assertLessEqual(row["observed_prompt_tokens"], 40_000)
        with self.assertRaises(SystemExit):
            select_requests(workload, 100, 25_000, 40_000)


if __name__ == "__main__":
    unittest.main()
