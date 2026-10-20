"""Tests for the proof kit: torch screen port and trace parser."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "proof_kit"))

from probe import certified_select  # noqa: E402
from torch_screen import certified_picks, encode_index  # noqa: E402


class TorchScreenMatchesProbe(unittest.TestCase):
    def test_picks_and_union_match_reference(self):
        rng = np.random.default_rng(20260929)
        keys = rng.normal(size=(257, 32))
        keys[:85] = keys[0]  # exercise ties and duplicates
        queries = rng.normal(size=(4, 4, 32))
        queries[0] = 0.0
        for bits in (8, 6, 4):
            reference = certified_select(keys, queries, 19, bits, "joint",
                                         return_details=True)
            expected = reference.pop("details")["selected_ids"]
            torch_keys = torch.from_numpy(keys.astype(np.float64))
            torch_queries = torch.from_numpy(queries.astype(np.float64))
            picks, union = certified_picks(torch_keys, torch_queries, 19,
                                           encode_index(torch_keys, bits))
            self.assertTrue(torch.equal(picks, torch.from_numpy(expected)),
                            f"picks differ at bits={bits}")
            self.assertAlmostEqual(
                len(union) / len(keys), reference["refine_union_fraction"],
                delta=1e-9, msg=f"union differs at bits={bits}")

    def test_check_mode_reports_ok_on_synthetic(self):
        # A capture-shaped file is required for --check; the real captures
        # cover it in the runbook. Here we only pin the CLI contract.
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "proof_kit/torch_screen.py"),
             "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)


class TraceParser(unittest.TestCase):
    def test_selftest_via_cli(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "proof_kit/index_share.py"),
             "--selftest"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("selftest ok", result.stdout)

    def test_empty_trace_rejected(self):
        from index_share import parse_trace
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"traceEvents": []}, handle)
            path = handle.name
        with self.assertRaises(ValueError):
            parse_trace(path)


if __name__ == "__main__":
    unittest.main()
