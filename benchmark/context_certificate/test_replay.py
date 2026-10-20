"""CPU checks for masking, native-reference disagreements and bad captures."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from probe import certified_select, scores, topk
from replay import load_capture, replay, write_fixture


class ReplayChecks(unittest.TestCase):
    def test_causal_selection_and_capture_contract(self):
        rng = np.random.default_rng(22)
        keys = rng.normal(size=(30, 8))
        queries = rng.normal(size=(3, 4, 8))
        # Invisible future keys have much larger scores; none may win early.
        keys[20:] *= 1000
        valid = np.array([10, 20, 30])
        for mode in ("basic", "joint"):
            result = certified_select(keys, queries, 5, 6, mode,
                                      valid_counts=valid, return_details=True)
            for row, picked in enumerate(result["details"]["selected_ids"]):
                expected = topk(scores(keys[:valid[row]], queries[row:row + 1])[0], 5)
                np.testing.assert_array_equal(picked, expected)
        for bad in (np.array([4, 20, 30]), np.array([10.0, 20.0, 30.0])):
            with self.assertRaises(ValueError):
                certified_select(keys, queries, 5, 6, valid_counts=bad)
        with self.assertRaises(ValueError):
            certified_select(keys * np.nan, queries, 5, 6)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.npz"
            write_fixture(path, key_count=40, positions=3, top_k=8)
            original = {name: value.copy() for name, value in np.load(path, allow_pickle=False).items()}
            report = replay(path, bits=(4, 6, 8), positions=[1, 3])
            self.assertEqual(report["status"], "cpu_replay_checks_passed")
            self.assertEqual(len(report["cases"]), 12)
            self.assertEqual(report["metadata"]["source_kind"], "synthetic")
            for options in ({"bits": []}, {"positions": []}):
                with self.assertRaises(ValueError):
                    replay(path, **options)

            changed = dict(original)
            changed["valid_counts"] = np.array([39, 38, 40], dtype=np.uint64)
            np.savez(path, **changed)
            with self.assertRaises(ValueError):
                load_capture(path)

            # Native output order is not equivalent to membership correctness.
            changed = dict(original)
            changed["native_selected_ids"] = original["native_selected_ids"][:, ::-1]
            np.savez(path, **changed)
            report = replay(path, bits=(8,))
            self.assertEqual(report["status"], "cpu_replay_checks_passed")
            self.assertFalse(report["cases"][0]["native_selected_order_match"])

            # Simulate native numerical differences that escape the CPU bounds.
            changed = dict(original)
            changed["native_scores"] = original["native_scores"] + 1e6
            np.savez(path, **changed)
            self.assertEqual(replay(path, bits=(8,))["status"], "capture_mismatch")

            # Without dense scores, a different native set is still detected.
            changed = dict(original)
            del changed["native_scores"]
            selected = original["native_selected_ids"].copy()
            selected[0, 0] = next(i for i in range(38) if i not in selected[0])
            changed["native_selected_ids"] = selected
            np.savez(path, **changed)
            self.assertEqual(replay(path, bits=(8,))["status"], "capture_mismatch")

            for field, value in (("score_semantics", "different_scorer"),
                                 ("shared_key_bank_verified", False),
                                 ("query_dtype", "float32")):
                changed = dict(original)
                metadata = json.loads(original["metadata"].item())
                metadata[field] = value
                changed["metadata"] = json.dumps(metadata)
                np.savez(path, **changed)
                with self.assertRaises(ValueError):
                    load_capture(path)

            # No object arrays/pickle and no physical/padded/out-of-range IDs.
            changed = dict(original)
            changed["native_selected_ids"] = original["native_selected_ids"].copy()
            changed["native_selected_ids"][0, 0] = 39
            np.savez(path, **changed)
            with self.assertRaises(ValueError):
                load_capture(path)
            changed = dict(original)
            changed["keys"] = np.array([object()], dtype=object)
            np.savez(path, **changed)
            with self.assertRaises(ValueError):
                load_capture(path)


if __name__ == "__main__":
    unittest.main()
