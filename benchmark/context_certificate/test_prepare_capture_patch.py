"""Exercise patched call sites on CPU without importing a serving runtime."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from prepare_capture_patch import (
    ACCEPTANCE_HOOK, COMMIT, DEPENDENCY_HASHES, HELPER, INDEXER, PROCESSOR, SOURCE_HASHES,
    check_source_root, patch_source, prepare,
)


REPO = Path(__file__).resolve().parents[2]


def source(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=REPO, text=True)


class CapturePatchChecks(unittest.TestCase):
    def test_standalone_checker_in_shallow_container_path(self):
        # /tmp/check.py has the same parent depth as /capture/check.py.
        with tempfile.NamedTemporaryFile(suffix=".py", dir="/tmp") as script:
            script.write(Path(__file__).with_name("prepare_capture_patch.py").read_bytes())
            script.flush()
            result = subprocess.run([sys.executable, script.name, "--help"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_overlay_pin_and_output_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in SOURCE_HASHES | DEPENDENCY_HASHES:
                dest = root / "original" / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(source(path))
            check_source_root(root / "original")
            manifest = prepare(REPO, root / "overlay")
            self.assertEqual(manifest["source_commit"], COMMIT)
            self.assertEqual(manifest["original_sha256"], SOURCE_HASHES)
            self.assertEqual(manifest["dependency_sha256"], DEPENDENCY_HASHES)
            self.assertEqual(set(manifest["overlay_sha256"]), {INDEXER, PROCESSOR, HELPER})
            self.assertTrue((root / "overlay" / "capture.patch").is_file())
            for path, expected in manifest["overlay_sha256"].items():
                self.assertEqual(hashlib.sha256((root / "overlay" / path).read_bytes()).hexdigest(), expected)
            for path, expected in (SOURCE_HASHES | DEPENDENCY_HASHES).items():
                self.assertEqual(hashlib.sha256((root / "original" / path).read_bytes()).hexdigest(), expected)
            for path in DEPENDENCY_HASHES:
                self.assertFalse((root / "overlay" / path).exists())
            with self.assertRaises(ValueError):
                prepare(REPO, root / "overlay")
            (root / "original" / INDEXER).write_text(source(INDEXER) + "\n")
            with self.assertRaises(ValueError):
                check_source_root(root / "original")
            with self.assertRaises(ValueError):
                patch_source(INDEXER, source(INDEXER) + "\n")
            self.assertIn(HELPER, manifest["overlay_sha256"])

    def test_dependency_changes_and_missing_files_are_rejected(self):
        self.assertFalse(SOURCE_HASHES.keys() & DEPENDENCY_HASHES.keys())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in SOURCE_HASHES | DEPENDENCY_HASHES:
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(source(path))
            check_source_root(root)
            for path in DEPENDENCY_HASHES:
                with self.subTest(path=path):
                    destination = root / path
                    original = destination.read_bytes()
                    destination.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(ValueError, "source revision mismatch"):
                        check_source_root(root)
                    destination.write_bytes(original)
            (root / next(iter(DEPENDENCY_HASHES))).unlink()
            with self.assertRaises(FileNotFoundError):
                check_source_root(root)

    def test_native_selection_unchanged_and_correct_request_mapping(self):
        tree = ast.parse(patch_source(INDEXER, source(INDEXER)))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QSAIndexer")
        function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "select_decode_tokens")
        calls = []
        native = torch.arange(512).expand(4, -1).clone()
        scope = {
            "torch": torch, "_QSA_CAPTURE_ENABLED": True,
            "get_is_capture_mode": lambda: False, "get_world_rank": lambda: 0,
            "qsa_mqa_decode": lambda *args: torch.zeros(4, 512),
            "qsa_fast_topk": lambda *args, **kwargs: native,
            "expand_qsa_block_indices": lambda blocks, *args, **kwargs: blocks + 10,
            "capture_qsa": lambda **kwargs: calls.append(kwargs),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "patched-qsa", "exec"), scope)
        capture = scope["select_decode_tokens"]
        indexer = SimpleNamespace(layer_id=3, block_topk=512, compress_ratio=4,
                                  token_topk=2048, _qsa_capture_is_mtp=False)
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([9, 3]), rids=["request-a", "request-b"],
            forward_mode=SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: True))
        metadata = SimpleNamespace(req_pool_indices=torch.tensor([3, 3, 9, 9]), is_cuda_graph=False)
        q = torch.zeros(4, 8, 128, dtype=torch.bfloat16)
        args = (indexer, q, torch.zeros(64, 8, 1, 128, dtype=torch.bfloat16),
                torch.zeros(4, 64, dtype=torch.int32), torch.tensor([512] * 4),
                512, torch.tensor([2047, 2048, 2047, 2048]), torch.tensor([2049, 2049]))
        original = native.clone()
        result = capture(*args, forward_batch=batch, indexer_metadata=metadata)
        torch.testing.assert_close(result, original + 10)
        torch.testing.assert_close(native, original)
        self.assertEqual(calls[0]["row_request_ids"], ["request-b", "request-b", "request-a", "request-a"])
        self.assertIs(calls[0]["native_selected_ids"], native)
        self.assertEqual(calls[0]["forward_mode"], "target_verify")

        for enabled, rank, mtp in [(False, 0, False), (True, 1, False), (True, 0, True)]:
            calls.clear()
            scope["_QSA_CAPTURE_ENABLED"] = enabled
            scope["get_world_rank"] = lambda: rank
            indexer._qsa_capture_is_mtp = mtp
            torch.testing.assert_close(capture(*args, forward_batch=batch, indexer_metadata=metadata), original + 10)
            self.assertFalse(calls)
        indexer._qsa_capture_is_mtp = False
        scope["get_world_rank"] = lambda: 0
        metadata.is_cuda_graph = True
        with self.assertRaisesRegex(RuntimeError, "eager"):
            capture(*args, forward_batch=batch, indexer_metadata=metadata)

    def test_acceptance_cpu_fields_are_not_mislabeled_as_emission(self):
        patched = patch_source(PROCESSOR, source(PROCESSOR))
        self.assertLess(patched.index("        if (_QSA_CAPTURE_ENABLED and not batch.spec_algorithm"),
                        patched.index("        self.metrics_reporter.num_generated_tokens += len(batch.reqs)"))
        req = SimpleNamespace(rid="r", req_pool_idx=9, spec_verify_ct=3,
                              retraction_count=0, is_retracted=False, finished=lambda: False)
        calls = []
        scope = {
            "_QSA_CAPTURE_ENABLED": True, "get_world_rank": lambda: 0,
            "batch": SimpleNamespace(reqs=[req], forward_iter=11,
                                     spec_algorithm=SimpleNamespace(is_none=lambda: False)),
            "result": SimpleNamespace(num_correct_drafts_per_req_cpu=[3], speculative_num_draft_tokens=4),
            "next_token_ids": [[123]], "capture_acceptance": lambda **kwargs: calls.append(kwargs),
        }
        import textwrap
        exec(compile(textwrap.dedent(ACCEPTANCE_HOOK), "acceptance-hook", "exec"), scope)
        self.assertEqual(calls[0]["accepted_drafts"], [3])
        self.assertEqual(calls[0]["draft_positions"], 4)
        self.assertEqual(calls[0]["per_request_metadata"][0]["grammar_retained_count"], 1)
        self.assertNotIn("emitted_tokens", json.dumps(calls[0]))


if __name__ == "__main__":
    unittest.main()
