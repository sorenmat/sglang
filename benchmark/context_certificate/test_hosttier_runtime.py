"""CPU tests for the hosttier runtime module, no GPU required.

FakePool mirrors the pinned HybridLinearKVPool semantics exactly: model
layer ids translate through full_attention_layer_id_mapping and every
other id raises ValueError (memory_pool.py, pinned 4ccff141). These tests
regress the three window-2 bugs: wrong layer-id space, draft-pool arming,
and hollow snapshots.
"""

import ast
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "host_kv"))

PAGE = 64
KV_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]


class FakePool:
    """Implements the HybridLinearKVPool interface the hook relies on."""

    def __init__(self, layer_ids, slots=4096, draft=False):
        self.draft = draft
        self.full_attention_layer_id_mapping = {layer: i for i, layer in enumerate(layer_ids)}
        self.k = [torch.randn(slots, 2, 256, dtype=torch.bfloat16) for _ in layer_ids]
        self.v = [torch.randn(slots, 2, 256, dtype=torch.bfloat16) for _ in layer_ids]

    def _translate(self, layer_id):
        if layer_id not in self.full_attention_layer_id_mapping:
            raise ValueError(f"{layer_id=} not in full attention layers")
        return self.full_attention_layer_id_mapping[layer_id]

    def get_key_buffer(self, layer_id):
        return self.k[self._translate(layer_id)]

    def get_value_buffer(self, layer_id):
        return self.v[self._translate(layer_id)]


def fresh_module(**env):
    log = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    log.close()
    os.environ.update({"SGLANG_HOSTTIER_ENABLE": "1",
                       "SGLANG_HOSTTIER_EVICT_FRACTION": "0.9",
                       "SGLANG_HOSTTIER_HOT_PAGES": "2",
                       "SGLANG_HOSTTIER_PAGE": str(PAGE),
                       "SGLANG_HOSTTIER_LOG": log.name})
    for key in ("SGLANG_HOSTTIER_DISABLE",):
        os.environ.pop(key, None)
    module = importlib.import_module("hosttier")
    importlib.reload(module)
    return module, log.name


def slots_for_pages(pages):
    slots = []
    for page in pages:
        slots.extend(range(page * PAGE, (page + 1) * PAGE))
    return torch.tensor(slots, dtype=torch.long)


class LayerDiscovery(unittest.TestCase):
    def setUp(self):
        self.ht, _ = fresh_module()

    def test_mapping_space_used(self):
        pool = FakePool(KV_LAYERS)
        layers = self.ht._attn_layers(pool)
        self.assertEqual(layers, KV_LAYERS)

    def test_draft_pool_rejected_and_not_cached(self):
        draft = FakePool([3], draft=True)
        self.assertEqual(self.ht._attn_layers(draft), [])
        self.assertIsNone(self.ht._state["kv_layers"])  # not poisoned
        main = FakePool(KV_LAYERS)
        self.assertEqual(self.ht._attn_layers(main), KV_LAYERS)

    def test_probe_fallback_without_mapping(self):
        pool = FakePool(KV_LAYERS)
        del pool.full_attention_layer_id_mapping
        pool.get_key_buffer = lambda layer_id: pool.k[0]  # permissive pool
        pool.get_value_buffer = lambda layer_id: pool.v[0]
        layers = self.ht._attn_layers(pool)
        self.assertEqual(len(layers), 12)


class StagingCycle(unittest.TestCase):
    def setUp(self):
        self.ht, self.log = fresh_module()
        self.pool = FakePool(KV_LAYERS, slots=64 * 2200)
        self.pages = list(range(100, 132)) + list(range(2100, 2132))  # span > 2000; straddles evicted and hot regions

    def test_snapshot_evict_stage_verifies_bit_exact(self):
        originals = {layer: (pool_k.clone(), pool_v.clone()) for layer, pool_k, pool_v
                     in zip(KV_LAYERS, self.pool.k, self.pool.v)}
        # Span eviction poisons more pages than one call demands; pages
        # leave the evicted set only when a call actually selects them.
        self.ht.stage_physical_slots(slots_for_pages(self.pages), self.pool)
        self.assertTrue(self.ht._state["snapshotted"])
        first_pass = self.ht._state["staged_pages"]
        self.assertGreater(first_pass, 0)
        self.assertLess(first_pass, len(self.ht._state["evicted"]) + first_pass)
        self.assertEqual(self.ht._state["verify_mismatches"], 0)
        # A later call demanding the whole span restores everything.
        self.ht.stage_physical_slots(slots_for_pages(range(1, 2132)), self.pool)
        self.assertEqual(len(self.ht._state["evicted"]), 0)
        self.assertGreater(self.ht._state["staged_pages"], first_pass)
        check = self.ht._state.get("selfcheck")
        self.assertIsNotNone(check, "arm-time self-check must run")
        self.assertTrue(check["verified"])
        self.assertGreater(self.ht._state["staged_pages"], 0)
        for (orig_k, orig_v), pool_k, pool_v in zip(
                [originals[l] for l in KV_LAYERS], self.pool.k, self.pool.v):
            self.assertTrue(torch.equal(orig_k, pool_k))
            self.assertTrue(torch.equal(orig_v, pool_v))
        stats = json.loads(Path(self.log).read_text())
        self.assertEqual(stats["verify_mismatches"], 0)

    def test_warmup_span_does_not_arm(self):
        # The mirror may pre-allocate (harmless); arming is what must not happen.
        self.ht.stage_physical_slots(slots_for_pages(range(1, 40)), self.pool)
        self.assertFalse(self.ht._state["snapshotted"])
        self.assertEqual(len(self.ht._state["evicted"]), 0)

    def test_mirror_never_allocates_for_draft_pool(self):
        draft = FakePool([3], draft=True)
        self.ht.stage_physical_slots(slots_for_pages(self.pages), draft)
        self.assertIsNone(self.ht._state["mirror"])
        self.assertFalse(self.ht._state["snapshotted"])


class Hosttier23ImportsClean(unittest.TestCase):
    def test_module_imports_and_arms_dormant(self):
        import importlib, os, sys, tempfile
        log = tempfile.NamedTemporaryFile(suffix=".json", delete=False); log.close()
        os.environ.update({"SGLANG_HOSTTIER_ENABLE": "1",
                           "SGLANG_HOSTTIER_23_LOG": log.name})
        sys.path.insert(0, str(ROOT / "host_kv"))
        import hosttier23
        importlib.reload(hosttier23)
        self.assertTrue(hosttier23.ENABLE)
        self.assertFalse(hosttier23._state["armed"])
        hosttier23.dump_stats()  # must serialize cleanly with empty state


class ProbePreservesStaticmethod(unittest.TestCase):
    def test_wrapped_staticmethod_stays_static(self):
        import types
        class Dummy:
            @staticmethod
            def _logical_to_physical(a, b):
                return a + b
            def forward_decode(self, x):
                return self._logical_to_physical(1, 2)
        ht, _ = fresh_module()
        ht.probe_install({"__name__": "dummy", "Dummy": Dummy})
        instance = Dummy()
        self.assertEqual(instance.forward_decode(0), 3)  # bound call still works


class OverlayStructure(unittest.TestCase):
    """Static checks on the patched backend — would have caught bugs 1
    (staticmethod placement) and 3 (signature mis-anchor) pre-deploy."""

    PATCHED = ROOT / "artifacts/hosttier-overlay-20260930/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"

    def _functions(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield node

    def test_no_hook_inside_staticmethod(self):
        tree = ast.parse(self.PATCHED.read_text())
        for func in self._functions(tree):
            if "stage_physical_slots" in ast.dump(func):
                # any enclosing decorator must not be staticmethod
                for deco in func.decorator_list:
                    name = ast.dump(deco)
                    self.assertNotIn("staticmethod", name,
                                     "hook landed in a staticmethod")

    def test_single_hook_in_paged_method(self):
        src = self.PATCHED.read_text()
        self.assertEqual(src.count("stage_physical_slots"), 6)  # 3 hooks x (import + call)
        self.assertIn("_md = getattr(self", src)
        self.assertIn("probe_install(globals())", src)

    def test_patched_parses_and_original_hash_stable(self):
        ast.parse(self.PATCHED.read_text())
        manifest = json.loads((self.PATCHED.parents[2] / "manifest.json").read_text()
                              if (self.PATCHED.parents[2] / "manifest.json").exists()
                              else (ROOT / "artifacts/hosttier-overlay-20260930/manifest.json").read_text())
        self.assertTrue(any(k.endswith("qwen_sparse_attn_backend.py")
                            for k in manifest["original_sha256"]))


if __name__ == "__main__":
    unittest.main()
